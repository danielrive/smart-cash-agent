from typing import Annotated, TypedDict, Literal, Union
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field,field_validator
from app.agent.state import AgentState, AgentStep  # Ensure this supports messages: Annotated[list[BaseMessage], add_messages]

# Tools
from app.tools import SearchTool
from app.tools.currency_tool import convert_currency

# Define the LLM
llm = init_chat_model(
    "us.amazon.nova-premier-v1:0",
    model_provider="bedrock_converse"
)

#### Define General Prompt
# Raw string, not need vars, it is static
SYSTEM_PROMPT = """
You are a precise assistant that MAY use tools (e.g., web_search). IMPORTANT RULES:
- If the user asks to provide a number (counts, how many, etc.), return ONLY the number as digits (e.g. `3`) and nothing else.
- Do not include explanations, context, or verbose text when the question expects a single number.
- Remove any internal reasoning/chain-of-thought from the final output.
- If you need to look up facts, call web_search (you may call it multiple times).
- Prefer low-variance answers: when possible use site:en.wikipedia.org if the user explicitly asked for Wikipedia.
"""

## Analysis prompt

analysis_prompt = ChatPromptTemplate.from_template("""
Analyze this question: {question}

Classify the type (factual, statistical, comparative, causal, direct_answer).
Select tools if needed (web_search for facts/searches,wikipedia_search, currency_converter for conversions, none).
Refine the question for better tool accuracy (make specific, add domain if helpful, keep concise).

Output exactly in this format:
TYPE: <type>
SELECTED_TOOLS: <tool1,tool2 or none>
redefined_question: <refined question or original>
REASON: <brief explanation>
""")

def init_agent_state(question: str) -> AgentState:
    return AgentState(
        question=question,
        messages=[],
        selected_tools=[],
        current_step=AgentStep.ANALYZE_QUESTION.value,
        final_answer="",
        input_file=None,
        question_type="",
        redefined_question="",
        reason=""
    )

# Tools
@tool
def web_search(query: str) -> str:
    """Search the web for information to answer questions."""
    return SearchTool().run(query)

@tool
def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert currency from one type to another."""
    return convert_currency(amount, from_currency, to_currency)

tools = [web_search, currency_converter]
llm_with_tools = llm.bind_tools(tools)

# Pydantic Model for Structured Output
class AnalysisOutput(BaseModel):
    question_type: str = Field(..., enum=["factual", "statistical", "comparative", "causal", "direct_answer"])
    selected_tools: list[str] = Field(..., enum=["web_search", "currency_converter"])
    redefined_question: str
    reason: str

    @field_validator("selected_tools", mode="before")
    def ensure_list(cls, v):
        if isinstance(v, str):
            return [v]
        return v

def analyze_question_node(state: AgentState) -> AgentState:
    """Analyze the question to determine next steps."""
    try:
        analysis_response = analysis_prompt | llm.with_structured_output(AnalysisOutput) 
        results = analysis_response.invoke({"question": state["question"]})

        ## Update state with new info
        state["question_type"] = results.question_type
        state["selected_tools"] = results.selected_tools
        state["redefined_question"] = results.redefined_question
        state["messages"].append(AIMessage(content=f"Analysis: {results.model_dump()}"))
        state["current_step"] = AgentStep.SELECT_TOOLS.value
    
    except Exception as e:
        state["messages"].append(AIMessage(content=f"Error during analysis: {str(e)}"))
        state["current_step"] = AgentStep.ERROR_RECOVERY.value

    return state

## Select tool node

def select_tools_node(state: AgentState) -> AgentState:
    """Select tools based on the analysis."""
    state["current_step"] = AgentStep.SELECT_TOOLS.value
    if state["selected_tools"]:
        state["messages"].append(AIMessage(content=f"Selected tools: {', '.join(state['selected_tools'])} for query: {state['redefined_question']}"))
    else:
        state["messages"].append(AIMessage(content="No tools needed; proceeding to synthesis."))
    return state
    

## Execute tools node

def execute_tools_node(state: AgentState) -> AgentState:
    """Execute selected tools (AgentStep.EXECUTE_TOOLS)."""
    state["current_step"] = AgentStep.EXECUTE_TOOLS.value
    tool_node = ToolNode(tools)
    
    try:
        tool_calls = []
        for tool_name in state["selected_tools"]:
            if tool_name == "web_search":
                tool_calls.append({
                    "name": "web_search",
                    "args": {"query": state["redefined_question"]},
                    "id": f"call_{tool_name}",
                    "type": "tool"
                })
            elif tool_name == "currency_converter":
                # Parse currency query (simplified; improve for robustness)
                import re
                match = re.search(r'(\d+\.?\d*)\s*(\w+)\s*to\s*(\w+)', state["redefined_question"], re.IGNORECASE)
                if match:
                    amount, from_curr, to_curr = float(match.group(1)), match.group(2).upper(), match.group(3).upper()
                    tool_calls.append({
                        "name": "currency_converter",
                        "args": {"amount": amount, "from_currency": from_curr, "to_currency": to_curr},
                        "id": f"call_{tool_name}",
                        "type": "tool"
                    })
        
        # Execute tools
        tool_results = tool_node.invoke({"messages": state["messages"] + [AIMessage(tool_calls=tool_calls,content="")]})
        state["messages"].extend(tool_results["messages"])
        
    except Exception as e:
        state["messages"].append(ToolMessage(content=f"Tool execution error: {e}", tool_call_id="error", name="error"))
        state["reason"] += f" (Tool error: {e})"
    
    return state

def synthesize_answer_node(state: AgentState) -> AgentState:
    """Placeholder for synthesis (AgentStep.SYNTHESIZE_ANSWER)."""
    state["current_step"] = AgentStep.SYNTHESIZE_ANSWER.value
    state["messages"].append(AIMessage(content="Synthesizing answer from analysis (placeholder)."))
    return state

### Routes

def route_after_select(state: AgentState) -> Literal["execute_tools", "synthesize_answer"]:
    """Route after tool selection."""
    return "execute_tools" if state["selected_tools"] else "synthesize_answer"

def should_continue(state: AgentState) -> Literal["execute_tools", "synthesize_answer"]:
    """Route after synthesis (placeholder)."""
    return "synthesize_answer"  # For now, always end loop

### Building the Graph with the current components

graph_builder = StateGraph(AgentState)
# Register the node for analyze question
graph_builder.add_node(AgentStep.ANALYZE_QUESTION.value, analyze_question_node)
# Register node for select tools
graph_builder.add_node(AgentStep.SELECT_TOOLS.value, select_tools_node)
graph_builder.add_node(AgentStep.EXECUTE_TOOLS.value, execute_tools_node)
graph_builder.add_node(AgentStep.SYNTHESIZE_ANSWER.value, synthesize_answer_node)



# Create edge 
graph_builder.add_edge(START, AgentStep.ANALYZE_QUESTION.value)
graph_builder.add_edge(AgentStep.ANALYZE_QUESTION.value, AgentStep.SELECT_TOOLS.value)
graph_builder.add_conditional_edges(AgentStep.SELECT_TOOLS.value, route_after_select, {
    "execute_tools": AgentStep.EXECUTE_TOOLS.value,
    "synthesize_answer": AgentStep.SYNTHESIZE_ANSWER.value
})
graph_builder.add_edge(AgentStep.EXECUTE_TOOLS.value, AgentStep.SYNTHESIZE_ANSWER.value)
graph_builder.add_edge(AgentStep.SYNTHESIZE_ANSWER.value, END)

graph = graph_builder.compile()


def run_agent(question: str) -> AgentState:
    state = init_agent_state(question)
    result = graph.invoke(state)
    print("Result State:", result)
    print(f"Question Type: {result['question_type']}")
    print(f"Selected Tools: {result['selected_tools']}")
    print(f"Refined Question: {result['redefined_question']}")
    print(f"Messages: {[m.content for m in result['messages']]}")
    return result