from typing import Annotated, Literal, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator
from app.agent.state import AgentState, AgentStep
import logging
import re

# Tools
from app.tools import SearchTool
from app.tools.currency_tool import convert_currency

logger = logging.getLogger(__name__)

# Define the LLM
llm = init_chat_model(
    "us.amazon.nova-premier-v1:0",
    model_provider="bedrock_converse"
)

## Analysis prompt
analysis_prompt = ChatPromptTemplate.from_template("""
You are analyzing a question to determine the approach needed to answer it accurately.

Question: {question}

Your analysis:

1. **Classify the question type:** factual, statistical, comparative, causal, mathematical, or currency_convert

2. **Decide if tools are needed** (Critical decision):
   
   Ask yourself: "Am I CERTAIN of the answer from general knowledge?"
   
   NO tools needed if:
   - Common knowledge facts you're 100% certain about
     Examples: "What is the capital of France?" → You know it's Paris
               "What is 2+2?" → You can calculate: 4
   - Standard definitions, meanings, well-known history
   - Simple calculations you can do mentally
   
   USE web_search if:
   - You need to VERIFY specific information you're uncertain about
   - Questions about specific counts, numbers, or dates
   - Questions about specific people/events where precision matters
   - Questions explicitly asking for Wikipedia or verified information
     Examples: "How many albums did Mercedes Sosa release 2000-2009?" → You don't know this exact count, needs search
               "Who won the Nobel Prize in Physics in 2023?" → Recent specific data, needs search
   
   USE currency_converter if:
   - Explicit currency conversion requested
     Example: "Convert 100 USD to EUR"

3. **Refine the question:**
   - If tools needed: Make it specific and search-optimized
   - If complex: Break into clear sub-questions
   - Otherwise: Keep as-is or clean up slightly

4. **Provide brief reasoning**

Key principle: Use tools when you need verification or don't have certain knowledge. Don't use tools for information you confidently know.
""")


def init_agent_state(question: str) -> AgentState:
    logger.info(f"Initializing agent state for question: {question}")
    return AgentState(
        question=question,
        messages=[HumanMessage(content=question)],
        selected_tools=[],
        current_step=AgentStep.ANALYZE_QUESTION.value,
        final_answer="",
        input_file=None,
        question_type="",
        redefined_question="",
        reason="",
        tool_loop_count = 0,
        last_tool_results=[]
    )

# Tools
@tool
def web_search(question: str, question_type: str) -> str:
    """Search the web for information to answer questions."""
    return SearchTool().run(question,question_type)

@tool
def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert currency from one type to another."""
    return convert_currency(amount, from_currency, to_currency)

tools = [web_search, currency_converter]
llm_with_tools = llm.bind_tools(tools)

# Pydantic Model for Structured Output
class AnalysisOutput(BaseModel):
    question_type: Literal["factual", "statistical", "comparative", "causal", "direct_answer", "mathematical", "currency_convert"]
    selected_tools: Optional[list[str]] = Field(default=None)
    redefined_question: str
    reason: str

    @field_validator("selected_tools", mode="before")
    @classmethod
    def normalize_tools(cls, v):
        """Ensure selected_tools is always None or a list of strings."""
        if v in (None, "None", "", [], {}):
            return None
        if isinstance(v, str):
            v = v.strip()
            # Sometimes the model returns 'web_search, currency_converter'
            if "," in v:
                return [x.strip() for x in v.split(",") if x.strip()]
            # If it returns a single tool as string
            if v in ["web_search", "currency_converter"]:
                return [v]
            # If it returned something nonsensical, drop it
            return None
        if isinstance(v, list):
            return [str(x).strip() for x in v if x]
        return None

def analyze_question_node(state: AgentState) -> AgentState:
    """Analyze the question to determine next steps."""
    try:
        logger.info(f"Analyzing question: {state['question']}")
        analysis_response = analysis_prompt | llm.with_structured_output(AnalysisOutput)
        results = analysis_response.invoke({"question": state["question"]})
        logger.info(f"Analysis RAW response {results}") 

        ## Update state with new info
        state["question_type"] = results.question_type
        state["selected_tools"] = results.selected_tools
        state["redefined_question"] = results.redefined_question
        state["messages"].append(AIMessage(content=f"Analysis: {results.model_dump()}"))
        state["current_step"] = AgentStep.SELECT_TOOLS.value
    
    except Exception as e:
        logger.exception("Error during analysis")
        state["messages"].append(AIMessage(content=f"Error during analysis: {str(e)}"))
        state["current_step"] = AgentStep.ERROR_RECOVERY.value

    return state

## Select tool node
def select_tools_node(state: AgentState) -> AgentState:
    """Select tools based on the analysis."""
    state["current_step"] = AgentStep.EXECUTE_TOOLS.value
    if state["selected_tools"]:
        state["messages"].append(AIMessage(content=f"Selected tools: {', '.join(state['selected_tools'])} for question: {state['redefined_question']}"))
        logger.info(f"tools selected {state['selected_tools']}")
    else:
        state["messages"].append(AIMessage(content="No tools needed; proceeding to synthesis."))
        logger.info("No tools selected")
    return state
    

## Execute tools node
def execute_tools_node(state: AgentState) -> AgentState:
    """Execute selected tools (AgentStep.EXECUTE_TOOLS)."""
    state["current_step"] = AgentStep.EXECUTE_TOOLS.value
    tool_node = ToolNode(tools)
    logger.info("Executing tools")

    try:
        tool_calls = []
        for tool_name in state["selected_tools"]:
            if tool_name == "web_search":
                logger.info("Executing web_search tool")
                tool_calls.append({
                    "name": "web_search",
                    "args": {
                        "question": state["redefined_question"],
                        "question_type": state["question_type"]
                    },
                    "id": f"call_{tool_name}",
                    "type": "tool_call",
                })
            elif tool_name == "currency_converter":
                logger.info("Executing currency_converter tool")
                # Better regex to handle more patterns
                # Matches: "100 USD to EUR", "convert 100 USD to EUR", "100 dollars in euros"
                pattern = r'(?:convert\s+)?(\d+(?:\.\d+)?)\s*(\w+)\s*(?:to|in|into|→)\s*(\w+)'
                match = re.search(pattern, state["redefined_question"], re.IGNORECASE)
                if match:
                    amount, from_curr, to_curr = float(match.group(1)), match.group(2).upper(), match.group(3).upper()
                    logger.debug(f"Currency conversion: {amount} {from_curr} → {to_curr}")
                    tool_calls.append({
                        "name": "currency_converter",
                        "args": {
                            "amount": amount,
                            "from_currency": from_curr,
                            "to_currency": to_curr,
                        },
                        "id": f"call_{tool_name}",
                        "type": "tool_call",
                    })
                else:
                    logger.warning(f"Failed to parse currency conversion from: {state['redefined_question']}")

        if not tool_calls:
            logger.info("No tool calls to execute.")
            return state

        # Send the tool call to the model
        ai_tool_message = AIMessage(content="", tool_calls=tool_calls)
        state["messages"].append(ai_tool_message)

        # Run the tools
        tool_results = tool_node.invoke({"messages": state["messages"]})
        new_tool_messages = [m for m in tool_results["messages"] if isinstance(m, ToolMessage)]

        logger.debug(f"Raw tool response: {[m.content for m in new_tool_messages]}")
        logger.info(f"Executed {len(new_tool_messages)} tools: {[m.name for m in new_tool_messages]}")

        # Format tool results for Bedrock Converse API
        tool_result_messages = [format_bedrock_tool_result(m) for m in new_tool_messages]
        
        logger.debug(f"Formatted {len(tool_result_messages)} tool results for Bedrock")
        if tool_result_messages and logger.isEnabledFor(logging.DEBUG):
            import json
            logger.debug("First tool result:\n%s", json.dumps(tool_result_messages[0], indent=2))

        # Update  state
        state["messages"].extend(tool_result_messages)
        state["last_tool_results"] = [
            json.dumps(m["content"][0]["toolResult"]["content"][0]["json"], ensure_ascii=False)
            for m in tool_result_messages
        ]

    except Exception as e:
        logger.exception("Error executing tools")
        state["messages"].append(
            ToolMessage(content=f"Tool execution error: {e}", tool_call_id="error", name="error")
        )
        state["reason"] += f" (Tool error: {e})"

    return state



def clean_model_output(text: str) -> str:
    """Remove thinking tags from model output."""
    return re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)


def format_bedrock_tool_result(tool_message: ToolMessage) -> dict:
    """
    Format a LangChain ToolMessage into Bedrock Converse API format.
    
    Follows AWS Bedrock documentation format:
    https://docs.aws.amazon.com/bedrock/latest/userguide/tool-use-inference-call.html
    
    Expected format:
    {
        "role": "user",
        "content": [{
            "toolResult": {
                "toolUseId": "tooluse_xxx",
                "content": [{"json": {...}}],
                "status": "success"
            }
        }]
    }
    
    Args:
        tool_message: ToolMessage from LangChain tool execution
    
    Returns:
        Dict formatted for Bedrock Converse API (Step 3 in documentation)
    """
    import json
    
    # Extract toolUseId - must match the original request ID
    # LangChain stores this in different attributes depending on version
    tool_use_id = (
        getattr(tool_message, "tool_call_id", None)
        or getattr(tool_message, "tool_use_id", None)
        or (tool_message.additional_kwargs.get("tool_call_id") if hasattr(tool_message, "additional_kwargs") else None)
        or f"auto_{tool_message.name}"
    )
    
    # Format content - Bedrock expects array with json or text objects
    raw_content = tool_message.content
    
    if isinstance(raw_content, str):
        try:
            # Try parsing as JSON first (preferred format)
            content_obj = json.loads(raw_content)
        except json.JSONDecodeError:
            # Fall back to text format
            content_obj = {"text": raw_content}
    elif isinstance(raw_content, dict):
        content_obj = raw_content
    elif isinstance(raw_content, list) and raw_content:
        # Take first dict if available
        content_obj = raw_content[0] if isinstance(raw_content[0], dict) else {"text": str(raw_content)}
    else:
        content_obj = {"text": str(raw_content)}
    
    # Build Bedrock ToolResult message (matches documentation format)
    return {
        "role": "user",
        "content": [{
            "toolResult": {
                "toolUseId": tool_use_id,
                "content": [{"json": content_obj}],
                "status": "success",  # Use "error" for failures
            }
        }],
    }

def synthesize_answer_node(state: AgentState) -> AgentState:
    """Synthesize answer from tool results (AgentStep.SYNTHESIZE_ANSWER)."""
    state["current_step"] = AgentStep.SYNTHESIZE_ANSWER.value
    logger.info("Synthesizing answer")
    logger.debug(f"STATE BEFORE synthesis: {[m.content for m in state['messages']]}")
    try:
        synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a precise assistant producing final answers for the GAIA benchmark.

You will receive:
- Original question
- Question type (factual, statistical, mathematical, etc.)
- Tool results (if tools were used)
- Conversation history

**CRITICAL ANSWER FORMAT RULES:**

1. For NUMERIC questions (statistical, mathematical, or "how many"):
   → Output ONLY the number as digits: 42
   → NO units, NO words, NO punctuation: "42 albums" "The answer is 42"
   → Just the number: 42

2. For FACTUAL questions (names, places, things):
   → Output only the fact, no extra words: Paris (not "The capital is Paris")
   → Keep it minimal: 1-5 words maximum

3. For COMPARATIVE or CAUSAL questions:
   → One concise sentence maximum
   → Example: "Because of Rayleigh scattering of sunlight in the atmosphere"

4. If tool results are insufficient:
   → Call the tools again with a refined question
   → Use your tool calling capability

**FORBIDDEN in output:**
 "Answer:", "Final Answer:", "<answer>" or any labels
 XML, HTML, JSON formatting
 Markdown or code blocks
 Repeating the question
 Quotes around answers
 <thinking> tags or reasoning steps
 Explanations after the answer

**Valid Examples:**
Q: "What is the capital of France?" → Paris
Q: "How many moons does Mars have?" → 2  
Q: "How many studio albums by X in 2000-2009?" → 5
Q: "Why is the sky blue?" → Because of Rayleigh scattering of sunlight in the atmosphere

---

Question: {question}
Question Type: {question_type}
Tool Results: {tool_results}
"""),
    MessagesPlaceholder("messages"),
])

        tool_results = "\n".join(state.get("last_tool_results", []))
        logger.info(f"Tool results: {tool_results[:200]}...")  # Log first 200 chars
        logger.debug(f"Full tool results: {tool_results}")

        synthesis_chain = synthesis_prompt | llm_with_tools
        response = synthesis_chain.invoke({
            "question": state["question"],
            "question_type": state["question_type"],
            "tool_results": tool_results,
            "messages": state["messages"],
        })
        logger.debug(f"Synthesis response: {response.content}")
        logger.debug(f"STATE AFTER synthesis: {len(state['messages'])} messages")
        if response.tool_calls:
            logger.info("Additional tool calls requested")
            state["messages"].append(AIMessage(content="", tool_calls=response.tool_calls))
            state["selected_tools"] = [call["name"] for call in response.tool_calls]
            state["redefined_question"] = response.tool_calls[0]["args"].get("question", state["redefined_question"])
            state["reason"] = state.get("reason", "") + " (Additional tool calls requested)"
        else:
            logger.debug(f"response {response.content}")
            answer = clean_model_output(response.content)
            state["final_answer"] = answer
            state["messages"].append(AIMessage(content=answer))
            logger.info("Final answer synthesized")
            state["selected_tools"] = []
    
    except Exception as e:
        state["messages"].append(AIMessage(content=f"Synthesis error: {e}"))
        state["final_answer"] = "Error occurred during synthesis."
        state["reason"] = state.get("reason", "") + f" (Synthesis error: {e})"
        logger.exception("Synthesis error")
    
    return state

def complete_node(state: AgentState) -> AgentState:
    """Format final answer (AgentStep.COMPLETE)."""
    state["current_step"] = AgentStep.COMPLETE.value
    state["messages"].append(AIMessage(content=f"Final answer: {state['final_answer']}"))
    logger.info(f"Agent complete. Final answer: {state['final_answer']}")
    return state

def route_after_select(state: AgentState) -> Literal["execute_tools", "synthesize_answer"]:
    """
    After the SELECT_TOOLS node:
    - If tools were selected → go execute them
    - Otherwise → skip straight to synthesis
    """
    return "execute_tools" if state.get("selected_tools") else "synthesize_answer"


def route_after_synthesis(state: AgentState) -> Literal["execute_tools", "complete"]:
    """
    After SYNTHESIZE_ANSWER:
    - If new tool calls were requested by the model → go execute them again
    - Otherwise → finish (complete)
    - Includes a safety cap on looping to avoid infinite tool calls
    """
    max_loops = 3
    loop_count = state.get("tool_loop_count", 0)

    if loop_count >= max_loops:
        logger.warning("Tool loop limit reached, stopping.")
        return "complete"

    # Increment loop counter for the next cycle
    state["tool_loop_count"] = loop_count + 1

    if state.get("selected_tools"):
        logger.info(f"Model requested new tool calls (loop #{state['tool_loop_count']})")
        return "execute_tools"

    return "complete"


### --- Build the Graph ---

graph_builder = StateGraph(AgentState)

# Register nodes
graph_builder.add_node(AgentStep.ANALYZE_QUESTION.value, analyze_question_node)
graph_builder.add_node(AgentStep.SELECT_TOOLS.value, select_tools_node)
graph_builder.add_node(AgentStep.EXECUTE_TOOLS.value, execute_tools_node)
graph_builder.add_node(AgentStep.SYNTHESIZE_ANSWER.value, synthesize_answer_node)
graph_builder.add_node(AgentStep.COMPLETE.value, complete_node)

# Define routes
graph_builder.add_edge(START, AgentStep.ANALYZE_QUESTION.value)
graph_builder.add_edge(AgentStep.ANALYZE_QUESTION.value, AgentStep.SELECT_TOOLS.value)

# Conditional edge after tool selection
graph_builder.add_conditional_edges(
    AgentStep.SELECT_TOOLS.value,
    route_after_select,
    {
        "execute_tools": AgentStep.EXECUTE_TOOLS.value,
        "synthesize_answer": AgentStep.SYNTHESIZE_ANSWER.value,
    },
)

# Standard flow from execution to synthesis
graph_builder.add_edge(AgentStep.EXECUTE_TOOLS.value, AgentStep.SYNTHESIZE_ANSWER.value)

# Conditional edge after synthesis (the missing link!)
graph_builder.add_conditional_edges(
    AgentStep.SYNTHESIZE_ANSWER.value,
    route_after_synthesis,
    {
        "execute_tools": AgentStep.EXECUTE_TOOLS.value,
        "complete": AgentStep.COMPLETE.value,
    },
)

# End after completion
graph_builder.add_edge(AgentStep.COMPLETE.value, END)

# Compile
graph = graph_builder.compile()

def run_agent(question: str) -> AgentState:
    state = init_agent_state(question)
    result = graph.invoke(state)

    print("Result State:", result)
    print(f"Question Type: {result['question_type']}")
    print(f"Reason: {result['reason']}")
    print(f"Selected Tools: {result['selected_tools']}")
    print(f"Refined Question: {result['redefined_question']}")
    print(f"Messages: {[m.content for m in result['messages']]}")
    print(f"Final Answer: {result['final_answer']}")
    logger.info(f"Final Answer: {result['final_answer']}")
    return result

def draw_graph():
    from IPython.display import Image, display
    display(Image(graph.get_graph().draw_mermaid_png()))
