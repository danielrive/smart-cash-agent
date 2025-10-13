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

#### Define General Prompt
SYSTEM_PROMPT = """
You are a precise assistant that CAN use tools (e.g., web_search, currency_converter).

Strict Rules:
- If the user asks for a number (counts, how many, statistics, math results, etc.):
  - Output ONLY the number as digits (e.g., `3`), with no words, units, or explanations.
- If question can not be answered directly you can use lookups:
  - If needed, call web_search (can be called multiple times).
  - If user explicitly mentions Wikipedia, prefer `site:en.wikipedia.org`.
- For currency conversion:
  - Always call currency_converter instead of trying to calculate yourself.
- For direct factual or simple questions:
  - Provide the answer directly if you are confident, without calling tools unnecessarily.
- Never include <thinking>, reasoning steps, or verbose explanations in the final output.
- Never prepend "Answer:" or any formatting — only output the final response.
"""

## Analysis prompt
analysis_prompt = ChatPromptTemplate.from_template("""
You are a precise assistant. Your task is to analyze this question: {question}

Follow these rules carefully:

1. **Classify** the question into one of the following types:
   - factual
   - statistical
   - comparative
   - causal
   - mathematical
   - currency_convert

2. **Decide if the question should be split into multiple sub-questions** to produce a complete and accurate answer.
   - Split it only if the question is complex, has multiple parts, or requires information from different sources.
   - If splitting is needed, briefly rewrite it into clear, concise sub-questions under `redefined_question`.
   - Each sub-question should focus on one distinct factual or analytical point.
   - Note: Splitting a question does **not** automatically mean that external tools are required.

3. **Decide if tools are needed:**
   - If you can confidently answer the question directly from general or common knowledge,
     set `SELECTED_TOOLS: None`.
   - Only use tools when **external, current, or numerical data** is required.

4. **Tool selection rules:**
   - `web_search` → Use ONLY if the question needs **real-time**, **recent**, or **specific factual data** that is not general world knowledge.
     (Example: “current GDP of Japan”, “population of Paris in 2024”)
   - `currency_converter` → Use ONLY for currency conversions.

5. **Factual clarification:**
   - For standard factual questions (like definitions, capitals, meanings, or antonyms), 
     these are considered **directly answerable** — do **not** use any tools.

6. **Refine the question** (only if tools are needed):
   - Make it concise and specific.
   - Add context if useful.
   - Remove unnecessary words.

---

**Output Rules (mandatory):**

Output **only** in the following exact format.  
Do NOT include any reasoning, steps, or text outside this format.

If no tools are needed:
TYPE: <factual | statistical | comparative | causal | mathematical | currency_convert>
SELECTED_TOOLS: None
redefined_question: <refined or original question, or list of sub-questions if split>
REASON: <brief explanation>

If tools are needed:
TYPE: <factual | statistical | comparative | causal | mathematical | currency_convert>
SELECTED_TOOLS: <comma-separated list of tools, e.g. web_search or currency_converter>
redefined_question: <refined question or structured list of sub-questions>
REASON: <brief explanation of why these tools are needed>

Important:
- Never include a tool “just in case.”
- Factual ≠ automatic tool use.
- Only use tools if **information retrieval or live data** is truly required.
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
        logger.info(f"tools selected {state["selected_tools"]}")
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
                import re
                match = re.search(r'(\d+\.?\d*)\s*(\w+)\s*to\s*(\w+)', state["redefined_question"], re.IGNORECASE)
                if match:
                    amount, from_curr, to_curr = float(match.group(1)), match.group(2).upper(), match.group(3).upper()
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

        tool_result_messages = []
        import json

        for m in new_tool_messages:
            # ✅ Extract tool_use_id safely (Bedrock expects this to match the original toolUseId)
            tool_use_id = (
                getattr(m, "tool_call_id", None)
                or getattr(m, "tool_use_id", None)
                or (m.additional_kwargs.get("tool_call_id") if hasattr(m, "additional_kwargs") else None)
                or f"auto_{m.name}"
            )

            # ✅ Normalize tool output to valid Bedrock ToolResult content
            raw_content = m.content
            content_obj = None

            if isinstance(raw_content, str):
                try:
                    content_obj = json.loads(raw_content)
                except json.JSONDecodeError:
                    content_obj = {"text": raw_content}

            elif isinstance(raw_content, list):
                # If list of dicts → pick first (Bedrock only allows one JSON object)
                if len(raw_content) > 0 and isinstance(raw_content[0], dict):
                    content_obj = raw_content[0]
                else:
                    content_obj = {"text": str(raw_content)}

            elif isinstance(raw_content, dict):
                content_obj = raw_content

            else:
                content_obj = {"text": str(raw_content)}

            # ✅ Build valid Bedrock-style ToolResult message
            tool_result_message = {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": tool_use_id,
                            "content": [{"json": content_obj}],
                            "status": "success",
                        }
                    }
                ],
            }

            logger.debug("ToolResult message to Bedrock:\n%s", json.dumps(tool_result_message, indent=2))
            tool_result_messages.append(tool_result_message)

        # ✅ Update conversation state
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
    return re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)

def synthesize_answer_node(state: AgentState) -> AgentState:
    """Synthesize answer from tool results (AgentStep.SYNTHESIZE_ANSWER)."""
    state["current_step"] = AgentStep.SYNTHESIZE_ANSWER.value
    logger.info("Synthesizing answer node")
    logger.debug(f"STATE BEFORE  CALLS LLM to sinthesize: {[m.content for m in state['messages']]}")
    try:
        synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a precise assistant. 
You will receive a user question, the classified question type, and the results from tools (if any).

Your job is to produce the **final answer only**, with no formatting or labels.

Strict Rules:
- For numeric questions (question_type: statistical, mathematical, or factual expecting a number):
  - Output ONLY the number as digits (e.g., 42), with no units, symbols, or words.
- For direct factual questions (e.g., "What is the capital of France?"):
  - Output only the fact (e.g., Paris).
- For comparative or causal questions:
  - Provide a concise natural-language answer — one sentence at most, no explanations.
- If tool results are insufficient or unclear rephrase the question and call the tools again.
- Never include "Answer:", "Final Answer:", "<answer>", markdown, or any other label.
- Never output XML, HTML, or JSON.
- Never repeat or restate the question.
- Never use punctuation or quotes around numeric answers.
- Never output reasoning or hidden text.

**Output Format (MANDATORY):**
Your entire output must be **only one of**:
1. A single number (e.g., 42)
2. A single short factual string (1–5 words)
3. A concise sentence (for comparative/causal)
4. The exact string: NEED_MORE_TOOL_CALLS

Anything else (including "Final Answer:", tags, or markup) is considered invalid.

Examples:
Q: "What is the capital of France?" → Paris
Q: "How many moons does Mars have?" → 2
Q: "Why is the sky blue?" → Because of Rayleigh scattering of sunlight in the atmosphere.

Inputs:
Question: {question}
Question Type: {question_type}
Tool Results: {tool_results}
"""),
    MessagesPlaceholder("messages"),
])

        tool_results = "\n".join(state.get("last_tool_results", []))
        logger.info(f"Toools ressuults {tool_results}")
        logger.debug(f"Tool results concatenated: {tool_results}")

        synthesis_chain = synthesis_prompt | llm_with_tools
        response = synthesis_chain.invoke({
            "question": state["question"],
            "question_type": state["question_type"],
            "tool_results": tool_results,
            "messages": state["messages"],
        })
        logger.debug(f"Synthesis response: {response.content}")
        logger.debug(f"STATE AFTER IF TOOL CALLS SYNTESIZE RESPONSE messages: {[m.content for m in state['messages']]}") ##REMOVE JUST TO CHECK THE LOOP
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
    
    #if state["question_type"] in ["statistical", "factual"] and state["final_answer"]:
    #    match = re.search(r'\b(\d+)\b', state["final_answer"])
    #    if match:
    #        state["final_answer"] = match.group(1)
    state["messages"].append(AIMessage(content=f"Final answer: {state['final_answer']}"))
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
