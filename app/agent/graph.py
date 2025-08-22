from langgraph.graph import StateGraph,START, END
from langgraph.prebuilt import ToolNode
from app.agent.state import AgentState
from app.tools import get_exchange_rate, add_expense, get_expense, extract_text
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langchain.schema import HumanMessage

# Define the LLM
llm = init_chat_model(
    "us.amazon.nova-lite-v1:0", 
    model_provider="bedrock_converse"
)

# Bind tools
tools = [get_exchange_rate, add_expense, get_expense, extract_text]
llm_with_tools = llm.bind_tools(tools)

# Define nodes
def call_llm(state: AgentState):
    messages = state["messages"]
    image = state["input_file"]
    if "input_file" in state and state["input_file"]:
        messages = messages + [
            HumanMessage(content=f"User provided an image file: {state['input_file']}")
        ]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def llm_router(state: AgentState) -> str:
    """Route LLM output based on whether it requested a tool."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "default"

tool_node = ToolNode(tools)

graph_builder = StateGraph(AgentState)

graph_builder.add_node("llm", call_llm)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "llm") 
graph_builder.add_conditional_edges(
    "llm",
    llm_router,
    {"tools": "tools", "default": END}
)
graph_builder.add_edge("tools", "llm")   # Send tool results back to LLM

graph = graph_builder.compile()
