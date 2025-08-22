from typing import Annotated, TypedDict, List, Optional
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    # Keep track of conversation history
    messages: Annotated[List[BaseMessage], add_messages]
    input_file: Optional[str] # Contains file path to image

    # Optional: track additional context
    user_id: str
    username: str
    last_tool_result: str | None
    currency: str | None

