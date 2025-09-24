from typing import Annotated, TypedDict, List, Optional
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from enum import Enum

class AgentState(TypedDict):
    question: str  ## get question to analyze firsts before answer
    messages: Annotated[List[BaseMessage], add_messages]  # To keep message history 
    redefined_question: str  # Question after analysis, if modified
    question_type: str  # Type of question (e.g., definitional, statistical)
    selected_tools: List[str]  # Name of the selected tool, if any
    current_step: str  # Current step in the agent workflow
    final_answer: str
    input_file: Optional[str] # Contains file path to image
    reason: str  # Reason for the selected tool
    last_tool_results: str


class AgentStep(Enum):
    ANALYZE_QUESTION = "analyze_question"
    SELECT_TOOLS = "select_tools"
    EXECUTE_TOOLS = "execute_tools"
    SYNTHESIZE_ANSWER = "synthesize_answer"
    ERROR_RECOVERY = "error_recovery"
    COMPLETE = "complete"

