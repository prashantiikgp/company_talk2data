from typing import TypedDict

class AgentState(TypedDict):
    messages: list
    metadata: dict
    rag_result: str
    graph_result: str
    evaluation: str

# Updated FilterState class
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

class AgentState(TypedDict):
    messages: List[BaseMessage]
    query: str
    filters: Optional[Dict[str, Any]]
    k: Optional[int]
    results: Optional[List[str]]