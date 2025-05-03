from typing import TypedDict

class AgentState(TypedDict):
    messages: list
    metadata: dict
    rag_result: str
    graph_result: str
    evaluation: str