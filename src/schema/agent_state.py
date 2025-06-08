from typing import TypedDict, List, Dict, Optional

class AgentState(TypedDict, total=False):
    """Shared memory state for LangGraph agents."""

    input_query: str
    enhanced_query: Optional[str]
    filters: Optional[dict]
    k: Optional[int]
    actions: Optional[List[str]]
    observations: Optional[List[str]]
    messages: List[dict] 
    final_response: Optional[str]
    agent_name: Optional[str]
    tools_Calls: Optional[List[Dict[str, str]]]
    tools_results: Optional[List[Dict[str, str]]]