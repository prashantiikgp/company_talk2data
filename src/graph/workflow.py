from langgraph.graph import StateGraph
from agents import rag_agent, graph_agent, evaluator_agent, enhancer_agent, supervisor
from typing import TypedDict
from schema.agent_state import AgentState

class AgentState(TypedDict):
    messages: list

builder = StateGraph(AgentState)

builder.add_node("rag", rag_agent.rag_agent)
builder.add_node("graph", graph_agent.graph_agent)
builder.add_node("evaluator", evaluator_agent.evaluator_agent)
builder.add_node("enhancer", enhancer_agent.enhancer_agent)
builder.add_node("supervisor", supervisor.supervisor_node)

builder.set_entry_point("supervisor")
builder.add_edge("supervisor", "__end__")
for name in ["enhancer", "rag", "graph", "evaluator"]:
    builder.add_edge(name, "supervisor")

workflow = builder.compile()