from typing_extensions import TypedDict
from typing import Literal
from config.setup import get_llm
from langgraph.types import Command

class Router(TypedDict):
    next: Literal["enhancer", "rag", "graph", "evaluator", "END"]

supervisor_prompt = (
    "Given the user query and progress so far, decide the next step in the pipeline."
)

llm = get_llm()

def supervisor_node(state: dict) -> Command:
    messages = [{"role": "system", "content": supervisor_prompt}] + state["messages"]
    decision = llm.with_structured_output(Router).invoke(messages)
    return Command(goto=decision["next"] if decision["next"] != "END" else "__end__")
