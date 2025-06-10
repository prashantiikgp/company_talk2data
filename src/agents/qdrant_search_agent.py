# %%
import sys, os
try:
    # ✅ Running from a Python script (.py file)
    TOOLS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",))
except NameError:
    # ✅ Running from a Jupyter notebook (__file__ is not defined)
    TOOLS_PATH = os.path.abspath(os.path.join(os.getcwd(), ".."))

SRC_PATH = os.path.join(TOOLS_PATH)

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
    print(f"✅ SRC path added: {SRC_PATH}")
else:
    print(f"🔁 SRC path already in sys.path: {SRC_PATH}")

# %%
# 2) Import LangChain and your tools
from langchain_openai import OpenAI
from langchain.agents import create_react_agent
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# %%
# 🔁 Import all tools from registry
from tools.qdrant_tools_registry import qdrant_search_tool

# %%
# Define tools for the qdrant agent

qdrant_agent_tools = [
    qdrant_search_tool,
]

# Define tool names for the agent
tool_names = [tool.name for tool in qdrant_agent_tools]

# Define the tool descriptions
tool_descriptions = [tool.description for tool in qdrant_agent_tools]

# Build readable tool help text for the prompt
tool_help_text = "\n".join(
    [f"{i+1}. {tool.name} - {tool.description}" for i, tool in enumerate(qdrant_agent_tools)]
)

# Define system prompt used during agent creation
qdrant_agent_prompt_template = PromptTemplate.from_template(
    """
Role:
You are the Qdrant Search Agent.
You have one tool:  {tool_names}

     - takes a Python dictionary with keys "query", "filters", and "k"
     
     
Description & Purpose:
-Your job is to:
    - Embed the query via OpenAI.
    - Translate filters into Qdrant payload conditions.
    - Execute a hybrid semantic + metadata search.
    - Return the top-k hits, each with its id, similarity score, and full payload metadata.

Inputs (Parameters):
You will receive a single Python dict as {input}, containing keys:
  • input["query"]   – the enhanced natural‑language query
  • input["filters"] – a dict of exact/range filters
  • input["k"]       – the integer top‑K

Follow exactly this ReAct format (no extra braces!):

Question: {input}
Thought: decide how to call the tool
Action: {tool_names}
Action Input: {{"query": "{{input[query]}}", "filters": {{input[filters]}}, "k": {{input[k]}} }}
Observation: <tool output>
Thought: I have the results
Final Answer: <a Python list of result dicts>

Begin!

Question: {input}
{agent_scratchpad}
"""
)


# Format the prompt with tool descriptions and names
formatted_prompt = qdrant_agent_prompt_template.partial(
    tools=tool_help_text,
    tool_names=", ".join(tool.name for tool in qdrant_agent_tools),
)

# 🔧 Define the React-style agent
llm = ChatOpenAI(model="gpt-4o",temperature=0.0) 


# Create the agent
qdrant_agent = create_react_agent(
    llm=llm,
    tools=qdrant_agent_tools,
    prompt=formatted_prompt,
    )



# src/nodes/quadrant_search_node.py

# src/nodes/quadrant_search_node.py

# src/nodes/quadrant_search_node.py
from typing import Any, Dict, List, Literal
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from schema.agent_state import AgentState
# Use the existing qdrant_agent defined in previous cells

def normalize_messages(raw_msgs: List[Any]) -> List[Dict[str, str]]:
    """
    Same normalization as in enhancer_node.
    """
    normalized = []
    for m in raw_msgs:
        if isinstance(m, HumanMessage):
            normalized.append({
                "role": m.name,
                "content": m.content
            })
        else:
            normalized.append(m)
    return normalized

def quadrant_search_node(state: AgentState) -> Command[Literal["__end__"]]:
    """
    1) Normalize history
    2) Build a single-string search_input from enhanced_query, filters, k
    3) Stream qdrant_search_agent over {"input":search_input, "intermediate_steps":[]}
    4) Parse results & reasoning
    5) Append a new dict message and update state
    6) Route to "__end__"
    """
    # ─── 1) Normalize chat history ────────────────────────
    msgs = normalize_messages(state.get("messages", []))

    # ─── 2) Extract search parameters ────────────────────
    q = state.get("enhanced_query", "")
    f = state.get("filters", {})
    k = state.get("k", 5)

    search_input = (
        f"QUERY:\n{q}\n\n"
        f"FILTERS:\n{f}\n\n"
        f"K:\n{k}"
    )

    # ─── 3) Stream the agent ──────────────────────────────
    actions: List[str] = []
    observations: List[str] = []
    final_output: Any = None

    payload = {"input": search_input, "intermediate_steps": []}
    for step in qdrant_agent.stream(payload):
        if isinstance(step, AgentAction):
            actions.append(str(step.log))
        elif isinstance(step, AgentFinish):
            final_output = step.return_values.get("output")
        else:
            observations.append(str(step))

    # ─── 4) Parse the final output ───────────────────────
    if isinstance(final_output, dict):
        results   = final_output.get("results", [])
        reasoning = final_output.get("reasoning", "")
        msg_text  = str(final_output)
    else:
        results, reasoning = [], ""
        msg_text = str(final_output) if final_output is not None else ""

    # ─── 5) Build logs & append new message ──────────────
    new_actions      = state.get("actions", []) + actions + ["Qdrant search completed"]
    new_observations = (
        state.get("observations", [])
        + observations
        + [
            f"Results count: {len(results)}",
            f"Reasoning: {reasoning}"
        ]
    )
    msgs.append({"role": "qdrant_search", "content": msg_text})

    # ─── 6) Return update & end the graph ───────────────
    return Command(
        update={
            "messages":          msgs,
            "retrieved_results": results,
            "final_response":    reasoning,
            "actions":           new_actions,
            "observations":      new_observations,
            "agent_name":        "qdrant_search",
        },
        goto="__end__"
    )
