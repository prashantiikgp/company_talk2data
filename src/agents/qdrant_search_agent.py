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
# Define tools for the enhancer agent
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
Role :
You are the Qdrant Search Tool, a micro-service that combines high-fidelity vector embeddings with rich, structured metadata filtering to retrieve the most relevant records from a Qdrant collection.

Description & Purpose :
-- Given a natural-language query, an optional dictionary of filters (exact matches or numeric ranges), and a desired result count k, your job is to:
-- Embed the query via OpenAI.
-- Translate filters into Qdrant payload conditions.
-- Execute a hybrid semantic + metadata search.
-- Return the top-k hits, each with its id, similarity score, and full payload metadata.

Inputs (Parameters) :
query (string) - free-text search string.
filters (dict) - {filters}
Exact: ( "state": "delhi", "industry_sector": "saas" )
Range: ( "year_founded": ("gte":2000,"lte":2010) )
k (integer) – the number of top results to return.

Examples :
Pure semantic (no filters)
qdrant_search(
  query="emerging agritech startups",
  filters=None,
  k=5
)
# → returns top-5 agritech vectors by relevance
Metadata only

qdrant_search(
  query="",
  filters=( "state": "karnataka", "industry_sector": "fintech" ),
  k=10
)
# → returns any fintech startups in Karnataka, ordered by vector‐default rank
Hybrid (semantic + filters + range)

qdrant_search(
  query="best B2B platforms",
  filters=(
    "state": "delhi",
    "year_founded": ("gte":2015),
    "industry_sector": "saas"
  ),
  k=3
)
# → returns top-3 SaaS B2B startups in Delhi founded ≥2015

Guidelines & Constraints
-- Must apply both vector similarity and all payload filters.
-- For textual filters use exact keyword match.
-- For numeric filters support gte / lte semantics.
-- If filters=None, perform a pure semantic lookup.
-- Always return at most k results.
-- Never omit an entry’s payload.
-- Ensure consistent lower-casing of filter values and field names.
-- Never return more than k results, even if multiple entries have the same score.

-- If no results match, return an empty list [].


Format:
Question: the input query
Thought: think step-by-step about what to extract
Action: the tool to use, from [{tool_names}]
Action Input: JSON string or plain text input to the tool
Observation: result returned by the tool
... (repeat Thought/Action/Observation as needed)
Thought: I have gathered all necessary structured data.
Final Answer: a dictionary of all extracted metadata and filters

Constraints:
- NEVER ask the user again
- ONLY use tools
- NEVER hallucinate missing data

Begin!

Question: {{input}}
{agent_scratchpad}"""
)


# Format the prompt with tool descriptions and names
formatted_prompt = qdrant_agent_prompt_template.partial(
    tools=tool_help_text,
    tool_names=", ".join(tool.name for tool in qdrant_agent_tools),
)

# 🔧 Define the React-style agent
llm = ChatOpenAI(model="gpt-4o") 


# Create the agent
qdrant_agent = create_react_agent(
    llm=llm,
    tools=qdrant_agent_tools,
    prompt=formatted_prompt,
)


# src/nodes/quadrant_search_node.py

from typing import Literal, List
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from schema.agent_state import AgentState

# Assumes you have already created and imported your React‑style Qdrant agent:
# from src.agents import qdrant_search_agent

def quadrant_search_node(state: AgentState) -> Command[Literal["__end__"]]:
    """
    React‑style Qdrant Search Node:
    - Takes enhanced_query, filters, and k from AgentState.
    - Streams the pre‑configured qdrant_search_agent (with verbose=False).
    - Captures internal AgentAction steps and tool Observations.
    - Parses the final Python dict output into `results` & `reasoning`.
    - Updates AgentState with:
        • messages          – the raw dict output as a HumanMessage
        • retrieved_results – the list of result dicts
        • final_response    – any summary the agent returned
        • actions/observations – full trace logs
    - Routes to "__end__" to terminate the workflow.
    """

    # Turn off built‑in console logging to avoid duplicate prints
    qdrant_agent.verbose = False

    # Collectors for this invocation
    actions: List[str] = []
    observations: List[str] = []
    final_output = None

    # Build the payload we send into the agent
    payload = {
        "messages": state.get("messages", []),
        "enhanced_query": state.get("enhanced_query"),
        "filters": state.get("filters"),
        "k": state.get("k"),
    }

    # Stream through the React loop
    for step in qdrant_agent.stream(payload):
        if isinstance(step, AgentAction):
            # Agent decided to call a tool or perform an internal action
            actions.append(str(step.log))
        elif isinstance(step, AgentFinish):
            # Agent finished: expect a Python dict in step.return_values["output"]
            final_output = step.return_values.get("output")
        else:
            # Intermediate tool output or observation
            observations.append(str(step))

    # Parse the final dict or fallback
    if isinstance(final_output, dict):
        results = final_output.get("results", [])
        reasoning = final_output.get("reasoning", "")
        message_content = str(final_output)
    else:
        results = []
        reasoning = ""
        message_content = str(final_output) if final_output is not None else ""

    # Build full trace logs
    new_actions = state.get("actions", []) + actions + ["Qdrant Search completed"]
    new_observations = (
        state.get("observations", [])
        + observations
        + [f"Results count: {len(results)}", f"Reasoning: {reasoning}"]
    )

    # Return updated state; "__end__" signals graph termination
    return Command(
        update={
            "messages": [
                HumanMessage(content=message_content, name="qdrant_search")
            ],
            "retrieved_results": results,
            "final_response": reasoning,
            "actions": new_actions,
            "observations": new_observations,
            "agent_name": "qdrant_search",
        },
        goto="__end__"
    )

