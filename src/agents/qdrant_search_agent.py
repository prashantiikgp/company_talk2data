

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
from tools.tools_registry import qdrant_search_tool

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


# %%
from langchain.agents import AgentExecutor

executor = AgentExecutor(agent=qdrant_agent, 
                         tools=qdrant_agent_tools, 
                         verbose=True, 
                         handle_parsing_errors=True)

query = """List D2C or SaaS companies in Delhi or Hyderabad"""
    
filters = {
        "total_funding_raised_inr": {"gte": 200},    # ₹200 cr+
        "hiring_status": "actively hiring",
        "industry_sector": "saas",                   # or "d2c"
        "lead_investors": "sequoia, accel",
        }

k=5


# 3️⃣ Invoke!

result = executor.invoke({"query": query, "filters": filters, "k": k})
print(result["output"])




