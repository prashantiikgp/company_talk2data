# %%
# -- Enhancer Agent --
# This script sets up an agent that enhances queries by extracting structured metadata and filters.
# It uses a set of tools to analyze and transform vague or unstructured queries into clear, structured metadata.
# The agent is designed to work with the LangChain framework and utilizes OpenAI's GPT-4 model.
# The agent is capable of using various tools such as keyword extractors, numeric constraint extractors, and category classifiers.

import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

print(f"✅ Added to sys.path: {BASE_DIR}")

from utils.path_config import get_base_dir

# %%
import os
import sys
from langchain_core.tools import Tool
from langchain.agents import create_react_agent
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from tools import rag_search_tool

# Define tools for the enhancer agent
rag_agent_tools = [rag_search_tool]


# Define tool names for the agent
tool_names = [tool.name for tool in rag_agent_tools]

# Define the tool descriptions
tool_descriptions = [tool.description for tool in rag_agent_tools]

# Build readable tool help text for the prompt
tool_help_text = "\n".join(
    [f"{i+1}. {tool.name} - {tool.description}" for i, tool in enumerate(rag_agent_tools)]
)

# %%
# 🧠 Prompt Template
rag_agent_prompt_template = PromptTemplate.from_template(
    """You are a RAG (Retrieval-Augmented Generation) Agent.

Your task is to:
1. Understand the user's enhanced query along with the associated structured filters.
2. Perform a semantic search using vector embeddings on the startup/company dataset.
3. Apply additional filters (like location, domain, team size, funding, or tech stack) to refine the retrieved results.
4. Return only the most relevant matches, based on both semantic similarity and filter alignment.

Rules:
- Do not generate new data. Only retrieve from indexed documents.
- Always use both the query and filters together if filters are present.
- Optimize the result set by ranking based on semantic closeness and matching metadata.
- Output only matched results with short, readable summaries. No commentary.

Example Use Cases:

🔹 1. Combined Semantic + Numeric Filter
Enhanced Query: "SaaS startups headquartered in Bengaluru with funding over ₹100 crore."
Structured Filters: `{{"headquarters_city": ["bengaluru"], "industry_sector": ["SaaS"], "total_funding_raised_inr": {{"gte": 100}}}}`
Result: Retrieve startups matching semantic intent of "SaaS + Bengaluru" and funding >= ₹100 Cr.

🔹 2. Tech Stack + Category Match
Query: "Fintech companies using AWS and React"
Filters: `{{"industry_sector": ["Fintech"], "tech_stack": ["aws", "react"]}}`
Result: Only return companies using both AWS and React in their stack with fintech positioning.

🔹 3. Hiring + Growth Filter
Query: "AI startups hiring for engineers"
Filters: `{{"industry_sector": ["AI"], "hiring_status": ["hiring"], "popular_roles_open": ["engineers"]}}`
Result: Return AI startups with open engineering roles and hiring intent.

🔹 4. Time + Revenue Constraints
Query: "E-commerce startups founded after 2018 with ₹25–75 crore revenue"
Filters: `{{"industry_sector": ["e-commerce"], "year_founded": {{"gte": 2018}}, "revenue_estimate_annual": {{"gte": 25, "lte": 75}}}}`
Result: Focus on recent e-com startups within the specified revenue band.

🔹 5. Multi-Category Intersection
Query: "High-growth B2B or B2C startups in HealthTech or EdTech"
Filters: `{{"product_categories": ["B2B", "B2C"], "industry_sector": ["HealthTech", "EdTech"]}}`
Result: Semantic match + multiple tag-based filters.

You have access to a semantic search tool:
{tools}

Format:
Question: enhanced semantic query
Filters: structured metadata extracted by previous agent
Thought: reason about what combination of filters and similarity to apply
Action: tool to invoke, from [{tool_names}]
Action Input: JSON with "query", "filters", "k"
Observation: result returned by the tool
... (repeat as needed)
Thought: I have retrieved the most relevant documents.
Final Answer: a list of relevant company entries (or document snippets)

Constraints:
- NEVER guess or synthesize information
- ALWAYS run the retrieval tool with filters if present
- Final output must be grounded in retrieved content

Begin!

Question: {input}
{agent_scratchpad}
"""
)



# Ensure that the cell defining `tool_help_text` is executed before running this cell.

# Format the prompt with tool descriptions and names
formatted_prompt = rag_agent_prompt_template.partial(
    tools=tool_help_text,
    tool_names=", ".join(tool.name for tool in rag_agent_tools)
)

# 🔧 Define the React-style agent
llm = ChatOpenAI(model="gpt-4o")  # Or use your preferred model


# Create the agent
rag_agent = create_react_agent(
    llm=llm,
    tools=[rag_search_tool],  # ✅ Wrap it in a list
    prompt=formatted_prompt,
)




# %%
def rag_node(state: dict) -> Command[Literal["supervisor"]]:
    """
    RAG node for retrieving relevant documents using semantic + filter-aware search.
    """
    system_prompt = (
        "You are a vector search agent. Your task is to:\n"
        "- Perform semantic search using the RAG tool.\n"
        "- Use filters if provided in the input.\n"
        "- Return the retrieved documents, nothing more.\n"
        "- Never generate answers, just retrieve."
    )

    messages = [{"role": "system", "content": system_prompt}] + state["messages"]

    response = rag_agent.invoke(messages)

    print("🔍 RAG Node executed. Returning to supervisor.")

    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=response.content,
                    name="rag_agent"
                )
            ]
        },
        goto="supervisor"
    )

# %%
from langchain.agents import AgentExecutor

executor = AgentExecutor(
                agent=rag_agent, 
                tools=rag_agent_tools, 
                verbose=True,
                max_iterations=2,
                handle_parsing_errors=True)

query = """

List D2C or SaaS companies in Delhi or Hyderabad that raised over ₹200 crore, are currently hiring for engineers and PMs, valued above $500 million, 

offer mobile apps or APIs, and are backed by Sequoia or Accel."""

result = executor.invoke({"input": query})
print(result["output"])


