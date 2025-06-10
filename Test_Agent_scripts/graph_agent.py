# -- Grapgh Agent --
# This agent is designed to interact with a graph database.
# It uses a language model to generate Cypher queries and retrieve structured data.

from langgraph.prebuilt import create_react_agent
from config.setup import get_llm
from tools.graph_query import graph_query_tool

llm = get_llm()

graph_agent = create_react_agent(
    llm,
    tools=[graph_query_tool],
    prompt="You are a Graph Search agent. Use graph_query_tool to retrieve structured data."
)