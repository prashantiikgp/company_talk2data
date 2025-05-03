from langgraph.prebuilt import create_react_agent
from config.setup import get_llm
from tools.vector_search import vector_search_tool

llm = get_llm()

rag_agent = create_react_agent(
    llm,
    tools=[vector_search_tool],
    prompt="Use the vector tool to retrieve matching documents from the dataset. Avoid reasoning."
)