# src/tools/qdrant_tools_registry.py
"""
Qdrant Tools Registry: Central place to register and wrap Qdrant-related tools for agent use.
"""
from langchain_openai import OpenAIEmbeddings
from langchain.tools import Tool

# Import the QdrantSearchTool 
from tools.qdrant_tools.qdrant_server_tool import QdrantSearchTool

# Utility: get collection name from config/loader if needed
from utils.qdrant_client_loader import get_qdrant_collection_name
COLLECTION_NAME = get_qdrant_collection_name()


# Instantiate the tool
embedding_model = OpenAIEmbeddings()
qdrant_search_tool_instance = QdrantSearchTool(
    host="localhost",
    port=6333,
    collection_name=COLLECTION_NAME,
    embedding_model=embedding_model
)

def wrapped_qdrant_search(inputs: dict) -> list:
    import json
    
    # 1️⃣ If they sent you a JSON string, parse it:
    if isinstance(inputs, str):
        try:
            inputs = json.loads(inputs)
        except json.JSONDecodeError:
            raise ValueError("Tool input is not a valid JSON dictionary.")
        
    if not isinstance(inputs, dict):
        raise ValueError("Tool input must be a dict or JSON string representing a dict.")
    
    # 2️⃣ Otherwise (a dict), just use it:
    query = inputs.get("query", "")
    filters = inputs.get("filters", None)
    k = inputs.get("k", 5)
    print(f"\n[DEBUG] Query: {query}")
    print(f"[DEBUG] Filters: {filters}")
    print(f"[DEBUG] Top K: {k}")
    try:
        results = qdrant_search_tool_instance.search(query=query, filters=filters, k=k)
        print(f"[DEBUG] Raw results: {results}")
        return results
    except Exception as e:
        print(f"[ERROR] Qdrant search failed: {e}")
        return []

# For LangChain agent compatibility
qdrant_search_tool = Tool(
    name="qdrant_search",
    description="Hybrid semantic and metadata search over the Indian startup Qdrant collection. Args: query (str), filters (dict), k (int).",
    func=wrapped_qdrant_search
)
