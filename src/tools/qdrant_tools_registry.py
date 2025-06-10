# Ensure src/ is in sys.path so 'tools' can be imported
# %%
import sys, os
try:
    # ✅ Running from a Python script (.py file)
    TOOLS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    # ✅ Running from a Jupyter notebook (__file__ is not defined)
    TOOLS_PATH = os.path.abspath(os.path.join(os.getcwd(), ".."))
SRC_PATH = os.path.join(TOOLS_PATH)


if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
    print(f"✅ SRC path added: {SRC_PATH}")
else:
    print(f"🔁 SRC path already in sys.path: {SRC_PATH}")  
   

"""
Qdrant Tools Registry: Central place to register and wrap Qdrant-related tools for agent use.
"""
from langchain_openai import OpenAIEmbeddings
from langchain.tools import Tool

# Import the QdrantSearchTool implementation from the actual server tool script
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

import ast
import json

def wrapped_qdrant_search(inputs):
    # 1) If it’s already a dict, great.
    print(f"[DEBUG] Type of tool input: {type(inputs)}")
    print(f"[DEBUG] Tool input value: {inputs}")
    if not isinstance(inputs, dict):
        try:
            inputs = json.loads(inputs)
        except Exception:
            try:
                inputs = ast.literal_eval(inputs)
            except Exception:
                raise ValueError(
                    "Tool input must be a dict or a valid Python dict literal."
                )
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
    func=wrapped_qdrant_search,
    description="""Perform hybrid semantic + metadata searches over our SuperVator startup knowledge base 🔍🚀

• **Semantic Retrieval** – Leverage OpenAI embeddings + Qdrant’s cosine‐distance vector index to find the most contextually relevant companies for ANY natural-language query (“emerging fintech players”, “best agritech startups”, etc.).

• **Metadata & Keyword Filters** – Narrow down results by exact or fuzzy matching on structured fields:
    – ▶️ *Categorical*: `state`, `industry_sector`, `hiring_status`, `tech_stack`, `founders`, etc.  
    – ▶️ *Numeric Ranges*: `year_founded`, `total_funding_raised_inr`, `number_of_employees_current`, etc. (supports `gte`/`lte` filters)

• **Fully Hybrid** – Mix & match: e.g. “top funded SaaS companies in Delhi founded after 2015”  
simply by passing your free-text query plus a `filters` dict:"""
   
) 
