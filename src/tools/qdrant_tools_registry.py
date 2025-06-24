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
#from langchain_openai import OpenAIEmbeddings
from langchain.tools import Tool
from langchain.tools import StructuredTool
#from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings

# Import the QdrantSearchTool implementation from the actual server tool script
from tools.qdrant_tools.qdrant_server_tool import QdrantSearchTool
 
# Utility: get collection name from config/loader if needed
from utils.qdrant_client_loader import get_qdrant_collection_name
COLLECTION_NAME = get_qdrant_collection_name()

from pydantic import BaseModel, Field

class QdrantSearchInput(BaseModel):
    query: str = Field(..., description="Enhanced natural‑language query")
    filters: dict | None = Field(None, description="Optional metadata filters (e.g. {'year_founded': {'gte':2015}}")
    k: int = Field(5, description="Number of top‑K results to return")

# Instantiate the tool
#embedding_model = OpenAIEmbeddings()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Instantiate the tool
qdrant_search_tool_instance = QdrantSearchTool(
    host="localhost",
    port=6333,
    collection_name=COLLECTION_NAME,
    embedding_model=embeddings
)

#qdrant_search_tool_instance = QdrantSearchTool(
#    host="localhost",
#    port=6333,
#    collection_name=COLLECTION_NAME,
#    embedding_model=embedding_model
#)

#def wrapped_qdrant_search(inputs: QdrantSearchInput) -> list:
#    print(f"\n[DEBUG] Query: {input.query}")
#    print(f"[DEBUG] Filters: {input.filters}")
#    print(f"[DEBUG] Top K: {input.k}")
#    try:
#        results = qdrant_search_tool_instance.search(
#                query=input.query,
#                filters=input.filters, 
#                k=input.k
#            )
#        print(f"[DEBUG] Raw results: {results}")
#        return results
#    except Exception as e:
#        print(f"[ERROR] Qdrant search failed: {e}")
#        return []

# For LangChain agent compatibility
#qdrant_search_tool = Tool(
#    name="qdrant_search",
#    func=wrapped_qdrant_search,
#    argument_schema=QdrantSearchInput,
#    description="""Perform hybrid semantic + metadata searches over our SuperVator startup knowledge base 🔍🚀

#• **Semantic Retrieval** – Leverage OpenAI embeddings + Qdrant’s cosine‐distance vector index to find the most contextually relevant companies for ANY natural-language query (“emerging fintech players”, “best agritech startups”, etc.).

#• **Metadata & Keyword Filters** – Narrow down results by exact or fuzzy matching on structured fields:
#    – ▶️ *Categorical*: `state`, `industry_sector`, `hiring_status`, `tech_stack`, `founders`, etc.  
#    – ▶️ *Numeric Ranges*: `year_founded`, `total_funding_raised_inr`, `number_of_employees_current`, etc. (supports `gte`/`lte` filters)

#• **Fully Hybrid** – Mix & match: e.g. “top funded SaaS companies in Delhi founded after 2015”  
#simply by passing your free-text query plus a `filters` dict:"""
   
#) 

# For LangChain agent compatibility
qdrant_search_tool = StructuredTool.from_function(
    name="qdrant_search",
    func=qdrant_search_tool_instance.search,
    argument_schema=QdrantSearchInput,
    description="""Semantic + metadata search over Qdrant.  
                    Embed a text query, apply optional filters, and return the top‑K matching documents.
                """
                )
qdrant_tools = [qdrant_search_tool]