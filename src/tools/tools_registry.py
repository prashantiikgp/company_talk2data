# %%
# %%
import sys, os

try:
    # ✅ Running from a Python script (.py file)
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    # ✅ Running from a Jupyter notebook (__file__ is not defined)
    base_path = os.path.abspath(os.path.join(os.getcwd(), ".."))

SRC_PATH = os.path.join(base_path)

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
    print(f"✅ SRC path added: {SRC_PATH}")
else:
    print(f"🔁 SRC path already in sys.path: {SRC_PATH}")

# %%
from tools.enhancer_agent_tools.keyword_extractor import keyword_extractor_fn
from tools.enhancer_agent_tools.numeric_extractor import extract_numeric_constraints
from tools.enhancer_agent_tools.filter_composer import compose_filters


# %%
# Keyword extractor tool
# This tool is designed to extract relevant company-related fields from the user's query based on keyword matching.

from langchain.agents import Tool
keyword_extractor_tool = Tool( 
    name="keyword_extractor",
    func=keyword_extractor_fn,
    description="""Extracts relevant company-related fields from the user's query based on keyword matching. Use this tool to identify which structured metadata fields—like headquarters city, tech stack, product category, funding stage, hiring status, or founder information—are being asked about. 
    Returns a dictionary mapping each detected field to the matched keyword.""".strip()
    )   


# %%
# Extract Numeric Constraints
# This tool identifies numeric constraints in the user's query, such as funding amount, revenue, employee size, valuation, or year of founding.

from langchain.agents import Tool

extract_numeric_constraints_tool = Tool(
    name="Numeric Constraint Extraction",
    func=extract_numeric_constraints,
    description=""""Extract numeric constraints from the query such as funding amount, revenue, employee size, valuation, or year of founding. 
    Use this tool to detect conditions like 'raised over $5 million', 'less than 100 employees', or 
    'founded after 2018' and convert them into structured filters for downstream company search.""".strip()
    
)

# %%
# Entity extraction tool
# This tool is used to extract named entities from the user's query.

#from langchain.agents import Tool
#entity_extractor_tool = Tool(
#    name="Entities Extraction",
#    func=extract_entities,
#    description="""Extracts named entities like actor names, directors, production companies, locations, or specific movie/show titles from the input query. 
#                    Use this when the query mentions people, places, or titles explicitly.""".strip()
#)

# %%
# Category Classification Tool

# This tool classifies the user query into one or more predefined company categories
# such as Fintech, SaaS, HealthTech, B2B, Logistics, etc.

#from langchain.agents import Tool
#classify_categories_tool = Tool(
#    name="Classify Categories",
#    func=classify_categories,
#    description="""Classifies the input query into predefined categories like SaaS, FinTech, Edtech, etc. 
#                    Returns a dictionary with the key 'industry_category'.",""".strip()
#)

# %%
# Compose Filters Tool

##  This tool combines the outputs of the previous tools into a structured filter object.
##  This is the final step in the pipeline, so it should be called after all other tools have been applied.
##  It takes the outputs of the keyword extractor, numeric constraints, entity extractor, and category classifier
##  and combines them into a single filter object that can be used for downstream search agents. 
from langchain.agents import Tool

filter_composer_tool = Tool(
    name="filter_composer",
    func=compose_filters,
    description="""Combines extracted metadata from enhancer tools into a unified dictionary of filters. It merges outputs from keyword, numeric, entity, and category extractors into a structured format. 
                    Should be used after all extraction tools have run. 
                    Accepts valid JSON-like inputs and returns a single filter object for downstream search agents.""".strip()

)

# %%
# RAG Search Tool
# This tool performs a semantic search over startup documents using FAISS.


#from langchain_core.tools import Tool
#from enhancer_agent_tools.rag_search_tool import RagSearchInput  # Import RagSearchInput

#rag_search_tool = Tool(
#    name="rag_search_tool",
#    func=rag_search_fn,
#    description="Semantic search tool over startup documents using FAISS. Supports filters like sector, funding, location.",
#    args_schema=RagSearchInput,
#)

# %%
# Qdrant Search Tool
from langchain.tools import Tool
from langchain_openai import OpenAIEmbeddings  # Add this import
from qdrant_tools.qdrant_server_tool import COLLECTION_NAME, QdrantSearchTool

# 4️⃣ Instantiate once
embedding_model = OpenAIEmbeddings()

qdrant_tool = QdrantSearchTool(
    host="localhost",
    port=6333,
    collection_name=COLLECTION_NAME,
    embedding_model=embedding_model
)

# Wrapper for LangChain agent compatibility
def wrapped_qdrant_search(inputs: dict) -> list:
    query = inputs.get("query", "")
    filters = inputs.get("filters", None)
    k = inputs.get("k", 5)
    print(f"\n[DEBUG] Query: {query}")
    print(f"[DEBUG] Filters: {filters}")
    print(f"[DEBUG] Top K: {k}")
    try:
        results = qdrant_tool.search(query=query, filters=filters, k=k)
        print(f"[DEBUG] Raw results: {results}")
        return results
    except Exception as e:
        print(f"[ERROR] Qdrant search failed: {e}")
        return []


# LangChain Tool registration
qdrant_search_tool = Tool(
    name="qdrant_search",
    func=wrapped_qdrant_search,
    description=(
        """Perform hybrid semantic + metadata searches over our SuperVator startup knowledge base 🔍🚀

• **Semantic Retrieval** – Leverage OpenAI embeddings + Qdrant’s cosine‐distance vector index to find the most contextually relevant companies for ANY natural-language query (“emerging fintech players”, “best agritech startups”, etc.).

• **Metadata & Keyword Filters** – Narrow down results by exact or fuzzy matching on structured fields:
    – ▶️ *Categorical*: `state`, `industry_sector`, `hiring_status`, `tech_stack`, `founders`, etc.  
    – ▶️ *Numeric Ranges*: `year_founded`, `total_funding_raised_inr`, `number_of_employees_current`, etc. (supports `gte`/`lte` filters)

• **Fully Hybrid** – Mix & match: e.g. “top funded SaaS companies in Delhi founded after 2015”  
simply by passing your free-text query plus a `filters` dict:"""
    )
)


# %%
