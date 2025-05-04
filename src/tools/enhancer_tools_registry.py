# %%
from enhancer_agent_tools.keyword_extractor import keyword_extractor_fn
from enhancer_agent_tools.numeric_extractor import extract_numeric_constraints
from enhancer_agent_tools.classify_category import classify_categories
from enhancer_agent_tools.entity_extractor import extract_entities
from enhancer_agent_tools.filter_composer import compose_filters
from enhancer_agent_tools.rag_search_tool import rag_search_fn, RagSearchInput

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

from langchain.agents import Tool
entity_extractor_tool = Tool(
    name="Entities Extraction",
    func=extract_entities,
    description="""Extracts named entities like actor names, directors, production companies, locations, or specific movie/show titles from the input query. 
                    Use this when the query mentions people, places, or titles explicitly.""".strip()
)

# %%
# Category Classification Tool

# This tool classifies the user query into one or more predefined company categories
# such as Fintech, SaaS, HealthTech, B2B, Logistics, etc.

from langchain.agents import Tool
classify_categories_tool = Tool(
    name="Classify Categories",
    func=classify_categories,
    description="""Classifies the input query into predefined categories like SaaS, FinTech, Edtech, etc. 
                    Returns a dictionary with the key 'industry_category'.""".strip()
)

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



# RAG Search Tool
# This tool performs a semantic search over startup documents using FAISS.


from langchain_core.tools import Tool
from enhancer_agent_tools.rag_search_tool import RagSearchInput  # Import RagSearchInput

rag_search_tool = Tool(
    name="rag_search_tool",
    func=rag_search_fn,
    description="Semantic search tool over startup documents using FAISS. Supports filters like sector, funding, location.",
    args_schema=RagSearchInput,
)
