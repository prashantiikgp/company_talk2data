# src/tools/enhancer_tools_registry.py
"""
Enhancer Tools Registry: Register and wrap all enhancer-related tools for agent use.
"""
# Example: You can add more tools as needed
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
    
from langchain.tools import Tool

# Import actual enhancer tools from enhancer_agent_tools
from tools.enhancer_agent_tools.keyword_extractor import keyword_extractor_fn
from tools.enhancer_agent_tools.numeric_extractor import extract_numeric_constraints
from tools.enhancer_agent_tools.filter_composer import compose_filters


def keyword_extractor_tool_func(inputs):
    if isinstance(inputs, dict):
        query = inputs.get("query", "")
    else:
        query = inputs
    return keyword_extractor_fn(query)


def extract_numeric_constraints_tool_func(inputs):
    if isinstance(inputs, dict):
        query = inputs.get("query", "")
    else:
        query = inputs
    return extract_numeric_constraints(query)


def filter_composer_tool_func(inputs):
    if isinstance(inputs, dict):
        tools_outputs = inputs.get("tools_outputs", [])
    else:
        tools_outputs = []
    return compose_filters(*tools_outputs)


keyword_extractor_tool = Tool(
    name="keyword_extractor",
    description="""Extracts relevant company-related fields from the user's query based on keyword matching. Use this tool to identify which structured metadata fields—like headquarters city, tech stack, product category, funding stage, hiring status, or founder information—are being asked about. 
    Returns a dictionary mapping each detected field to the matched keyword.""".strip(),
    func=keyword_extractor_tool_func,
)

extract_numeric_constraints_tool = Tool(
    name="extract_numeric_constraints",
    description="""Extract numeric constraints from the query such as funding amount, revenue, employee size, valuation, or year of founding. 
    Use this tool to detect conditions like 'raised over $5 million', 'less than 100 employees', or 
    'founded after 2018' and convert them into structured filters for downstream company search.""".strip(),
    func=extract_numeric_constraints_tool_func,
)

filter_composer_tool = Tool(
    name="filter_composer",
    description="""Combines extracted metadata from enhancer tools into a unified dictionary of filters. It merges outputs from keyword, numeric, entity, and category extractors into a structured format. 
                    Should be used after all extraction tools have run. 
                    Accepts valid JSON-like inputs and returns a single filter object for downstream search agents.""".strip(),
    func=filter_composer_tool_func,
)

# Registry list for easy import
enhancer_tools = [
    keyword_extractor_tool,
    extract_numeric_constraints_tool,
    filter_composer_tool,
]

# %%
