# %%
import sys, os
try:
    # ✅ Running from a Python script (.py file)
    TOOLS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))
except NameError:
    # ✅ Running from a Jupyter notebook (__file__ is not defined)
    TOOLS_PATH = os.path.abspath(os.path.join(os.getcwd()))

SRC_PATH = os.path.join(TOOLS_PATH)

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
    print(f"✅ SRC path added: {SRC_PATH}")
else:
    print(f"🔁 SRC path already in sys.path: {SRC_PATH}")

# main.py
from agents.enhancer_agent import enhancer_agent
from agents.qdrant_search_agent import qdrant_search_agent
from tools.enhancer_tools_registry import enhancer_tools
from tools.qdrant_tools_registry import qdrant_tools
from langchain.agents import AgentExecutor
from typing import Dict, Any

def run_enhancer_agent(query: str) -> Dict[str, Any]:
    # Enhancer agent setup
    enhancer_executor = AgentExecutor(
        agent=enhancer_agent,
        tools=enhancer_tools,
        verbose=False,
        handle_parsing_errors=True,
    )

    enhanced_query = enhancer_executor.invoke({"input": query})
    payload = enhanced_query.get("output", {})

    # Extract field
    
    query   = payload.get("query", query)
    filters = payload.get("filters", None)
    k       = payload.get("k", 5)

    # Qdrant agent setup
    qdrant_executor = AgentExecutor(
        agent=qdrant_search_agent,
        tools=qdrant_tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=3
    )

    result = qdrant_executor.invoke({
        "input": query,
        "filters": filters,
        "k": k,
    })

    return {
        "type": "card",
        "payload": result["output"]
    }

# For CLI testing
if __name__ == "__main__":
    import sys, json
    query = sys.argv[1] if len(sys.argv) > 1 else "Find 3 top startups in healthcare in India"
    result = run_enhancer_agent(query)
    print(json.dumps(result, indent=2))
