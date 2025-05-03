# %%
# rag_search_tool.py

import os
from typing import List, Optional, Dict
from pydantic import BaseModel
from langchain_core.tools import Tool
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from typing import List, Optional, Dict, Any

# Path setup
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "database", "vector_store", "faiss_fullrow_index")


# %%
# Caching FAISS index
_vector_cache = {}

def load_faiss_index(path: str = VECTOR_STORE_PATH) -> FAISS:
    if path in _vector_cache:
        return _vector_cache[path]

    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS index not found at: {path}")

# Load OpenAI embeddings
    embeddings = OpenAIEmbeddings()

# Load the FAISS index
    faiss_index = FAISS.load_local(
        path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    _vector_cache[path] = faiss_index
    return faiss_index

# %%
# ✅ Input Schema
from typing import Any
from pydantic import BaseModel, Field

class RagSearchInput(BaseModel):
    query: str = Field(..., description="User's semantic query")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Structured filters extracted")
    k: Optional[int] = Field(default=5, description="Number of results to return")


# ✅ Output Schema
class RagSearchResult(BaseModel):
    results: List[str]
    message: str = ""

    def dict(self):
        return {
            "results": self.results,
            "message": self.message
        }

# %%
# ✅ Core tool function
def rag_search_tool(query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> Dict:
    # Load vector store from cache or disk
    if "vectorstore" in _vector_cache:
        vectorstore = _vector_cache["vectorstore"]
    else:
        if not os.path.exists(VECTOR_STORE_PATH):
            return RagSearchResult(results=[], message="Vector store not found.").dict()

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        _vector_cache["vectorstore"] = vectorstore

    # Perform the search (get more docs than needed for filtering)
    raw_docs = vectorstore.similarity_search(query=query, k=20)

    # Manual filter
    def matches_filters(doc_metadata: Dict[str, Any]) -> bool:
        if not filters:
            return True
        for key, expected_value in filters.items():
            if key not in doc_metadata:
                return False
            doc_val = doc_metadata[key]
            # Support exact match, list match, or numeric comparison
            if isinstance(expected_value, dict) and isinstance(doc_val, (int, float)):
                if "gte" in expected_value and doc_val < expected_value["gte"]:
                    return False
                if "lte" in expected_value and doc_val > expected_value["lte"]:
                    return False
            elif isinstance(expected_value, list):
                if doc_val not in expected_value:
                    return False
            elif doc_val != expected_value:
                return False
        return True

    # Apply filtering
    filtered_docs = [doc for doc in raw_docs if matches_filters(doc.metadata)]
    final_docs = filtered_docs[:k]

    if not final_docs:
        return RagSearchResult(results=[], message="No matching documents after filtering.").dict()

    return RagSearchResult(
        results=[doc.page_content for doc in final_docs],
        message=f"{len(final_docs)} documents matched filters."
    ).dict()


# %%
# ✅ Example usage
if __name__ == "__main__":
    query = "Find B2C , B2B or e-commerce startups in the SaaS and logistics space"
    example_filters = {
    "industry_sector": ["SaaS", "Logistics"],
    "product_categories": ["B2B", "B2C", "E-commerce"]
}
    result = rag_search_tool(query=query, k=5, filters=example_filters)
    

    if result["results"]:
        print(f"✅ Found {len(result['results'])} documents.")
        for doc in result["results"]:
            print(f"📄 Content: {doc[:100]}...")
    else:
        print("❌ No valid documents found.")

    print("ℹ️ Full Output:", result)


