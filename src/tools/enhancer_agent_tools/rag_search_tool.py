# %%
# rag_search_tool.py

import os
from typing import List, Optional, Dict
from pydantic import BaseModel
from langchain_core.tools import Tool
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from typing import List, Optional, Dict, Any

from utils.path_config import get_base_dir ,get_vector_store_path


# Path setup
BASE_DIR = get_base_dir()

os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
vector_path = get_vector_store_path()


print(f"Vector store path: {vector_path}")
print(f"Base directory: {BASE_DIR}")

# %%
# ✅ Input Schema
from typing import Any
from pydantic import BaseModel, Field

class RagSearchInput(BaseModel):
    query: str = Field(..., description="User's semantic query")
    #filters: Optional[Dict[str, Any]] = Field(default=None, description="Structured filters extracted")
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
# Caching FAISS index
_vector_cache = {}

def load_vector_store(path: str = vector_path) -> FAISS:
    if path in _vector_cache:
        vectorstore = _vector_cache[VECTOR_STORE_PATH]

    if not os.path.exists(path):
        return RagSearchResult(results=[], message="Vector store not found.").dict()

# Load OpenAI embeddings
    embeddings = OpenAIEmbeddings()

# Load the FAISS index
    vector_store = FAISS.load_local(
        path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    _vector_cache[path] = vector_store
    return vector_store

  
print("FAISS index loaded successfully.")


# %%
def rag_search_fn(query: str, k: int = 5) -> Dict:
    print(f"[RAG Tool] Query: {query}")
    print(f"[RAG Tool] Top K: {k}")

    if vector_path in _vector_cache:
        print(f"[RAG Tool] Using cached vector store")
        vectorstore = _vector_cache[vector_path]
    else:
        if not os.path.exists(vector_path):
            print(f"[ERROR] FAISS directory not found at {vector_path}")
            return RagSearchResult(results=[], message="Vector store not found.").dict()

        print(f"[RAG Tool] Loading vector store from disk...")
        embeddings = OpenAIEmbeddings()
        try:
            vectorstore = FAISS.load_local(
                vector_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            _vector_cache[vector_path] = vectorstore
            print("[RAG Tool] FAISS vector store loaded.")
        except Exception as e:
            print(f"[ERROR] Failed to load FAISS: {e}")
            return RagSearchResult(results=[], message="Error loading vector store").dict()

    try:
        raw_docs = vectorstore.similarity_search(query=query, k=k)
        print(f"[RAG Tool] Retrieved {len(raw_docs)} documents.")
    except Exception as e:
        print(f"[ERROR] Similarity search failed: {e}")
        return RagSearchResult(results=[], message="Vector search failed").dict()

    if not raw_docs:
        return RagSearchResult(results=[], message="No results found.").dict()

    return RagSearchResult(
        results=[doc.page_content for doc in raw_docs],
        message=f"Top {len(raw_docs)} documents returned."
    ).dict()


# %%
# ✅ Example usage
if __name__ == "__main__":
    query = "Find B2C and e-commerce startups in the SaaS and logistics space"
    example_filters = {
    "industry_sector": ["SaaS", "Logistics"],
    "product_categories": ["B2B", "B2C", "E-commerce"]
}
    result = rag_search_fn(query=query, k=5)
    

    if result["results"]:
        print(f"✅ Found {len(result['results'])} documents.")
        for doc in result["results"]:
            print(f"📄 Content: {doc[:100]}...")
    else:
        print("❌ No valid documents found.")

    print("ℹ️ Full Output:", result)


