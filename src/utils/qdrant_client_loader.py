# src/utils/qdrant_client_loader.py

"""
Qdrant client loader utility.

✅ Features:
- Returns a singleton QdrantClient for local storage
- Points to localhost:6333 (Docker REST endpoint)
- Centralized access to collection name

✅ Usage:
     from utils.qdrant_client_loader import get_qdrant_client, get_qdrant_collection_name
"""

from qdrant_client import QdrantClient


# 🔁 Singleton client instance
_qdrant_client = None

# 📦 Centralized collection name
_QDRANT_COLLECTION_NAME = "indian_startups"


def get_qdrant_client() -> QdrantClient:
    """
    Returns a cached singleton QdrantClient instance pointing to the local Qdrant store path.
    """
    global _qdrant_client

    if _qdrant_client is None:
        print("[Qdrant Client] Connecting to Docker server at http://localhost:6333")
        _qdrant_client = QdrantClient(url="http://localhost:6333")

    return _qdrant_client


def get_qdrant_collection_name() -> str:
    """
    Returns the name of the Qdrant collection to use for queries.
    """
    return _QDRANT_COLLECTION_NAME

