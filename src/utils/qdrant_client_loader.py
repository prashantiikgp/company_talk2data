# src/utils/qdrant_client_loader.py

"""
Qdrant client loader utility.

✅ Features:
- Returns a singleton QdrantClient for local storage
- Prevents multiple instances from causing file lock errors
- Provides centralized access to the collection name

✅ Usage:
    from utils.qdrant_client_loader import get_qdrant_client, get_qdrant_collection_name
    client = get_qdrant_client()
    collection_name = get_qdrant_collection_name()
"""

from qdrant_client import QdrantClient
from utils.path_config import get_qdrant_store_path

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
        store_path = get_qdrant_store_path()
        print(f"[Qdrant Client] Initializing client with path: {store_path}")
        _qdrant_client = QdrantClient(path=store_path)

    return _qdrant_client


def get_qdrant_collection_name() -> str:
    """
    Returns the name of the Qdrant collection to use for queries.
    """
    return _QDRANT_COLLECTION_NAME
