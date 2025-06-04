# %%
import sys, os

try:
    # ✅ Running from a Python script (.py file)
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
except NameError:
    # ✅ Running from a Jupyter notebook (__file__ is not defined)
    base_path = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))

SRC_PATH = os.path.join(base_path)

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
    print(f"✅ SRC path added: {SRC_PATH}")
else:
    print(f"🔁 SRC path already in sys.path: {SRC_PATH}")

# %%
# 🚀 Import your utility loaders
from utils.qdrant_client_loader import get_qdrant_collection_name


# 📂 Define paths and configurations

COLLECTION_NAME = get_qdrant_collection_name()

print(f"📌 Collection Name: {COLLECTION_NAME}")


# %%
# --- Utility: Normalization ---
def normalize_field_name(field: str) -> str:
    return (
        field.strip().lower()
        .replace(" ", "_").replace("(", "").replace(")", "")
        .replace("/", "_")
    )

def normalize_field_value(value) -> str:
    return str(value).strip().lower()

# %%
# src/tools/qdrant_tool.py

import re
from typing import List, Dict, Any, Union

from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, MatchValue, Range, Filter
from langchain_openai import OpenAIEmbeddings


class QdrantSearchTool:
    """
    Tool for performing hybrid semantic + metadata searches against a Qdrant collection.
    """

    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        embedding_model: OpenAIEmbeddings,
    ):
        self.client = QdrantClient(host=host, port=port)
        self.collection = collection_name
        self.embedding_model = embedding_model

    @staticmethod
    def _normalize_field_name(field: str) -> str:
        f = field.strip().lower()
        f = re.sub(r"[ ()/]", "_", f)
        return re.sub(r"[^a-z0-9_]", "", f)

    @staticmethod
    def _normalize_field_value(value: Any) -> str:
        return str(value).strip().lower()

    def _build_filter(self, filters: Dict[str, Union[str, int, float, Dict[str, Any]]]) -> Filter:
        """
        Convert a user-provided dict of filters into a Qdrant Filter object.
        Supports:
          - exact match: {"state": "delhi"}
          - range match: {"year_founded": {"gte": 2000, "lte": 2010}}
        """
        conditions = []
        for raw_field, cond in filters.items():
            key = self._normalize_field_name(raw_field)

            if isinstance(cond, dict) and ("gte" in cond or "lte" in cond):
                conditions.append(
                    FieldCondition(
                        key=key,
                        range=Range(gte=cond.get("gte"), lte=cond.get("lte")),
                    )
                )
            else:
                val = self._normalize_field_value(cond)
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=val))
                )

        return Filter(must=conditions)

    def search(
        self,
        query: str,
        filters: Dict[str, Union[str, int, float, Dict[str, Any]]] = None,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Perform a similarity search with optional metadata filtering.
        Returns a list of dicts: { "id", "score", "payload" }.
        """
        # 1. Embed the query
        vector = self.embedding_model.embed_query(query)

        # 2. Build Qdrant filter if provided
        q_filter = self._build_filter(filters) if filters else None

        # 3. Execute search
        results = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            query_filter=q_filter,
            limit=k,
            with_payload=True,
        )

        # 4. Format output
        output = []
        for pt in results:
            output.append({
                "id": pt.id,
                "score": pt.score,
                "payload": pt.payload,
            })
        return output