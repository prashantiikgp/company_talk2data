    # src/schema/types.py
from __future__ import annotations
from typing import Dict, Any, List
from pydantic import BaseModel, Field, RootModel

# %%
import sys, os
try:
    # ✅ Running from a Python script (.py file)
    TOOLS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",))
except NameError:
    # ✅ Running from a Jupyter notebook (__file__ is not defined)
    TOOLS_PATH = os.path.abspath(os.path.join(os.getcwd(), ".."))

SRC_PATH = os.path.join(TOOLS_PATH)

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
    print(f"✅ SRC path added: {SRC_PATH}")
else:
    print(f"🔁 SRC path already in sys.path: {SRC_PATH}")   
    


# ────────────────────────────────────
# 0‑A. Keyword‑extractor
# ────────────────────────────────────
class KeywordExtractInput(BaseModel):
    """Just pass the raw user query."""
    text: str = Field(
        ...,
        description="Original natural‑language question",
        alias="query",              # <- extra accepted key
    )
class KeywordExtractOutput(RootModel[Dict[str, List[str]]]):
    """
    Maps each recognised payload‑field to the list of keyword‑strings
    that appeared in the query. Example:

        {
          "state": ["karnataka"],
          "industry_sector": ["fintech", "b2c"]
        }
    """
    pass


# ────────────────────────────────────
# 0‑B. Numeric‑constraint extractor
# ────────────────────────────────────
class NumericConstraintInput(BaseModel):
    text: str = Field(..., description="Original natural‑language question")

class RangeConstraint(BaseModel):
    gte: float | int | None = None
    lte: float | int | None = None

class NumericConstraintOutput(RootModel[Dict[str, RangeConstraint]]):
    """
    Dict[field → constraint‑dict].
    Example:
        {
          "total_funding_raised_inr": {"gte": 100, "lte": 500},
          "year_founded": {"gte": 2015}
        }
    """
    pass


# ────────────────────────────────────
# 0-C.  Filter‑composer
# ────────────────────────────────────
class SingleFilterDict(RootModel[Dict[str, Any]]):
    """One tool’s already‑validated filter dict."""
    pass

class FilterComposeInput(BaseModel):
    """Pass a *list* of filter‑dicts that need to be merged."""
    filters: List[SingleFilterDict] = Field(
        ...,
        description="Outputs from keyword‑extractor / numeric‑constraint tools."
    )

class FilterComposeOutput(RootModel[Dict[str, Any]]):
    """Merged filters ready for QdrantSearchInput.filters."""
    pass


# ────────────────────────────────────
# 1.  Enhancer stage  (keyword extraction etc.)
# ────────────────────────────────────
class EnhanceInput(BaseModel):
    """Raw user text that needs keyword/constant extraction."""
    text: str = Field(..., description="Full natural‑language question")


class EnhanceOutput(BaseModel):
    """What the enhancer passes downstream."""
    query: str = Field(..., description="Cleaned query ready for embedding")
    filters: Dict[str, Any] | None = Field(
        default=None,
        description="Dict of exact / range filters inferred from the text",
    )
    k: int = Field(5, description="How many results the next agent should fetch")


# ────────────────────────────────────
# 2.  Qdrant search stage
# ────────────────────────────────────
class QdrantSearchInput(BaseModel):
    """Input schema for qdrant_search tool / agent."""
    query: str = Field(..., description="Natural‑language similarity query")
    filters: Dict[str, Any] | None = Field(
        default=None,
        description="Metadata filters, e.g. {'state':'Maharashtra'}",
    )
    k: int = Field(5, description="Top‑K hits to return")


class QdrantSearchHit(BaseModel):
    """One point returned from Qdrant."""
    id: int | str = Field(..., description="Point ID in Qdrant")
    score: float = Field(..., description="Cosine similarity score (0‑1)")
    payload: Dict[str, Any] = Field(..., description="Full metadata for the point")


#class QdrantSearchOutput(BaseModel):
#    """List wrapper so LangChain can serialise/validate the result."""
#    __root__: List[QdrantSearchHit]


# ────────────────────────────────────
# 3.  Answer / polishing stage
# ────────────────────────────────────
class AnswerInput(BaseModel):
    """What your AnswerAgent receives from Qdrant + user question."""
    question: str
    docs: List[Dict[str, Any]]  # probably EnhanceOutput + qdrant payload merged


class AnswerOutput(BaseModel):
    """Final concise answer returned to the front‑end."""
    answer: str
    citations: List[int] | None = None   # optional: ids of supporting docs

# %%
