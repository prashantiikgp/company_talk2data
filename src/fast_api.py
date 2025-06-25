# fastapi_app.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Optional: import your actual enhancer agent
from .main import run_enhancer_agent

app = FastAPI()

# Allow frontend (Flutter web or emulator) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Backend is alive!"}


# ✅ This supports your current Flutter UI
@app.get("/get_companies")
async def get_companies():
    return [
        {
            "name": "Zyphra AI",
            "location": "Bangalore, India",
            "sector": "B2B SaaS",
            "funding": "10M USD",
            "founded_year": 2021,
            "description": "AI-powered document intelligence platform."
        },
        {
            "name": "ClariPay",
            "location": "Delhi, India",
            "sector": "Cross-border Payments",
            "funding": "5M USD",
            "founded_year": 2020,
            "description": "Simplifies global vendor payments for exporters."
        }
    ]


# ✅ This is future-facing (for chat input / enhanced search)
class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat(request: QueryRequest):
    result = run_enhancer_agent(request.query)
    return result
