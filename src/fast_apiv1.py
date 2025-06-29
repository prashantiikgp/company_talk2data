# %%
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union, Literal

app = FastAPI()

# Import your agent logic from main.py
from .main import run_enhancer_agent

# %%
from fastapi.middleware.cors import CORSMiddleware

# Allow frontend (Flutter web or emulator) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# %%
# ✅ Pydantic models for structured responses
class Company(BaseModel):
    name: str
    location: str
    sector: str
    funding: str
    founded_year: int
    description: str

class CompanyCardResponse(BaseModel):
    type: str  # should be "company_cards"
    payload: List[Company]

class MessageResponse(BaseModel):
    type: str  # should be "message"
    payload: str

# ✅ Input model from Flutter
class QueryRequest(BaseModel):
    query: str

# %%
# ✅ Root health check
@app.get("/")
def read_root():
    return {"message": "Backend is alive!"}

# %%
# ✅ Main dynamic chat endpoint
@app.post("/chat", response_model=Union[CompanyCardResponse, MessageResponse])
async def chat(request: QueryRequest):
    result = run_enhancer_agent(request.query)

    # 🎯 If result is a list of company-like dicts
    if isinstance(result, list) and all("name" in r for r in result):
        return {
            "type": "company_cards",
            "payload": result
        }
    
    # 🧠 Else fallback to message
    return {
        "type": "message",
        "payload": str(result)
    }


