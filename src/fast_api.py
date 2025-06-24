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


# fastapi_app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
from .main import run_enhancer_agent  # 👈 import from main.py

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Your backend is alive!"}

class QueryRequest(BaseModel):
    query: str


@app.post("/chat")
async def chat(request: QueryRequest):
    result= run_enhancer_agent(request.query)
    return result