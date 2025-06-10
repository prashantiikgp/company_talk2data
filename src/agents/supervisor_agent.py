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
    #print(f"✅ SRC path added: {SRC_PATH}")
else:
    pass
    #print(f"🔁 SRC path already in sys.path: {SRC_PATH}")


# src/agents/supervisor_agent.py

from typing import Literal, List, Optional
from pydantic import BaseModel, Field
from langchain import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from schema.agent_state import AgentState


# 1️⃣ Define the structured output schema for the Supervisor Agent
class SupervisorOutput(BaseModel):
    next: Literal["enhancer", "qdrant_search", "__end__"] = Field(
        description="The name of the next node to run"
    )
    reason: str = Field(
        description="Brief rationale for why this node was selected"
    )

# 2️⃣ Build the system prompt template for the Supervisor
supervisor_prompt = PromptTemplate.from_template(
    """You are a workflow Supervisor. You oversee two agents:
1) enhancer — improves and structures the user's query
2) qdrant_search — retrieves results from the Qdrant database

Based on the conversation history and the current state, choose which agent should run next.

State summary:
- enhanced_query   : {enhanced_query}
- filters          : {filters}
- retrieved_results: {retrieved_results}

Provide a JSON object matching this schema:
{{ 
  "next": "<enhancer|qdrant_search|__end__>", 
  "reason": "Why you chose that node" 
}}

Conversation history:
{messages}
"""
)



# 3️⃣ Instantiate the chat model (zero temperature for deterministic routing)
llm = ChatOpenAI(model="gpt-4o", temperature=0)

def run_supervisor_agent(
    messages: List[dict],
    enhanced_query: Optional[str],
    filters: Optional[dict],
    retrieved_results: Optional[list]
) -> SupervisorOutput:
    """
    Calls the Supervisor LLM with structured output, returning a SupervisorOutput.
    """
    prompt = supervisor_prompt.format(
        messages="\n".join(f"{m['role']}: {m['content']}" for m in messages),
        enhanced_query=enhanced_query or "",
        filters=filters or {},
        retrieved_results=retrieved_results or []
    )
    # The LLM will return an object matching SupervisorOutput
    return llm.with_structured_output(SupervisorOutput).invoke([{"role": "system", "content": prompt}])

# 4️⃣ Wrap it all in a LangGraph node
def supervisor_node(state: AgentState) -> Command[Literal["enhancer", "qdrant_search", "__end__"]]:
    """
    LangGraph node that:
    1. Extracts state fields.
    2. Invokes the Supervisor agent.
    3. Appends the agent's reason to state.messages.
    4. Returns a Command routing to the chosen next node.
    """

    # Extract the pieces of state the supervisor needs
    messages          = state.get("messages", [])
    enhanced_query    = state.get("enhanced_query")
    filters           = state.get("filters")
    retrieved_results = state.get("retrieved_results")

    # Invoke the Supervisor LLM
    supervisor_out = run_supervisor_agent(
        messages,
        enhanced_query,
        filters,
        retrieved_results
    )

    # Append the supervisor's rationale to the conversation trace
    updated_messages = messages + [
        HumanMessage(content=supervisor_out.reason, name="supervisor")
    ]

    # Return the updated state and route to the chosen node
    return Command(
        update={
            "messages": updated_messages
        },
        goto=supervisor_out.next
    )
