# %%
# Ensure src/ is in sys.path so 'tools' can be imported
# %%
import sys, os
try:
    # ✅ Running from a Python script (.py file)
    TOOLS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    # ✅ Running from a Jupyter notebook (__file__ is not defined)
    TOOLS_PATH = os.path.abspath(os.path.join(os.getcwd(), ".."))
SRC_PATH = os.path.join(TOOLS_PATH)


if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
    print(f"✅ SRC path added: {SRC_PATH}")
else:
    print(f"🔁 SRC path already in sys.path: {SRC_PATH}")


# %%
# -- Enhancer Agent --
# This script sets up an agent that enhances queries by extracting structured metadata and filters.
# It uses a set of tools to analyze and transform vague or unstructured queries into clear, structured metadata.
# The agent is designed to work with the LangChain framework and utilizes OpenAI's GPT-4 model.
# The agent is capable of using various tools such as keyword extractors, numeric constraint extractors, and category classifiers.

import os
import sys
from langchain_core.tools import Tool
from langchain.agents import create_react_agent
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


# 🔁 Import all tools from registry
from tools.enhancer_tools_registry import (
    keyword_extractor_tool,
    extract_numeric_constraints_tool,
    filter_composer_tool
)

# Define tools for the enhancer agent
enhancer_tools = [
    keyword_extractor_tool,
    extract_numeric_constraints_tool,
    filter_composer_tool
]


# Define tool names for the agent
tool_names = [tool.name for tool in enhancer_tools]

# Define the tool descriptions
tool_descriptions = [tool.description for tool in enhancer_tools]

# Build readable tool help text for the prompt
tool_help_text = "\n".join(
    [f"{i+1}. {tool.name} - {tool.description}" for i, tool in enumerate(enhancer_tools)]
)


# Define system prompt used during agent creation

enhancer_agent_prompt_template = PromptTemplate.from_template(
    """You are a Query Enhancer Agent.

Your task is to:
1. Understand the user's query and its intent.
2. Clarify any ambiguities or vague instructions.
3. Add contextual details such as location, industry, amount, category, or other metadata if implied.
4. Output an improved version of the query, optimized for semantic and structured search.

Rules:
- Do not ask follow-up questions.
- Infer likely meanings when user input is ambiguous.
- Your output should be a standalone, precise query in natural language.
- Do not include explanations or formatting like “Here’s the improved query.”

Example:
1. Fuzzy Funding Reference → Structured Numeric + Stage Filter
User Query: "Startups with big funding"
Enhanced Prompt: "Indian startups that raised over $10 million in Series A or later funding rounds."
Focus: Adds monetary threshold and filters funding stage.

2. Location Mention → Structured Geographic Filter
User Query: "Top SaaS companies in Bangalore"
Enhanced Prompt: "Top-performing SaaS startups headquartered in Bengaluru with recent traction or funding activity."
Focus: Resolves city + domain, adds optional temporal or traction filter.

🔹 3. Team Size Mention → Employee Count Filter
User Query: "Companies with large teams working in AI"
Enhanced Prompt: "AI-focused startups in India with over 200 employees and active hiring status."
Focus: Converts “large team” into a numeric filter + hiring signal.

🔹 4. Recent Growth Mention → Year + Funding Filter
User Query: "Fast-growing e-commerce startups"
Enhanced Prompt: "E-commerce startups founded after 2018 with rapid growth and multiple funding rounds."
Focus: Infers growth using founding year and number of rounds.

🔹 5. Tech Stack Mention → Tool + Sector Filters
User Query: "Fintech companies using AWS and React"
Enhanced Prompt: "Fintech startups with a tech stack including AWS and React, offering scalable B2C solutions."
Focus: Extracts cloud/backend/frontend tech + sector.



You have access to the following tools:
{tools}

Format:
Question: the input query
Thought: think step-by-step about what to extract
Action: the tool to use, from [{tool_names}]
Action Input: JSON string or plain text input to the tool
Observation: result returned by the tool
... (repeat Thought/Action/Observation as needed)
Thought: I have gathered all necessary structured data.
Final Answer: a dictionary of all extracted metadata and filters

Constraints:
- NEVER ask the user again
- ONLY use tools
- NEVER hallucinate missing data

Begin!

Question: {input}
{agent_scratchpad}"""
)

# Format the prompt with tool descriptions and names
formatted_prompt = enhancer_agent_prompt_template.partial(
    tools=tool_help_text,
    tool_names=", ".join(tool.name for tool in enhancer_tools)
)

# 🔧 Define the React-style agent
llm = ChatOpenAI(model="gpt-4o")  # Or use your preferred model


# Create the agent
enhancer_agent = create_react_agent(
    llm=llm,
    tools=enhancer_tools,
    prompt=formatted_prompt,
    )


from typing import Literal, List, Dict
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import HumanMessage
from schema.agent_state import AgentState

def enhancer_node(state: AgentState) -> Command[Literal["supervisor"]]:
    """
    React‑style Enhancer Agent Node.

    1. Streams through the pre‑configured enhancer_agent (with your tool prompt).
    2. Captures internal actions and observations.
    3. Parses its final Python dict output: enhanced_query, filters, k.
    4. Updates the shared AgentState with full trace logs.
    5. Routes control back to 'supervisor'.
    """

    # ─── Disable the agent’s built‑in verbose logging ───────────────────────
    enhancer_agent.verbose = False

    # ─── Prepare collectors for this invocation ────────────────────────────
    actions: List[str] = []
    observations: List[str] = []
    final_output = None

    # ─── Stream the React‑style loop ──────────────────────────────────────
    # We pass in only the conversation history; the agent already knows its system prompt & tools.
    for step in enhancer_agent.stream({"messages": state.get("messages", [])}):
        if isinstance(step, AgentAction):
            # Tool choice or internal thought
            actions.append(str(step.log))
        elif isinstance(step, AgentFinish):
            # Final result from the agent: expected to be a Python dict
            final_output = step.return_values.get("output")
        else:
            # Intermediate tool output
            observations.append(str(step))

    # ─── Parse the agent’s final output dict (or fallback) ─────────────────
    if isinstance(final_output, dict):
        enhanced_query = final_output.get("enhanced_query", "")
        filters = final_output.get("filters", {})
        k = final_output.get("k", None)
        message_content = str(final_output)
    else:
        # Fallback if agent returned something unexpected
        enhanced_query = ""
        filters = {}
        k = None
        message_content = str(final_output) if final_output is not None else ""

    # ─── Append a completion marker and summary to the logs ────────────────
    new_actions = (
        state.get("actions", [])
        + actions
        + ["Enhancer agent completed"]
    )
    new_observations = (
        state.get("observations", [])
        + observations
        + [
            f"Enhanced Query: {enhanced_query}",
            f"Filters: {filters}",
            f"k: {k}",
        ]
    )

    # ─── Return the updated state and route back to the supervisor ─────────
    return Command(
        update={
            "messages": [
                HumanMessage(content=message_content, name="enhancer")
            ],
            "enhanced_query": enhanced_query,
            "filters": filters,
            "k": k,
            "actions": new_actions,
            "observations": new_observations,
            "agent_name": "enhancer",
        },
        goto="supervisor"
    )

