# %%
import os
import sys

# Go up one level to reach src/
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "tools"))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

print(f"BASE_DIR: {BASE_DIR}")


# 🔁 Import all tools from registry
from tools.enhancer_tools_registry import (
    keyword_extractor_tool,
    extract_numeric_constraints_tool,
    filter_composer_tool
)

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



# Go up one level to reach src/
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "tools"))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

print(f"BASE_DIR: {BASE_DIR}")


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
    """You are an advanced query enhancer agent.
Your role is to analyze vague or unstructured queries, extract key information, and transform them into clear, structured metadata and filters.
Use tools like keyword extractor, numeric constraint extractor, and category classifiers to accomplish this.
Only use the tools to extract structure. Do not generate final answers or ask the user again.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must analyze
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the structured metadata
Final Answer: the structured metadata or filters extracted

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


# %%
def enhancer_node(state: dict) -> Command[Literal["supervisor"]]:
    """
    Enhancer node for refining and structuring vague user queries.

    Returns:
        Command: Enhanced message sent back to the supervisor with reasoning.
    """

    # 🧾 Instructional system message — passed as part of prompt to guide behavior
    system_prompt = (
        "You are an advanced query enhancer. Your task is to:\n"
        "- Clarify and refine vague or ambiguous user inputs.\n"
        "- Extract metadata using tools (industry, funding stage, filters).\n"
        "- NEVER generate final answers, only enhance the query.\n"
        "- Return your reasoning and structured results back to the supervisor."
    )

    # 🧠 Compose messages for the agent
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]

    # 🚀 Invoke the agent (you can use `invoke()` or stream if needed)
    enhanced_response = enhancer_agent.invoke(messages)

    print("🧩 Enhancer Node executed. Routing back to: supervisor")

    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=enhanced_response.content,
                    name="enhancer"
                )
            ]
        },
        goto="supervisor"
    )



# %%
from langchain.agents import AgentExecutor

executor = AgentExecutor(agent=enhancer_agent, tools=enhancer_tools, verbose=True, handle_parsing_errors=True)

query = """

List D2C or SaaS companies in Delhi or Hyderabad that raised over ₹200 crore, are currently hiring for engineers and PMs, valued above $500 million, 

offer mobile apps or APIs, and are backed by Sequoia or Accel."""

result = executor.invoke({"input": query})
print(result["output"])


