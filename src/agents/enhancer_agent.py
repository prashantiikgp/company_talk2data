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

# -- Enhancer Agent --
# This script sets up an agent that enhances queries by extracting structured metadata and filters.
# It uses a set of tools to analyze and transform vague or unstructured queries into clear, structured metadata.
# The agent is designed to work with the LangChain framework and utilizes OpenAI's GPT-4 model.
# The agent is capable of using various tools such as keyword extractors, numeric constraint extractors, and category classifiers.

from langchain_core.tools import Tool
from langchain.agents import create_react_agent
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from typing import Literal
#from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()



# 🔁 Import all tools from registry
from tools.enhancer_tools_registry import (
    keyword_extractor_tool,
    numeric_constraint_tool,
    filter_composer_tool
)

# Define tools for the enhancer agent
enhancer_tools = [
    keyword_extractor_tool,
    numeric_constraint_tool,
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
"""You are the **Query‑Enhancer Agent**.
You have access to the following tools:
{tool_names}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
Valid "action" values: "Final Answer" or {tool_names}
Provide only ONE action per $JSON_BLOB, as shown:

```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}


Follow this format:
Question: {input}
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)

Thought: I know what to respond
Action:
```
{{
  "action": "Final Answer",
  "action_input": {{
    "query":   "<string>",
    "filters": {{ ... }} | null,
    "k":       <int>
  }}
}}
```
Example final answer:
Action:
```json
{{
  "action": "Final Answer",
  "action_input": {{
    "query": "B2B SaaS startups",
    "filters": {{
      "industry_sector": ["b2b", "saas"],
      "location": "India",
      "company_type": "startup"
    }},
    "k": 5
  }}
}}
IMPORTANT RULE:
- Only call "filter_composer" when you already have at least one filter.
- If numeric extractor returns {{}}, proceed directly to the Final Answer.
Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation


Human!
{input}

{agent_scratchpad}
 (reminder to respond in a JSON blob no matter what)
""")

# Format the prompt with tool descriptions and names
formatted_prompt = enhancer_agent_prompt_template.partial(
    tools=tool_help_text,
    tool_names=", ".join(tool.name for tool in enhancer_tools)
)

# 🔧 Define the React-style agent
#llm = ChatOpenAI(model="gpt-4o",temperature=0)  # Or use your preferred model


llm_enhancer = ChatTogether(model="mistralai/Mistral-7B-Instruct-v0.2",
                            temperature=0,
                            api_key=os.getenv("together_ai_api_key"))


#create the agent with the tools and formatted prompt
from langchain.agents import create_structured_chat_agent, AgentExecutor
from schema.tools_schema import EnhanceInput, EnhanceOutput 

enhancer_agent = create_structured_chat_agent(
    llm=llm_enhancer,
    tools=enhancer_tools,
    prompt=formatted_prompt,
)


from langchain.agents import AgentExecutor

enhancer_executor = AgentExecutor(agent=enhancer_agent, 
                         tools=enhancer_tools, 
                         verbose=True, 
                         handle_parsing_errors=True)


# src/nodes/enhancer_node.py

from typing import Any, Dict, List, Literal
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from schema.agent_state import AgentState

def normalize_messages(raw_msgs: List[Any]) -> List[Dict[str, str]]:
    """
    Turn any HumanMessage objects into plain dicts {"role","content"},
    and pass through existing dicts as-is.
    """
    normalized = []
    for m in raw_msgs:
        if isinstance(m, HumanMessage):
            normalized.append({
                "role": m.name,       # who spoke
                "content": m.content  # what they said
            })
        else:
            normalized.append(m)     # assume it's already a dict
    return normalized


def enhancer_node(state: AgentState) -> Command[Literal["supervisor"]]:
    # 1) Normalize history
    msgs = normalize_messages(state.get("messages", []))

    # 2) Extract user input
    user_input = ""
    for m in reversed(msgs):
        if m.get("role") == "user":
            user_input = m["content"]
            break
    print(f"\n[DEBUG] Enhancer Node received user_input: {user_input}")

    # 3) Stream the agent
    actions: List[str] = []
    observations: List[str] = []
    final_output: Any = None

    payload = {"input": user_input, "intermediate_steps": []}
    print(f"[DEBUG] Enhancer payload: {payload}")

    for step in enhancer_agent.stream(payload):
        if isinstance(step, AgentAction):
            print(f"[DEBUG] AgentAction: {step.log}")
            actions.append(str(step.log))
        elif isinstance(step, AgentFinish):
            print(f"[DEBUG] AgentFinish output: {step.return_values.get('output')}")
            final_output = step.return_values.get("output")
        else:
            print(f"[DEBUG] Observation: {step}")
            observations.append(str(step))

    # 4) Parse final_output
    print(f"[DEBUG] raw final_output: {final_output}")
    if isinstance(final_output, dict):
        enhanced_query = final_output.get("enhanced_query", "")
        filters        = final_output.get("filters", {})
        k              = final_output.get("k", None)
        msg_text       = str(final_output)
    else:
        enhanced_query = ""
        filters        = {}
        k              = None
        msg_text       = str(final_output) if final_output is not None else ""

    # 5) Log summary
    new_actions      = state.get("actions", []) + actions + ["Enhancer completed"]
    new_observations = (
        state.get("observations", [])
        + observations
        + [
            f"Enhanced Query: {enhanced_query}",
            f"Filters: {filters}",
            f"k: {k}",
        ]
    )
    print(f"[DEBUG] Parsed enhanced_query: {enhanced_query}")
    print(f"[DEBUG] Parsed filters: {filters}")
    print(f"[DEBUG] Parsed k: {k}")

    # 6) Append message and return
    msgs.append({"role": "enhancer", "content": msg_text})

    return Command(
        update={
            "messages":       msgs,
            "enhanced_query": enhanced_query,
            "filters":        filters,
            "k":              k,
            "actions":        new_actions,
            "observations":   new_observations,
            "agent_name":     "enhancer",
        },
        goto="supervisor"
    )
