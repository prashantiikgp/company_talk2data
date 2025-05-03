from config.setup import init_env
from graph.workflow import workflow
from schema.agent_state import AgentState

init_env()

result = workflow.invoke({
    "messages": [{"role": "user", "content": "Find GDP of NY and CA, then average it"}]
})

for msg in result["messages"]:
    print(f"[{msg.type}] {msg.content}")