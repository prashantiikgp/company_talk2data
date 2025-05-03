import os
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any, Dict, List

class Handler(BaseCallbackHandler):
    """
    Custom debug handler for tracing all major LangChain events during execution.
    Logs chain calls, tool usage, LLM prompts/responses, agent actions, and errors.
    """

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        """
        🔗 Triggered when a Chain starts.
        Shows the chain name and input arguments.
        """
        print(f"\n🔗 [Chain Start] {serialized.get('name', 'Unnamed Chain')} with inputs:")
        print(inputs)

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        """
        ✅ Triggered when a Chain successfully ends.
        Shows the output returned by the chain.
        """
        print(f"\n✅ [Chain End] Output:")
        print(outputs)

    def on_chain_error(self, error: Exception, **kwargs):
        """
        ❌ Triggered when a Chain fails.
        Shows the raised exception.
        """
        print(f"\n❌ [Chain Error] {str(error)}")

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        """
        🛠️ Triggered when a Tool is called.
        Shows tool name and input.
        """
        print(f"\n🛠️ [Tool Start] {serialized.get('name', 'Unnamed Tool')} with input: {input_str}")

    def on_tool_end(self, output: str, **kwargs):
        """
        ✅ Triggered when a Tool finishes.
        Shows the tool's output.
        """
        print(f"\n✅ [Tool End] Output: {output}")

    def on_tool_error(self, error: Exception, **kwargs):
        """
        ❌ Triggered when a Tool throws an error.
        Logs the exception.
        """
        print(f"\n❌ [Tool Error] {str(error)}")

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        """
        📤 Triggered before sending a prompt to the LLM.
        Shows the exact prompt(s) being passed.
        """
        print(f"\n📤 [LLM Start] Prompt:")
        for prompt in prompts:
            print(prompt)

    def on_llm_end(self, response: Any, **kwargs):
        """
        ✅ Triggered when the LLM returns a response.
        Shows the LLM output.
        """
        print(f"\n✅ [LLM End] Response:")
        print(response)

    def on_llm_error(self, error: Exception, **kwargs):
        """
        ❌ Triggered if the LLM call fails.
        Shows the exception.
        """
        print(f"\n❌ [LLM Error] {str(error)}")

    def on_agent_action(self, action: Any, **kwargs):
        """
        🤖 Triggered when an agent selects a tool.
        Shows the chosen tool and its input.
        """
        print(f"\n🤖 [Agent Action] Tool: {action.tool}, Input: {action.tool_input}")

    def on_agent_finish(self, finish: Any, **kwargs):
        """
        🏁 Triggered when an agent finishes execution.
        Logs final return values.
        """
        print(f"\n🏁 [Agent Finish] Return values:")
        print(finish.return_values)


def init_env():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "company_info"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

def get_llm():
    return ChatOpenAI(
        model="gpt-4",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        callbacks=[Handler()]
    )