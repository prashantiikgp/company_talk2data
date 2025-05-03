## -- Evaluator Agent --
# This agent is designed to evaluate the output of other agents.
# It uses a language model to analyze the output and provide feedback.

from langgraph.prebuilt import create_react_agent
from config.setup import get_llm
from tools.grader import response_grader_tool

llm = get_llm()

evaluator_agent = create_react_agent(
    llm,
    tools=[response_grader_tool],
    prompt="You are an Evaluator agent. Use response_grader_tool to compare responses and determine which one better answers the query."
)