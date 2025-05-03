# stream_app.py
import streamlit as st
from langgraph.graph import StateGraph
from agents import rag_agent, graph_agent, evaluator_agent, enhancer_agent, supervisor
from workflow import workflow  # precompiled LangGraph workflow

# For state management
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(layout="wide")
st.title("🧠 Multi-Agent Chat + Stream Viewer")

# Layout: Chat on left, Backend Logs on right
col1, col2 = st.columns([2, 1])

# Chat Interface (col1)
with col1:
    user_input = st.text_input("You:", key="user_input")
    if st.button("Send") and user_input:
        st.session_state.chat_history.append(("user", user_input))

        with st.spinner("Running agents..."):
            stream = workflow.stream(
                {"messages": [{"role": "user", "content": user_input}], "metadata": {}},
                config={"configurable": {"return_intermediate_steps": True}}
            )

            # Display outputs in col2 log viewer
            logs = []
            final_answer = ""
            for step in stream:
                for k, v in step.items():
                    if k == "messages":
                        for msg in v:
                            if msg["role"] == "assistant":
                                final_answer = msg["content"]
                    logs.append(str(step))

            st.session_state.chat_history.append(("assistant", final_answer))

# Display chat history
    for sender, message in st.session_state.chat_history:
        with st.chat_message(sender):
            st.markdown(message)

# Stream Output Column (col2)
with col2:
    st.markdown("### 🔍 Agent Logs")
    if st.button("Clear Logs"):
        st.session_state.logs = []
    if "logs" in locals():
        for log in logs:
            st.code(log, language="python")
