import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
import pandas as pd
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_core.output_parsers import StrOutputParser


# ------------------ 🔐 Environment Setup ------------------
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ------------------ 🧠 Load LLM ------------------
llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=OPENAI_API_KEY
)

# ------------------ 🗃️ Connect to Neo4j ------------------
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=False,
    allow_dangerous_requests=True
)


# ------------------ 3. 🤖 Resolver Agent Setup ------------------
resolver_prompt = ChatPromptTemplate.from_template("""
You are an AI assistant tasked with clarifying ambiguous or vague user queries for a Neo4j movie graph dataset.

Instructions:
- Reformulate the query only when necessary for disambiguation.
- Do not hallucinate facts or future dates.
- The dataset contains movies only up to the year 2020.
- Valid node labels include: Show, Person, Genre
- Use properties like type = "Movie" to filter movies.

Chat History:
{chat_history}

User Question:
{question}

Clarified and grounded question:
""")

resolver_agent = resolver_prompt | llm | StrOutputParser()

resolver_wrapper = RunnableLambda(
    lambda x: {
        "query": resolver_agent.invoke({
            "question": x["question"],
            "chat_history": x["chat_history"]
        })
    }
)

# ------------------ 4. 🧠 Fuzzy Logic ------------------
def fuzzy_genre_enhancer(data: dict) -> dict:
    query = data["query"].lower()  # coming from resolver

    genre_map = {
        "action": "Action & Adventure",
        "horror": "Horror Movies",
        "romcom": "Romantic Comedies",
        "sci fi": "Sci-Fi & Fantasy",
        "thriller": "Thrillers",
        "sports": "Sports Movies",
        "documentary": "Documentaries",
        "comedy": "Comedies",
    }

    for key, value in genre_map.items():
        if key in query:
            # Replace exact match with fuzzy Cypher-friendly condition
            data["query"] = query.replace(key, f"toLower(g.name) CONTAINS '{key}'")
            break

    return data
fuzzy_genre_enhancer = RunnableLambda(fuzzy_genre_enhancer)

# ------------------ 5. 🔗 Memory-Aware Graph QA Chain ------------------
cypher_prompt = ChatPromptTemplate.from_template("""
You are a Cypher query generation assistant.

Your ONLY job is to generate Cypher queries — never include explanations, comments, markdown, or natural language.

Schema:
{schema}

Rules:
- Use node label `Show` to refer to movies, and always filter with s.type = "Movie".
- Never use `Movie` as a label — it doesn't exist.
- For "latest", sort with ORDER BY s.release_year DESC LIMIT 1.
- For genres, apply: toLower(g.name) CONTAINS '<genre>'.
- Do NOT hallucinate future years (no 2023 or later).
- If the user does **not** specify how many results they want, default to LIMIT 15.
- If the user specifies a number of results (e.g., "top 10", "100 movies"), use that number **instead of** the default.

Only return a valid Cypher query.

User Question:
{question}

Cypher:
""")
# Initialize the GraphCypherQAChain
graph_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    allow_dangerous_requests=True,
    verbose=True,
    enhanced_schema=True,
    cypher_prompt=cypher_prompt,
)

# Update pipeline to include fuzzy logic
memory_aware_pipeline = resolver_wrapper | fuzzy_genre_enhancer | graph_chain


# ------------------ 🎛️ Streamlit UI ------------------
st.set_page_config(page_title="Netflix Q&A", layout="wide")

# 🧠 Initialize session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_df" not in st.session_state:
    st.session_state.last_df = None

# ------------------ Layout Columns ------------------
left_col, center_col, right_col = st.columns([1, 3, 1])

# ------------------ Sidebar (Left): Chat Memory ------------------
with left_col:
    st.markdown("### 🧠 Chat Memory")
    for msg in reversed(st.session_state.chat_history):
        st.markdown(f"""
        <div style='color: white; padding:10px; font-size:13px;'>
            <b>🧑 You:</b> {msg['query']}<br>
            <b>🤖 Assistant:</b> {msg['answer']}
        </div>
        <hr style='border: 0.5px solid #ddd;' />
        """, unsafe_allow_html=True)

# ------------------ Main Chat Interface ------------------
with center_col:
    st.title("🎬 Data_Agent: Ask Netflix Dataset")
    st.markdown("Ask anything about the Netflix movies dataset.")

    user_input = st.text_input("Enter your query:", placeholder="e.g. Find the most recent Indian movie")

    if user_input:
        with st.spinner("Thinking..."):
            try:
                result = memory_aware_pipeline.invoke({
                    "question": user_input,
                    "chat_history": st.session_state.chat_history
                })

                st.success("Answer:")
                answer = result.get("result", "No result found.")

                if isinstance(answer, list):
                    df = pd.DataFrame(answer)
                    st.dataframe(df)
                    st.session_state.last_df = df
                    st.session_state.chat_history.append({
                        "query": user_input,
                        "answer": f"{len(df)} results shown below ⬇️"
                    })
                else:
                    st.write(answer)
                    st.session_state.last_df = None
                    st.session_state.chat_history.append({
                        "query": user_input,
                        "answer": answer
                    })

            except Exception as e:
                st.error(f"❌ Error: {e}")

# ------------------ Right Column: CSV Download ------------------
with right_col:
    if st.session_state.last_df is not None:
        csv_data = st.session_state.last_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Result as CSV",
            data=csv_data,
            file_name="netflix_query_result.csv",
            mime="text/csv"
        )