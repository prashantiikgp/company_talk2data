# ------------------ 1. 🔐 Environment Setup ------------------
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from neo4j import GraphDatabase
import pandas as pd
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
import re

# ------------------ 🔐 Load Secrets ------------------
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ------------------ 2. 🔗 LangChain Setup ------------------
llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)
graph_store = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

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
            "question": x.get("question", ""),
            "chat_history": x.get("chat_history", [])
        })
    }
)

# ------------------ 4. 🧠 Fuzzy Logic ------------------
def fuzzy_genre_enhancer(data: dict) -> dict:
    query = data["query"].lower()
    genre_map = {
        "action": "Action & Adventure", "horror": "Horror Movies", "romcom": "Romantic Comedies",
        "sci fi": "Sci-Fi & Fantasy", "thriller": "Thrillers", "sports": "Sports Movies",
        "documentary": "Documentaries", "comedy": "Comedies"
    }
    for key in genre_map:
        if key in query:
            data["query"] = query.replace(key, f"toLower(g.name) CONTAINS '{key}'")
            break
    return data
fuzzy_genre_enhancer = RunnableLambda(fuzzy_genre_enhancer)

# ------------------ 5. 📜 Cypher Prompt ------------------
cypher_prompt = ChatPromptTemplate.from_template("""
You are a Cypher query generation assistant.

Your ONLY job is to return a Cypher query. Do not include explanations, markdown, comments, or natural language.

---
Schema:
{schema}

---
📌 General Rules:
- Use node label `Show` for all movies. Always include `s.type = "Movie"` in the WHERE clause.
- Never use a non-existent label like `Movie`.
- Do **NOT** hallucinate future years (no 2023 or beyond).
- Use `toLower(g.name) CONTAINS '<genre>'` to filter genres.
- For recency, sort by: `ORDER BY s.release_year DESC`.
- If the user gives no count, default to `LIMIT 15`.
- If the user gives a number (e.g. "top 50", "100 movies"), ALWAYS use that number in the `LIMIT`.
- Never reduce the requested limit. If the user says "100 movies", return a Cypher with `LIMIT 100` — not less.
- If user asks for movies from multiple countries, use `c.name IN [...]` instead of trying to match both individually.
- Be strict: only generate one Cypher query.
                                                 
- ⚠️ Important:
    - If you use WITH, every field (e.g., s.rating, s.release_year) must be aliased using AS.
    - If you use aggregation (like collect, count), ORDER BY must happen **after** a WITH.
    - You must always pass variables correctly through WITH before using them later.

---
📌 Few-shot Examples:

User Question: "List 100 movies produced in India and United States"
Cypher:
MATCH (s:Show)-[:PRODUCED_IN]->(c:Country)
WHERE s.type = "Movie" AND c.name IN ["India", "United States"]
RETURN DISTINCT s.title, s.release_year
ORDER BY s.release_year DESC
LIMIT 100

User Question: "List 50 movies released in India and United States"
Cypher:
MATCH (s:Show)-[:PRODUCED_IN]->(c:Country)
WHERE s.type = "Movie" AND c.name IN ["India", "United States"]
RETURN s.title, s.release_year
ORDER BY s.release_year DESC
LIMIT 50                                

---

User Question:
{question}

Cypher:
""")

# ------------------ 6. 🧠 Graph Chain ------------------
graph_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    top_k=100,
    graph=graph_store,
    allow_dangerous_requests=True,
    verbose=True,
    return_intermediate_steps=True,
    enhanced_schema=True,
    cypher_prompt=cypher_prompt,
)

# ------------------ 7. 🔗 Pipeline ------------------
memory_aware_pipeline = resolver_wrapper | fuzzy_genre_enhancer | graph_chain

# ------------------ 8. 💾 CSV Export Function ------------------
def save_neo4j_results_to_csv(results: list[dict], output_path: str = "neo4j_results.csv") -> str:
    if not results:
        raise ValueError("Empty result set. No data to save.")
    try:
        flattened = [entry['s'] if 's' in entry else entry for entry in results]
    except Exception as e:
        raise ValueError(f"Malformed data: {e}")
    df = pd.DataFrame(flattened)
    preferred_order = ['title', 'release_year', 'rating', 'duration', 'description', 'type', 'date_added']
    df = df[[col for col in preferred_order if col in df.columns] + 
            [col for col in df.columns if col not in preferred_order]]
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ Saved {len(df)} entries to: {output_path}")
    return output_path


# ------------------ 9. 🧪 Query ------------------

user_question = "List 100 movies released in India and United states"


match = re.search(r"\b(\d{2,4})\b", user_question)
top_k = int(match.group(1)) if match else 15
graph_chain.top_k = top_k
print(f"🔍 Searching for top {graph_chain.top_k} movies...")


# Execute the pipeline

response = memory_aware_pipeline.invoke({"question": user_question, "chat_history": []})

cypher_query = response["intermediate_steps"][0].get("query", "N/A")
context_data = response["intermediate_steps"][1].get("context", [])

print("Cypher query:", cypher_query)
print("Returned rows:", len(context_data))

# Optional: save to DataFrame + CSV
df = pd.DataFrame(context_data)
print(df.head())
df.to_csv("exports/indian_horror_movies.csv", index=False)



# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Netflix Graph QA", layout="wide")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_df" not in st.session_state:
    st.session_state.last_df = None

col1, col2, col3 = st.columns([0.8, 2.4, 0.8])

with col1:
    st.header("📁 Session")
    session_name = st.text_input("Session Name")
    if session_name:
        st.session_state["session_name"] = session_name
        st.success(f"Current Session: {session_name}")

    st.markdown("---")
    st.subheader("📜 Last Cypher Query")
    if "last_query" in st.session_state:
        st.code(st.session_state["last_query"], language="cypher")

    st.subheader("📊 Data Snapshot")
    if isinstance(st.session_state.last_df, pd.DataFrame):
        st.dataframe(st.session_state.last_df.head(10))

    if st.session_state.last_df is not None:
        csv_data = st.session_state.last_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", data=csv_data, file_name="netflix_results.csv", mime="text/csv")

with col2:
    st.title("🎬 Ask Netflix Dataset")
    user_query = st.text_input("Ask a question:", placeholder="e.g. List 100 horror movies")

    if user_query:
        match = re.search(r"\b(\d{2,4})\b", user_query)
        top_k = int(match.group(1)) if match else 15
        graph_chain.top_k = top_k

        with st.spinner("Thinking..."):
            try:
                response = memory_aware_pipeline.invoke({"question": user_query, "chat_history": st.session_state.chat_history})
                query = response["intermediate_steps"][0].get("query", "")
                data = response["intermediate_steps"][1].get("context", [])
                st.session_state.last_query = query

                if data:
                    df = pd.DataFrame(data)
                    st.dataframe(df)
                    st.session_state.last_df = df
                    st.session_state.chat_history.append({"query": user_query, "answer": f"Returned {len(df)} rows."})
                else:
                    st.warning("No results found.")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    

with col3:
    st.subheader("🗣️ Chat History")
    for msg in reversed(st.session_state.chat_history):
        st.markdown(f"""
<div style='
    background-color: #f7f7f7;
    padding: 12px 16px;
    border-radius: 10px;
    margin-bottom: 10px;
    font-size: 14px;
    line-height: 1.5;
    color: #333;
'>
    <p><strong>🧑 You:</strong><br>{msg['query']}</p>
    <p><strong>🤖 Assistant:</strong><br>{msg['answer']}</p>
</div>
""", unsafe_allow_html=True)
   
   
