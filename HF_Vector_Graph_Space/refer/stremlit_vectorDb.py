# ------------------ 1. 📦 Install Packages ------------------
#%%capture
#%pip install pandas langchain langchain-openai faiss-cpu langchain-community streamlit python-dotenv

# ------------------ 2. 📂 Import Libraries ------------------
import os
import pandas as pd
import streamlit as st
import re
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ------------------ 3. 🔑 Load API Keys ------------------
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("❌ OPENAI_API_KEY is missing. Please check your .env file.")

# ------------------ 4. 🏢 CompanyVectorDB Class ------------------
class CompanyVectorDB:
    def __init__(self):
        self.base_path = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "Data"))
        self.vectorstore_path = os.path.join(self.base_path, "faiss_index")
        self.documents = []
        self.dataframes = {}

        if not os.path.exists(self.base_path):
            raise FileNotFoundError(f"❌ Data folder not found: {self.base_path}")

        self.load_csvs(self.base_path)
        if not self.documents:
            raise ValueError("❌ No documents loaded. Please check your CSV files.")

        self.vectorstore = self.load_or_create_vectorstore()

    def load_csvs(self, folder_path):
        print(f"📂 Loading CSVs from {folder_path}")
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                path = os.path.join(folder_path, filename)
                try:
                    df = pd.read_csv(path)
                    df.columns = [col.strip().lower().replace(" ", "_").replace("-", "_") for col in df.columns]
                    self.dataframes[filename] = df
                    for _, row in df.iterrows():
                        metadata = row.to_dict()
                        content = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
                        self.documents.append(Document(page_content=content, metadata=metadata))
                    print(f"✅ Loaded {filename} with {len(df)} rows.")
                except Exception as e:
                    print(f"❌ Error reading {filename}: {e}")

        print(f"📄 Total documents loaded: {len(self.documents)}")

    def load_or_create_vectorstore(self):
        embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
        if os.path.exists(self.vectorstore_path):
            try:
                print("📦 Loading cached vectorstore...")
                return FAISS.load_local(
                    folder_path=self.vectorstore_path,
                    embeddings=embedding_model,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"⚠️ Failed to load cached vectorstore. Rebuilding... ({e})")

        print("⚙️ Creating new vectorstore...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(self.documents)
        if not docs:
            raise ValueError("❌ No documents created after splitting. Check your data or splitter settings.")
        vs = FAISS.from_documents(docs, embedding_model)
        vs.save_local(self.vectorstore_path)
        print("✅ Vectorstore created and saved.")
        return vs

    def search(self, query: str, k: int = 5):
        return self.vectorstore.similarity_search(query, k=k)

# ------------------ 5. 🧠 Helper Functions ------------------

# --- Field extractor ---
def extract_relevant_fields(user_query: str) -> list:
    field_mapping = {
        "title": ["title", "movie", "film", "show"],
        "director": ["director", "directed by"],
        "release_year": ["year", "release", "released"],
        "country": ["country", "origin"],
        "rating": ["rating", "rated"],
        "duration": ["duration", "length", "time"],
        "description": ["description", "summary", "plot"],
        "cast": ["cast", "actor", "actress"],
        "genre": ["genre", "category", "type"]
    }
    detected_fields = set()
    query_lower = user_query.lower()

    for field, keywords in field_mapping.items():
        if any(keyword in query_lower for keyword in keywords):
            detected_fields.add(field)

    # Always include title by default
    if "title" not in detected_fields:
        detected_fields.add("title")

    return list(detected_fields)

# --- Document Filter ---
def filter_docs_by_fields(docs: list, fields: list) -> pd.DataFrame:
    clean_data = []
    for doc in docs:
        clean_entry = {field: doc.metadata.get(field, None) for field in fields}
        clean_data.append(clean_entry)
    return pd.DataFrame(clean_data)

# --- Top-k extractor ---
def extract_top_k(user_query: str, default_k: int = 15) -> int:
    numbers = re.findall(r'\d+', user_query)
    if numbers:
        return min(int(numbers[0]), 500)  # Limit to 500 max
    return default_k

# ------------------ 6. 🌐 Streamlit App ------------------

st.set_page_config(page_title="Netflix Vector QA", layout="wide")

# Load Vector DB
@st.cache_resource
def load_vector_db():
    return CompanyVectorDB()

db = load_vector_db()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_df" not in st.session_state:
    st.session_state.last_df = None

# Layout
col1, col2, col3 = st.columns([0.8, 2.4, 0.8])

with col1:
    st.header("📁 Session Info")
    session_name = st.text_input("Session Name")
    if session_name:
        st.session_state["session_name"] = session_name
        st.success(f"Current Session: {session_name}")

    st.markdown("---")
    st.subheader("📊 Download Last Search")
    if st.session_state.last_df is not None:
        csv_data = st.session_state.last_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", data=csv_data, file_name="vector_results.csv", mime="text/csv")

with col2:
    st.title("🎬 Ask Netflix Dataset")
    user_query = st.text_input("Ask a question:", placeholder="e.g. List 100 movies directed by women")

    if user_query:
        top_k = extract_top_k(user_query)
        with st.spinner(f"Searching top {top_k} results..."):
            try:
                retrieved_docs = db.search(user_query, k=top_k)

                if retrieved_docs:
                    relevant_fields = extract_relevant_fields(user_query)
                    result_df = filter_docs_by_fields(retrieved_docs, relevant_fields)

                    if not result_df.empty:
                        st.caption(f"📋 Fields detected: {', '.join(relevant_fields)}")
                        st.success(f"✅ Retrieved {len(result_df)} results.")
                        st.dataframe(result_df, use_container_width=True)
                        st.session_state.last_df = result_df
                        st.session_state.chat_history.append({"query": user_query, "answer": f"Returned {len(result_df)} results."})
                    else:
                        st.warning("No fields matched for display.")

                else:
                    st.warning("No documents found.")

            except Exception as e:
                st.error(f"❌ Error: {e}")

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
