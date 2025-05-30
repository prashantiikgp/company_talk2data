# %%
##%%capture
##%pip install -U langchain langchain-community faiss-cpu openai langchain-openai

# %%
# -- Build full-row vector store for Indian Startup Dataset --
# This script creates a vector store from the Indian Startup Dataset, where each document represents a full row of data.
# The vector store is saved locally for later use in retrieval tasks.


import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import os

# Define base paths
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
DATA_PATH = os.path.join(BASE_DIR, "Data", "Enriched_Indian_Startup_Dataset.csv")
INDEX_PATH = os.path.join(BASE_DIR, "database","vector_store", "faiss_full_row_index" )  # 🔄 Changed the index name to reflect strategy

print(f"Base Directory: {BASE_DIR}")
print(f"Data Path: {DATA_PATH}")
print(f"Index Path: {INDEX_PATH}")


# %%

def build_fullrow_vectorstore():
    df = pd.read_csv(DATA_PATH)
    docs = []

    for idx, row in df.iterrows():
        row_id = idx
        metadata = row.to_dict()

        # Join all columns into one document
        content = "\n".join(
            f"{col}: {row[col]}" for col in df.columns
            if pd.notna(row[col]) and str(row[col]).strip()
        )

        doc = Document(
            page_content=content,
            metadata={"row_id": row_id, **metadata}
        )
        docs.append(doc)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(INDEX_PATH)

    print(f"✅ Full-row vector store created with {len(docs)} documents at: {INDEX_PATH}")

if __name__ == "__main__":
    build_fullrow_vectorstore()


# %%
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document

# 🔧 Change to your actual index path
# Paths
INDEX_DIR = "src/Data/faiss_field_chunk_index"
FAISS_INDEX_PATH = f"{INDEX_DIR}/index.faiss"
PKL_INDEX_PATH = f"{INDEX_DIR}/index.pkl"

embeddings = OpenAIEmbeddings()

# 🔁 Load existing vector store
vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# 🎯 Sample query
query = "tell me the all the details of 5  related to funding in Bengaluru-based SaaS companies with over ₹1000 Cr funding"

# 🔍 Run similarity search
results = vectorstore.similarity_search_with_score(query, k=10)

# 🧪 Output test results
for i, (doc, score) in enumerate(results):
    print(f"\n🔹 Result #{i+1}")
    print(f"📄 Content:\n{doc.page_content}")
    print(f"📏 Score: {score:.4f}")

# %%


# %%



