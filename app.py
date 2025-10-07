import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings

from modules.loader import load_docs
from modules.preprocessing import preprocess_docs
from modules.vectorstore import build_vectorstore
from modules.rag_chain import build_rag_chain

load_dotenv()

st.title("📘 Falcon via OpenRouter + RAG")

# ----------------------------
# File Upload
# ----------------------------

uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])

if uploaded_file is not None:
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    docs = load_docs(file_path)
else:
    docs = load_docs(os.path.join("data", "sample.txt"))

# ----------------------------
# Preprocess & Embeddings
# ----------------------------
split_docs = preprocess_docs(docs)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = build_vectorstore(split_docs, embeddings)
retriever = vectorstore.as_retriever()

# ----------------------------
# OpenRouter LLM Setup
# ----------------------------
llm = ChatOpenAI(
    model="mistralai/mistral-small-3.2-24b-instruct:free",  # Falcon on OpenRouter
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.7,
    max_tokens=512
)

# ----------------------------
# Build RAG Chain
# ----------------------------
qa_chain = build_rag_chain(llm, retriever)

# ----------------------------
# Query
# ----------------------------
query = st.text_input("Enter your question")

if query:
    response = qa_chain({"query": query})
    st.write("### Answer")
    st.write(response["result"])

    with st.expander("Sources"):
        for doc in response["source_documents"]:
            st.write(doc.page_content[:200] + "....")
