# app.py
import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings

from modules.loader import load_docs
from modules.preprocessing import preprocess_docs
from modules.vectorstore import build_vectorstore
from modules.rag_chain import build_rag_chain


# Load environment variables

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    st.error("OPENROUTER_API_KEY not found! Check your .env file.")
    st.stop()
else:
    st.success("API key loaded successfully.")

st.title("📘 Falcon via OpenRouter + RAG")


# Ensure default sample file exists

default_file = os.path.join("data", "sample.txt")
if not os.path.exists(default_file):
    os.makedirs("data", exist_ok=True)
    with open(default_file, "w", encoding="utf-8") as f:
        f.write("This is a sample document. You can ask questions about it.")

# File Upload

uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])

if uploaded_file is not None:
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    docs = load_docs(file_path)
else:
    docs = load_docs(default_file)


# Preprocess & Embeddings

split_docs = preprocess_docs(docs)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = build_vectorstore(split_docs, embeddings)
retriever = vectorstore.as_retriever()


# OpenRouter LLM Setup

llm = ChatOpenAI(
    model="google/gemini-2.0-flash-001",  # You can change to any available OpenRouter model
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.7,
    max_tokens=512
)


# Build RAG Chain

qa_chain = build_rag_chain(llm, retriever)  # ensure return_source_documents=True in rag_chain.py


# Safe LLM call with retries

def safe_qa_call(chain, query, retries=5):
    for i in range(retries):
        try:
            return chain({"query": query})
        except Exception as e:
            st.warning(f"LLM call failed (attempt {i+1}/{retries}): {e}")
            time.sleep(2 ** i)  # exponential backoff
    st.error("LLM is currently unavailable. Try again later.")
    return None


# Query

query = st.text_input("Enter your question")

if query:
    response = safe_qa_call(qa_chain, query)
    if response:
        st.write("### Answer")
        st.write(response.get("result", "No answer returned."))

        # Show sources safely
        if "source_documents" in response:
            with st.expander("Sources"):
                for doc in response["source_documents"]:
                    st.write(doc.page_content[:200] + "....")
        else:
            st.info("No source documents returned.")
