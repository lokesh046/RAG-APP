from langchain.vectorstores import FAISS

def build_vectorstore(docs, embeddings):
    """
    creates a FAISS vectorestore from given documents.
    """
    return FAISS.from_documents(docs,embeddings)
