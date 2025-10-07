from langchain.chains import RetrievalQA

def build_rag_chain(llm, retriever):
    """
    Builds Retrievals-Augumented Generative (RAG) chain.
    """

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
        
    )