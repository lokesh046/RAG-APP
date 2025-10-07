from langchain.text_splitter  import RecursiveCharacterTextSplitter

def preprocess_docs(docs,chunk_size = 500, chunk_overlap =50):
    """
    splits documents into smaller chunks for better retrieval
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size  = chunk_size,
        chunk_overlap = chunk_overlap
    )

    return text_splitter.split_documents(docs)