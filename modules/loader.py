from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
import os

def load_docs(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist. Please provide a valid file.")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")  # specify encoding
    elif ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError("Unsupported file format. Use TXT, PDF, or DOCX.")
    
    try:
        return loader.load()
    except Exception as e:
        raise RuntimeError(f"Error loading {file_path}") from e
