import os 
from langchain_community.document_loaders import PyPDFLoader , UnstructuredWordDocumentLoader, TextLoader 
from langchain_core.documents import Document 


def load_file(file_path: str)-> list[Document]:
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    print(f"Attempting to load file :{file_path} with extension: {ext}")
    
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = UnstructuredWordDocumentLoader(file_path)
    elif ext in [".txt", ".md"]:
        loader = TextLoader(file_path, encoding='utf-8' , errors='ignore')

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return loader.load()            


