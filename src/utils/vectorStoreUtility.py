import os

from modules.docReader import read_pdf, read_docx
from modules.textSplitter import split_documents
from modules.VectorStore import  addDocsToVectorStore,get_retriever
from utils.ConfigUtility import ConfigUtility
from modules.docReader import load_all_docs
from pathlib import Path


def initializeVectorStore(forceRecreate: bool):

    
    path =os.path.join(Path(__file__).parent.parent.parent,'Docs')
    print(f"Initializing Vector store from path: {path}")
    

    docs = load_all_docs(path)

    split_docs = split_documents(docs)

    addDocsToVectorStore(split_docs, forceRecreate=forceRecreate)


