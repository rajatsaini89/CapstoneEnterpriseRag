from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.text_splitter import TextSplitter
import utils.ConfigUtility as config

def split_documents(documents: List[Document]) -> List[Document]:
    chunk_size = config.ConfigUtility().getChunkSize()
    chunk_overlap = config.ConfigUtility().getChunkOverlap()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = []
    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            meta = dict(doc.metadata) if doc.metadata else {}
            meta.update({"chunk_index": i})
            split_docs.append(Document(page_content=chunk, metadata=meta))
    return split_docs