from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.text_splitter import TextSplitter
import utils.ConfigUtility as config

def split_documents(documents: List[Document]) -> List[Document]:
    if not isinstance(documents, list):
        raise ValueError("Documents must be provided as a list")

    try:
        config_utility = config.ConfigUtility()
        chunk_size = config_utility.getChunkSize()
        chunk_overlap = config_utility.getChunkOverlap()
        min_chunk_size = config_utility.getMinChunkSize()
    except Exception as error:
        raise RuntimeError("Unable to load document splitting configuration") from error

    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("Chunk size must be a positive integer")
    if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        raise ValueError("Chunk overlap must be a non-negative integer")
    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size")
    if not isinstance(min_chunk_size, int) or min_chunk_size < 1:
        raise ValueError("Minimum chunk size must be a positive integer")

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        split_docs = []
        for document_index, doc in enumerate(documents):
            if not isinstance(doc, Document):
                raise ValueError(
                    f"Document at index {document_index} is not a valid Document"
                )
            if not isinstance(doc.page_content, str):
                raise ValueError(
                    f"Document at index {document_index} has invalid text content"
                )

            chunks = splitter.split_text(doc.page_content)
            for chunk_index, chunk in enumerate(chunks):
                if len(chunk) >= min_chunk_size:
                    meta = dict(doc.metadata) if doc.metadata else {}
                    meta.update({"chunk_index": chunk_index})
                    split_docs.append(Document(page_content=chunk, metadata=meta))
        return split_docs
    except ValueError:
        raise
    except Exception as error:
        raise RuntimeError("Unable to split documents into chunks") from error