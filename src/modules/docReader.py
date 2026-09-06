import os
from pathlib import Path
from langchain_core.documents import Document
import pypdf
import docx

def load_all_docs(folder_path: str) -> list[Document]:
    print(f"Loading documents from folder: {folder_path}")
    docs = []
    
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)

        if file.lower().endswith('.pdf'):
            docs.extend(read_pdf(path))
        elif file.lower().endswith('.docx'):
            docs.extend(read_docx(path))
        elif file.lower().endswith('.txt'):
            docs.extend(read_txt(path))

    return docs


def read_pdf(file_path: str) -> list[Document]:
    """
    Load a PDF file and return a list of Document objects.

    Args:
        file_path (str): The path to the PDF file.

    """

    reader = pypdf.PdfReader(file_path)
    documents = []
    for i , page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            documents.append(Document(page_content=text, metadata={"page": i + 1, "FileName": file_path.split("/")[-1]}))

    return documents

def read_docx(file_path: str) -> list[Document]:
    """
    Load a DOCX file and return a list of Document objects.

    Args:
        file_path (str): The path to the DOCX file.

    """

    doc = docx.Document(file_path)
    documents = []
    for i, paragraph in enumerate(doc.paragraphs):
        text = paragraph.text
        if text:
            documents.append(Document(page_content=text, metadata={"paragraph": i + 1, "FileName": file_path.split("/")[-1]}))

    return documents


def read_txt(file_path: str) -> list[Document]:
    """
    Load a text file and return it as a Document object.

    Args:
        file_path (str): The path to the text file.

    """

    with open(file_path, "r", encoding="utf-8") as text_file:
        text = text_file.read()

    if not text:
        return []

    return [Document(page_content=text, metadata={"FileName": os.path.basename(file_path)})]