
import os
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from utils.ConfigUtility import ConfigUtility
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers import MultiQueryRetriever


try:
    config = ConfigUtility()
    store_type = config.getVectorStoreType().lower()
    provider = config.getEmbeddingProvider().lower()
    persist_dir = os.path.join(
        Path(__file__).parent.parent,
        "vectorestore",
        f"{provider}_{store_type}_index",
    )
    index_path = os.path.join(persist_dir, "index.faiss")
    metadata_path = os.path.join(persist_dir, "index.pkl")
except Exception as error:
    raise RuntimeError("Unable to initialize vector store configuration") from error

def get_embeddings_model(provider: str):
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("An embedding provider is required")

    provider = provider.lower()

    if provider == "openai":
        apikey = config.getOpenAIKey()
        if not apikey:
            raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
        try:
            print("Loading OpenAI embeddings...")
            return OpenAIEmbeddings(api_key=apikey)
        except Exception as error:
            raise RuntimeError("Unable to initialize OpenAI embeddings") from error
    elif provider in ["google", "gemini"]:
        apikey = config.getGeminiKey()
        if not apikey:
            raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
        try:
            print("Loading Gemini embeddings...")
            return GoogleGenerativeAIEmbeddings(
                model=config.getEmbeddingModel(), google_api_key=apikey
            )
        except Exception as error:
            raise RuntimeError("Unable to initialize Gemini embeddings") from error
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def addDocsToVectorStore(docs: List[Document], forceRecreate: bool = False):
    if not isinstance(docs, list):
        raise ValueError("Documents must be provided as a list")

    try:
        store_type = config.getVectorStoreType().lower()
        provider = config.getEmbeddingProvider().lower()
        embedding_model = get_embeddings_model(provider)

        if store_type == "faiss":
            print(f"Using persist dir: {persist_dir} for vector store")
            os.makedirs(persist_dir, exist_ok=True)

            if forceRecreate and os.path.exists(persist_dir):
                for file_path in (index_path, metadata_path):
                    if os.path.exists(file_path):
                        os.remove(file_path)

            if os.path.exists(index_path):
                vectorestore = FAISS.load_local(
                    persist_dir,
                    embedding_model,
                    allow_dangerous_deserialization=True,
                )
                if docs:
                    print("Updating existing FAISS")
                    vectorestore.add_documents(docs)
                else:
                    print("No new documents to add to the existing FAISS store.")
            else:
                if not docs:
                    raise ValueError("No documents provided and no FAISS index was found")
                print(f"Creating new FAISS store at {persist_dir}")
                vectorestore = FAISS.from_documents(docs, embedding_model)

            vectorestore.save_local(persist_dir)
        else:
            raise ValueError(f"Unsupported vector store: {store_type}")
    except ValueError:
        raise
    except Exception as error:
        raise RuntimeError("Unable to update the vector store") from error

def __getVectorStore():
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index was not found at '{persist_dir}'. Add documents before retrieving."
        )

    try:
        embedding_model = get_embeddings_model(provider)
        print("Loading existing FAISS vector store")
        return FAISS.load_local(
            persist_dir,
            embedding_model,
            allow_dangerous_deserialization=True,
        )
    except Exception as error:
        raise RuntimeError("Unable to load the FAISS vector store") from error




def get_retriever():
    try:
        vectorstore = __getVectorStore()
        retConfig = config.getRetrieverConfig()
        retriever_type = retConfig.type.lower()

        if retriever_type == "base":
            print("Using base retriever (simple similarity search)")
            return vectorstore.as_retriever(search_kwargs={"k": retConfig.top_k})
        elif retriever_type == "multi_query":
            print("Using multi-query retriever")
            llmProvider = retConfig.llm_provider.lower()
            if llmProvider == "openai":
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            elif llmProvider in ["google", "gemini"]:
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
            else:
                raise ValueError(
                    f"Unsupported LLM provider for multi-query retriever: {llmProvider}"
                )
            print(f"Using LLM provider: {llmProvider} for multi-query retriever")
            return MultiQueryRetriever.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(
                    search_kwargs={"k": retConfig.top_k}
                ),
            )
        else:
            raise ValueError(f"Unsupported retriever type: {retriever_type}")
    except (FileNotFoundError, ValueError):
        raise
    except Exception as error:
        raise RuntimeError("Unable to create the configured document retriever") from error



    




