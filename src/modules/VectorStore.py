
import os
from typing import List, Optional

from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from utils.ConfigUtility import ConfigUtility
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers import MultiQueryRetriever


config = ConfigUtility()

store_type = config.getVectorStoreType().lower()
provider = config.getEmbeddingProvider().lower()
persist_dir = os.path.join(os.getcwd(), "vectorestore",f"{provider}_{store_type}_index")
index_path =os.path.join(persist_dir, "index.faiss")
metadata_path =os.path.join(persist_dir, "index.pkl")

def get_embeddings_model(provider: str):
    provider = provider.lower()

    if provider == "openai":
        apikey = config.getOpenAIKey()
        if not apikey:
            raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
        print("Loading OpenAI embeddings...")
        return OpenAIEmbeddings(api_key=apikey)
    elif provider in ["google", "gemini"]:
        apikey = config.getGeminiKey()
        if not apikey:
            raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
        print("Loading Gemini embeddings...")
        return GoogleGenerativeAIEmbeddings(
            model=config.getEmbeddingModel(), google_api_key=apikey
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def addDocsToVectorStore(  docs: List[Document]):

    store_type = config.getVectorStoreType().lower()
    provider = config.getEmbeddingProvider().lower()
    embedding_model = get_embeddings_model(provider)

    if store_type == "faiss":
        persist_dir = os.path.join(os.getcwd(), "vectorestore",f"{provider}_{store_type}_index")
        print(f'Using persist dir: {persist_dir} for vectore store')
        os.makedirs(persist_dir, exist_ok=True)

       

        if os.path.exists(index_path):
            vectorestore = FAISS.load_local(persist_dir, embedding_model, allow_dangerous_deserialization=True)
            if docs:
                print(f'Updating existing FAISS')
                vectorestore.add_documents(docs)
            else:
                print(f'No new documents to add to the existing FAISS store.')
            
        else:
            if not docs:
                raise ValueError(f'No documents provided and no FAISS index was found')
            print(f'Creating new FAISS store at {persist_dir}')
            vectorestore = FAISS.from_documents(docs, embedding_model)

        vectorestore.save_local(persist_dir)
    else:
        raise ValueError(f"Unsupported vector store: {store_type}")

def __getVectorStore():
    embedding_model = get_embeddings_model(provider)
    
    if os.path.exists(index_path):
        print(f'Updating existing FAISS')
        vectorestore = FAISS.load_local(persist_dir, embedding_model, allow_dangerous_deserialization=True)
        return vectorestore




def get_retriever():

    vectorstore = __getVectorStore()
    retConfig = config.getRetrieverConfig()
    retriever_type =  retConfig.type.lower()
   

    if retriever_type == "base":
        print("using base retriever (simple similarity search)") 
        return vectorstore.as_retriever( search_kwargs={"k": retConfig.top_k})
    elif retriever_type == "multi_query":
        print("using multi-query retriever")
        llmProvider = retConfig.llm_provider.lower()
        # todo: Move to config file
        if llmProvider == "openai":
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        elif llmProvider == "gemini":
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        else:
            raise ValueError(f"Unsupported LLM provider for multi-query retriever: {llmProvider}")
        return MultiQueryRetriever.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": retConfig.top_k})
        )
    else:
        raise ValueError(f"Unsupported retriever type: {retriever_type}")



    




