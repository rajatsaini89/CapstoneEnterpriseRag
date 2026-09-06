from rank_bm25 import BM25Okapi
from typing import Any, List
import re
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr
import numpy as np
from models import bm25Config
from utils.ConfigUtility import ConfigUtility



bm25_config: bm25Config.BM25Config
configUtility = ConfigUtility()

def simple_tokenizer(text: str):
    text = text.lower()

    token = re.findall(r'\b\w+\b', text)    # gets only words, ignoring punctuation

    return token

    

class BM25Retriever:
    def  __init__(self, documents):
        bm25_config = configUtility.getBM25Config()
        self.documents = documents
        self.tokenized_corpus = [simple_tokenizer(doc.page_content) for doc in documents]
       
       #initialize BM25 with the tokenized corpus
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=bm25_config.k1,
            b=bm25_config.b
                              )
        

    def get_top_k(self, query:str):
        bm25_config = configUtility.getBM25Config()
        print(f"Retrieving top BM25 matches {bm25_config.top_k} documents for query: {query}")
        query_tokens = simple_tokenizer(query)
        doc_scores = self.bm25.get_scores(query_tokens)

        top_k_idx =np.argsort(doc_scores)[::-1][:bm25_config.top_k]  # Get indices of top k scores in descending order

        results = []

        for idx in top_k_idx:
            doc = self.documents[idx]
            score = doc_scores[idx]
            results.append((doc, score))
        return results
