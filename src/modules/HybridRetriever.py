import os
from pathlib import Path
from modules.BM25Retriever import BM25Retriever
from modules.VectorStore import get_retriever as faiss_retriever
from utils.ConfigUtility import ConfigUtility
from langchain_core.documents import Document
import numpy as np
from modules.docReader import load_all_docs
from models import HybridConfigModel


def normalize_scores(scores):
    """
    Normalizes a list/array of scores into the range [0, 1].
    Handles edge cases where all scores are equal.
    """

    try:
        scores = np.array(scores, dtype=float)
        if scores.size == 0:
            raise ValueError("At least one score is required for normalization")
        if not np.isfinite(scores).all():
            raise ValueError("Scores must contain only finite values")

        max_score = scores.max()
        min_score = scores.min()

        if max_score - min_score == 0:
            return np.ones_like(scores)

        return (scores - min_score) / (max_score - min_score)
    except ValueError:
        raise
    except Exception as error:
        raise RuntimeError("Unable to normalize retrieval scores") from error


def retrieve_documents(query: str)-> list[Document]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("The retrieval query must be a non-empty string")

    try:
        print(f"Retrieving hybrid results for query: {query}")
        config = ConfigUtility()
        hybridConfig: HybridConfigModel = config.getHybridWeightConfig()

        faiss_retriever_instance = faiss_retriever()
        faiss_Docs = faiss_retriever_instance.invoke(query)
        faiss_scores = np.linspace(1, 0.1, len(faiss_Docs))
        print(f"FAISS retrieved {len(faiss_Docs)} documents for query: {query}")

        folderPath = os.path.join(Path(__file__).parent.parent.parent, "Docs")
        documents = load_all_docs(folderPath)

        bm25_retriever_instance = BM25Retriever(documents)
        bm25Results = bm25_retriever_instance.get_top_k(query)
        bm25Scores = [score for _, score in bm25Results]
        bm25Docs = [doc for doc, _ in bm25Results]
        print(f"BM25 retrieved {len(bm25Docs)} documents for query: {query}")

        faiss_scores = normalize_scores(faiss_scores)
        bm25_scores = normalize_scores(bm25Scores)

        print(f"Normalized FAISS scores: {faiss_scores}")
        print(f"Normalized BM25 scores: {bm25_scores}")
        combined_docs = faiss_Docs + bm25Docs

        combined_scores = (
            hybridConfig.bm25_weight
            * np.concatenate([bm25_scores, np.zeros(len(faiss_scores))])
            + hybridConfig.embedding_weight
            * np.concatenate([np.zeros(len(bm25_scores)), faiss_scores])
        )

        idx_sorted = np.argsort(combined_scores)[::-1][:hybridConfig.top_k]
        ranked_results = [
            (combined_docs[idx], combined_scores[idx]) for idx in idx_sorted
        ]

        print(f"Hybrid retrieved {len(ranked_results)} documents for query: {query}")
        return ranked_results
    except ValueError:
        raise
    except Exception as error:
        raise RuntimeError(
            f"Unable to retrieve hybrid documents for query '{query}'"
        ) from error


    


    