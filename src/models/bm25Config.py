from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(init=False)
class BM25Config:
    """Configuration for the BM25 retriever."""

    k1: float
    b: float
    stopwords: list[str]
    top_k: int

    def __init__(self, config: Mapping[str, Any]) -> None:
        """Initialize the model from the full config or BM25 section."""
        bm25_config = config.get("bm25", config)
        if not isinstance(bm25_config, Mapping):
            raise ValueError("The 'bm25' configuration must be an object")

        self.k1 = bm25_config.get("k1", 1.5)
        self.b = bm25_config.get("b", 0.75)
        self.stopwords = bm25_config.get("stopwords", [])
        self.top_k = bm25_config.get("top_k", 5)
        self._validate()

    def _validate(self) -> None:
        if self.k1 <= 0:
            raise ValueError("BM25 k1 must be greater than 0")
        if not 0 <= self.b <= 1:
            raise ValueError("BM25 b must be between 0 and 1")
        if not isinstance(self.stopwords, list):
            raise ValueError("BM25 stopwords must be a list")
        if self.top_k < 1:
            raise ValueError("BM25 top_k must be at least 1")