from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(init=False)
class HybridConfigModel:
	"""Configuration for combining BM25 and embedding scores."""

	bm25_weight: float
	embedding_weight: float
	top_k: int

	def __init__(self, config: Mapping[str, Any]) -> None:
		"""Initialize the model from the full config or hybrid weight section."""
		weight_config = config.get("hybridWeight", config)
		if not isinstance(weight_config, Mapping):
			raise ValueError("The 'hybridWeight' configuration must be an object")

		self.bm25_weight = weight_config.get("bm25_weight", 0.6)
		self.embedding_weight = weight_config.get("embedding_weight", 0.4)
		self.top_k = weight_config.get("top_k", 5)
		self._validate()

	def _validate(self) -> None:
		if not 0 <= self.bm25_weight <= 1:
			raise ValueError("BM25 weight must be between 0 and 1")
		if not 0 <= self.embedding_weight <= 1:
			raise ValueError("Embedding weight must be between 0 and 1")
		if abs(self.bm25_weight + self.embedding_weight - 1) > 1e-9:
			raise ValueError("BM25 and embedding weights must sum to 1")
		if self.top_k < 1:
			raise ValueError("Hybrid top_k must be at least 1")
