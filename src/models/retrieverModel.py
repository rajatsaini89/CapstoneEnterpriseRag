from dataclasses import dataclass
from typing import Any, Mapping



@dataclass(init=False)
class RetrieverModel:
	"""Configuration for a document retriever."""

	type: str
	top_k: int
	score_threshold: float
	use_score_threshold: bool
	llm_provider: str
	llm_model: str

	def __init__(self, config: Mapping[str, Any]) -> None:
		"""Initialize the model from the full config or retriever section."""
		retriever_config = config.get("retriver", config)
		if not isinstance(retriever_config, Mapping):
			raise ValueError("The 'retriver' configuration must be an object")

		self.type = retriever_config.get("type", "base")
		self.top_k = retriever_config.get("top_k", 5)
		self.score_threshold = retriever_config.get("score_threshold", 0.5)
		self.use_score_threshold = retriever_config.get("use_score_threshold", False)
		self.llm_provider = retriever_config.get("llm_provider", "openai")
		self.llm_model = retriever_config.get("llm_model", "gpt-4o")
		self._validate()

	def _validate(self) -> None:
		if not self.type:
			raise ValueError("Retriever type must not be empty")
		if self.top_k < 1:
			raise ValueError("top_k must be at least 1")
		if not 0 <= self.score_threshold <= 1:
			raise ValueError("score_threshold must be between 0 and 1")
		if not self.llm_provider:
			raise ValueError("LLM provider must not be empty")
		if not self.llm_model:
			raise ValueError("LLM model must not be empty")

   