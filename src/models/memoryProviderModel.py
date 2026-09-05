from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(init=False)
class MemoryProviderModel:
    """Configuration for the memory provider."""

    llm_provider: str
    llm_model: str

    def __init__(self, config: Mapping[str, Any]) -> None:
        """Initialize the model from the full config or memoryProvider section."""
        memory_config = config.get("memoryProvider", config)
        if not isinstance(memory_config, Mapping):
            raise ValueError("The 'memoryProvider' configuration must be an object")

        self.llm_provider = memory_config.get("llm_provider", "openai")
        self.llm_model = memory_config.get("llm_model", "gpt-4o-mini")
        self._validate()

    def _validate(self) -> None:
        if not self.llm_provider:
            raise ValueError("Memory provider LLM provider must not be empty")
        if not self.llm_model:
            raise ValueError("Memory provider LLM model must not be empty")
