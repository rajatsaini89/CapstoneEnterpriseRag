import json
import os
from pathlib import Path
from typing import Any
import models.memoryProviderModel as memoryProviderModel
import models.retrieverModel as retrieverModel
import models.bm25Config as bm25Config
import models.HybridConfigModel as hybridWeightModel

class ConfigUtility:
    def __init__(self) -> None:
        self.configPath = Path(__file__).resolve().parents[2] / "config.json"

    def readConfig(self) -> dict[str, Any]:
        with self.configPath.open("r", encoding="utf-8") as config_file:
            return json.load(config_file)

    def getEmbeddingProvider(self) -> str:
        return self.readConfig()["embedding_provider"]

    def getLLMProvider(self) -> str:
        return self.readConfig()["llm_provider"]

    def getLLMModel(self) -> str:
        return self.readConfig()["llm_model"]

    def getEmbeddingModel(self) -> str:
        return self.readConfig()["embedding_model"]

    def getOpenAIKey(self) -> str | None:
        return os.getenv("OPENAI_API_KEY")

    def getGeminiKey(self) -> str | None:
        return os.getenv("GEMINI_API_KEY")

    def getVectorStoreType(self) -> str:
        return self.readConfig().get("vector_store_type")

    def getChunkSize(self) -> int:
        return self.readConfig().get("chunk_size",200)

    def getMinChunkSize(self) -> int:
        return self.readConfig().get("min_chunk_size")

    def getChunkOverlap(self) -> int:
        return self.readConfig().get("chunk_overlap",75)

    def getVectorStoreDimension(self) -> int:
        return self.readConfig().get("vector_store_dimention")

    def getLLMTemperature(self) -> float:
        return self.readConfig().get("llm_temperature", 0.3)

    def getRecentMemoryTopK(self) -> int:
        return self.readConfig().get("recent_memory_top_k", 2)

    def getBM25Config(self) -> bm25Config.BM25Config:
        return bm25Config.BM25Config(self.readConfig().get("bm25", {}))

    def getHybridWeightConfig(self) -> hybridWeightModel.HybridConfigModel:
        return hybridWeightModel.HybridConfigModel(self.readConfig().get("hybridRetrieverConfig"))

    def getRetrieverConfig(self) -> retrieverModel: 
        model:retrieverModel = retrieverModel.RetrieverModel(self.readConfig().get("retriver"))
        
        return model

    def getMemoryProviderConfig(self) -> memoryProviderModel:
        model:memoryProviderModel = memoryProviderModel.MemoryProviderModel(self.readConfig().get("memoryProvider"))

        return model
