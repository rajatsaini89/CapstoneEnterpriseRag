import json
import os
from pathlib import Path
from typing import Any
import models.memoryProviderModel as memoryProviderModel
import models.retrieverModel as retrieverModel
import models.bm25Config as bm25Config
import models.HybridConfigModel as hybridWeightModel
from .ConfigDatabaseUtility import ConfigDatabaseUtility

class ConfigUtility:
    databaseConfigKeys = (
        "embedding_provider",
        "llm_provider",
        "llm_model",
        "llm_temperature",
        "chunk_size",
        "min_chunk_size",
        "chunk_overlap",
        "hybridRetrieverConfig",
    )

    def __init__(self) -> None:
        self.configPath = Path(__file__).resolve().parents[2] / "config.json"
        self.configDatabase = ConfigDatabaseUtility()
        self.configDatabase.migrate_from_json(self.configPath, self.databaseConfigKeys)

    def readConfig(self) -> dict[str, Any]:
        with self.configPath.open("r", encoding="utf-8") as config_file:
            return json.load(config_file)

    def getEmbeddingProvider(self) -> str:
        return self.configDatabase.get("embedding_provider")

    def getLLMProvider(self) -> str:
        return self.configDatabase.get("llm_provider")

    def getLLMModel(self) -> str:
        return self.configDatabase.get("llm_model")

    def getEmbeddingModel(self) -> str:
        return self.readConfig()["embedding_model"]

    def getOpenAIKey(self) -> str | None:
        return os.getenv("OPENAI_API_KEY")

    def getGeminiKey(self) -> str | None:
        return os.getenv("GEMINI_API_KEY")

    def getVectorStoreType(self) -> str:
        return self.readConfig().get("vector_store_type")

    def getChunkSize(self) -> int:
        return self.configDatabase.get("chunk_size")

    def getMinChunkSize(self) -> int:
        return self.configDatabase.get("min_chunk_size")

    def getChunkOverlap(self) -> int:
        return self.configDatabase.get("chunk_overlap")

    def getVectorStoreDimension(self) -> int:
        return self.readConfig().get("vector_store_dimention")

    def getLLMTemperature(self) -> float:
        return self.configDatabase.get("llm_temperature")

    def getRecentMemoryTopK(self) -> int:
        return self.readConfig().get("recent_memory_top_k", 2)

    def getBM25Config(self) -> bm25Config.BM25Config:
        return bm25Config.BM25Config(self.readConfig().get("bm25", {}))

    def getHybridWeightConfig(self) -> hybridWeightModel.HybridConfigModel:
        return hybridWeightModel.HybridConfigModel(
            self.configDatabase.get("hybridRetrieverConfig")
        )

    def getRetrieverConfig(self) -> retrieverModel: 
        model:retrieverModel = retrieverModel.RetrieverModel(self.readConfig().get("retriver"))
        
        return model

    def getMemoryProviderConfig(self) -> memoryProviderModel:
        model:memoryProviderModel = memoryProviderModel.MemoryProviderModel(self.readConfig().get("memoryProvider"))

        return model
