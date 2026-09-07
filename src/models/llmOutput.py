from typing import  TypedDict

class LLMOutput(TypedDict):
    answer:str
    sources: list[str]
    retrieved_contexts: list[str]