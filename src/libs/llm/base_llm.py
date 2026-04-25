from abc import ABC, abstractmethod
from typing import Mapping, Sequence


class BaseLLM(ABC):
    @abstractmethod
    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        raise NotImplementedError


class UnavailableLLM(BaseLLM):
    def __init__(self, provider: str):
        self.provider = provider

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        raise NotImplementedError(f"LLM provider '{self.provider}' is not implemented yet")
