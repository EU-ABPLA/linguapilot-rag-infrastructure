from abc import ABC, abstractmethod
from typing import Any, Optional, Union


class BaseVisionLLM(ABC):
    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image_path: Union[str, bytes],
        trace: Optional[Any] = None,
    ) -> str:
        raise NotImplementedError


class UnavailableVisionLLM(BaseVisionLLM):
    def __init__(self, provider: str):
        self.provider = provider

    def chat_with_image(
        self,
        text: str,
        image_path: Union[str, bytes],
        trace: Optional[Any] = None,
    ) -> str:
        raise NotImplementedError(
            "Vision LLM provider '" + self.provider + "' is not implemented yet"
        )
