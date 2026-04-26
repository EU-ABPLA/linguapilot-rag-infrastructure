from abc import ABC, abstractmethod

from core.types import Document


class BaseLoader(ABC):
    @abstractmethod
    def load(self, path: str) -> Document:
        raise NotImplementedError
