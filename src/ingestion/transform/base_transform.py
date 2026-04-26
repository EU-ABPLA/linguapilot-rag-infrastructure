from abc import ABC, abstractmethod
from typing import Any, List, Optional

from core.types import Chunk


class BaseTransform(ABC):
    @abstractmethod
    def transform(
        self, chunks: List[Chunk], trace: Optional[Any] = None
    ) -> List[Chunk]:
        raise NotImplementedError
