from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence


class BaseEmbedding(ABC):
	@abstractmethod
	def embed(
		self, texts: Sequence[str], trace: Optional[Any] = None
	) -> List[List[float]]:
		raise NotImplementedError


class UnavailableEmbedding(BaseEmbedding):
	def __init__(self, provider: str):
		self.provider = provider

	def embed(
		self, texts: Sequence[str], trace: Optional[Any] = None
	) -> List[List[float]]:
		raise NotImplementedError(
			f"Embedding provider '{self.provider}' is not implemented yet"
		)
