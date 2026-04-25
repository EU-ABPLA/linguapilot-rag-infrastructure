from abc import ABC, abstractmethod
from typing import Any, List, Mapping, Optional, Sequence


class BaseVectorStore(ABC):
	@abstractmethod
	def upsert(
		self, records: Sequence[Mapping[str, Any]], trace: Optional[Any] = None
	) -> None:
		raise NotImplementedError

	@abstractmethod
	def query(
		self,
		vector: Sequence[float],
		top_k: int,
		filters: Optional[Mapping[str, Any]] = None,
		trace: Optional[Any] = None,
	) -> List[Mapping[str, Any]]:
		raise NotImplementedError


class UnavailableVectorStore(BaseVectorStore):
	def __init__(self, provider: str):
		self.provider = provider

	def upsert(
		self, records: Sequence[Mapping[str, Any]], trace: Optional[Any] = None
	) -> None:
		raise NotImplementedError(
			"VectorStore provider '" + self.provider + "' is not implemented yet"
		)

	def query(
		self,
		vector: Sequence[float],
		top_k: int,
		filters: Optional[Mapping[str, Any]] = None,
		trace: Optional[Any] = None,
	) -> List[Mapping[str, Any]]:
		raise NotImplementedError(
			"VectorStore provider '" + self.provider + "' is not implemented yet"
		)
