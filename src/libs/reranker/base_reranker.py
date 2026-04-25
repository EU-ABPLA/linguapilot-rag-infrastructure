from abc import ABC, abstractmethod
from typing import Any, List, Mapping, Optional, Sequence


class BaseReranker(ABC):
	@abstractmethod
	def rerank(
		self,
		query: str,
		candidates: Sequence[Mapping[str, Any]],
		trace: Optional[Any] = None,
	) -> List[Mapping[str, Any]]:
		raise NotImplementedError


class NoneReranker(BaseReranker):
	def rerank(
		self,
		query: str,
		candidates: Sequence[Mapping[str, Any]],
		trace: Optional[Any] = None,
	) -> List[Mapping[str, Any]]:
		return list(candidates)


class UnavailableReranker(BaseReranker):
	def __init__(self, backend: str):
		self.backend = backend

	def rerank(
		self,
		query: str,
		candidates: Sequence[Mapping[str, Any]],
		trace: Optional[Any] = None,
	) -> List[Mapping[str, Any]]:
		raise NotImplementedError(
			"Reranker backend '" + self.backend + "' is not implemented yet"
		)
