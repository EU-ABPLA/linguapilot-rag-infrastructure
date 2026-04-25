from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence


class BaseEvaluator(ABC):
	@abstractmethod
	def evaluate(
		self,
		query: str,
		retrieved_ids: Sequence[str],
		golden_ids: Sequence[str],
		trace: Optional[Any] = None,
	) -> Dict[str, float]:
		raise NotImplementedError


class UnavailableEvaluator(BaseEvaluator):
	def __init__(self, backend: str):
		self.backend = backend

	def evaluate(
		self,
		query: str,
		retrieved_ids: Sequence[str],
		golden_ids: Sequence[str],
		trace: Optional[Any] = None,
	) -> Dict[str, float]:
		raise NotImplementedError(
			"Evaluator backend '" + self.backend + "' is not implemented yet"
		)
