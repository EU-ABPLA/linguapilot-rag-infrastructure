from abc import ABC, abstractmethod
from typing import Any, List, Optional


class BaseSplitter(ABC):
	@abstractmethod
	def split_text(self, text: str, trace: Optional[Any] = None) -> List[str]:
		raise NotImplementedError


class UnavailableSplitter(BaseSplitter):
	def __init__(self, provider: str):
		self.provider = provider

	def split_text(self, text: str, trace: Optional[Any] = None) -> List[str]:
		raise NotImplementedError(
			"Splitter provider '" + self.provider + "' is not implemented yet"
		)
