from typing import Any, Callable, Dict, Mapping

from libs.splitter.base_splitter import BaseSplitter, UnavailableSplitter
from libs.splitter.recursive_splitter import RecursiveSplitter

SplitterBuilder = Callable[[], BaseSplitter]


class SplitterFactory:
	_registry: Dict[str, SplitterBuilder] = {}

	@classmethod
	def register(cls, provider: str, builder: SplitterBuilder) -> None:
		normalized = _normalize_provider(provider)
		cls._registry[normalized] = builder

	@classmethod
	def unregister(cls, provider: str) -> None:
		normalized = _normalize_provider(provider)
		cls._registry.pop(normalized, None)

	@classmethod
	def create(cls, settings: Any) -> BaseSplitter:
		provider = _extract_provider(settings)
		builder = cls._registry.get(provider)
		if builder is None:
			raise ValueError("Unknown splitter provider: " + provider)
		instance = builder()
		if not isinstance(instance, BaseSplitter):
			raise TypeError(
				"Factory builder for '" + provider + "' must return BaseSplitter"
			)
		return instance


def _normalize_provider(provider: str) -> str:
	if not isinstance(provider, str) or not provider.strip():
		raise ValueError("Invalid splitter provider")
	return provider.strip().lower().replace("-", "_")


def _extract_provider(settings: Any) -> str:
	provider: Any
	if isinstance(settings, Mapping):
		provider = _provider_from_mapping(settings)
	else:
		provider = _provider_from_object(settings)
	return _normalize_provider(provider)


def _provider_from_mapping(settings: Mapping[str, Any]) -> Any:
	splitter = settings.get("splitter")
	if isinstance(splitter, Mapping) and "provider" in splitter:
		return splitter["provider"]
	if "provider" in settings:
		return settings["provider"]
	raise ValueError("Missing required field: splitter.provider")


def _provider_from_object(settings: Any) -> Any:
	splitter = getattr(settings, "splitter", None)
	if splitter is not None and hasattr(splitter, "provider"):
		return getattr(splitter, "provider")
	if hasattr(settings, "provider"):
		return getattr(settings, "provider")
	raise ValueError("Missing required field: splitter.provider")


SplitterFactory.register("recursive", RecursiveSplitter)
for _provider in ("semantic", "fixed_length"):
	SplitterFactory.register(
		_provider, lambda provider=_provider: UnavailableSplitter(provider)
	)
