from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from libs.splitter.base_splitter import BaseSplitter, UnavailableSplitter
from libs.splitter.recursive_splitter import RecursiveSplitter

SplitterBuilder = Callable[..., BaseSplitter]


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
        instance = _build_with_settings(builder, settings)
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


def _build_with_settings(builder: SplitterBuilder, settings: Any) -> BaseSplitter:
    if _builder_accepts_settings(builder):
        return builder(settings)
    return builder()


def _builder_accepts_settings(builder: SplitterBuilder) -> bool:
    try:
        signature = inspect.signature(builder)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            return True
    return False


def _build_recursive(settings: Any) -> BaseSplitter:
    config = _extract_splitter_config(settings)
    chunk_size = _coerce_positive_int(config.get("chunk_size"), 800)
    chunk_overlap = _coerce_non_negative_int(config.get("chunk_overlap"), 100)
    separators = _coerce_optional_separators(config.get("separators"))
    return RecursiveSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )


def _extract_splitter_config(settings: Any) -> Dict[str, Any]:
    if isinstance(settings, Mapping):
        splitter = settings.get("splitter")
        if isinstance(splitter, Mapping):
            return dict(splitter)
        return {}
    splitter = getattr(settings, "splitter", None)
    if splitter is None:
        return {}
    if isinstance(splitter, Mapping):
        return dict(splitter)
    output: Dict[str, Any] = {}
    for field in ("provider", "chunk_size", "chunk_overlap", "separators"):
        if hasattr(splitter, field):
            output[field] = getattr(splitter, field)
    return output


def _coerce_positive_int(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("splitter.chunk_size must be an integer")
    if value <= 0:
        raise ValueError("splitter.chunk_size must be greater than 0")
    return value


def _coerce_non_negative_int(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("splitter.chunk_overlap must be an integer")
    if value < 0:
        raise ValueError("splitter.chunk_overlap must be greater than or equal to 0")
    return value


def _coerce_optional_separators(value: Any) -> Optional[Sequence[str]]:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("splitter.separators must be a sequence of strings")
    output = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("splitter.separators must contain only strings")
        output.append(item)
    return tuple(output)


SplitterFactory.register("recursive", _build_recursive)
for _provider in ("semantic", "fixed_length"):
    SplitterFactory.register(
        _provider, lambda provider=_provider: UnavailableSplitter(provider)
    )
