from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml


class SettingsError(ValueError):
    pass


@dataclass
class LLMSettings:
    provider: str
    model: str
    api_key: str = ""


@dataclass
class EmbeddingSettings:
    provider: str
    model: str
    dimensions: int = 0


@dataclass
class SplitterSettings:
    provider: str = "recursive"
    chunk_size: int = 800
    chunk_overlap: int = 100
    separators: Optional[List[str]] = None


@dataclass
class VectorStoreSettings:
    provider: str
    collection: str = "default"


@dataclass
class RetrievalSettings:
    top_k: int
    use_hybrid: bool = True


@dataclass
class RerankSettings:
    enabled: bool
    provider: str = "none"
    model: str = ""


@dataclass
class EvaluationSettings:
    backends: List[str]
    golden_test_set: str = "tests/fixtures/golden_test_set.json"


@dataclass
class ObservabilitySettings:
    enabled: bool
    log_file: str = "logs/traces.jsonl"


@dataclass
class ChunkRefinerSettings:
    use_llm: bool = False
    prompt_path: str = "config/prompts/chunk_refinement.txt"


@dataclass
class IngestionSettings:
    chunk_refiner: ChunkRefinerSettings


@dataclass
class Settings:
    llm: LLMSettings
    embedding: EmbeddingSettings
    splitter: SplitterSettings
    vector_store: VectorStoreSettings
    retrieval: RetrievalSettings
    rerank: RerankSettings
    evaluation: EvaluationSettings
    observability: ObservabilitySettings
    ingestion: IngestionSettings


def load_settings(path: str) -> Settings:
    raw = _load_yaml(path)
    settings = Settings(
        llm=LLMSettings(
            provider=_require_str(raw, "llm.provider"),
            model=_require_str(raw, "llm.model"),
            api_key=_optional_str(raw, "llm.api_key", ""),
        ),
        embedding=EmbeddingSettings(
            provider=_require_str(raw, "embedding.provider"),
            model=_require_str(raw, "embedding.model"),
            dimensions=_optional_int(raw, "embedding.dimensions", 0),
        ),
        splitter=SplitterSettings(
            provider=_optional_str(raw, "splitter.provider", "recursive"),
            chunk_size=_optional_int(raw, "splitter.chunk_size", 800),
            chunk_overlap=_optional_int(raw, "splitter.chunk_overlap", 100),
            separators=_optional_str_list(raw, "splitter.separators", None),
        ),
        vector_store=VectorStoreSettings(
            provider=_require_str(raw, "vector_store.provider"),
            collection=_optional_str(raw, "vector_store.collection", "default"),
        ),
        retrieval=RetrievalSettings(
            top_k=_require_int(raw, "retrieval.top_k"),
            use_hybrid=_optional_bool(raw, "retrieval.use_hybrid", True),
        ),
        rerank=RerankSettings(
            enabled=_require_bool(raw, "rerank.enabled"),
            provider=_optional_str(raw, "rerank.provider", "none"),
            model=_optional_str(raw, "rerank.model", ""),
        ),
        evaluation=EvaluationSettings(
            backends=_require_str_list(raw, "evaluation.backends"),
            golden_test_set=_optional_str(
                raw,
                "evaluation.golden_test_set",
                "tests/fixtures/golden_test_set.json",
            ),
        ),
        observability=ObservabilitySettings(
            enabled=_require_bool(raw, "observability.enabled"),
            log_file=_optional_str(raw, "observability.log_file", "logs/traces.jsonl"),
        ),
        ingestion=IngestionSettings(
            chunk_refiner=ChunkRefinerSettings(
                use_llm=_optional_bool(raw, "ingestion.chunk_refiner.use_llm", False),
                prompt_path=_optional_str(
                    raw,
                    "ingestion.chunk_refiner.prompt_path",
                    "config/prompts/chunk_refinement.txt",
                ),
            )
        ),
    )
    validate_settings(settings)
    return settings


def validate_settings(settings: Settings) -> None:
    if not settings.llm.provider:
        raise SettingsError("Missing required field: llm.provider")
    if not settings.embedding.provider:
        raise SettingsError("Missing required field: embedding.provider")
    if not settings.splitter.provider:
        raise SettingsError("Missing required field: splitter.provider")
    if settings.splitter.chunk_size <= 0:
        raise SettingsError("Invalid value for field: splitter.chunk_size")
    if settings.splitter.chunk_overlap < 0:
        raise SettingsError("Invalid value for field: splitter.chunk_overlap")
    if settings.retrieval.top_k <= 0:
        raise SettingsError("Invalid value for field: retrieval.top_k")
    if not settings.ingestion.chunk_refiner.prompt_path:
        raise SettingsError("Invalid value for field: ingestion.chunk_refiner.prompt_path")
    if not settings.evaluation.backends:
        raise SettingsError("Missing required field: evaluation.backends")


def _load_yaml(path: str) -> Dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise SettingsError(f"Config file not found: {path}")
    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SettingsError("Invalid config root type: expected mapping")
    return data


def _get(raw: Dict[str, Any], field: str) -> Any:
    current: Any = raw
    for part in field.split("."):
        if not isinstance(current, dict) or part not in current:
            raise SettingsError(f"Missing required field: {field}")
        current = current[part]
    return current


def _optional_get(raw: Dict[str, Any], field: str, default: Any) -> Any:
    current: Any = raw
    for part in field.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _require_str(raw: Dict[str, Any], field: str) -> str:
    value = _get(raw, field)
    if not isinstance(value, str) or not value.strip():
        raise SettingsError(f"Invalid value for field: {field}")
    return value


def _optional_str(raw: Dict[str, Any], field: str, default: str) -> str:
    value = _optional_get(raw, field, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise SettingsError(f"Invalid value for field: {field}")
    return value


def _require_int(raw: Dict[str, Any], field: str) -> int:
    value = _get(raw, field)
    if not isinstance(value, int):
        raise SettingsError(f"Invalid value for field: {field}")
    return value


def _optional_int(raw: Dict[str, Any], field: str, default: int) -> int:
    value = _optional_get(raw, field, default)
    if not isinstance(value, int):
        raise SettingsError(f"Invalid value for field: {field}")
    return value


def _require_bool(raw: Dict[str, Any], field: str) -> bool:
    value = _get(raw, field)
    if not isinstance(value, bool):
        raise SettingsError(f"Invalid value for field: {field}")
    return value


def _optional_bool(raw: Dict[str, Any], field: str, default: bool) -> bool:
    value = _optional_get(raw, field, default)
    if not isinstance(value, bool):
        raise SettingsError(f"Invalid value for field: {field}")
    return value


def _require_str_list(raw: Dict[str, Any], field: str) -> List[str]:
    value = _get(raw, field)
    if not isinstance(value, list) or not value:
        raise SettingsError(f"Invalid value for field: {field}")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise SettingsError(f"Invalid value for field: {field}")
    return value


def _optional_str_list(
    raw: Dict[str, Any], field: str, default: Optional[List[str]]
) -> Optional[List[str]]:
    value = _optional_get(raw, field, default)
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SettingsError(f"Invalid value for field: {field}")
    output: List[str] = []
    for item in value:
        if not isinstance(item, str):
            raise SettingsError(f"Invalid value for field: {field}")
        output.append(item)
    return output
