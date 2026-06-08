from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence

import yaml


class SettingsError(ValueError):
    pass


_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass
class LLMSettings:
    provider: str
    model: str
    api_key: str = ""
    base_url: str = ""


@dataclass
class VisionLLMSettings:
    provider: str = ""
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    endpoint: str = ""
    api_version: str = "2024-02-15-preview"
    max_image_size: int = 2048


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
class MetadataEnricherSettings:
    use_llm: bool = False


@dataclass
class ImageCaptionerSettings:
    use_vision_llm: bool = False
    prompt_path: str = "config/prompts/image_captioning.txt"


@dataclass
class IngestionSettings:
    chunk_refiner: ChunkRefinerSettings
    metadata_enricher: MetadataEnricherSettings
    image_captioner: ImageCaptionerSettings


@dataclass
class Settings:
    llm: LLMSettings
    vision_llm: VisionLLMSettings
    embedding: EmbeddingSettings
    splitter: SplitterSettings
    vector_store: VectorStoreSettings
    retrieval: RetrievalSettings
    rerank: RerankSettings
    evaluation: EvaluationSettings
    observability: ObservabilitySettings
    ingestion: IngestionSettings


def load_settings(path: str) -> Settings:
    _load_dotenv_from_config_path(path)
    raw = _load_yaml(path)
    settings = Settings(
        llm=LLMSettings(
            provider=_require_str(raw, "llm.provider"),
            model=_require_str(raw, "llm.model"),
            api_key=_optional_str(raw, "llm.api_key", ""),
            base_url=_optional_str(raw, "llm.base_url", ""),
        ),
        vision_llm=VisionLLMSettings(
            provider=_optional_str(raw, "vision_llm.provider", ""),
            model=_optional_str(raw, "vision_llm.model", ""),
            api_key=_optional_str(raw, "vision_llm.api_key", ""),
            base_url=_optional_str(raw, "vision_llm.base_url", ""),
            endpoint=_optional_str(raw, "vision_llm.endpoint", ""),
            api_version=_optional_str(
                raw, "vision_llm.api_version", "2024-02-15-preview"
            ),
            max_image_size=_optional_int(raw, "vision_llm.max_image_size", 2048),
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
            ),
            metadata_enricher=MetadataEnricherSettings(
                use_llm=_optional_bool(raw, "ingestion.metadata_enricher.use_llm", False)
            ),
            image_captioner=ImageCaptionerSettings(
                use_vision_llm=_optional_bool(
                    raw, "ingestion.image_captioner.use_vision_llm", False
                ),
                prompt_path=_optional_str(
                    raw,
                    "ingestion.image_captioner.prompt_path",
                    "config/prompts/image_captioning.txt",
                ),
            ),
        ),
    )
    validate_settings(settings)
    return settings


def validate_settings(settings: Settings) -> None:
    if not settings.llm.provider:
        raise SettingsError("Missing required field: llm.provider")
    if settings.ingestion.image_captioner.use_vision_llm:
        if not settings.vision_llm.provider:
            raise SettingsError("Missing required field: vision_llm.provider")
        if not settings.vision_llm.model:
            raise SettingsError("Missing required field: vision_llm.model")
        if settings.vision_llm.max_image_size <= 0:
            raise SettingsError("Invalid value for field: vision_llm.max_image_size")
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
    if not settings.ingestion.image_captioner.prompt_path:
        raise SettingsError("Invalid value for field: ingestion.image_captioner.prompt_path")
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


def _load_dotenv_from_config_path(config_path: str) -> None:
    config_file = Path(config_path).resolve()
    for folder in [config_file.parent, *config_file.parents]:
        dotenv_path = folder / ".env"
        if dotenv_path.exists() and dotenv_path.is_file():
            _merge_dotenv_into_environ(dotenv_path)
            return


def _merge_dotenv_into_environ(dotenv_path: Path) -> None:
    with dotenv_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            normalized_key = key.strip()
            if not normalized_key:
                continue
            normalized_value = value.strip()
            if (
                len(normalized_value) >= 2
                and normalized_value[0] == normalized_value[-1]
                and normalized_value[0] in ("'", '"')
            ):
                normalized_value = normalized_value[1:-1]
            existing_value = os.environ.get(normalized_key)
            if existing_value is None or str(existing_value).strip() == "":
                os.environ[normalized_key] = normalized_value


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
    return _resolve_env_placeholders(value, field)


def _optional_str(raw: Dict[str, Any], field: str, default: str) -> str:
    value = _optional_get(raw, field, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise SettingsError(f"Invalid value for field: {field}")
    return _resolve_env_placeholders(value, field)


def _resolve_env_placeholders(value: str, field: str) -> str:
    if "${" not in value:
        return value
    missing = [name for name in _ENV_VAR_PATTERN.findall(value) if name not in os.environ]
    if missing:
        raise SettingsError(
            "Missing environment variable(s) for field "
            + field
            + ": "
            + ", ".join(sorted(set(missing)))
        )
    return _ENV_VAR_PATTERN.sub(lambda match: os.environ[match.group(1)], value)


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
