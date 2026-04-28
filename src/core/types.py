from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.id, "id")
        _validate_str(self.text, "text")
        _validate_metadata(self.metadata)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": _clone_mapping(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Document":
        return cls(
            id=str(data.get("id", "")),
            text=str(data.get("text", "")),
            metadata=_coerce_metadata(data.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, raw: str) -> "Document":
        parsed = json.loads(raw)
        if not isinstance(parsed, Mapping):
            raise ValueError("Document json must be an object")
        return cls.from_dict(parsed)


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_offset: int = 0
    end_offset: int = 0
    source_ref: Optional[str] = None

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.id, "id")
        _validate_str(self.text, "text")
        _validate_metadata(self.metadata)
        _validate_int(self.start_offset, "start_offset")
        _validate_int(self.end_offset, "end_offset")
        if self.end_offset < self.start_offset:
            raise ValueError("end_offset must be greater than or equal to start_offset")
        if self.source_ref is not None:
            _validate_non_empty_str(self.source_ref, "source_ref")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": _clone_mapping(self.metadata),
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "source_ref": self.source_ref,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Chunk":
        return cls(
            id=str(data.get("id", "")),
            text=str(data.get("text", "")),
            metadata=_coerce_metadata(data.get("metadata", {})),
            start_offset=_coerce_int(data.get("start_offset", 0), "start_offset"),
            end_offset=_coerce_int(data.get("end_offset", 0), "end_offset"),
            source_ref=_coerce_optional_str(data.get("source_ref")),
        )

    @classmethod
    def from_json(cls, raw: str) -> "Chunk":
        parsed = json.loads(raw)
        if not isinstance(parsed, Mapping):
            raise ValueError("Chunk json must be an object")
        return cls.from_dict(parsed)


@dataclass(frozen=True)
class ChunkRecord:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    dense_vector: Optional[List[float]] = None
    sparse_vector: Optional[List[float]] = None

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.id, "id")
        _validate_str(self.text, "text")
        _validate_metadata(self.metadata)
        _validate_optional_vector(self.dense_vector, "dense_vector")
        _validate_optional_vector(self.sparse_vector, "sparse_vector")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": _clone_mapping(self.metadata),
            "dense_vector": _clone_vector(self.dense_vector),
            "sparse_vector": _clone_vector(self.sparse_vector),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChunkRecord":
        return cls(
            id=str(data.get("id", "")),
            text=str(data.get("text", "")),
            metadata=_coerce_metadata(data.get("metadata", {})),
            dense_vector=_coerce_optional_vector(data.get("dense_vector"), "dense_vector"),
            sparse_vector=_coerce_optional_vector(data.get("sparse_vector"), "sparse_vector"),
        )

    @classmethod
    def from_json(cls, raw: str) -> "ChunkRecord":
        parsed = json.loads(raw)
        if not isinstance(parsed, Mapping):
            raise ValueError("ChunkRecord json must be an object")
        return cls.from_dict(parsed)


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.chunk_id, "chunk_id")
        _validate_float(self.score, "score")
        _validate_str(self.text, "text")
        _validate_generic_mapping(self.metadata, "metadata")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "score": float(self.score),
            "text": self.text,
            "metadata": _clone_mapping(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RetrievalResult":
        return cls(
            chunk_id=str(data.get("chunk_id", data.get("id", ""))),
            score=_coerce_float(data.get("score", 0.0), "score"),
            text=str(data.get("text", data.get("content", ""))),
            metadata=_coerce_generic_mapping(data.get("metadata", {}), "metadata"),
        )

    @classmethod
    def from_json(cls, raw: str) -> "RetrievalResult":
        parsed = json.loads(raw)
        if not isinstance(parsed, Mapping):
            raise ValueError("RetrievalResult json must be an object")
        return cls.from_dict(parsed)


def _validate_metadata(metadata: Mapping[str, Any]) -> None:
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    source_path = metadata.get("source_path")
    if not isinstance(source_path, str) or not source_path.strip():
        raise ValueError("metadata.source_path must be non-empty")
    images = metadata.get("images")
    if images is None:
        return
    if not isinstance(images, list):
        raise ValueError("metadata.images must be a list")
    for item in images:
        _validate_image_ref(item)


def _validate_image_ref(item: Any) -> None:
    if not isinstance(item, Mapping):
        raise ValueError("metadata.images item must be an object")
    _validate_non_empty_str(item.get("id"), "metadata.images[].id")
    _validate_non_empty_str(item.get("path"), "metadata.images[].path")
    if "page" in item:
        _validate_int(item.get("page"), "metadata.images[].page")
    if "text_offset" in item:
        _validate_int(item.get("text_offset"), "metadata.images[].text_offset")
    if "text_length" in item:
        _validate_int(item.get("text_length"), "metadata.images[].text_length")
    if "position" in item and not isinstance(item.get("position"), Mapping):
        raise ValueError("metadata.images[].position must be an object")


def _validate_non_empty_str(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(field_name + " must be non-empty")


def _validate_str(value: Any, field_name: str) -> None:
    if not isinstance(value, str):
        raise ValueError(field_name + " must be a string")


def _validate_int(value: Any, field_name: str) -> None:
    if not isinstance(value, int):
        raise ValueError(field_name + " must be an integer")


def _validate_float(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(field_name + " must be numeric")


def _validate_generic_mapping(value: Any, field_name: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(field_name + " must be a mapping")


def _validate_optional_vector(vector: Optional[Sequence[Any]], field_name: str) -> None:
    if vector is None:
        return
    if not isinstance(vector, Sequence):
        raise ValueError(field_name + " must be a sequence")
    for dim in vector:
        if not isinstance(dim, (int, float)):
            raise ValueError(field_name + " must contain numeric values")


def _coerce_metadata(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be a mapping")
    return _clone_mapping(value)


def _coerce_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(field_name + " must be an integer")
    return value


def _coerce_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("source_ref must be a string")
    return value


def _coerce_optional_vector(value: Any, field_name: str) -> Optional[List[float]]:
    if value is None:
        return None
    if not isinstance(value, Sequence):
        raise ValueError(field_name + " must be a sequence")
    output: List[float] = []
    for dim in value:
        if not isinstance(dim, (int, float)):
            raise ValueError(field_name + " must contain numeric values")
        output.append(float(dim))
    return output


def _coerce_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(field_name + " must be numeric")
    return float(value)


def _coerce_generic_mapping(value: Any, field_name: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(field_name + " must be a mapping")
    return _clone_mapping(value)


def _clone_mapping(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(dict(mapping), ensure_ascii=True))


def _clone_vector(value: Optional[Sequence[float]]) -> Optional[List[float]]:
    if value is None:
        return None
    return [float(item) for item in value]
