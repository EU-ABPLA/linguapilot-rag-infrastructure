from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


def get_tool_schema() -> Dict[str, Any]:
    return {
        "name": "get_document_summary",
        "description": "Get summary metadata by document id.",
        "inputSchema": {
            "type": "object",
            "properties": {"doc_id": {"type": "string"}},
            "required": ["doc_id"],
        },
    }


def call_get_document_summary(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    return get_document_summary(
        doc_id=arguments.get("doc_id"),
        metadata_index=arguments.get("metadata_index"),
        metadata_path=arguments.get("metadata_path", "data/db/document_metadata.json"),
    )


def get_document_summary(
    doc_id: Any,
    metadata_index: Optional[Mapping[str, Any]] = None,
    metadata_path: str = "data/db/document_metadata.json",
) -> Dict[str, Any]:
    normalized_doc_id = _normalize_doc_id(doc_id)
    index = (
        _normalize_index(metadata_index)
        if metadata_index is not None
        else _load_metadata_index(metadata_path)
    )
    payload = index.get(normalized_doc_id)
    if payload is None:
        raise ValueError("doc_id not found")
    if not isinstance(payload, Mapping):
        raise ValueError("document metadata is invalid")
    title = _normalize_text(payload.get("title"), "title")
    summary = _normalize_text(payload.get("summary"), "summary")
    tags = _normalize_tags(payload.get("tags"))
    result: Dict[str, Any] = {
        "doc_id": normalized_doc_id,
        "title": title,
        "summary": summary,
        "tags": tags,
    }
    created_at = payload.get("created_at")
    if isinstance(created_at, str) and created_at.strip():
        result["created_at"] = created_at.strip()
    return result


def _normalize_doc_id(doc_id: Any) -> str:
    if not isinstance(doc_id, str):
        raise ValueError("doc_id must be a string")
    normalized = doc_id.strip()
    if not normalized:
        raise ValueError("doc_id must be non-empty")
    return normalized


def _load_metadata_index(metadata_path: str) -> Dict[str, Dict[str, Any]]:
    path = Path(metadata_path)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return _normalize_index(raw)


def _normalize_index(value: Any) -> Dict[str, Dict[str, Any]]:
    if isinstance(value, Mapping):
        output: Dict[str, Dict[str, Any]] = {}
        for raw_key, raw_item in value.items():
            key = _normalize_doc_id(raw_key)
            if not isinstance(raw_item, Mapping):
                raise ValueError("document metadata is invalid")
            output[key] = dict(raw_item)
        return output
    if isinstance(value, list):
        output = {}
        for raw_item in value:
            if not isinstance(raw_item, Mapping):
                raise ValueError("document metadata is invalid")
            key = _normalize_doc_id(raw_item.get("doc_id"))
            output[key] = dict(raw_item)
        return output
    raise ValueError("document metadata index must be a mapping or list")


def _normalize_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(field_name + " must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(field_name + " must be non-empty")
    return normalized


def _normalize_tags(value: Any) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("tags must be a list")
    tags: List[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("tags item must be a string")
        normalized = item.strip()
        if normalized:
            tags.append(normalized)
    return tags
