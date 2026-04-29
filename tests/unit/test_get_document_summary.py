from __future__ import annotations

import json

import pytest

from mcp_server.tools.get_document_summary import (
    call_get_document_summary,
    get_document_summary,
    get_tool_schema,
)


def test_get_tool_schema_declares_doc_id_required() -> None:
    schema = get_tool_schema()
    assert schema["name"] == "get_document_summary"
    assert schema["inputSchema"]["required"] == ["doc_id"]


def test_get_document_summary_returns_structured_payload() -> None:
    payload = get_document_summary(
        doc_id="doc-1",
        metadata_index={
            "doc-1": {
                "title": "Azure Setup",
                "summary": "How to configure Azure OpenAI settings.",
                "tags": ["azure", "openai"],
                "created_at": "2026-04-29T10:30:00Z",
            }
        },
    )
    assert payload == {
        "doc_id": "doc-1",
        "title": "Azure Setup",
        "summary": "How to configure Azure OpenAI settings.",
        "tags": ["azure", "openai"],
        "created_at": "2026-04-29T10:30:00Z",
    }


def test_get_document_summary_raises_for_missing_doc_id() -> None:
    with pytest.raises(ValueError) as exc_info:
        get_document_summary(
            doc_id="missing",
            metadata_index={
                "doc-1": {
                    "title": "t",
                    "summary": "s",
                    "tags": ["x"],
                }
            },
        )
    assert "doc_id not found" in str(exc_info.value)


def test_get_document_summary_raises_for_invalid_doc_id_type() -> None:
    with pytest.raises(ValueError) as exc_info:
        get_document_summary(
            doc_id=123,
            metadata_index={
                "doc-1": {
                    "title": "t",
                    "summary": "s",
                    "tags": ["x"],
                }
            },
        )
    assert "doc_id must be a string" in str(exc_info.value)


def test_get_document_summary_loads_metadata_from_json_file(tmp_path) -> None:
    metadata_path = tmp_path / "document_metadata.json"
    metadata_path.write_text(
        json.dumps(
            [
                {
                    "doc_id": "doc-from-file",
                    "title": "File Doc",
                    "summary": "Read from metadata file.",
                    "tags": ["file", "metadata"],
                }
            ],
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    payload = get_document_summary(
        doc_id="doc-from-file",
        metadata_path=str(metadata_path),
    )
    assert payload == {
        "doc_id": "doc-from-file",
        "title": "File Doc",
        "summary": "Read from metadata file.",
        "tags": ["file", "metadata"],
    }


def test_call_get_document_summary_uses_doc_id_argument() -> None:
    payload = call_get_document_summary(
        {"doc_id": "doc-1", "metadata_index": {"doc-1": {"title": "t", "summary": "s"}}}
    )
    assert payload["doc_id"] == "doc-1"
