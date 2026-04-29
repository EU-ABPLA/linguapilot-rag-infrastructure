from __future__ import annotations

from core.response.response_builder import ResponseBuilder
from core.types import RetrievalResult


def test_response_builder_returns_markdown_and_structured_citations() -> None:
    builder = ResponseBuilder()
    results = [
        RetrievalResult(
            chunk_id="chunk-1",
            score=0.91,
            text="Azure endpoint setup details for authentication and retry policy.",
            metadata={"source_path": "docs/azure.md", "page": 2},
        ),
        RetrievalResult(
            chunk_id="chunk-2",
            score=0.82,
            text="OpenAI deployment settings and model compatibility notes.",
            metadata={"source_path": "docs/openai.md"},
        ),
    ]
    payload = builder.build(results, "How to configure Azure?")
    assert "content" in payload
    assert "structuredContent" in payload
    text = payload["content"][0]["text"]
    assert "[1]" in text
    assert "[2]" in text
    citations = payload["structuredContent"]["citations"]
    assert len(citations) == 2
    assert citations[0]["source"] == "docs/azure.md"
    assert citations[0]["chunk_id"] == "chunk-1"
    assert isinstance(citations[0]["score"], float)


def test_response_builder_returns_friendly_message_when_empty() -> None:
    builder = ResponseBuilder()
    payload = builder.build([], "empty query")
    text = payload["content"][0]["text"]
    assert "未找到相关文档" in text
    assert payload["structuredContent"]["citations"] == []
