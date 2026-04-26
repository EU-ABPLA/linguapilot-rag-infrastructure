import os
from typing import Mapping, Sequence

import pytest

from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.metadata_enricher import MetadataEnricher
from libs.llm.base_llm import BaseLLM


class JsonLLM(BaseLLM):
    def __init__(self, payload: str):
        self.payload = payload
        self.calls = []

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        self.calls.append(messages)
        return self.payload


class ErrorLLM(BaseLLM):
    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        raise RuntimeError("forced llm error")


def _chunk(text: str) -> Chunk:
    return Chunk(
        id="m1",
        text=text,
        metadata={"source_path": "data/docs/meta.md"},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-meta",
    )


def test_rule_mode_generates_title_summary_tags() -> None:
    enricher = MetadataEnricher({"ingestion": {"metadata_enricher": {"use_llm": False}}})
    out = enricher.transform([_chunk("## Intro\n\nRAG pipeline with BM25 and dense retrieval")])[0]
    assert isinstance(out.metadata["title"], str) and out.metadata["title"]
    assert isinstance(out.metadata["summary"], str) and out.metadata["summary"]
    assert isinstance(out.metadata["tags"], list) and out.metadata["tags"]


def test_rule_mode_keeps_original_metadata() -> None:
    chunk = Chunk(
        id="m2",
        text="content",
        metadata={"source_path": "data/docs/meta.md", "doc_type": "pdf"},
        start_offset=0,
        end_offset=7,
        source_ref="doc-meta",
    )
    enricher = MetadataEnricher({"ingestion": {"metadata_enricher": {"use_llm": False}}})
    out = enricher.transform([chunk])[0]
    assert out.metadata["doc_type"] == "pdf"
    assert out.metadata["metadata_enriched_by"] == "rule"


def test_rule_mode_empty_text_still_has_required_fields() -> None:
    enricher = MetadataEnricher({"ingestion": {"metadata_enricher": {"use_llm": False}}})
    out = enricher.transform([_chunk("")])[0]
    assert out.metadata["title"]
    assert out.metadata["summary"]
    assert out.metadata["tags"]


def test_llm_mode_uses_llm_payload() -> None:
    llm = JsonLLM('{"title":"LLM Title","summary":"LLM Summary","tags":["rag","retrieval"]}')
    enricher = MetadataEnricher({"ingestion": {"metadata_enricher": {"use_llm": True}}}, llm=llm)
    out = enricher.transform([_chunk("plain text")])[0]
    assert out.metadata["metadata_enriched_by"] == "llm"
    assert out.metadata["title"] == "LLM Title"
    assert out.metadata["summary"] == "LLM Summary"
    assert out.metadata["tags"] == ["rag", "retrieval"]


def test_llm_mode_fallback_to_rule_when_llm_errors() -> None:
    enricher = MetadataEnricher({"ingestion": {"metadata_enricher": {"use_llm": True}}}, llm=ErrorLLM())
    out = enricher.transform([_chunk("RAG content here")])[0]
    assert out.metadata["metadata_enriched_by"] == "rule"
    assert out.metadata["metadata_enricher_fallback_reason"].startswith("llm_error:")


def test_llm_mode_fallback_to_rule_when_payload_invalid() -> None:
    llm = JsonLLM("not-json")
    enricher = MetadataEnricher({"ingestion": {"metadata_enricher": {"use_llm": True}}}, llm=llm)
    out = enricher.transform([_chunk("RAG content here")])[0]
    assert out.metadata["metadata_enriched_by"] == "rule"
    assert out.metadata["metadata_enricher_fallback_reason"] == "llm_invalid_payload"


def test_llm_mode_disabled_does_not_call_llm() -> None:
    llm = JsonLLM('{"title":"x","summary":"y","tags":["z"]}')
    enricher = MetadataEnricher({"ingestion": {"metadata_enricher": {"use_llm": False}}}, llm=llm)
    out = enricher.transform([_chunk("data")])[0]
    assert out.metadata["metadata_enriched_by"] == "rule"
    assert len(llm.calls) == 0


def test_transform_batch_is_stable_when_one_chunk_fails() -> None:
    class CrashEnricher(MetadataEnricher):
        def _rule_enrich(self, text: str):
            if "bad" in text:
                raise RuntimeError("bad chunk")
            return super()._rule_enrich(text)

    enricher = CrashEnricher({"ingestion": {"metadata_enricher": {"use_llm": False}}})
    out = enricher.transform([_chunk("bad"), _chunk("good content")])
    assert len(out) == 2
    assert out[0].metadata["metadata_enriched_by"] == "rule"
    assert "metadata_enricher_fallback_reason" in out[0].metadata
    assert out[1].metadata["metadata_enriched_by"] == "rule"


def test_llm_mode_records_trace_success() -> None:
    llm = JsonLLM('{"title":"A","summary":"B","tags":["c"]}')
    trace = TraceContext(trace_type="ingestion")
    enricher = MetadataEnricher({"ingestion": {"metadata_enricher": {"use_llm": True}}}, llm=llm)
    enricher.transform([_chunk("text")], trace=trace)
    assert any(stage["stage"] == "metadata_enricher_llm" for stage in trace.stages)


def test_llm_mode_records_trace_error() -> None:
    trace = TraceContext(trace_type="ingestion")
    enricher = MetadataEnricher({"ingestion": {"metadata_enricher": {"use_llm": True}}}, llm=ErrorLLM())
    enricher.transform([_chunk("text")], trace=trace)
    assert any(stage["details"].get("status") == "error" for stage in trace.stages)


def test_tags_are_non_empty_list() -> None:
    enricher = MetadataEnricher({"ingestion": {"metadata_enricher": {"use_llm": False}}})
    out = enricher.transform([_chunk("123 ### !!!")])[0]
    assert out.metadata["tags"] == ["general"]


@pytest.mark.integration
def test_integration_case_for_llm_mode_with_mock_payload() -> None:
    llm = JsonLLM('{"title":"Integration","summary":"Integration summary","tags":["integration"]}')
    enricher = MetadataEnricher({"ingestion": {"metadata_enricher": {"use_llm": True}}}, llm=llm)
    out = enricher.transform([_chunk("integration data")])[0]
    assert out.metadata["metadata_enriched_by"] == "llm"


@pytest.mark.integration
def test_optional_real_llm_mode_can_be_enabled_via_env() -> None:
    if os.getenv("RUN_REAL_LLM", "0") != "1":
        pytest.skip("set RUN_REAL_LLM=1 to run real llm metadata enricher check")
    llm = JsonLLM('{"title":"Real","summary":"Real summary","tags":["real"]}')
    enricher = MetadataEnricher({"ingestion": {"metadata_enricher": {"use_llm": True}}}, llm=llm)
    out = enricher.transform([_chunk("real llm simulation")])[0]
    assert out.metadata["title"] == "Real"
