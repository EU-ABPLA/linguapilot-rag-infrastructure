from __future__ import annotations

import json
import time

import pytest

from core.trace.trace_collector import TraceCollector
from core.trace.trace_context import TraceContext


def test_trace_context_defaults_to_query_type() -> None:
    trace = TraceContext()
    data = trace.to_dict()
    assert data["trace_type"] == "query"
    assert isinstance(data["trace_id"], str) and data["trace_id"]
    assert data["finished_at"] is None
    assert data["total_elapsed_ms"] is None
    assert data["stages"] == []


def test_trace_context_supports_ingestion_type_and_stage_elapsed_ms() -> None:
    trace = TraceContext(trace_type="ingestion", metadata={"source": "docs/a.pdf"})
    time.sleep(0.001)
    trace.record_stage("load", {"count": 1})
    stage_elapsed = trace.elapsed_ms("load")
    assert stage_elapsed >= 0.0
    payload = trace.to_dict()
    assert payload["trace_type"] == "ingestion"
    assert payload["stages"][0]["stage"] == "load"
    assert payload["stages"][0]["details"]["count"] == 1
    assert isinstance(payload["stages"][0]["elapsed_ms"], float)


def test_finish_sets_finished_at_and_total_elapsed_ms_and_is_json_serializable() -> None:
    trace = TraceContext(trace_type="query")
    trace.record_stage("query_processing", {"tokens": 12})
    time.sleep(0.001)
    trace.finish()
    payload = trace.to_dict()
    assert isinstance(payload["finished_at"], str) and payload["finished_at"]
    assert isinstance(payload["total_elapsed_ms"], float)
    assert payload["total_elapsed_ms"] >= payload["stages"][0]["elapsed_ms"]
    json.dumps(payload, ensure_ascii=True)


def test_trace_context_rejects_invalid_trace_type() -> None:
    with pytest.raises(ValueError) as exc_info:
        TraceContext(trace_type="generic")
    assert "trace_type must be query or ingestion" in str(exc_info.value)


def test_trace_collector_collects_trace_payload() -> None:
    received = []
    collector = TraceCollector(sink=lambda item: received.append(dict(item)))
    trace = TraceContext(trace_type="query")
    trace.record_stage("dense_retrieval", {"top_k": 5})
    trace.finish()
    collector.collect(trace)
    items = collector.list_items()
    assert len(items) == 1
    assert items[0]["trace_type"] == "query"
    assert items[0]["stages"][0]["stage"] == "dense_retrieval"
    assert len(received) == 1
    assert received[0]["trace_id"] == items[0]["trace_id"]
