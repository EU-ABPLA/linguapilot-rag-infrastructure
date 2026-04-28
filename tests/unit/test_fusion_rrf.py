import math

import pytest

from core.query_engine.fusion import Fusion
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult


def _result(chunk_id: str, score: float = 0.0, text: str = "", source: str = "a.md") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text or chunk_id,
        metadata={"source_path": source},
    )


def test_rrf_fusion_is_deterministic_and_stable() -> None:
    fusion = Fusion(settings={"retrieval": {"top_k": 10}}, rrf_k=60)
    dense = [_result("chunk-a"), _result("chunk-b"), _result("chunk-c")]
    sparse = [_result("chunk-b"), _result("chunk-c"), _result("chunk-d")]
    first = fusion.fuse(dense, sparse, top_k=4)
    second = fusion.fuse(dense, sparse, top_k=4)
    assert [item.chunk_id for item in first] == ["chunk-b", "chunk-c", "chunk-a", "chunk-d"]
    assert [item.chunk_id for item in second] == [item.chunk_id for item in first]
    assert [item.score for item in second] == [item.score for item in first]


def test_rrf_k_is_configurable_from_settings() -> None:
    dense = [_result("chunk-a")]
    fusion = Fusion(settings={"retrieval": {"top_k": 5, "fusion": {"k": 10}}})
    results = fusion.fuse(dense, [], top_k=1)
    assert len(results) == 1
    assert math.isclose(results[0].score, 1.0 / 11.0, rel_tol=1e-9, abs_tol=1e-12)


def test_rrf_uses_default_top_k_from_settings() -> None:
    fusion = Fusion(settings={"retrieval": {"top_k": 2}}, rrf_k=50)
    dense = [_result("chunk-a"), _result("chunk-b"), _result("chunk-c")]
    results = fusion.fuse(dense, [])
    assert len(results) == 2


def test_rrf_tie_breaks_by_chunk_id() -> None:
    fusion = Fusion(settings={"retrieval": {"top_k": 5}}, rrf_k=1)
    dense = [_result("chunk-b"), _result("chunk-a")]
    sparse = [_result("chunk-a"), _result("chunk-b")]
    results = fusion.fuse(dense, sparse, top_k=2)
    assert len(results) == 2
    assert math.isclose(results[0].score, results[1].score, rel_tol=1e-9, abs_tol=1e-12)
    assert [item.chunk_id for item in results] == ["chunk-a", "chunk-b"]


def test_rrf_records_trace_stage() -> None:
    fusion = Fusion(settings={"retrieval": {"top_k": 5}}, rrf_k=60)
    trace = TraceContext(trace_type="query")
    results = fusion.fuse([_result("chunk-a")], [_result("chunk-a")], trace=trace)
    assert len(results) == 1
    stages = [stage for stage in trace.stages if stage["stage"] == "fusion"]
    assert len(stages) == 1
    assert stages[0]["details"]["result_count"] == 1
    assert stages[0]["details"]["rrf_k"] == 60


def test_rrf_rejects_invalid_rrf_k() -> None:
    with pytest.raises(ValueError) as exc_info:
        Fusion(settings={"retrieval": {"top_k": 5}}, rrf_k=0)
    assert "rrf_k must be positive" in str(exc_info.value)


def test_rrf_rejects_invalid_dense_results_type() -> None:
    fusion = Fusion(settings={"retrieval": {"top_k": 5}})
    with pytest.raises(ValueError) as exc_info:
        fusion.fuse("bad", [])  # type: ignore[arg-type]
    assert "dense_results must be a sequence of RetrievalResult" in str(exc_info.value)
