from typing import Any, Dict, List, Mapping, Optional, Sequence

from core.query_engine.reranker import Reranker
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult
from libs.reranker.base_reranker import BaseReranker


class SuccessBackend(BaseReranker):
    def rerank(
        self,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
        trace: Optional[Any] = None,
    ) -> List[Mapping[str, Any]]:
        by_id: Dict[str, Mapping[str, Any]] = {str(item["id"]): item for item in candidates}
        return [by_id["b"], by_id["a"], by_id["c"]]


class FailingBackend(BaseReranker):
    def rerank(
        self,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
        trace: Optional[Any] = None,
    ) -> List[Mapping[str, Any]]:
        raise RuntimeError("cross_encoder fallback required: timeout: slow model")


def _candidate(chunk_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=chunk_id + "-text",
        metadata={"source_path": "docs/" + chunk_id + ".md"},
    )


def test_reranker_applies_backend_order() -> None:
    reranker = Reranker(
        settings={"retrieval": {"top_k": 5}, "rerank": {"enabled": True}},
        backend=SuccessBackend(),
    )
    candidates = [_candidate("a", 0.9), _candidate("b", 0.8), _candidate("c", 0.7)]
    out = reranker.rerank("query", candidates, top_k=2)
    assert [item.chunk_id for item in out] == ["b", "a"]
    assert all(item.metadata.get("rerank_fallback") is None for item in out)


def test_reranker_fallback_returns_original_and_marks_metadata() -> None:
    reranker = Reranker(
        settings={"retrieval": {"top_k": 5}, "rerank": {"enabled": True}},
        backend=FailingBackend(),
    )
    candidates = [_candidate("a", 0.9), _candidate("b", 0.8), _candidate("c", 0.7)]
    out = reranker.rerank("query", candidates, top_k=2)
    assert [item.chunk_id for item in out] == ["a", "b"]
    assert out[0].metadata["rerank_fallback"] is True
    assert "timeout" in str(out[0].metadata["rerank_fallback_reason"])


def test_reranker_records_fallback_stage() -> None:
    reranker = Reranker(
        settings={"retrieval": {"top_k": 5}, "rerank": {"enabled": True}},
        backend=FailingBackend(),
    )
    trace = TraceContext(trace_type="query")
    out = reranker.rerank("query", [_candidate("a", 0.9)], trace=trace)
    assert len(out) == 1
    stages = [stage for stage in trace.stages if stage["stage"] == "reranker"]
    assert len(stages) == 1
    assert stages[0]["details"]["fallback"] is True


def test_reranker_skips_when_disabled() -> None:
    reranker = Reranker(
        settings={"retrieval": {"top_k": 5}, "rerank": {"enabled": False}},
        backend=SuccessBackend(),
    )
    candidates = [_candidate("a", 0.9), _candidate("b", 0.8)]
    out = reranker.rerank("query", candidates)
    assert [item.chunk_id for item in out] == ["a", "b"]
