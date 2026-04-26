from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.embedding.sparse_encoder import SparseEncoder


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={"source_path": "data/docs/sample.md"},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-1",
    )


def test_sparse_encoder_builds_term_weights_structure() -> None:
    encoder = SparseEncoder()
    chunks = [_chunk("c1", "RAG rag BM25"), _chunk("c2", "dense retrieval dense")]
    out = encoder.encode(chunks)
    assert len(out) == 2
    assert out[0]["chunk_id"] == "c1"
    assert out[0]["doc_length"] == 3
    assert out[0]["term_weights"] == {"rag": 2, "bm25": 1}
    assert out[1]["term_weights"]["dense"] == 2
    assert out[1]["term_weights"]["retrieval"] == 1


def test_sparse_encoder_handles_empty_text() -> None:
    encoder = SparseEncoder()
    out = encoder.encode([_chunk("c1", "   ")])
    assert out[0]["chunk_id"] == "c1"
    assert out[0]["doc_length"] == 0
    assert out[0]["term_weights"] == {}


def test_sparse_encoder_normalizes_case_and_filters_non_tokens() -> None:
    encoder = SparseEncoder()
    out = encoder.encode([_chunk("c1", "Hello, HELLO!! 123 @@ test_case")])
    assert out[0]["doc_length"] == 4
    assert out[0]["term_weights"] == {"hello": 2, "123": 1, "test_case": 1}


def test_sparse_encoder_returns_empty_for_empty_chunks() -> None:
    encoder = SparseEncoder()
    assert encoder.encode([]) == []


def test_sparse_encoder_records_trace_stage() -> None:
    encoder = SparseEncoder()
    trace = TraceContext(trace_type="ingestion")
    encoder.encode([_chunk("c1", "alpha beta"), _chunk("c2", "")], trace=trace)
    assert any(stage["stage"] == "sparse_encoder" for stage in trace.stages)
