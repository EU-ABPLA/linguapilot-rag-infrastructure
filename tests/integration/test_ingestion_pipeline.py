import re
from pathlib import Path
from typing import Any, List, Optional

import pytest

from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.embedding.batch_processor import BatchEncodingResult
from ingestion.pipeline import IngestionPipeline, IngestionResult
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from libs.loader.file_integrity import SQLiteIntegrityChecker
from libs.loader.pdf_loader import PdfLoader
from libs.vector_store.chroma_store import ChromaStore


class FakeBatchProcessor:
    def process(
        self, chunks: List[Chunk], trace: Optional[TraceContext] = None
    ) -> List[BatchEncodingResult]:
        output: List[BatchEncodingResult] = []
        for chunk in chunks:
            tokens = [token.lower() for token in re.findall(r"[A-Za-z0-9_]+", chunk.text)]
            term_weights = {}
            for token in tokens:
                term_weights[token] = term_weights.get(token, 0) + 1
            output.append(
                BatchEncodingResult(
                    chunk_id=chunk.id,
                    text=chunk.text,
                    metadata=dict(chunk.metadata),
                    dense_vector=[float(len(tokens) or 1), float(len(chunk.text) or 1)],
                    sparse_stats={
                        "chunk_id": chunk.id,
                        "doc_length": len(tokens),
                        "term_weights": term_weights,
                    },
                )
            )
        if trace is not None:
            trace.record_stage(
                "fake_batch_processor",
                {"status": "ok", "chunk_count": len(output)},
            )
        return output


class BrokenBatchProcessor:
    def process(
        self, chunks: List[Chunk], trace: Optional[TraceContext] = None
    ) -> List[BatchEncodingResult]:
        raise RuntimeError("forced batch failure")


def _create_pipeline(tmp_path: Path, batch_processor: Any) -> IngestionPipeline:
    settings = {
        "splitter": {"provider": "recursive", "chunk_size": 120, "chunk_overlap": 0},
        "vector_store": {"provider": "chroma", "collection": "default"},
        "ingestion": {
            "chunk_refiner": {"use_llm": False},
            "metadata_enricher": {"use_llm": False},
            "image_captioner": {"use_vision_llm": False},
        },
    }
    return IngestionPipeline(
        settings=settings,
        integrity_checker=SQLiteIntegrityChecker(str(tmp_path / "db" / "ingestion_history.db")),
        loader=PdfLoader(image_output_root=str(tmp_path / "extracted_images")),
        batch_processor=batch_processor,
        vector_store=ChromaStore(
            persist_directory=str(tmp_path / "db" / "chroma"),
            collection="default",
        ),
        bm25_indexer=BM25Indexer(index_dir=str(tmp_path / "db" / "bm25")),
        image_storage=ImageStorage(
            db_path=str(tmp_path / "db" / "image_index.db"),
            image_root=str(tmp_path / "images"),
        ),
    )


def _write_complex_pdf(path: Path) -> None:
    text = (
        b"%PDF-1.4\n"
        b"1 0 obj\n"
        b"Chapter 1 introduction\n"
        b"Chunking and retrieval design.\n"
        b"/Subtype /Image\n"
        b"endobj\n"
        b"2 0 obj\n"
        b"Chapter 2 indexing and search.\n"
        b"/Subtype /Image\n"
        b"endobj\n"
        b"3 0 obj\n"
        b"Chapter 3 pipeline orchestration.\n"
        b"/Subtype /Image\n"
        b"endobj\n"
        b"%%EOF"
    )
    path.write_bytes(text)


@pytest.mark.integration
def test_ingestion_pipeline_runs_full_flow_and_persists_outputs(tmp_path: Path) -> None:
    source_pdf = tmp_path / "complex_technical_doc.pdf"
    _write_complex_pdf(source_pdf)
    pipeline = _create_pipeline(tmp_path, batch_processor=FakeBatchProcessor())
    progress_events: List[str] = []
    trace = TraceContext(trace_type="ingestion")
    result = pipeline.run(
        str(source_pdf),
        collection="default",
        on_progress=lambda stage, current, total: progress_events.append(
            stage + ":" + str(current) + "/" + str(total)
        ),
        trace=trace,
    )
    assert isinstance(result, IngestionResult)
    assert result.status == "success"
    assert result.skipped is False
    assert result.chunk_count > 0
    assert result.vector_count == result.chunk_count
    assert result.image_count == 3
    assert (tmp_path / "db" / "bm25" / "bm25_index.pkl").exists()
    assert (tmp_path / "db" / "chroma" / "default.json").exists()
    saved_images = ImageStorage(
        db_path=str(tmp_path / "db" / "image_index.db"),
        image_root=str(tmp_path / "images"),
    ).list_images(collection="default")
    assert len(saved_images) == 3
    assert set(item.split(":")[0] for item in progress_events) == {
        "load",
        "split",
        "transform",
        "embed",
        "upsert",
    }
    required_stages = {"load", "split", "transform", "embed", "upsert"}
    by_stage = {}
    for item in trace.stages:
        name = item.get("stage")
        if name in required_stages:
            by_stage[name] = item
    assert set(by_stage.keys()) == required_stages
    for stage_name in required_stages:
        stage_item = by_stage[stage_name]
        assert isinstance(stage_item.get("elapsed_ms"), float)
        details = stage_item.get("details", {})
        assert isinstance(details.get("method"), str) and details["method"]
        assert isinstance(details.get("provider"), str) and details["provider"]
    payload = trace.to_dict()
    assert payload["trace_type"] == "ingestion"


@pytest.mark.integration
def test_ingestion_pipeline_second_run_skips_when_file_unchanged(tmp_path: Path) -> None:
    source_pdf = tmp_path / "simple.pdf"
    _write_complex_pdf(source_pdf)
    pipeline = _create_pipeline(tmp_path, batch_processor=FakeBatchProcessor())
    first = pipeline.run(str(source_pdf), collection="default")
    second = pipeline.run(str(source_pdf), collection="default")
    assert first.status == "success"
    assert second.status == "skipped"
    assert second.skipped is True
    assert second.file_hash == first.file_hash


@pytest.mark.integration
def test_ingestion_pipeline_reports_stage_error_clearly(tmp_path: Path) -> None:
    source_pdf = tmp_path / "broken.pdf"
    _write_complex_pdf(source_pdf)
    pipeline = _create_pipeline(tmp_path, batch_processor=BrokenBatchProcessor())
    with pytest.raises(RuntimeError) as exc_info:
        pipeline.run(str(source_pdf), collection="default")
    message = str(exc_info.value)
    assert "ingestion pipeline error at embed:" in message
    assert "forced batch failure" in message
