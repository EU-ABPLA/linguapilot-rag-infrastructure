from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.types import Chunk, Document
from ingestion.embedding.batch_processor import BatchEncodingResult
from ingestion.pipeline import IngestionPipeline


class FakeIntegrityChecker:
    def __init__(self) -> None:
        self.marked_success = False
        self.marked_failed = False

    def compute_sha256(self, path: str) -> str:
        return "fake-hash-" + Path(path).name

    def should_skip(self, file_hash: str) -> bool:
        return False

    def mark_success(
        self,
        file_hash: str,
        file_path: str,
        file_size: int,
        chunk_count: int,
    ) -> None:
        self.marked_success = True

    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        self.marked_failed = True


class FakeLoader:
    def load(self, path: str) -> Document:
        return Document(
            id="doc-1",
            text="hello world",
            metadata={"source_path": path},
        )


class FakeChunker:
    def split_document(self, document: Document) -> List[Chunk]:
        return [
            Chunk(
                id="chunk-1",
                text="hello world",
                metadata={"source_path": document.metadata["source_path"]},
            )
        ]


class FakeTransform:
    def __init__(self, name: str):
        self._name = name

    def transform(self, chunks: List[Chunk], trace: Optional[Any] = None) -> List[Chunk]:
        output: List[Chunk] = []
        for item in chunks:
            metadata = dict(item.metadata)
            metadata[self._name] = True
            output.append(
                Chunk(
                    id=item.id,
                    text=item.text,
                    metadata=metadata,
                    start_offset=item.start_offset,
                    end_offset=item.end_offset,
                    source_ref=item.source_ref,
                )
            )
        return output


class FakeBatchProcessor:
    def process(
        self,
        chunks: List[Chunk],
        trace: Optional[Any] = None,
    ) -> List[BatchEncodingResult]:
        output: List[BatchEncodingResult] = []
        for item in chunks:
            output.append(
                BatchEncodingResult(
                    chunk_id=item.id,
                    text=item.text,
                    metadata=dict(item.metadata),
                    dense_vector=[1.0, 2.0],
                    sparse_stats={
                        "chunk_id": item.id,
                        "doc_length": 2,
                        "term_weights": {"hello": 1, "world": 1},
                    },
                )
            )
        return output


class FakeVectorUpserter:
    def upsert(
        self,
        chunks: List[Chunk],
        dense_vectors: List[List[float]],
        trace: Optional[Any] = None,
    ) -> List[str]:
        return ["vec-1"]


class FakeBM25Indexer:
    def update(
        self,
        sparse_stats: List[Dict[str, Any]],
        persist: bool = True,
    ) -> Dict[str, Any]:
        return {"doc_count": len(sparse_stats)}


class FakeImageStorage:
    def save_images(
        self,
        images: Any,
        collection: str,
        doc_hash: str,
    ) -> List[str]:
        return []


def _build_pipeline(tmp_path: Path, transforms: List[Any]) -> IngestionPipeline:
    return IngestionPipeline(
        settings={
            "splitter": {"provider": "recursive"},
            "embedding": {"provider": "fake-embedding"},
            "vector_store": {"provider": "fake-store"},
        },
        integrity_checker=FakeIntegrityChecker(),
        loader=FakeLoader(),
        chunker=FakeChunker(),
        transforms=transforms,
        batch_processor=FakeBatchProcessor(),
        vector_upserter=FakeVectorUpserter(),
        bm25_indexer=FakeBM25Indexer(),
        image_storage=FakeImageStorage(),
    )


def test_pipeline_progress_callback_reports_stage_order_and_counts(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"pdf")
    pipeline = _build_pipeline(
        tmp_path,
        transforms=[FakeTransform("t1"), FakeTransform("t2")],
    )
    events: List[tuple[str, int, int]] = []
    result = pipeline.run(
        str(source),
        on_progress=lambda stage, current, total: events.append((stage, current, total)),
    )
    assert result.status == "success"
    assert events == [
        ("load", 1, 1),
        ("split", 1, 1),
        ("transform", 1, 2),
        ("transform", 2, 2),
        ("embed", 1, 1),
        ("upsert", 1, 1),
    ]


def test_pipeline_progress_callback_is_optional(tmp_path: Path) -> None:
    source = tmp_path / "source-no-callback.pdf"
    source.write_bytes(b"pdf")
    pipeline = _build_pipeline(tmp_path, transforms=[])
    result = pipeline.run(str(source), on_progress=None)
    assert result.status == "success"
    assert result.chunk_count == 1
