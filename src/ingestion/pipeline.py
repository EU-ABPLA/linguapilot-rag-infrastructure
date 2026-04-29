from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

from core.trace.trace_context import TraceContext
from core.types import Chunk, Document
from ingestion.chunking.document_chunker import DocumentChunker
from ingestion.embedding.batch_processor import BatchEncodingResult, BatchProcessor
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from ingestion.storage.vector_upserter import VectorUpserter
from ingestion.transform.base_transform import BaseTransform
from ingestion.transform.chunk_refiner import ChunkRefiner
from ingestion.transform.image_captioner import ImageCaptioner
from ingestion.transform.metadata_enricher import MetadataEnricher
from libs.loader.base_loader import BaseLoader
from libs.loader.file_integrity import FileIntegrityChecker, SQLiteIntegrityChecker
from libs.loader.pdf_loader import PdfLoader
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.vector_store_factory import VectorStoreFactory

ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class IngestionResult:
    status: str
    source_path: str
    collection: str
    file_hash: str
    skipped: bool
    document_id: Optional[str]
    chunk_count: int
    vector_count: int
    image_count: int
    bm25_doc_count: int


class IngestionPipeline:
    def __init__(
        self,
        settings: Any,
        integrity_checker: Optional[FileIntegrityChecker] = None,
        loader: Optional[BaseLoader] = None,
        chunker: Optional[DocumentChunker] = None,
        transforms: Optional[Sequence[BaseTransform]] = None,
        batch_processor: Optional[BatchProcessor] = None,
        vector_upserter: Optional[VectorUpserter] = None,
        vector_store: Optional[BaseVectorStore] = None,
        bm25_indexer: Optional[BM25Indexer] = None,
        image_storage: Optional[ImageStorage] = None,
    ):
        self._settings = settings
        self._integrity_checker = integrity_checker or SQLiteIntegrityChecker()
        self._loader = loader or PdfLoader()
        self._chunker = chunker or DocumentChunker(settings)
        self._transforms = list(transforms) if transforms is not None else _default_transforms(
            settings
        )
        self._batch_processor = batch_processor or BatchProcessor(settings)
        self._vector_upserter = vector_upserter
        self._vector_store = vector_store
        self._bm25_indexer = bm25_indexer or BM25Indexer()
        self._image_storage = image_storage or ImageStorage()

    def run(
        self,
        source_path: str,
        collection: str = "default",
        force: bool = False,
        on_progress: Optional[ProgressCallback] = None,
        trace: Optional[TraceContext] = None,
    ) -> IngestionResult:
        normalized_path = str(Path(source_path))
        file_hash = _run_stage(
            "integrity",
            lambda: self._integrity_checker.compute_sha256(normalized_path),
        )
        if not force and self._integrity_checker.should_skip(file_hash):
            return IngestionResult(
                status="skipped",
                source_path=normalized_path,
                collection=collection,
                file_hash=file_hash,
                skipped=True,
                document_id=None,
                chunk_count=0,
                vector_count=0,
                image_count=0,
                bm25_doc_count=0,
            )
        try:
            document = _run_stage("load", lambda: self._loader.load(normalized_path))
            _record_trace(
                trace,
                "load",
                {
                    "status": "ok",
                    "method": "document_loader",
                    "provider": _resolve_loader_provider(self._loader),
                    "source_path": normalized_path,
                },
            )
            _emit_progress(on_progress, "load", 1, 1)
            chunks = _run_stage("split", lambda: self._chunker.split_document(document))
            _record_trace(
                trace,
                "split",
                {
                    "status": "ok",
                    "method": "chunking",
                    "provider": _resolve_splitter_provider(self._settings),
                    "chunk_count": len(chunks),
                },
            )
            _emit_progress(on_progress, "split", 1, 1)
            transformed = _run_stage(
                "transform",
                lambda: self._apply_transforms(chunks, on_progress=on_progress, trace=trace),
            )
            _record_trace(
                trace,
                "transform",
                {
                    "status": "ok",
                    "method": "transform_chain",
                    "provider": "->".join(
                        [item.__class__.__name__ for item in self._transforms]
                    )
                    if self._transforms
                    else "none",
                    "chunk_count": len(transformed),
                },
            )
            encoded = _run_stage(
                "embed",
                lambda: self._batch_processor.process(transformed, trace=trace),
            )
            _record_trace(
                trace,
                "embed",
                {
                    "status": "ok",
                    "method": "batch_embedding",
                    "provider": _resolve_embedding_provider(self._settings),
                    "chunk_count": len(encoded),
                },
            )
            _emit_progress(on_progress, "embed", 1, 1)
            upserted = _run_stage(
                "upsert",
                lambda: self._persist_outputs(
                    document=document,
                    chunks=transformed,
                    encoded=encoded,
                    collection=collection,
                    trace=trace,
                ),
            )
            _record_trace(
                trace,
                "upsert",
                {
                    "status": "ok",
                    "method": "vector_store_upsert",
                    "provider": _resolve_vector_store_provider(self._settings),
                    "vector_count": len(upserted["vector_ids"]),
                    "image_count": len(upserted["saved_images"]),
                },
            )
            _emit_progress(on_progress, "upsert", 1, 1)
            file_size = Path(normalized_path).stat().st_size
            self._integrity_checker.mark_success(
                file_hash=file_hash,
                file_path=normalized_path,
                file_size=file_size,
                chunk_count=len(transformed),
            )
            return IngestionResult(
                status="success",
                source_path=normalized_path,
                collection=collection,
                file_hash=file_hash,
                skipped=False,
                document_id=document.id,
                chunk_count=len(transformed),
                vector_count=len(upserted["vector_ids"]),
                image_count=len(upserted["saved_images"]),
                bm25_doc_count=int(upserted["bm25_doc_count"]),
            )
        except Exception as exc:
            self._safe_mark_failed(file_hash, str(exc))
            raise

    def _apply_transforms(
        self,
        chunks: List[Chunk],
        on_progress: Optional[ProgressCallback],
        trace: Optional[TraceContext],
    ) -> List[Chunk]:
        if not self._transforms:
            _emit_progress(on_progress, "transform", 1, 1)
            return chunks
        current = chunks
        total = len(self._transforms)
        for index, transform in enumerate(self._transforms):
            try:
                current = transform.transform(current, trace=trace)
            except Exception as exc:
                raise RuntimeError(
                    "transform failed in "
                    + transform.__class__.__name__
                    + ": "
                    + str(exc)
                ) from exc
            _emit_progress(on_progress, "transform", index + 1, total)
        return current

    def _persist_outputs(
        self,
        document: Document,
        chunks: List[Chunk],
        encoded: List[BatchEncodingResult],
        collection: str,
        trace: Optional[TraceContext],
    ) -> dict:
        dense_vectors = [list(item.dense_vector) for item in encoded]
        sparse_stats = [dict(item.sparse_stats) for item in encoded]
        upserter = self._resolve_upserter(collection)
        vector_ids = upserter.upsert(chunks, dense_vectors, trace=trace)
        bm25_snapshot = self._bm25_indexer.update(sparse_stats, persist=True)
        images = document.metadata.get("images", [])
        saved_images = self._image_storage.save_images(
            images,
            collection=collection,
            doc_hash=document.id,
        )
        return {
            "vector_ids": vector_ids,
            "saved_images": saved_images,
            "bm25_doc_count": bm25_snapshot.get("doc_count", 0),
        }

    def _resolve_upserter(self, collection: str) -> VectorUpserter:
        if self._vector_upserter is not None:
            return self._vector_upserter
        if self._vector_store is None:
            self._vector_store = _create_vector_store(self._settings, collection)
        return VectorUpserter(
            self._settings,
            vector_store=self._vector_store,
            collection=collection,
        )

    def _safe_mark_failed(self, file_hash: str, error_msg: str) -> None:
        try:
            self._integrity_checker.mark_failed(file_hash, error_msg)
        except Exception:
            return


def _default_transforms(settings: Any) -> List[BaseTransform]:
    return [ChunkRefiner(settings), MetadataEnricher(settings), ImageCaptioner(settings)]


def _create_vector_store(settings: Any, collection: str) -> BaseVectorStore:
    if not isinstance(settings, dict):
        return VectorStoreFactory.create(settings)
    cloned = dict(settings)
    vector_store = cloned.get("vector_store")
    if isinstance(vector_store, dict):
        next_vector_store = dict(vector_store)
    else:
        next_vector_store = {}
    next_vector_store["collection"] = collection
    cloned["vector_store"] = next_vector_store
    return VectorStoreFactory.create(cloned)


def _run_stage(stage: str, fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception as exc:
        raise RuntimeError("ingestion pipeline error at " + stage + ": " + str(exc)) from exc


def _emit_progress(
    callback: Optional[ProgressCallback], stage: str, current: int, total: int
) -> None:
    if callback is None:
        return
    callback(stage, current, total)


def _record_trace(
    trace: Optional[TraceContext], stage: str, details: Optional[dict] = None
) -> None:
    if trace is None:
        return
    trace.record_stage(stage, details or {})


def _resolve_loader_provider(loader: BaseLoader) -> str:
    name = loader.__class__.__name__.lower()
    if "pdf" in name:
        return "pdf"
    return loader.__class__.__name__


def _resolve_splitter_provider(settings: Any) -> str:
    if isinstance(settings, dict):
        splitter = settings.get("splitter")
        if isinstance(splitter, dict):
            value = splitter.get("provider")
            if isinstance(value, str) and value.strip():
                return value.strip()
    splitter = getattr(settings, "splitter", None)
    if splitter is not None:
        value = getattr(splitter, "provider", None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _resolve_embedding_provider(settings: Any) -> str:
    if isinstance(settings, dict):
        embedding = settings.get("embedding")
        if isinstance(embedding, dict):
            value = embedding.get("provider")
            if isinstance(value, str) and value.strip():
                return value.strip()
    embedding = getattr(settings, "embedding", None)
    if embedding is not None:
        value = getattr(embedding, "provider", None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _resolve_vector_store_provider(settings: Any) -> str:
    if isinstance(settings, dict):
        vector_store = settings.get("vector_store")
        if isinstance(vector_store, dict):
            value = vector_store.get("provider")
            if isinstance(value, str) and value.strip():
                return value.strip()
    vector_store = getattr(settings, "vector_store", None)
    if vector_store is not None:
        value = getattr(vector_store, "provider", None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"
