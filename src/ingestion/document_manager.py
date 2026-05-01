from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from libs.loader.file_integrity import SQLiteIntegrityChecker
from libs.vector_store.chroma_store import ChromaStore


@dataclass(frozen=True)
class DocumentInfo:
    doc_id: str
    source_path: str
    collection: str
    chunk_count: int
    image_count: int


@dataclass(frozen=True)
class DocumentDetail:
    doc_id: str
    source_path: str
    collection: str
    chunk_ids: List[str]
    chunk_count: int
    image_count: int
    images: List[Dict[str, Any]]
    status: Optional[str]
    processed_at: Optional[str]


@dataclass(frozen=True)
class DeleteResult:
    source_path: str
    collection: str
    deleted_chunks: int
    deleted_bm25_docs: int
    deleted_images: int
    deleted_integrity_records: int
    success: bool


@dataclass(frozen=True)
class CollectionStats:
    collection: str
    document_count: int
    chunk_count: int
    image_count: int
    unique_sources: int


class DocumentManager:
    def __init__(
        self,
        chroma_store: Optional[ChromaStore] = None,
        bm25_indexer: Optional[BM25Indexer] = None,
        image_storage: Optional[ImageStorage] = None,
        file_integrity: Optional[SQLiteIntegrityChecker] = None,
    ):
        self._chroma_store = chroma_store or ChromaStore()
        self._bm25_indexer = bm25_indexer or BM25Indexer()
        self._image_storage = image_storage or ImageStorage()
        self._file_integrity = file_integrity or SQLiteIntegrityChecker()

    def list_documents(self, collection: Optional[str] = None) -> List[DocumentInfo]:
        records = self._list_chunk_records(collection=collection)
        grouped: Dict[tuple[str, str], Dict[str, Any]] = {}
        for item in records:
            metadata = item["metadata"]
            source_path = _pick_source_path(metadata)
            if source_path is None:
                continue
            row_collection = _pick_collection(metadata, default=collection)
            key = (source_path, row_collection)
            row = grouped.get(key)
            if row is None:
                row = {
                    "doc_id": source_path,
                    "source_path": source_path,
                    "collection": row_collection,
                    "chunk_ids": [],
                    "doc_hash": _pick_doc_hash(metadata),
                }
                grouped[key] = row
            chunk_id = item["id"]
            row["chunk_ids"].append(chunk_id)
            if row.get("doc_hash") is None:
                row["doc_hash"] = _pick_doc_hash(metadata)
        output: List[DocumentInfo] = []
        for key in sorted(grouped.keys()):
            row = grouped[key]
            doc_hash = row.get("doc_hash")
            image_count = _count_images(
                self._image_storage,
                collection=row["collection"],
                doc_hash=doc_hash,
            )
            output.append(
                DocumentInfo(
                    doc_id=row["doc_id"],
                    source_path=row["source_path"],
                    collection=row["collection"],
                    chunk_count=len(row["chunk_ids"]),
                    image_count=image_count,
                )
            )
        return output

    def get_document_detail(self, doc_id: str) -> DocumentDetail:
        normalized_doc_id = _normalize_non_empty(doc_id, "doc_id")
        records = self._chroma_store.get_by_metadata({"source_path": normalized_doc_id})
        if not records:
            raise ValueError("document not found: " + normalized_doc_id)
        first_metadata = _extract_metadata(records[0])
        collection = _pick_collection(first_metadata, default=None)
        chunk_ids: List[str] = []
        doc_hash = _pick_doc_hash(first_metadata)
        for item in records:
            chunk_ids.append(str(item.get("id", "")))
            metadata = _extract_metadata(item)
            if doc_hash is None:
                doc_hash = _pick_doc_hash(metadata)
        chunk_ids = sorted([item for item in chunk_ids if item])
        images = self._image_storage.list_images(collection=collection, doc_hash=doc_hash)
        status, processed_at = _find_integrity_status(
            self._file_integrity,
            source_path=normalized_doc_id,
        )
        return DocumentDetail(
            doc_id=normalized_doc_id,
            source_path=normalized_doc_id,
            collection=collection,
            chunk_ids=chunk_ids,
            chunk_count=len(chunk_ids),
            image_count=len(images),
            images=images,
            status=status,
            processed_at=processed_at,
        )

    def delete_document(self, source_path: str, collection: str) -> DeleteResult:
        normalized_source_path = _normalize_non_empty(source_path, "source_path")
        normalized_collection = _normalize_non_empty(collection, "collection")
        records = self._chroma_store.get_by_metadata(
            {"source_path": normalized_source_path, "collection": normalized_collection}
        )
        chunk_ids: List[str] = []
        doc_hashes: List[str] = []
        for item in records:
            item_id = item.get("id")
            if isinstance(item_id, str) and item_id.strip():
                chunk_ids.append(item_id.strip())
            metadata = _extract_metadata(item)
            doc_hash = _pick_doc_hash(metadata)
            if doc_hash is not None and doc_hash not in doc_hashes:
                doc_hashes.append(doc_hash)
        deleted_chunks = self._chroma_store.delete_by_metadata(
            {"source_path": normalized_source_path, "collection": normalized_collection}
        )
        deleted_bm25_docs = 0
        for chunk_id in chunk_ids:
            if self._bm25_indexer.remove_document(chunk_id, persist=False):
                deleted_bm25_docs += 1
        self._bm25_indexer.save()
        deleted_images = 0
        if doc_hashes:
            for doc_hash in doc_hashes:
                deleted_images += self._image_storage.delete_images(
                    collection=normalized_collection,
                    doc_hash=doc_hash,
                )
        else:
            deleted_images += self._image_storage.delete_images(collection=normalized_collection)
        deleted_integrity_records = self._file_integrity.remove_record(
            file_path=normalized_source_path
        )
        return DeleteResult(
            source_path=normalized_source_path,
            collection=normalized_collection,
            deleted_chunks=deleted_chunks,
            deleted_bm25_docs=deleted_bm25_docs,
            deleted_images=deleted_images,
            deleted_integrity_records=deleted_integrity_records,
            success=(
                deleted_chunks > 0
                or deleted_bm25_docs > 0
                or deleted_images > 0
                or deleted_integrity_records > 0
            ),
        )

    def get_collection_stats(self, collection: Optional[str] = None) -> CollectionStats:
        normalized_collection: str
        if collection is None:
            normalized_collection = self._chroma_store.collection
        else:
            normalized_collection = _normalize_non_empty(collection, "collection")
        docs = self.list_documents(collection=normalized_collection)
        chunk_count = sum(item.chunk_count for item in docs)
        image_count = sum(item.image_count for item in docs)
        unique_sources = len({item.source_path for item in docs})
        return CollectionStats(
            collection=normalized_collection,
            document_count=len(docs),
            chunk_count=chunk_count,
            image_count=image_count,
            unique_sources=unique_sources,
        )

    def _list_chunk_records(self, collection: Optional[str]) -> List[Mapping[str, Any]]:
        if collection is None:
            return self._chroma_store.get_by_metadata()
        normalized = _normalize_non_empty(collection, "collection")
        return self._chroma_store.get_by_metadata({"collection": normalized})


def _normalize_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(field_name + " must be non-empty")
    return value.strip()


def _extract_metadata(item: Mapping[str, Any]) -> Dict[str, Any]:
    value = item.get("metadata", {})
    if not isinstance(value, Mapping):
        return {}
    return dict(value)


def _pick_source_path(metadata: Mapping[str, Any]) -> Optional[str]:
    value = metadata.get("source_path")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _pick_collection(metadata: Mapping[str, Any], default: Optional[str]) -> str:
    value = metadata.get("collection")
    if isinstance(value, str) and value.strip():
        return value.strip()
    if default is not None:
        return default
    return "default"


def _pick_doc_hash(metadata: Mapping[str, Any]) -> Optional[str]:
    value = metadata.get("source_ref")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _count_images(image_storage: ImageStorage, collection: str, doc_hash: Optional[str]) -> int:
    if doc_hash is None:
        return 0
    return len(image_storage.list_images(collection=collection, doc_hash=doc_hash))


def _find_integrity_status(
    checker: SQLiteIntegrityChecker,
    source_path: str,
) -> tuple[Optional[str], Optional[str]]:
    for item in checker.list_processed():
        if str(item.get("file_path", "")) == source_path:
            return item.get("status"), item.get("processed_at")
    return None, None
