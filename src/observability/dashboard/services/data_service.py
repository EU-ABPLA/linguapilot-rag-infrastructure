from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from core.settings import load_settings
from ingestion.storage.image_storage import ImageStorage
from libs.loader.file_integrity import SQLiteIntegrityChecker
from libs.vector_store.chroma_store import ChromaStore


class DataService:
    def __init__(
        self,
        chroma_store: Optional[ChromaStore] = None,
        image_storage: Optional[ImageStorage] = None,
        file_integrity: Optional[SQLiteIntegrityChecker] = None,
    ):
        self._chroma_store = chroma_store or _build_default_chroma_store()
        self._image_storage = image_storage or ImageStorage()
        self._file_integrity = file_integrity or SQLiteIntegrityChecker()

    def list_documents(self, collection: Optional[str] = None) -> List[Dict[str, Any]]:
        records = self._list_chunk_records(collection=collection)
        status_by_source = self._status_by_source()
        grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for item in records:
            metadata = _extract_metadata(item)
            source_path = _pick_source_path(metadata)
            if source_path is None:
                continue
            item_collection = _pick_collection(metadata, default=collection)
            key = (source_path, item_collection)
            row = grouped.get(key)
            if row is None:
                status_row = status_by_source.get(source_path, {})
                row = {
                    "source_path": source_path,
                    "collection": item_collection,
                    "chunk_count": 0,
                    "chunk_ids": [],
                    "doc_hash": _pick_doc_hash(metadata),
                    "status": status_row.get("status"),
                    "processed_at": status_row.get("processed_at"),
                }
                grouped[key] = row
            row["chunk_count"] += 1
            chunk_id = str(item.get("id", "")).strip()
            if chunk_id:
                row["chunk_ids"].append(chunk_id)
            if row.get("doc_hash") is None:
                row["doc_hash"] = _pick_doc_hash(metadata)
        output: List[Dict[str, Any]] = []
        for key in sorted(grouped.keys()):
            row = grouped[key]
            doc_hash = row.get("doc_hash")
            image_count = 0
            if doc_hash is not None:
                image_count = len(
                    self._image_storage.list_images(
                        collection=row["collection"],
                        doc_hash=doc_hash,
                    )
                )
            output.append(
                {
                    "source_path": row["source_path"],
                    "collection": row["collection"],
                    "chunk_count": int(row["chunk_count"]),
                    "image_count": int(image_count),
                    "status": row.get("status"),
                    "processed_at": row.get("processed_at"),
                }
            )
        return output

    def list_collections(self) -> List[str]:
        rows = self.list_documents()
        values = sorted({str(item["collection"]) for item in rows})
        return values

    def get_document_detail(
        self,
        source_path: str,
        collection: Optional[str] = None,
    ) -> Dict[str, Any]:
        normalized_source = _normalize_non_empty(source_path, "source_path")
        filters: Dict[str, Any] = {"source_path": normalized_source}
        if collection is not None:
            filters["collection"] = _normalize_non_empty(collection, "collection")
        rows = self._chroma_store.get_by_metadata(filters)
        chunks: List[Dict[str, Any]] = []
        doc_hash: Optional[str] = None
        resolved_collection = "default"
        for row in rows:
            metadata = _extract_metadata(row)
            resolved_collection = _pick_collection(metadata, default=collection)
            if doc_hash is None:
                doc_hash = _pick_doc_hash(metadata)
            chunks.append(
                {
                    "chunk_id": str(row.get("id", "")),
                    "text": str(row.get("text", row.get("content", ""))),
                    "metadata": metadata,
                }
            )
        chunks.sort(
            key=lambda item: (
                _to_int(item["metadata"].get("chunk_index")),
                str(item["chunk_id"]),
            )
        )
        images: List[Dict[str, Any]] = []
        if doc_hash is not None:
            images = self._image_storage.list_images(
                collection=resolved_collection,
                doc_hash=doc_hash,
            )
        status_row = self._status_by_source().get(normalized_source, {})
        return {
            "source_path": normalized_source,
            "collection": resolved_collection,
            "chunk_count": len(chunks),
            "chunks": chunks,
            "images": images,
            "status": status_row.get("status"),
            "processed_at": status_row.get("processed_at"),
        }

    def _list_chunk_records(self, collection: Optional[str]) -> List[Mapping[str, Any]]:
        if collection is None:
            return self._chroma_store.get_by_metadata()
        normalized = _normalize_non_empty(collection, "collection")
        return self._chroma_store.get_by_metadata({"collection": normalized})

    def _status_by_source(self) -> Dict[str, Dict[str, Any]]:
        output: Dict[str, Dict[str, Any]] = {}
        for item in self._file_integrity.list_processed():
            source = str(item.get("file_path", "")).strip()
            if not source:
                continue
            if source in output:
                continue
            output[source] = {
                "status": item.get("status"),
                "processed_at": item.get("processed_at"),
            }
        return output


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


def _normalize_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(field_name + " must be non-empty")
    return value.strip()


def _to_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _build_default_chroma_store() -> ChromaStore:
    try:
        settings = load_settings("config/settings.yaml")
        return ChromaStore(collection=settings.vector_store.collection)
    except Exception:
        return ChromaStore()
