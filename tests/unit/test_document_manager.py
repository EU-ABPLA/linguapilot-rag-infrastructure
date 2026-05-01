from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.document_manager import DocumentManager
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from libs.loader.file_integrity import SQLiteIntegrityChecker
from libs.vector_store.chroma_store import ChromaStore


def _write_png(path: Path, marker: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + marker.encode("utf-8"))


def _setup_manager(tmp_path: Path) -> DocumentManager:
    chroma = ChromaStore(
        persist_directory=str(tmp_path / "db" / "chroma"),
        collection="cloud",
    )
    records = [
        {
            "id": "vec-a1",
            "vector": [1.0, 0.0],
            "content": "alpha guide",
            "metadata": {
                "source_path": "docs/a.pdf",
                "collection": "cloud",
                "source_ref": "doc-a",
            },
        },
        {
            "id": "vec-a2",
            "vector": [0.8, 0.2],
            "content": "alpha notes",
            "metadata": {
                "source_path": "docs/a.pdf",
                "collection": "cloud",
                "source_ref": "doc-a",
            },
        },
        {
            "id": "vec-b1",
            "vector": [0.0, 1.0],
            "content": "beta guide",
            "metadata": {
                "source_path": "docs/b.pdf",
                "collection": "cloud",
                "source_ref": "doc-b",
            },
        },
    ]
    chroma.upsert(records)

    bm25 = BM25Indexer(index_dir=str(tmp_path / "db" / "bm25"))
    bm25.build(
        [
            {"chunk_id": "vec-a1", "doc_length": 2, "term_weights": {"alpha": 2}},
            {"chunk_id": "vec-a2", "doc_length": 2, "term_weights": {"alpha": 1}},
            {"chunk_id": "vec-b1", "doc_length": 2, "term_weights": {"beta": 2}},
        ],
        persist=True,
    )

    image_storage = ImageStorage(
        db_path=str(tmp_path / "db" / "image_index.db"),
        image_root=str(tmp_path / "images"),
    )
    src_a = tmp_path / "src" / "a.png"
    src_b = tmp_path / "src" / "b.png"
    _write_png(src_a, "a")
    _write_png(src_b, "b")
    image_storage.save_images(
        [{"id": "img-a", "path": str(src_a), "page": 1}],
        collection="cloud",
        doc_hash="doc-a",
    )
    image_storage.save_images(
        [{"id": "img-b", "path": str(src_b), "page": 1}],
        collection="cloud",
        doc_hash="doc-b",
    )

    integrity = SQLiteIntegrityChecker(str(tmp_path / "db" / "ingestion_history.db"))
    integrity.mark_success("hash-a", "docs/a.pdf", file_size=100, chunk_count=2)
    integrity.mark_success("hash-b", "docs/b.pdf", file_size=100, chunk_count=1)

    return DocumentManager(
        chroma_store=chroma,
        bm25_indexer=bm25,
        image_storage=image_storage,
        file_integrity=integrity,
    )


def test_document_manager_list_documents_returns_source_chunk_and_image_counts(
    tmp_path: Path,
) -> None:
    manager = _setup_manager(tmp_path)
    docs = manager.list_documents(collection="cloud")
    assert len(docs) == 2
    assert docs[0].source_path == "docs/a.pdf"
    assert docs[0].chunk_count == 2
    assert docs[0].image_count == 1
    assert docs[1].source_path == "docs/b.pdf"
    assert docs[1].chunk_count == 1
    assert docs[1].image_count == 1


def test_document_manager_get_document_detail_returns_chunks_images_and_status(
    tmp_path: Path,
) -> None:
    manager = _setup_manager(tmp_path)
    detail = manager.get_document_detail("docs/a.pdf")
    assert detail.source_path == "docs/a.pdf"
    assert detail.collection == "cloud"
    assert detail.chunk_count == 2
    assert detail.image_count == 1
    assert detail.status == "success"
    assert detail.chunk_ids == ["vec-a1", "vec-a2"]
    assert detail.images[0]["image_id"] == "img-a"


def test_document_manager_delete_document_removes_vector_bm25_images_and_integrity(
    tmp_path: Path,
) -> None:
    manager = _setup_manager(tmp_path)
    result = manager.delete_document("docs/a.pdf", "cloud")
    assert result.success is True
    assert result.deleted_chunks == 2
    assert result.deleted_bm25_docs == 2
    assert result.deleted_images == 1
    assert result.deleted_integrity_records == 1
    docs = manager.list_documents(collection="cloud")
    assert [item.source_path for item in docs] == ["docs/b.pdf"]
    assert manager.get_collection_stats("cloud").document_count == 1


def test_document_manager_get_collection_stats_returns_aggregated_counts(
    tmp_path: Path,
) -> None:
    manager = _setup_manager(tmp_path)
    stats = manager.get_collection_stats("cloud")
    assert stats.collection == "cloud"
    assert stats.document_count == 2
    assert stats.chunk_count == 3
    assert stats.image_count == 2
    assert stats.unique_sources == 2


def test_document_manager_get_document_detail_raises_for_missing_doc(tmp_path: Path) -> None:
    manager = _setup_manager(tmp_path)
    with pytest.raises(ValueError) as exc_info:
        manager.get_document_detail("docs/missing.pdf")
    assert "document not found" in str(exc_info.value)
