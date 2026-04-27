import sqlite3
from pathlib import Path

import pytest

from ingestion.storage.image_storage import ImageStorage


def _write_png(path: Path, marker: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + marker.encode("utf-8"))


def test_image_storage_save_images_writes_files_and_records_mapping(tmp_path: Path) -> None:
    source_a = tmp_path / "source" / "a.png"
    source_b = tmp_path / "source" / "b.png"
    _write_png(source_a, "a")
    _write_png(source_b, "b")
    storage = ImageStorage(
        db_path=str(tmp_path / "db" / "image_index.db"),
        image_root=str(tmp_path / "images"),
    )
    saved = storage.save_images(
        [
            {"id": "img_a", "path": str(source_a), "page": 1},
            {"id": "img_b", "path": str(source_b), "page": 2},
        ],
        collection="default",
        doc_hash="doc-1",
    )
    assert len(saved) == 2
    target_a = tmp_path / "images" / "default" / "img_a.png"
    target_b = tmp_path / "images" / "default" / "img_b.png"
    assert target_a.exists()
    assert target_b.exists()
    assert storage.get_path("img_a") == str(target_a)
    assert storage.get_path("img_b") == str(target_b)


def test_image_storage_lookup_missing_returns_none(tmp_path: Path) -> None:
    storage = ImageStorage(
        db_path=str(tmp_path / "db" / "image_index.db"),
        image_root=str(tmp_path / "images"),
    )
    assert storage.get_path("not_exists") is None


def test_image_storage_mapping_persists_after_reload(tmp_path: Path) -> None:
    source = tmp_path / "source" / "persist.png"
    _write_png(source, "persist")
    db_path = tmp_path / "db" / "image_index.db"
    image_root = tmp_path / "images"
    storage = ImageStorage(db_path=str(db_path), image_root=str(image_root))
    storage.save_images(
        [{"id": "img_persist", "path": str(source), "page": 1}],
        collection="lesson",
        doc_hash="doc-persist",
    )
    reloaded = ImageStorage(db_path=str(db_path), image_root=str(image_root))
    assert reloaded.get_path("img_persist") == str(
        tmp_path / "images" / "lesson" / "img_persist.png"
    )


def test_image_storage_uses_wal_and_creates_table(tmp_path: Path) -> None:
    db_path = tmp_path / "db" / "image_index.db"
    storage = ImageStorage(db_path=str(db_path), image_root=str(tmp_path / "images"))
    with storage._connect() as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "wal"
    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='image_index'"
        ).fetchone()
    assert row is not None
    assert row[0] == "image_index"


def test_image_storage_list_images_supports_collection_and_doc_hash(
    tmp_path: Path,
) -> None:
    source_a = tmp_path / "source" / "a.png"
    source_b = tmp_path / "source" / "b.png"
    _write_png(source_a, "a")
    _write_png(source_b, "b")
    storage = ImageStorage(
        db_path=str(tmp_path / "db" / "image_index.db"),
        image_root=str(tmp_path / "images"),
    )
    storage.save_images(
        [{"id": "img_1", "path": str(source_a), "page": 1}],
        collection="c1",
        doc_hash="doc-a",
    )
    storage.save_images(
        [{"id": "img_2", "path": str(source_b), "page": 1}],
        collection="c2",
        doc_hash="doc-b",
    )
    only_c1 = storage.list_images(collection="c1")
    assert [item["image_id"] for item in only_c1] == ["img_1"]
    only_doc_b = storage.list_images(doc_hash="doc-b")
    assert [item["image_id"] for item in only_doc_b] == ["img_2"]


def test_image_storage_delete_images_removes_files_and_rows(tmp_path: Path) -> None:
    source = tmp_path / "source" / "x.png"
    _write_png(source, "x")
    storage = ImageStorage(
        db_path=str(tmp_path / "db" / "image_index.db"),
        image_root=str(tmp_path / "images"),
    )
    storage.save_images(
        [{"id": "img_x", "path": str(source), "page": 1}],
        collection="to_delete",
        doc_hash="doc-x",
    )
    deleted = storage.delete_images(collection="to_delete", doc_hash="doc-x")
    assert deleted == 1
    assert storage.get_path("img_x") is None
    assert not (tmp_path / "images" / "to_delete" / "img_x.png").exists()


def test_image_storage_raises_when_source_file_missing(tmp_path: Path) -> None:
    storage = ImageStorage(
        db_path=str(tmp_path / "db" / "image_index.db"),
        image_root=str(tmp_path / "images"),
    )
    with pytest.raises(FileNotFoundError):
        storage.save_images(
            [{"id": "img_missing", "path": str(tmp_path / "none.png"), "page": 1}],
            collection="default",
            doc_hash="doc-1",
        )
