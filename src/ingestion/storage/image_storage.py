from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


class ImageStorage:
    def __init__(
        self,
        db_path: str = "data/db/image_index.db",
        image_root: str = "data/images",
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.image_root = Path(image_root)
        self.image_root.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def save_images(
        self,
        images: Sequence[Mapping[str, Any]],
        collection: str,
        doc_hash: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        normalized_collection = _normalize_collection(collection)
        normalized_doc_hash = _normalize_optional_str(doc_hash)
        output: List[Dict[str, Any]] = []
        target_dir = self.image_root / normalized_collection
        target_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            for item in images:
                normalized = _normalize_image_item(item)
                source_path = Path(normalized["path"])
                if not source_path.exists() or not source_path.is_file():
                    raise FileNotFoundError("image file not found: " + str(source_path))
                suffix = source_path.suffix if source_path.suffix else ".png"
                target_path = target_dir / (normalized["image_id"] + suffix)
                if source_path.resolve() != target_path.resolve():
                    shutil.copy2(str(source_path), str(target_path))
                resolved_doc_hash = (
                    normalized_doc_hash
                    if normalized_doc_hash is not None
                    else normalized["doc_hash"]
                )
                conn.execute(
                    """
                    INSERT INTO image_index
                    (image_id, file_path, collection, doc_hash, page_num, created_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(image_id) DO UPDATE SET
                        file_path=excluded.file_path,
                        collection=excluded.collection,
                        doc_hash=excluded.doc_hash,
                        page_num=excluded.page_num,
                        created_at=CURRENT_TIMESTAMP
                    """,
                    (
                        normalized["image_id"],
                        str(target_path),
                        normalized_collection,
                        resolved_doc_hash,
                        normalized["page_num"],
                    ),
                )
                output.append(
                    {
                        "image_id": normalized["image_id"],
                        "file_path": str(target_path),
                        "collection": normalized_collection,
                        "doc_hash": resolved_doc_hash,
                        "page_num": normalized["page_num"],
                    }
                )
        return output

    def get_path(self, image_id: str) -> Optional[str]:
        normalized = _normalize_non_empty_str(image_id, "image_id")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT file_path FROM image_index WHERE image_id = ?",
                (normalized,),
            ).fetchone()
        if row is None:
            return None
        return str(row[0])

    def list_images(
        self,
        collection: Optional[str] = None,
        doc_hash: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        where: List[str] = []
        params: List[Any] = []
        if collection is not None:
            where.append("collection = ?")
            params.append(_normalize_collection(collection))
        if doc_hash is not None:
            where.append("doc_hash = ?")
            params.append(_normalize_non_empty_str(doc_hash, "doc_hash"))
        sql = (
            "SELECT image_id, file_path, collection, doc_hash, page_num, created_at "
            "FROM image_index"
        )
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC, image_id ASC"
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        output: List[Dict[str, Any]] = []
        for row in rows:
            output.append(
                {
                    "image_id": row[0],
                    "file_path": row[1],
                    "collection": row[2],
                    "doc_hash": row[3],
                    "page_num": row[4],
                    "created_at": row[5],
                }
            )
        return output

    def delete_images(self, collection: str, doc_hash: Optional[str] = None) -> int:
        normalized_collection = _normalize_collection(collection)
        where = ["collection = ?"]
        params: List[Any] = [normalized_collection]
        if doc_hash is not None:
            where.append("doc_hash = ?")
            params.append(_normalize_non_empty_str(doc_hash, "doc_hash"))
        sql = (
            "SELECT image_id, file_path FROM image_index WHERE "
            + " AND ".join(where)
            + " ORDER BY image_id ASC"
        )
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
            if rows:
                conn.execute(
                    "DELETE FROM image_index WHERE " + " AND ".join(where),
                    tuple(params),
                )
        for row in rows:
            file_path = Path(str(row[1]))
            if file_path.exists() and file_path.is_file():
                file_path.unlink()
        return len(rows)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS image_index (
                    image_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    collection TEXT,
                    doc_hash TEXT,
                    page_num INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_collection ON image_index(collection)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_doc_hash ON image_index(doc_hash)"
            )


def _normalize_image_item(item: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(item, Mapping):
        raise ValueError("image item must be a mapping")
    image_id = item.get("id")
    if image_id is None:
        image_id = item.get("image_id")
    image_path = item.get("path")
    if image_path is None:
        image_path = item.get("file_path")
    page_num = item.get("page")
    if page_num is None:
        page_num = item.get("page_num")
    normalized_page: Optional[int]
    if page_num is None:
        normalized_page = None
    elif isinstance(page_num, bool) or not isinstance(page_num, int) or page_num < 0:
        raise ValueError("page_num must be a non-negative integer")
    else:
        normalized_page = page_num
    return {
        "image_id": _normalize_non_empty_str(image_id, "image_id"),
        "path": _normalize_non_empty_str(image_path, "path"),
        "doc_hash": _normalize_optional_str(item.get("doc_hash")),
        "page_num": normalized_page,
    }


def _normalize_non_empty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(field_name + " must be non-empty")
    return value.strip()


def _normalize_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("doc_hash must be a string")
    stripped = value.strip()
    if not stripped:
        return None
    return stripped


def _normalize_collection(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("collection must be non-empty")
    return value.strip()
