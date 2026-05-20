import hashlib
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


class FileIntegrityChecker(ABC):
    @abstractmethod
    def compute_sha256(self, path: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def should_skip(self, file_hash: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def mark_success(
        self,
        file_hash: str,
        file_path: str,
        file_size: Optional[int] = None,
        chunk_count: Optional[int] = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def remove_record(
        self,
        file_hash: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> int:
        raise NotImplementedError


class SQLiteIntegrityChecker(FileIntegrityChecker):
    def __init__(self, db_path: str = "data/db/ingestion_history.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def compute_sha256(self, path: str) -> str:
        file_path = Path(path)
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError("file not found: " + path)
        hasher = hashlib.sha256()
        with file_path.open("rb") as handle:
            while True:
                chunk = handle.read(8192)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    def should_skip(self, file_hash: str) -> bool:
        normalized_hash = _normalize_hash(file_hash)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM ingestion_history WHERE file_hash = ?",
                (normalized_hash,),
            ).fetchone()
        return row is not None and str(row[0]) == "success"

    def mark_success(
        self,
        file_hash: str,
        file_path: str,
        file_size: Optional[int] = None,
        chunk_count: Optional[int] = None,
    ) -> None:
        normalized_hash = _normalize_hash(file_hash)
        normalized_path = str(file_path)
        if not normalized_path.strip():
            raise ValueError("file_path must be non-empty")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_history
                (file_hash, file_path, file_size, status, error_msg, chunk_count, processed_at)
                VALUES (?, ?, ?, 'success', NULL, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(file_hash) DO UPDATE SET
                    file_path=excluded.file_path,
                    file_size=excluded.file_size,
                    status='success',
                    error_msg=NULL,
                    chunk_count=excluded.chunk_count,
                    processed_at=CURRENT_TIMESTAMP
                """,
                (
                    normalized_hash,
                    normalized_path,
                    file_size,
                    chunk_count,
                ),
            )

    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        normalized_hash = _normalize_hash(file_hash)
        message = str(error_msg)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_history
                (file_hash, file_path, file_size, status, error_msg, chunk_count, processed_at)
                VALUES (?, '', NULL, 'failed', ?, NULL, CURRENT_TIMESTAMP)
                ON CONFLICT(file_hash) DO UPDATE SET
                    status='failed',
                    error_msg=excluded.error_msg,
                    processed_at=CURRENT_TIMESTAMP
                """,
                (normalized_hash, message),
            )

    def remove_record(
        self,
        file_hash: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> int:
        if file_hash is None and file_path is None:
            raise ValueError("file_hash or file_path must be provided")
        clauses: List[str] = []
        params: List[Any] = []
        if file_hash is not None:
            clauses.append("file_hash = ?")
            params.append(_normalize_hash(file_hash))
        if file_path is not None:
            normalized_path = str(file_path).strip()
            if not normalized_path:
                raise ValueError("file_path must be non-empty")
            clauses.append("file_path = ?")
            params.append(normalized_path)
        sql = "DELETE FROM ingestion_history WHERE " + " AND ".join(clauses)
        with self._connect() as conn:
            cursor = conn.execute(sql, tuple(params))
            deleted = int(cursor.rowcount if cursor.rowcount is not None else 0)
        return deleted

    def list_processed(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT file_hash, file_path, file_size, status, processed_at, error_msg, chunk_count
                FROM ingestion_history
                ORDER BY processed_at DESC
                """
            ).fetchall()
        output: List[Dict[str, Any]] = []
        for row in rows:
            output.append(
                {
                    "file_hash": row[0],
                    "file_path": row[1],
                    "file_size": row[2],
                    "status": row[3],
                    "processed_at": row[4],
                    "error_msg": row[5],
                    "chunk_count": row[6],
                }
            )
        return output

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_history (
                    file_hash TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_size INTEGER,
                    status TEXT NOT NULL CHECK(status IN ('success', 'failed', 'processing')),
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    error_msg TEXT,
                    chunk_count INTEGER
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_status ON ingestion_history(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_processed_at ON ingestion_history(processed_at)"
            )


def _normalize_hash(file_hash: str) -> str:
    if not isinstance(file_hash, str) or not file_hash.strip():
        raise ValueError("file_hash must be non-empty")
    return file_hash.strip().lower()
