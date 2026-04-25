import sqlite3
from pathlib import Path

from libs.loader.file_integrity import SQLiteIntegrityChecker


def test_compute_sha256_is_stable(tmp_path: Path) -> None:
    data_file = tmp_path / "sample.txt"
    data_file.write_text("hello integrity", encoding="utf-8")
    checker = SQLiteIntegrityChecker(str(tmp_path / "ingestion_history.db"))
    first = checker.compute_sha256(str(data_file))
    second = checker.compute_sha256(str(data_file))
    assert first == second
    assert len(first) == 64


def test_should_skip_returns_true_after_mark_success(tmp_path: Path) -> None:
    db_path = tmp_path / "ingestion_history.db"
    checker = SQLiteIntegrityChecker(str(db_path))
    file_hash = "abc123"
    assert checker.should_skip(file_hash) is False
    checker.mark_success(
        file_hash=file_hash,
        file_path="data/docs/a.pdf",
        file_size=100,
        chunk_count=5,
    )
    assert checker.should_skip(file_hash) is True


def test_database_is_created_and_has_required_table(tmp_path: Path) -> None:
    db_path = tmp_path / "db" / "ingestion_history.db"
    checker = SQLiteIntegrityChecker(str(db_path))
    assert db_path.exists()
    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ingestion_history'"
        ).fetchone()
    assert row is not None
    assert row[0] == "ingestion_history"
    assert checker.list_processed() == []


def test_database_uses_wal_mode(tmp_path: Path) -> None:
    db_path = tmp_path / "ingestion_history.db"
    checker = SQLiteIntegrityChecker(str(db_path))
    with checker._connect() as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "wal"


def test_mark_failed_updates_status_and_message(tmp_path: Path) -> None:
    db_path = tmp_path / "ingestion_history.db"
    checker = SQLiteIntegrityChecker(str(db_path))
    checker.mark_failed("hash-1", "network error")
    rows = checker.list_processed()
    assert len(rows) == 1
    assert rows[0]["file_hash"] == "hash-1"
    assert rows[0]["status"] == "failed"
    assert rows[0]["error_msg"] == "network error"
