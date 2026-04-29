from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from observability.logger import JSONFormatter, get_trace_logger, write_trace


def test_json_formatter_serializes_mapping_payload() -> None:
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="trace",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg={"trace_id": "abc", "trace_type": "query"},
        args=(),
        exc_info=None,
    )
    raw = formatter.format(record)
    payload = json.loads(raw)
    assert payload["trace_id"] == "abc"
    assert payload["trace_type"] == "query"


def test_get_trace_logger_writes_jsonl_line(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "traces.jsonl"
    logger = get_trace_logger(log_file=str(log_file), name="linguapilot.trace.test.a")
    logger.info({"trace_id": "id-1", "trace_type": "ingestion", "stages": []})
    for handler in logger.handlers:
        handler.flush()
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["trace_type"] == "ingestion"
    assert payload["trace_id"] == "id-1"


def test_write_trace_appends_trace_dict(tmp_path: Path) -> None:
    log_file = tmp_path / "trace" / "events.jsonl"
    write_trace(
        {
            "trace_id": "id-2",
            "trace_type": "query",
            "started_at": "2026-04-29T12:00:00+00:00",
            "stages": [{"stage": "query_processing", "elapsed_ms": 1.2}],
        },
        log_file=str(log_file),
    )
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["trace_type"] == "query"
    assert payload["stages"][0]["stage"] == "query_processing"


def test_write_trace_rejects_non_mapping(tmp_path: Path) -> None:
    with pytest.raises(ValueError) as exc_info:
        write_trace(["bad"], log_file=str(tmp_path / "x.jsonl"))  # type: ignore[arg-type]
    assert "trace_dict must be a mapping" in str(exc_info.value)
