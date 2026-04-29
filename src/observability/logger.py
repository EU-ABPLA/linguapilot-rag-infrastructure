import logging
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def get_logger(name: str = "linguapilot") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = record.msg
        if isinstance(message, Mapping):
            payload = dict(message)
        else:
            payload = {
                "message": record.getMessage(),
                "level": record.levelname,
                "logger": record.name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def get_trace_logger(
    log_file: str = "logs/traces.jsonl",
    name: str = "linguapilot.trace",
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    resolved_path = str(Path(log_file).resolve())
    if not _has_file_handler(logger, resolved_path):
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(file_path, encoding="utf-8")
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
    return logger


def write_trace(trace_dict: Mapping[str, Any], log_file: str = "logs/traces.jsonl") -> None:
    if not isinstance(trace_dict, Mapping):
        raise ValueError("trace_dict must be a mapping")
    trace_logger = get_trace_logger(log_file=log_file)
    trace_logger.info(dict(trace_dict))


def _has_file_handler(logger: logging.Logger, resolved_path: str) -> bool:
    for handler in logger.handlers:
        if not isinstance(handler, logging.FileHandler):
            continue
        base = getattr(handler, "baseFilename", "")
        if str(base) == resolved_path:
            return True
    return False
