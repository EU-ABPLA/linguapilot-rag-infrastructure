from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class TraceContext:
    trace_type: str = "query"
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: str = field(default_factory=lambda: _utc_now_iso())
    stages: List[Dict[str, Any]] = field(default_factory=list)
    finished_at: Optional[str] = None
    total_elapsed_ms: Optional[float] = None
    error: Optional[str] = None
    _started_monotonic: float = field(default_factory=time.perf_counter, init=False, repr=False)

    def __post_init__(self) -> None:
        self.trace_type = _normalize_trace_type(self.trace_type)
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a mapping")

    def record_stage(self, stage: str, details: Optional[Dict[str, Any]] = None) -> None:
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError("stage must be non-empty")
        entry: Dict[str, Any] = {
            "stage": stage.strip(),
            "timestamp": _utc_now_iso(),
            "details": details or {},
            "elapsed_ms": self.elapsed_ms(),
        }
        self.stages.append(entry)

    def finish(self, error: Optional[str] = None) -> None:
        self.finished_at = _utc_now_iso()
        self.total_elapsed_ms = self.elapsed_ms()
        self.error = error

    def elapsed_ms(self, stage_name: Optional[str] = None) -> float:
        if stage_name is None:
            if self.total_elapsed_ms is not None:
                return float(self.total_elapsed_ms)
            return float((time.perf_counter() - self._started_monotonic) * 1000.0)
        if not isinstance(stage_name, str) or not stage_name.strip():
            raise ValueError("stage_name must be non-empty")
        target = stage_name.strip()
        for item in reversed(self.stages):
            if item.get("stage") == target:
                value = item.get("elapsed_ms")
                if isinstance(value, (int, float)):
                    return float(value)
                break
        raise ValueError("stage not found: " + target)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "metadata": dict(self.metadata),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_elapsed_ms": self.total_elapsed_ms,
            "error": self.error,
            "stages": list(self.stages),
        }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_trace_type(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("trace_type must be non-empty")
    normalized = value.strip().lower()
    if normalized not in {"query", "ingestion"}:
        raise ValueError("trace_type must be query or ingestion")
    return normalized
