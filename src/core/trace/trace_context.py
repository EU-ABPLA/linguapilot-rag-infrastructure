from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class TraceContext:
    trace_type: str = "generic"
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: str = field(default_factory=lambda: _utc_now_iso())
    stages: List[Dict[str, Any]] = field(default_factory=list)
    finished_at: Optional[str] = None
    duration_ms: Optional[int] = None
    error: Optional[str] = None
    _started_monotonic: float = field(default_factory=time.perf_counter, init=False, repr=False)

    def record_stage(self, stage: str, details: Optional[Dict[str, Any]] = None) -> None:
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError("stage must be non-empty")
        entry: Dict[str, Any] = {
            "stage": stage.strip(),
            "timestamp": _utc_now_iso(),
            "details": details or {},
        }
        self.stages.append(entry)

    def finish(self, error: Optional[str] = None) -> Dict[str, Any]:
        self.finished_at = _utc_now_iso()
        self.duration_ms = int((time.perf_counter() - self._started_monotonic) * 1000)
        self.error = error
        return self.to_dict()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "metadata": dict(self.metadata),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "stages": list(self.stages),
        }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
