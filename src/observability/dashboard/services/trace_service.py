from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


class TraceService:
    def __init__(self, trace_file: str = "logs/traces.jsonl"):
        self._trace_file = Path(trace_file)

    def list_traces(
        self,
        trace_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        rows = self._read_all()
        if trace_type is not None:
            normalized = _normalize_optional_str(trace_type)
            rows = [item for item in rows if str(item.get("trace_type", "")) == normalized]
        rows.sort(
            key=lambda item: (
                str(item.get("started_at", "")),
                str(item.get("trace_id", "")),
            ),
            reverse=True,
        )
        if limit is not None and limit > 0:
            return rows[:limit]
        return rows

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        normalized = _normalize_non_empty_str(trace_id, "trace_id")
        for item in self._read_all():
            if str(item.get("trace_id", "")) == normalized:
                return item
        return None

    def stage_rows(self, trace: Mapping[str, Any]) -> List[Dict[str, Any]]:
        stages = trace.get("stages", [])
        if not isinstance(stages, list):
            return []
        output: List[Dict[str, Any]] = []
        for item in stages:
            if not isinstance(item, Mapping):
                continue
            stage = item.get("stage")
            elapsed = item.get("elapsed_ms")
            details = item.get("details", {})
            if not isinstance(stage, str) or not stage.strip():
                continue
            if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
                continue
            method = ""
            provider = ""
            if isinstance(details, Mapping):
                method = str(details.get("method", ""))
                provider = str(details.get("provider", ""))
            output.append(
                {
                    "stage": stage.strip(),
                    "elapsed_ms": float(elapsed),
                    "method": method,
                    "provider": provider,
                }
            )
        return output

    def stage_duration_rows(self, trace: Mapping[str, Any]) -> List[Dict[str, Any]]:
        raw_rows = self.stage_rows(trace)
        if not raw_rows:
            return []
        counts: Dict[str, int] = {}
        for item in raw_rows:
            stage = item["stage"]
            counts[stage] = counts.get(stage, 0) + 1
        occurrences: Dict[str, int] = {}
        output: List[Dict[str, Any]] = []
        previous_elapsed = 0.0
        for item in raw_rows:
            stage = item["stage"]
            occurrences[stage] = occurrences.get(stage, 0) + 1
            occurrence = occurrences[stage]
            elapsed = float(item["elapsed_ms"])
            duration = elapsed - previous_elapsed
            previous_elapsed = elapsed
            if duration < 0:
                duration = 0.0
            stage_label = stage
            if counts[stage] > 1:
                stage_label = stage + " #" + str(occurrence)
            output.append(
                {
                    "stage_label": stage_label,
                    "stage": stage,
                    "occurrence": occurrence,
                    "duration_ms": duration,
                    "cumulative_elapsed_ms": elapsed,
                    "method": item["method"],
                    "provider": item["provider"],
                }
            )
        return output

    def _read_all(self) -> List[Dict[str, Any]]:
        if not self._trace_file.exists() or not self._trace_file.is_file():
            return []
        output: List[Dict[str, Any]] = []
        for raw_line in self._trace_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except Exception:
                continue
            if not isinstance(parsed, Mapping):
                continue
            output.append(dict(parsed))
        return output


def _normalize_non_empty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(field_name + " must be non-empty")
    return value.strip()


def _normalize_optional_str(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("trace_type must be non-empty")
    return value.strip()
