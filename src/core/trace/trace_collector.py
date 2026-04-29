from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from core.trace.trace_context import TraceContext


class TraceCollector:
    def __init__(self, sink: Optional[Callable[[Dict[str, Any]], None]] = None):
        self._sink = sink
        self._items: List[Dict[str, Any]] = []

    def collect(self, trace: TraceContext) -> None:
        if not isinstance(trace, TraceContext):
            raise ValueError("trace must be a TraceContext")
        payload = trace.to_dict()
        self._items.append(payload)
        if self._sink is not None:
            self._sink(dict(payload))

    def list_items(self) -> List[Dict[str, Any]]:
        return [dict(item) for item in self._items]
