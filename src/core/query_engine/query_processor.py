from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from core.trace.trace_context import TraceContext

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
_INLINE_FILTER_PATTERN = re.compile(r"([A-Za-z_][A-Za-z0-9_.-]*)\s*:\s*([^\s]+)")
_SPACE_PATTERN = re.compile(r"\s+")
_DEFAULT_STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "are",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "how",
    "why",
    "into",
    "about",
    "your",
    "you",
    "a",
    "an",
    "is",
    "of",
    "to",
    "in",
    "on",
}


@dataclass(frozen=True)
class ProcessedQuery:
    query: str
    keywords: List[str]
    filters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "keywords": list(self.keywords),
            "filters": dict(self.filters),
        }


class QueryProcessor:
    def __init__(
        self,
        stop_words: Optional[set[str]] = None,
        min_keyword_length: int = 2,
    ):
        if isinstance(min_keyword_length, bool) or not isinstance(min_keyword_length, int):
            raise ValueError("min_keyword_length must be an integer")
        if min_keyword_length <= 0:
            raise ValueError("min_keyword_length must be positive")
        self._stop_words = set(stop_words) if stop_words is not None else set(_DEFAULT_STOP_WORDS)
        self._min_keyword_length = min_keyword_length

    def process(
        self,
        query: str,
        filters: Optional[Mapping[str, Any]] = None,
        trace: Optional[TraceContext] = None,
    ) -> ProcessedQuery:
        normalized_query = _normalize_query(query)
        inline_filters, keyword_query = _extract_inline_filters(normalized_query)
        merged_filters = dict(inline_filters)
        merged_filters.update(_normalize_filters(filters))
        keywords = _extract_keywords(
            keyword_query,
            stop_words=self._stop_words,
            min_keyword_length=self._min_keyword_length,
        )
        if trace is not None:
            trace.record_stage(
                "query_processor",
                {
                    "status": "ok",
                    "keyword_count": len(keywords),
                    "filter_count": len(merged_filters),
                },
            )
        return ProcessedQuery(
            query=normalized_query,
            keywords=keywords,
            filters=merged_filters,
        )


def _normalize_query(query: Any) -> str:
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    normalized = _SPACE_PATTERN.sub(" ", query.strip())
    if not normalized:
        raise ValueError("query must be non-empty")
    return normalized


def _extract_inline_filters(query: str) -> tuple[Dict[str, Any], str]:
    filters: Dict[str, Any] = {}
    for match in _INLINE_FILTER_PATTERN.finditer(query):
        key = match.group(1).strip().lower()
        raw_value = match.group(2).strip()
        value = _strip_quoted(raw_value)
        if key and value:
            filters[key] = value
    stripped_query = _INLINE_FILTER_PATTERN.sub(" ", query)
    stripped_query = _SPACE_PATTERN.sub(" ", stripped_query).strip()
    return filters, stripped_query


def _extract_keywords(query: str, stop_words: set[str], min_keyword_length: int) -> List[str]:
    raw_tokens = [token.lower() for token in _TOKEN_PATTERN.findall(query)]
    keywords: List[str] = []
    seen = set()
    for token in raw_tokens:
        if len(token) < min_keyword_length:
            continue
        if token in stop_words:
            continue
        if token in seen:
            continue
        seen.add(token)
        keywords.append(token)
    if keywords:
        return keywords
    fallback = [token.lower() for token in raw_tokens if token.lower() not in seen]
    if fallback:
        deduped: List[str] = []
        dedup_seen = set()
        for token in fallback:
            if token in dedup_seen:
                continue
            dedup_seen.add(token)
            deduped.append(token)
        if deduped:
            return deduped
    compact = query.strip().lower()
    if compact:
        return [compact]
    return []


def _normalize_filters(filters: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if filters is None:
        return {}
    if not isinstance(filters, Mapping):
        raise ValueError("filters must be a mapping")
    output: Dict[str, Any] = {}
    for key, value in filters.items():
        if not isinstance(key, str):
            continue
        normalized_key = key.strip()
        if not normalized_key:
            continue
        output[normalized_key] = _normalize_filter_value(value)
    return output


def _normalize_filter_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, Mapping):
        nested: Dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and key.strip():
                nested[key.strip()] = _normalize_filter_value(item)
        return nested
    if isinstance(value, list):
        return [_normalize_filter_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_filter_value(item) for item in value]
    return str(value)


def _strip_quoted(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1].strip()
    return value
