from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional

from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform
from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory

_WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_STOP_WORDS = {
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
    "you",
    "your",
    "into",
    "about",
    "have",
    "has",
    "had",
}


class MetadataEnricher(BaseTransform):
    def __init__(self, settings: Any, llm: Optional[BaseLLM] = None):
        self._settings = settings
        self._use_llm = _extract_use_llm(settings)
        self._llm = llm
        self._llm_init_error: Optional[str] = None
        self._last_fallback_reason: Optional[str] = None
        if self._llm is None and self._use_llm:
            try:
                self._llm = LLMFactory.create(settings)
            except Exception as exc:
                self._llm = None
                self._llm_init_error = str(exc)

    def transform(
        self, chunks: List[Chunk], trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        output: List[Chunk] = []
        for chunk in chunks:
            try:
                enriched = self._rule_enrich(chunk.text)
                metadata = dict(chunk.metadata)
                metadata["title"] = enriched["title"]
                metadata["summary"] = enriched["summary"]
                metadata["tags"] = enriched["tags"]
                metadata["metadata_enriched_by"] = "rule"
                metadata.pop("metadata_enricher_fallback_reason", None)
                if self._use_llm:
                    llm_metadata = self._llm_enrich(chunk.text, trace)
                    if llm_metadata is None:
                        metadata["metadata_enricher_fallback_reason"] = (
                            self._last_fallback_reason or "llm_unavailable"
                        )
                    else:
                        metadata["title"] = llm_metadata["title"]
                        metadata["summary"] = llm_metadata["summary"]
                        metadata["tags"] = llm_metadata["tags"]
                        metadata["metadata_enriched_by"] = "llm"
                output.append(
                    Chunk(
                        id=chunk.id,
                        text=chunk.text,
                        metadata=metadata,
                        start_offset=chunk.start_offset,
                        end_offset=chunk.end_offset,
                        source_ref=chunk.source_ref,
                    )
                )
            except Exception as exc:
                cleaned = _normalize_text(chunk.text)
                metadata = dict(chunk.metadata)
                metadata["title"] = _pick_title(cleaned)
                metadata["summary"] = _pick_summary(cleaned)
                metadata["tags"] = _extract_tags(cleaned)
                metadata["metadata_enriched_by"] = "rule"
                metadata["metadata_enricher_fallback_reason"] = "enricher_error: " + str(
                    exc
                )
                output.append(
                    Chunk(
                        id=chunk.id,
                        text=chunk.text,
                        metadata=metadata,
                        start_offset=chunk.start_offset,
                        end_offset=chunk.end_offset,
                        source_ref=chunk.source_ref,
                    )
                )
        return output

    def _rule_enrich(self, text: str) -> Dict[str, Any]:
        cleaned = _normalize_text(text)
        title = _pick_title(cleaned)
        summary = _pick_summary(cleaned)
        tags = _extract_tags(cleaned)
        return {"title": title, "summary": summary, "tags": tags}

    def _llm_enrich(
        self, text: str, trace: Optional[TraceContext] = None
    ) -> Optional[Dict[str, Any]]:
        self._last_fallback_reason = None
        if not self._use_llm:
            self._last_fallback_reason = "llm_disabled"
            return None
        if self._llm is None:
            self._last_fallback_reason = self._llm_init_error or "llm_unavailable"
            if trace is not None:
                trace.record_stage(
                    "metadata_enricher_llm",
                    {"status": "skipped", "reason": self._last_fallback_reason},
                )
            return None
        prompt = (
            "Generate JSON with keys title, summary, tags for the following chunk. "
            "tags must be an array of short keywords.\n\n"
            + text
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            raw = self._llm.chat(messages)
        except Exception as exc:
            self._last_fallback_reason = "llm_error: " + str(exc)
            if trace is not None:
                trace.record_stage(
                    "metadata_enricher_llm",
                    {"status": "error", "error": str(exc)},
                )
            return None
        parsed = _parse_llm_json(raw)
        if parsed is None:
            self._last_fallback_reason = "llm_invalid_payload"
            if trace is not None:
                trace.record_stage(
                    "metadata_enricher_llm",
                    {"status": "error", "error": "invalid_payload"},
                )
            return None
        if trace is not None:
            trace.record_stage(
                "metadata_enricher_llm",
                {"status": "ok", "title_length": len(parsed["title"]), "tag_count": len(parsed["tags"])},
            )
        return parsed


def _extract_use_llm(settings: Any) -> bool:
    if isinstance(settings, Mapping):
        ingestion = settings.get("ingestion")
        if isinstance(ingestion, Mapping):
            enricher = ingestion.get("metadata_enricher")
            if isinstance(enricher, Mapping):
                value = enricher.get("use_llm")
                if isinstance(value, bool):
                    return value
        return False
    ingestion = getattr(settings, "ingestion", None)
    if ingestion is None:
        return False
    enricher = getattr(ingestion, "metadata_enricher", None)
    if enricher is None:
        return False
    value = getattr(enricher, "use_llm", False)
    return bool(value)


def _parse_llm_json(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    candidate = raw.strip()
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", candidate)
        if match is None:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(data, Mapping):
        return None
    title = data.get("title")
    summary = data.get("summary")
    tags = data.get("tags")
    if not isinstance(title, str) or not title.strip():
        return None
    if not isinstance(summary, str) or not summary.strip():
        return None
    if not isinstance(tags, list):
        return None
    normalized_tags: List[str] = []
    for item in tags:
        if isinstance(item, str) and item.strip():
            normalized_tags.append(item.strip())
    if not normalized_tags:
        return None
    return {
        "title": title.strip(),
        "summary": summary.strip(),
        "tags": normalized_tags[:5],
    }


def _normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _pick_title(text: str) -> str:
    if not text:
        return "Untitled Chunk"
    lines = [line.strip("# ").strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return "Untitled Chunk"
    title = lines[0]
    if len(title) > 80:
        title = title[:80].rstrip()
    return title or "Untitled Chunk"


def _pick_summary(text: str) -> str:
    if not text:
        return "No summary available."
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= 180:
        return compact
    return compact[:180].rstrip() + "..."


def _extract_tags(text: str) -> List[str]:
    words = [w.lower() for w in _WORD_PATTERN.findall(text)]
    filtered = [w for w in words if w not in _STOP_WORDS]
    if not filtered:
        return ["general"]
    counts = Counter(filtered)
    ordered = [item for item, _ in counts.most_common(5)]
    return ordered or ["general"]
