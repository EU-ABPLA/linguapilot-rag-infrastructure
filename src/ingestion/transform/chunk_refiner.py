from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform
from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory

_CODE_FENCE_PATTERN = re.compile(r"(```[\s\S]*?```)")


class ChunkRefiner(BaseTransform):
    def __init__(
        self, settings: Any, llm: Optional[BaseLLM] = None, prompt_path: Optional[str] = None
    ):
        self._settings = settings
        self._use_llm = _extract_use_llm(settings)
        self._prompt = self._load_prompt(prompt_path)
        self._llm: Optional[BaseLLM] = llm
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
                refined_rule = self._rule_based_refine(chunk.text)
                metadata = dict(chunk.metadata)
                metadata["refined_by"] = "rule"
                metadata.pop("refiner_fallback_reason", None)
                refined_text = refined_rule
                if self._use_llm:
                    refined_llm = self._llm_refine(refined_rule, trace)
                    if refined_llm is None:
                        metadata["refiner_fallback_reason"] = (
                            self._last_fallback_reason or "llm_unavailable"
                        )
                    else:
                        refined_text = refined_llm
                        metadata["refined_by"] = "llm"
                output.append(
                    Chunk(
                        id=chunk.id,
                        text=refined_text,
                        metadata=metadata,
                        start_offset=chunk.start_offset,
                        end_offset=chunk.end_offset,
                        source_ref=chunk.source_ref,
                    )
                )
            except Exception as exc:
                metadata = dict(chunk.metadata)
                metadata["refined_by"] = "original"
                metadata["refiner_error"] = str(exc)
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

    def _rule_based_refine(self, text: str) -> str:
        if not isinstance(text, str):
            raise ValueError("chunk text must be a string")
        if not text.strip():
            return ""
        parts = _CODE_FENCE_PATTERN.split(text)
        cleaned: List[str] = []
        for part in parts:
            if not part:
                continue
            if part.startswith("```") and part.endswith("```"):
                cleaned.append(part)
                continue
            normalized = _clean_plain_text(part)
            if normalized:
                cleaned.append(normalized)
        merged = "\n\n".join(cleaned)
        merged = re.sub(r"\n{3,}", "\n\n", merged)
        return merged.strip()

    def _llm_refine(
        self, text: str, trace: Optional[TraceContext] = None
    ) -> Optional[str]:
        self._last_fallback_reason = None
        if not self._use_llm:
            self._last_fallback_reason = "llm_disabled"
            return None
        if self._llm is None:
            self._last_fallback_reason = self._llm_init_error or "llm_unavailable"
            if trace is not None:
                trace.record_stage(
                    "chunk_refiner_llm",
                    {"status": "skipped", "reason": self._last_fallback_reason},
                )
            return None
        prompt = self._prompt.replace("{text}", text)
        messages = [{"role": "user", "content": prompt}]
        try:
            result = self._llm.chat(messages)
        except Exception as exc:
            self._last_fallback_reason = "llm_error: " + str(exc)
            if trace is not None:
                trace.record_stage(
                    "chunk_refiner_llm",
                    {"status": "error", "error": str(exc)},
                )
            return None
        if not isinstance(result, str) or not result.strip():
            self._last_fallback_reason = "llm_empty_response"
            if trace is not None:
                trace.record_stage(
                    "chunk_refiner_llm",
                    {"status": "error", "error": "empty_response"},
                )
            return None
        output = result.strip()
        if trace is not None:
            trace.record_stage(
                "chunk_refiner_llm",
                {"status": "ok", "input_length": len(text), "output_length": len(output)},
            )
        return output

    def _load_prompt(self, prompt_path: Optional[str] = None) -> str:
        candidate = prompt_path or _extract_prompt_path(self._settings)
        fallback = "Refine the chunk while preserving all factual information.\n\n{text}"
        if not isinstance(candidate, str) or not candidate.strip():
            return fallback
        file_path = Path(candidate)
        if not file_path.exists():
            return fallback
        text = file_path.read_text(encoding="utf-8")
        if not text.strip():
            return fallback
        if "{text}" not in text:
            return text.rstrip() + "\n\n{text}"
        return text


def _extract_use_llm(settings: Any) -> bool:
    if isinstance(settings, Mapping):
        ingestion = settings.get("ingestion")
        if isinstance(ingestion, Mapping):
            chunk_refiner = ingestion.get("chunk_refiner")
            if isinstance(chunk_refiner, Mapping):
                value = chunk_refiner.get("use_llm")
                if isinstance(value, bool):
                    return value
        return False
    ingestion = getattr(settings, "ingestion", None)
    if ingestion is None:
        return False
    chunk_refiner = getattr(ingestion, "chunk_refiner", None)
    if chunk_refiner is None:
        return False
    value = getattr(chunk_refiner, "use_llm", False)
    return bool(value)


def _extract_prompt_path(settings: Any) -> str:
    default = "config/prompts/chunk_refinement.txt"
    if isinstance(settings, Mapping):
        ingestion = settings.get("ingestion")
        if not isinstance(ingestion, Mapping):
            return default
        chunk_refiner = ingestion.get("chunk_refiner")
        if not isinstance(chunk_refiner, Mapping):
            return default
        value = chunk_refiner.get("prompt_path")
        if isinstance(value, str) and value.strip():
            return value
        return default
    ingestion = getattr(settings, "ingestion", None)
    if ingestion is None:
        return default
    chunk_refiner = getattr(ingestion, "chunk_refiner", None)
    if chunk_refiner is None:
        return default
    value = getattr(chunk_refiner, "prompt_path", default)
    if isinstance(value, str) and value.strip():
        return value
    return default


def _clean_plain_text(text: str) -> str:
    working = text.replace("\r\n", "\n").replace("\r", "\n")
    working = re.sub(r"<!--[\s\S]*?-->", " ", working)
    working = re.sub(r"<[^>\n]+>", " ", working)
    lines = working.split("\n")
    output: List[str] = []
    previous_blank = False
    for line in lines:
        normalized = re.sub(r"[ \t]+", " ", line).strip()
        if _is_noise_line(normalized):
            continue
        if not normalized:
            if previous_blank:
                continue
            previous_blank = True
            output.append("")
            continue
        previous_blank = False
        output.append(normalized)
    return "\n".join(output).strip()


def _is_noise_line(line: str) -> bool:
    if not line:
        return False
    lowered = line.lower()
    if re.match(r"^[-_*]{3,}$", line):
        return True
    if re.match(r"^page\s+\d+(\s+of\s+\d+)?$", lowered):
        return True
    if re.match(r"^第?\s*\d+\s*页$", line):
        return True
    if lowered in {"confidential", "internal use only"}:
        return True
    return False
