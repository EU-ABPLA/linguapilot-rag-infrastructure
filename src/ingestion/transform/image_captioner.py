from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform
from libs.llm.base_vision_llm import BaseVisionLLM
from libs.llm.llm_factory import LLMFactory


class ImageCaptioner(BaseTransform):
    def __init__(
        self,
        settings: Any,
        vision_llm: Optional[BaseVisionLLM] = None,
        prompt_path: Optional[str] = None,
    ):
        self._settings = settings
        self._use_vision_llm = _extract_use_vision_llm(settings)
        self._vision_llm = vision_llm
        self._vision_init_error: Optional[str] = None
        self._prompt_template = self._load_prompt(prompt_path)
        if self._vision_llm is None and self._use_vision_llm:
            try:
                self._vision_llm = LLMFactory.create_vision_llm(settings)
            except Exception as exc:
                self._vision_llm = None
                self._vision_init_error = str(exc)

    def transform(
        self, chunks: List[Chunk], trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        output: List[Chunk] = []
        for chunk in chunks:
            metadata = dict(chunk.metadata)
            refs = _normalize_refs(metadata.get("image_refs"))
            if not refs:
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
                continue
            if not self._use_vision_llm or self._vision_llm is None:
                metadata["has_unprocessed_images"] = True
                metadata.pop("image_captions", None)
                if self._vision_init_error:
                    metadata["image_captioner_fallback_reason"] = self._vision_init_error
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
                continue
            image_paths = _extract_image_paths(metadata.get("images"))
            captions: Dict[str, str] = {}
            has_unprocessed = False
            fallback_reason: Optional[str] = None
            for image_id in refs:
                image_path = image_paths.get(image_id)
                if image_path is None:
                    has_unprocessed = True
                    fallback_reason = fallback_reason or "missing_image_path"
                    continue
                prompt = self._prompt_template.replace("{chunk_text}", chunk.text).replace(
                    "{image_id}", image_id
                )
                try:
                    caption = self._vision_llm.chat_with_image(
                        prompt, image_path, trace=trace
                    )
                except Exception as exc:
                    has_unprocessed = True
                    fallback_reason = fallback_reason or ("vision_llm_error: " + str(exc))
                    if trace is not None:
                        trace.record_stage(
                            "image_captioner_llm",
                            {"status": "error", "image_id": image_id, "error": str(exc)},
                        )
                    continue
                if isinstance(caption, str) and caption.strip():
                    captions[image_id] = caption.strip()
                    if trace is not None:
                        trace.record_stage(
                            "image_captioner_llm",
                            {"status": "ok", "image_id": image_id},
                        )
                else:
                    has_unprocessed = True
                    fallback_reason = fallback_reason or "empty_caption"
            if captions:
                metadata["image_captions"] = captions
            else:
                metadata.pop("image_captions", None)
            if has_unprocessed:
                metadata["has_unprocessed_images"] = True
                if fallback_reason:
                    metadata["image_captioner_fallback_reason"] = fallback_reason
            else:
                metadata.pop("has_unprocessed_images", None)
                metadata.pop("image_captioner_fallback_reason", None)
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

    def _load_prompt(self, prompt_path: Optional[str] = None) -> str:
        candidate = prompt_path or _extract_prompt_path(self._settings)
        fallback = "Describe image {image_id} in the chunk context.\n\n{chunk_text}"
        if not isinstance(candidate, str) or not candidate.strip():
            return fallback
        file_path = Path(candidate)
        if not file_path.exists():
            return fallback
        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            return fallback
        if "{chunk_text}" not in text:
            text = text + "\n\n{chunk_text}"
        if "{image_id}" not in text:
            text = text + "\n\nImage ID: {image_id}"
        return text


def _extract_use_vision_llm(settings: Any) -> bool:
    if isinstance(settings, Mapping):
        ingestion = settings.get("ingestion")
        if not isinstance(ingestion, Mapping):
            return False
        image_captioner = ingestion.get("image_captioner")
        if not isinstance(image_captioner, Mapping):
            return False
        value = image_captioner.get("use_vision_llm")
        if isinstance(value, bool):
            return value
        return False
    ingestion = getattr(settings, "ingestion", None)
    if ingestion is None:
        return False
    image_captioner = getattr(ingestion, "image_captioner", None)
    if image_captioner is None:
        return False
    return bool(getattr(image_captioner, "use_vision_llm", False))


def _extract_prompt_path(settings: Any) -> str:
    default = "config/prompts/image_captioning.txt"
    if isinstance(settings, Mapping):
        ingestion = settings.get("ingestion")
        if not isinstance(ingestion, Mapping):
            return default
        image_captioner = ingestion.get("image_captioner")
        if not isinstance(image_captioner, Mapping):
            return default
        value = image_captioner.get("prompt_path")
        if isinstance(value, str) and value.strip():
            return value
        return default
    ingestion = getattr(settings, "ingestion", None)
    if ingestion is None:
        return default
    image_captioner = getattr(ingestion, "image_captioner", None)
    if image_captioner is None:
        return default
    value = getattr(image_captioner, "prompt_path", default)
    if isinstance(value, str) and value.strip():
        return value
    return default


def _normalize_refs(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    refs: List[str] = []
    for item in value:
        if isinstance(item, str) and item.strip() and item not in refs:
            refs.append(item)
    return refs


def _extract_image_paths(images: Any) -> Dict[str, str]:
    if not isinstance(images, list):
        return {}
    output: Dict[str, str] = {}
    for item in images:
        if not isinstance(item, Mapping):
            continue
        image_id = item.get("id")
        image_path = item.get("path")
        if isinstance(image_id, str) and image_id and isinstance(image_path, str) and image_path:
            output[image_id] = image_path
    return output
