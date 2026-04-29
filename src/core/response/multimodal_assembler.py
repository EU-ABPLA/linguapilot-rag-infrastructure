from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from core.types import RetrievalResult


class MultimodalAssembler:
    def build_image_content(
        self,
        retrieval_results: Sequence[RetrievalResult],
    ) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in retrieval_results:
            if not isinstance(item, RetrievalResult):
                continue
            for image_path in _collect_image_paths(item.metadata):
                normalized = str(Path(image_path))
                if normalized in seen:
                    continue
                payload = _encode_image(normalized)
                if payload is None:
                    continue
                output.append(payload)
                seen.add(normalized)
        return output


def _collect_image_paths(metadata: Mapping[str, Any]) -> List[str]:
    refs = _normalize_refs(metadata.get("image_refs"))
    image_by_id = _image_map(metadata.get("images"))
    output: List[str] = []
    if refs:
        for ref in refs:
            path = image_by_id.get(ref)
            if path:
                output.append(path)
        return output
    for path in image_by_id.values():
        output.append(path)
    return output


def _normalize_refs(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    output: List[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            output.append(item.strip())
    return output


def _image_map(value: Any) -> Dict[str, str]:
    if not isinstance(value, list):
        return {}
    output: Dict[str, str] = {}
    for item in value:
        if not isinstance(item, Mapping):
            continue
        image_id = item.get("id")
        image_path = item.get("path")
        if not isinstance(image_id, str) or not image_id.strip():
            continue
        if not isinstance(image_path, str) or not image_path.strip():
            continue
        output[image_id.strip()] = image_path.strip()
    return output


def _encode_image(path: str) -> Optional[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return None
    mime_type = _guess_mime_type(file_path.suffix)
    if mime_type is None:
        return None
    encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return {"type": "image", "mimeType": mime_type, "data": encoded}


def _guess_mime_type(suffix: str) -> Optional[str]:
    key = suffix.lower()
    if key == ".png":
        return "image/png"
    if key == ".jpg" or key == ".jpeg":
        return "image/jpeg"
    if key == ".webp":
        return "image/webp"
    if key == ".gif":
        return "image/gif"
    return None
