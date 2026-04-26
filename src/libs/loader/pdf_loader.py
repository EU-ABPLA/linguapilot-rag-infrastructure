import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from core.types import Document
from libs.loader.base_loader import BaseLoader


class PdfLoader(BaseLoader):
    def __init__(self, image_output_root: str = "data/images"):
        self.image_output_root = Path(image_output_root)

    def load(self, path: str) -> Document:
        file_path = Path(path)
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError("pdf file not found: " + path)
        raw = file_path.read_bytes()
        if not raw:
            raise ValueError("pdf file is empty: " + path)
        doc_hash = hashlib.sha256(raw).hexdigest()[:16]
        text = _extract_text(raw)
        images = self._extract_images(raw, doc_hash)
        if images:
            text, images = _inject_image_placeholders(text, images)
        metadata: Dict[str, Any] = {
            "source_path": str(file_path),
            "doc_type": "pdf",
        }
        if images:
            metadata["images"] = images
        return Document(id=doc_hash, text=text, metadata=metadata)

    def _extract_images(self, raw: bytes, doc_hash: str) -> List[Dict[str, Any]]:
        marker = b"/Subtype /Image"
        count = raw.count(marker)
        if count <= 0:
            return []
        output_dir = self.image_output_root / doc_hash
        output_dir.mkdir(parents=True, exist_ok=True)
        results: List[Dict[str, Any]] = []
        for index in range(count):
            image_id = f"{doc_hash}_1_{index + 1}"
            image_path = output_dir / f"{image_id}.png"
            image_path.write_bytes(_placeholder_png_bytes(index))
            results.append(
                {
                    "id": image_id,
                    "path": str(image_path),
                    "page": 1,
                    "text_offset": 0,
                    "text_length": 0,
                    "position": {"x": 0, "y": 0, "w": 0, "h": 0},
                }
            )
        return results


def _extract_text(raw: bytes) -> str:
    decoded = raw.decode("latin1", errors="ignore")
    cleaned = re.sub(r"\s+", " ", decoded).strip()
    if cleaned:
        return cleaned
    return "PDF content"


def _inject_image_placeholders(
    text: str, images: List[Dict[str, Any]]
) -> Tuple[str, List[Dict[str, Any]]]:
    working_text = text
    updated_images: List[Dict[str, Any]] = []
    for image in images:
        placeholder = "[IMAGE: " + str(image["id"]) + "]"
        if working_text:
            working_text += "\n"
        offset = len(working_text)
        working_text += placeholder
        updated = dict(image)
        updated["text_offset"] = offset
        updated["text_length"] = len(placeholder)
        updated_images.append(updated)
    return working_text, updated_images


def _placeholder_png_bytes(seed: int) -> bytes:
    prefix = b"\x89PNG\r\n\x1a\n"
    marker = ("placeholder-" + str(seed)).encode("utf-8")
    return prefix + marker
