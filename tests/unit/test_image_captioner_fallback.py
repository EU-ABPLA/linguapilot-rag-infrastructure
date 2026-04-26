from typing import Any, Mapping, Optional, Sequence, Union

from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.image_captioner import ImageCaptioner
from libs.llm.base_vision_llm import BaseVisionLLM


class FakeVisionLLM(BaseVisionLLM):
    def __init__(self, raise_for: Optional[str] = None):
        self.raise_for = raise_for
        self.calls = []

    def chat_with_image(
        self,
        text: str,
        image_path: Union[str, bytes],
        trace: Optional[Any] = None,
    ) -> str:
        self.calls.append({"text": text, "image_path": image_path})
        if isinstance(image_path, str) and self.raise_for and self.raise_for in image_path:
            raise RuntimeError("caption failure")
        if not isinstance(image_path, str):
            return ""
        return "caption for " + image_path


def _chunk(
    text: str = "chunk text",
    image_refs: Optional[Sequence[str]] = None,
    images: Optional[list] = None,
) -> Chunk:
    metadata = {"source_path": "data/docs/sample.pdf"}
    if image_refs is not None:
        metadata["image_refs"] = list(image_refs)
    if images is not None:
        metadata["images"] = images
    return Chunk(
        id="img-chunk-1",
        text=text,
        metadata=metadata,
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-1",
    )


def test_image_captioner_enabled_generates_captions() -> None:
    llm = FakeVisionLLM()
    settings = {
        "ingestion": {
            "image_captioner": {
                "use_vision_llm": True,
                "prompt_path": "config/prompts/image_captioning.txt",
            }
        }
    }
    captioner = ImageCaptioner(settings, vision_llm=llm)
    chunk = _chunk(
        image_refs=["img1", "img2"],
        images=[{"id": "img1", "path": "data/images/img1.png"}, {"id": "img2", "path": "data/images/img2.png"}],
    )
    out = captioner.transform([chunk])[0]
    assert out.metadata["image_captions"]["img1"] == "caption for data/images/img1.png"
    assert out.metadata["image_captions"]["img2"] == "caption for data/images/img2.png"
    assert "has_unprocessed_images" not in out.metadata
    assert len(llm.calls) == 2


def test_image_captioner_disabled_marks_unprocessed() -> None:
    llm = FakeVisionLLM()
    settings = {"ingestion": {"image_captioner": {"use_vision_llm": False}}}
    captioner = ImageCaptioner(settings, vision_llm=llm)
    chunk = _chunk(
        image_refs=["img1"],
        images=[{"id": "img1", "path": "data/images/img1.png"}],
    )
    out = captioner.transform([chunk])[0]
    assert out.metadata["has_unprocessed_images"] is True
    assert "image_captions" not in out.metadata
    assert len(llm.calls) == 0


def test_image_captioner_llm_error_falls_back_without_blocking() -> None:
    llm = FakeVisionLLM(raise_for="img1.png")
    settings = {"ingestion": {"image_captioner": {"use_vision_llm": True}}}
    captioner = ImageCaptioner(settings, vision_llm=llm)
    chunk = _chunk(
        image_refs=["img1"],
        images=[{"id": "img1", "path": "data/images/img1.png"}],
    )
    out = captioner.transform([chunk])[0]
    assert out.metadata["has_unprocessed_images"] is True
    assert out.metadata["image_captioner_fallback_reason"].startswith("vision_llm_error:")
    assert "image_captions" not in out.metadata


def test_image_captioner_keeps_chunks_without_image_refs_unchanged() -> None:
    llm = FakeVisionLLM()
    settings = {"ingestion": {"image_captioner": {"use_vision_llm": True}}}
    captioner = ImageCaptioner(settings, vision_llm=llm)
    chunk = _chunk(image_refs=None, images=None)
    out = captioner.transform([chunk])[0]
    assert "has_unprocessed_images" not in out.metadata
    assert "image_captions" not in out.metadata
    assert len(llm.calls) == 0


def test_image_captioner_missing_image_path_marks_unprocessed() -> None:
    llm = FakeVisionLLM()
    settings = {"ingestion": {"image_captioner": {"use_vision_llm": True}}}
    captioner = ImageCaptioner(settings, vision_llm=llm)
    chunk = _chunk(image_refs=["img1"], images=[])
    out = captioner.transform([chunk])[0]
    assert out.metadata["has_unprocessed_images"] is True
    assert out.metadata["image_captioner_fallback_reason"] == "missing_image_path"
    assert "image_captions" not in out.metadata


def test_image_captioner_partial_success_keeps_captions_and_marks_unprocessed() -> None:
    llm = FakeVisionLLM()
    settings = {"ingestion": {"image_captioner": {"use_vision_llm": True}}}
    captioner = ImageCaptioner(settings, vision_llm=llm)
    chunk = _chunk(
        image_refs=["img1", "img2"],
        images=[{"id": "img1", "path": "data/images/img1.png"}],
    )
    out = captioner.transform([chunk])[0]
    assert out.metadata["image_captions"]["img1"] == "caption for data/images/img1.png"
    assert out.metadata["has_unprocessed_images"] is True
    assert out.metadata["image_captioner_fallback_reason"] == "missing_image_path"


def test_image_captioner_records_trace_stage() -> None:
    llm = FakeVisionLLM()
    settings = {"ingestion": {"image_captioner": {"use_vision_llm": True}}}
    captioner = ImageCaptioner(settings, vision_llm=llm)
    chunk = _chunk(
        image_refs=["img1"],
        images=[{"id": "img1", "path": "data/images/img1.png"}],
    )
    trace = TraceContext(trace_type="ingestion")
    captioner.transform([chunk], trace=trace)
    assert any(stage["stage"] == "image_captioner_llm" for stage in trace.stages)
