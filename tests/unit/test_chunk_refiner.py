import json
from pathlib import Path
from typing import Mapping, Sequence

import pytest

from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform
from ingestion.transform.chunk_refiner import ChunkRefiner
from libs.llm.base_llm import BaseLLM


class EchoLLM(BaseLLM):
    def __init__(self, response: str = "llm result"):
        self.response = response
        self.calls = []

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        self.calls.append(messages)
        return self.response


class EmptyLLM(BaseLLM):
    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        return "   "


class ErrorLLM(BaseLLM):
    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        raise RuntimeError("boom")


def _chunk(text: str = "hello") -> Chunk:
    return Chunk(
        id="c1",
        text=text,
        metadata={"source_path": "data/docs/test.md"},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-1",
    )


def _fixtures() -> dict:
    return json.loads(Path("tests/fixtures/noisy_chunks.json").read_text(encoding="utf-8"))


def test_01_base_transform_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseTransform()


def test_02_noisy_fixture_has_eight_scenarios() -> None:
    data = _fixtures()
    assert len(data) == 8


def test_03_load_prompt_from_file_with_placeholder(tmp_path: Path) -> None:
    prompt = tmp_path / "p.txt"
    prompt.write_text("Refine:\n{text}", encoding="utf-8")
    refiner = ChunkRefiner({"ingestion": {"chunk_refiner": {"use_llm": False}}}, prompt_path=str(prompt))
    assert "{text}" in refiner._prompt


def test_04_load_prompt_appends_placeholder_when_missing(tmp_path: Path) -> None:
    prompt = tmp_path / "p.txt"
    prompt.write_text("Refine only", encoding="utf-8")
    refiner = ChunkRefiner({"ingestion": {"chunk_refiner": {"use_llm": False}}}, prompt_path=str(prompt))
    assert "{text}" in refiner._prompt


def test_05_rule_refine_typical_noise() -> None:
    data = _fixtures()["typical_noise_scenario"]
    refiner = ChunkRefiner({"ingestion": {"chunk_refiner": {"use_llm": False}}})
    assert refiner._rule_based_refine(data["input"]) == data["expected"]


def test_06_rule_refine_ocr_noise() -> None:
    data = _fixtures()["ocr_errors"]
    refiner = ChunkRefiner({"ingestion": {"chunk_refiner": {"use_llm": False}}})
    assert refiner._rule_based_refine(data["input"]) == data["expected"]


def test_07_rule_refine_page_header_footer() -> None:
    data = _fixtures()["page_header_footer"]
    refiner = ChunkRefiner({"ingestion": {"chunk_refiner": {"use_llm": False}}})
    assert refiner._rule_based_refine(data["input"]) == data["expected"]


def test_08_rule_refine_excessive_whitespace() -> None:
    data = _fixtures()["excessive_whitespace"]
    refiner = ChunkRefiner({"ingestion": {"chunk_refiner": {"use_llm": False}}})
    assert refiner._rule_based_refine(data["input"]) == data["expected"]


def test_09_rule_refine_format_markers() -> None:
    data = _fixtures()["format_markers"]
    refiner = ChunkRefiner({"ingestion": {"chunk_refiner": {"use_llm": False}}})
    assert refiner._rule_based_refine(data["input"]) == data["expected"]


def test_10_rule_refine_clean_text_not_over_cleaned() -> None:
    data = _fixtures()["clean_text"]
    refiner = ChunkRefiner({"ingestion": {"chunk_refiner": {"use_llm": False}}})
    assert refiner._rule_based_refine(data["input"]) == data["expected"]


def test_11_rule_refine_preserves_code_blocks() -> None:
    data = _fixtures()["code_blocks"]
    refiner = ChunkRefiner({"ingestion": {"chunk_refiner": {"use_llm": False}}})
    cleaned = refiner._rule_based_refine(data["input"])
    assert data["expected_contains"] in cleaned


def test_12_rule_refine_mixed_noise() -> None:
    data = _fixtures()["mixed_noise"]
    refiner = ChunkRefiner({"ingestion": {"chunk_refiner": {"use_llm": False}}})
    assert refiner._rule_based_refine(data["input"]) == data["expected"]


def test_13_transform_rule_mode_sets_metadata_refined_by_rule() -> None:
    refiner = ChunkRefiner({"ingestion": {"chunk_refiner": {"use_llm": False}}})
    out = refiner.transform([_chunk("A    B")])
    assert out[0].metadata["refined_by"] == "rule"


def test_14_transform_rule_mode_keeps_batch_size() -> None:
    refiner = ChunkRefiner({"ingestion": {"chunk_refiner": {"use_llm": False}}})
    out = refiner.transform([_chunk("x"), _chunk("y")])
    assert len(out) == 2


def test_15_transform_rule_mode_handles_empty_list() -> None:
    refiner = ChunkRefiner({"ingestion": {"chunk_refiner": {"use_llm": False}}})
    assert refiner.transform([]) == []


def test_16_transform_rule_mode_keeps_source_path() -> None:
    refiner = ChunkRefiner({"ingestion": {"chunk_refiner": {"use_llm": False}}})
    out = refiner.transform([_chunk("x")])
    assert out[0].metadata["source_path"] == "data/docs/test.md"


def test_17_transform_rule_mode_ignores_llm_when_disabled() -> None:
    llm = EchoLLM("from llm")
    refiner = ChunkRefiner({"ingestion": {"chunk_refiner": {"use_llm": False}}}, llm=llm)
    out = refiner.transform([_chunk("A  B")])
    assert out[0].metadata["refined_by"] == "rule"
    assert len(llm.calls) == 0


def test_18_transform_llm_mode_uses_llm_output() -> None:
    llm = EchoLLM("LLM cleaned")
    refiner = ChunkRefiner({"ingestion": {"chunk_refiner": {"use_llm": True}}}, llm=llm)
    out = refiner.transform([_chunk("A  B")])
    assert out[0].text == "LLM cleaned"
    assert out[0].metadata["refined_by"] == "llm"


def test_19_transform_llm_mode_fallback_on_exception() -> None:
    refiner = ChunkRefiner({"ingestion": {"chunk_refiner": {"use_llm": True}}}, llm=ErrorLLM())
    out = refiner.transform([_chunk("A   B")])
    assert out[0].metadata["refined_by"] == "rule"
    assert out[0].metadata["refiner_fallback_reason"].startswith("llm_error:")


def test_20_transform_llm_mode_fallback_on_empty_response() -> None:
    refiner = ChunkRefiner({"ingestion": {"chunk_refiner": {"use_llm": True}}}, llm=EmptyLLM())
    out = refiner.transform([_chunk("A   B")])
    assert out[0].metadata["refined_by"] == "rule"
    assert out[0].metadata["refiner_fallback_reason"] == "llm_empty_response"


def test_21_transform_llm_mode_fallback_when_llm_missing() -> None:
    refiner = ChunkRefiner({"ingestion": {"chunk_refiner": {"use_llm": True}}}, llm=None)
    out = refiner.transform([_chunk("A   B")])
    assert out[0].metadata["refined_by"] == "rule"
    assert "refiner_fallback_reason" in out[0].metadata


def test_22_llm_refine_returns_none_when_disabled() -> None:
    refiner = ChunkRefiner({"ingestion": {"chunk_refiner": {"use_llm": False}}}, llm=EchoLLM("x"))
    assert refiner._llm_refine("abc") is None


def test_23_llm_refine_formats_prompt_with_text() -> None:
    llm = EchoLLM("ok")
    refiner = ChunkRefiner(
        {"ingestion": {"chunk_refiner": {"use_llm": True}}},
        llm=llm,
        prompt_path="config/prompts/chunk_refinement.txt",
    )
    refiner._llm_refine("payload")
    first = llm.calls[0][0]["content"]
    assert "payload" in first


def test_24_llm_refine_records_trace_on_success() -> None:
    llm = EchoLLM("ok")
    trace = TraceContext(trace_type="ingestion")
    refiner = ChunkRefiner({"ingestion": {"chunk_refiner": {"use_llm": True}}}, llm=llm)
    refiner._llm_refine("payload", trace=trace)
    assert any(stage["stage"] == "chunk_refiner_llm" for stage in trace.stages)


def test_25_llm_refine_records_trace_on_error() -> None:
    trace = TraceContext(trace_type="ingestion")
    refiner = ChunkRefiner({"ingestion": {"chunk_refiner": {"use_llm": True}}}, llm=ErrorLLM())
    refiner._llm_refine("payload", trace=trace)
    assert any(stage["details"].get("status") == "error" for stage in trace.stages)


def test_26_transform_single_chunk_failure_does_not_break_batch() -> None:
    class CrashRefiner(ChunkRefiner):
        def _rule_based_refine(self, text: str) -> str:
            if "crash" in text:
                raise RuntimeError("bad chunk")
            return super()._rule_based_refine(text)

    refiner = CrashRefiner({"ingestion": {"chunk_refiner": {"use_llm": False}}})
    chunks = [
        _chunk("crash payload"),
        Chunk(
            id="c2",
            text="good   payload",
            metadata={"source_path": "data/docs/test.md"},
            start_offset=0,
            end_offset=12,
            source_ref="doc-1",
        ),
    ]
    out = refiner.transform(chunks)
    assert out[0].metadata["refined_by"] == "original"
    assert out[1].metadata["refined_by"] == "rule"


def test_27_transform_llm_mode_mapping_switch_true() -> None:
    llm = EchoLLM("mapped")
    settings = {"ingestion": {"chunk_refiner": {"use_llm": True}}}
    refiner = ChunkRefiner(settings, llm=llm)
    out = refiner.transform([_chunk("x")])
    assert out[0].metadata["refined_by"] == "llm"
    assert out[0].text == "mapped"
