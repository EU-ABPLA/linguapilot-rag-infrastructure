import os
from typing import Mapping, Sequence

import pytest

from core.types import Chunk
from ingestion.transform.chunk_refiner import ChunkRefiner
from libs.llm.base_llm import BaseLLM
from libs.llm.openai_llm import OpenAILLM


class BrokenLLM(BaseLLM):
    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        raise RuntimeError("forced integration error")


def _chunk(text: str) -> Chunk:
    return Chunk(
        id="i1",
        text=text,
        metadata={"source_path": "data/docs/integration.md"},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-int",
    )


@pytest.mark.integration
def test_chunk_refiner_real_llm_refinement() -> None:
    if os.getenv("RUN_REAL_LLM", "0") != "1":
        pytest.skip("set RUN_REAL_LLM=1 to run real llm integration test")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        pytest.skip("OPENAI_API_KEY is required for real llm integration")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = OpenAILLM(model=model, api_key=api_key)
    settings = {
        "ingestion": {"chunk_refiner": {"use_llm": True, "prompt_path": "config/prompts/chunk_refinement.txt"}}
    }
    refiner = ChunkRefiner(settings, llm=llm)
    noisy = _chunk("Page 2 of 3\n\nThis   is  useful text.")
    out = refiner.transform([noisy])
    assert len(out) == 1
    assert out[0].metadata["refined_by"] in {"llm", "rule"}
    assert isinstance(out[0].text, str)
    assert out[0].text.strip()


@pytest.mark.integration
def test_chunk_refiner_invalid_llm_fallbacks_to_rule_mode() -> None:
    settings = {"ingestion": {"chunk_refiner": {"use_llm": True}}}
    refiner = ChunkRefiner(settings, llm=BrokenLLM())
    noisy = _chunk("Page 9\n\nA   B")
    out = refiner.transform([noisy])
    assert out[0].metadata["refined_by"] == "rule"
    assert out[0].metadata["refiner_fallback_reason"].startswith("llm_error:")
