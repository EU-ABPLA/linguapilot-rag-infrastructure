from pathlib import Path

import pytest

from core.settings import SettingsError, load_settings


def test_load_settings_success() -> None:
    settings = load_settings("config/settings.yaml")
    assert settings.llm.provider == "openai"
    assert settings.embedding.provider == "openai"
    assert settings.vector_store.provider == "chroma"
    assert settings.retrieval.top_k == 5


def test_load_settings_missing_field_error_path(tmp_path: Path) -> None:
    config = """
llm:
  provider: openai
  model: gpt-4o-mini
embedding:
  model: text-embedding-3-small
vector_store:
  provider: chroma
retrieval:
  top_k: 5
rerank:
  enabled: false
evaluation:
  backends: [custom]
observability:
  enabled: true
"""
    file_path = tmp_path / "settings.yaml"
    file_path.write_text(config, encoding="utf-8")

    with pytest.raises(SettingsError) as exc_info:
        load_settings(str(file_path))

    assert "embedding.provider" in str(exc_info.value)
