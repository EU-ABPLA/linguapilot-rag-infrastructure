from pathlib import Path

import pytest

from core.settings import SettingsError, load_settings


def test_load_settings_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    settings = load_settings("config/settings.yaml")
    assert isinstance(settings.llm.provider, str) and settings.llm.provider
    assert settings.llm.provider == "deepseek"
    assert settings.llm.base_url == "https://api.deepseek.com"
    assert settings.vision_llm.provider == ""
    assert settings.vision_llm.model == ""
    assert settings.ingestion.image_captioner.use_vision_llm is False
    assert isinstance(settings.embedding.provider, str) and settings.embedding.provider
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


def test_load_settings_resolves_env_placeholders(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key-123")
    config = """
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: ${LLM_API_KEY}
embedding:
  provider: openai
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

    settings = load_settings(str(file_path))
    assert settings.llm.api_key == "test-key-123"


def test_load_settings_raises_when_env_placeholder_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    config = """
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: ${LLM_API_KEY}
embedding:
  provider: openai
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
    assert "LLM_API_KEY" in str(exc_info.value)


def test_load_settings_reads_env_from_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / ".env").write_text("LLM_API_KEY=dotenv-key\n", encoding="utf-8")
    (config_dir / "settings.yaml").write_text(
        """
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: ${LLM_API_KEY}
embedding:
  provider: openai
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
""",
        encoding="utf-8",
    )

    settings = load_settings(str(config_dir / "settings.yaml"))
    assert settings.llm.api_key == "dotenv-key"


def test_load_settings_prefers_existing_env_over_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "shell-env-key")
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / ".env").write_text("LLM_API_KEY=dotenv-key\n", encoding="utf-8")
    (config_dir / "settings.yaml").write_text(
        """
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: ${LLM_API_KEY}
embedding:
  provider: openai
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
""",
        encoding="utf-8",
    )

    settings = load_settings(str(config_dir / "settings.yaml"))
    assert settings.llm.api_key == "shell-env-key"


def test_load_settings_uses_dotenv_when_env_value_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "")
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / ".env").write_text("LLM_API_KEY=dotenv-key\n", encoding="utf-8")
    (config_dir / "settings.yaml").write_text(
        """
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: ${LLM_API_KEY}
embedding:
  provider: openai
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
""",
        encoding="utf-8",
    )
    settings = load_settings(str(config_dir / "settings.yaml"))
    assert settings.llm.api_key == "dotenv-key"


def test_load_settings_requires_vision_config_when_captioner_enabled(tmp_path: Path) -> None:
    config = """
llm:
  provider: deepseek
  model: deepseek-v4-flash
embedding:
  provider: openai
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
ingestion:
  image_captioner:
    use_vision_llm: true
"""
    file_path = tmp_path / "settings.yaml"
    file_path.write_text(config, encoding="utf-8")

    with pytest.raises(SettingsError) as exc_info:
        load_settings(str(file_path))
    assert "vision_llm.provider" in str(exc_info.value)
