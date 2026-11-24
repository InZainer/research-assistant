import os

import pytest

from research_assistant.cli import main
from research_assistant.config import load_settings
from research_assistant.providers import LMSTUDIO_URL, PROVIDERS, describe_all, get_provider, resolve


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Переменные провайдеров и SDK (в том числе ANTHROPIC_BASE_URL, которую выставляет Claude Code)."""
    for name in list(os.environ):
        if name.startswith("LLM_") or name.endswith("_API_KEY") or name.endswith("_BASE_URL") or name in {
                "MODEL", "REASONING_EFFORT", "ANTHROPIC_MODEL", "EMBED_MODEL"}:
            monkeypatch.delenv(name, raising=False)


def test_without_provider_old_openai_variables_work(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "lm-studio")
    monkeypatch.setenv("MODEL", "qwen3.5-4b")
    settings = load_settings(env_file=None)
    assert settings.provider is None and settings.model == "qwen3.5-4b"
    assert settings.embed_base_url == "http://localhost:1234/v1"  # прежнее правило: эмбеддинги там же


def test_cloud_provider_uses_own_key_and_keeps_embeddings_local(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "inception")
    monkeypatch.setenv("INCEPTION_API_KEY", "test-key")
    settings = load_settings(env_file=None)
    assert settings.provider == "inception"
    assert settings.base_url == "https://api.inceptionlabs.ai/v1" and settings.model == "mercury-3"
    assert settings.api_key == "test-key" and settings.missing_llm_vars() == []
    assert settings.embed_base_url == LMSTUDIO_URL and settings.embed_api_key == "lm-studio"


def test_provider_argument_overrides_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "inception")
    settings = load_settings(env_file=None, provider="lmstudio")
    assert settings.provider == "lmstudio" and settings.api_key == "lm-studio"
    assert settings.model == "qwen3.5-4b" and settings.reasoning_effort == "none"
    assert settings.embed_base_url == LMSTUDIO_URL  # локальный провайдер: эмбеддинги у него же


def test_missing_key_and_model_are_named_by_provider_variables():
    settings = load_settings(env_file=None, provider="openrouter")
    assert settings.missing_llm_vars() == ["OPENROUTER_API_KEY", "LLM_OPENROUTER_MODEL"]


def test_sdk_variables_do_not_leak_into_profiles(monkeypatch):
    # Claude Code и SDK используют эти имена для своих целей: профиль их не читает.
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example")
    monkeypatch.setenv("ANTHROPIC_MODEL", "some-claude-code-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.inceptionlabs.ai/v1")
    assert resolve("anthropic").base_url == "https://api.anthropic.com/v1/"
    assert resolve("anthropic").model == "claude-sonnet-5-5"
    assert resolve("openai").base_url == "https://api.openai.com/v1"


def test_own_overrides(monkeypatch):
    monkeypatch.setenv("LLM_OPENROUTER_MODEL", "vendor/model")
    monkeypatch.setenv("LLM_OPENROUTER_API_KEY", "own-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "standard-key")
    monkeypatch.setenv("LLM_LMSTUDIO_REASONING_EFFORT", "")  # пустая строка — не передавать reasoning_effort
    chat = resolve("openrouter")
    assert chat.model == "vendor/model" and chat.api_key == "own-key" and chat.key_var == "LLM_OPENROUTER_API_KEY"
    assert resolve("lmstudio").reasoning_effort is None
    monkeypatch.setenv("LLM_CUSTOM_BASE_URL", "https://llm.example/v1")
    assert resolve("custom").base_url == "https://llm.example/v1"
    assert resolve("custom").base_url and resolve("Custom").provider.name == "custom"


def test_unknown_provider():
    with pytest.raises(ValueError, match="неизвестный провайдер"):
        get_provider("nosuch")
    assert main(["--provider", "nosuch", "providers"]) == 2


def test_describe_never_prints_key_values(monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-value-123")
    rows = {row["name"]: row for row in describe_all()}
    assert rows["anthropic"]["key"] == "есть" and rows["lmstudio"]["key"] == "не нужен"
    assert set(rows) == set(PROVIDERS)
    assert main(["--provider", "anthropic", "providers"]) == 0
    out = capsys.readouterr().out
    assert "secret-value-123" not in out and "* anthropic" in out
