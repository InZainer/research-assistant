from pathlib import Path

import pytest

from research_assistant.config import load_settings

ENV_VARS = ["OPENAI_BASE_URL", "OPENAI_API_KEY", "MODEL", "EMBED_MODEL", "DATA_DIR", "RUNS_DIR", "MAX_STEPS", "CONTACT_EMAIL", "OPENALEX_API_KEY",
            "EMBED_BASE_URL", "EMBED_API_KEY", "REASONING_EFFORT", "LLM_PROVIDER"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_defaults():
    settings = load_settings(env_file=None)
    assert settings.max_steps == 12
    assert settings.db_path == Path("data") / "library.sqlite"
    assert settings.missing_llm_vars() == ["OPENAI_API_KEY", "MODEL"]


def test_reads_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "lm-studio")
    monkeypatch.setenv("MODEL", "qwen3.5-4b")
    monkeypatch.setenv("MAX_STEPS", "5")
    settings = load_settings(env_file=None)
    assert settings.base_url == "http://localhost:1234/v1"
    assert settings.model == "qwen3.5-4b"
    assert settings.max_steps == 5
    assert settings.missing_llm_vars() == []


@pytest.mark.parametrize("value", ["abc", "0"])
def test_invalid_max_steps(monkeypatch, value):
    monkeypatch.setenv("MAX_STEPS", value)
    with pytest.raises(ValueError, match="MAX_STEPS"):
        load_settings(env_file=None)


def test_user_agent_mentions_contact(monkeypatch):
    assert "student project" in load_settings(env_file=None).user_agent
    monkeypatch.setenv("CONTACT_EMAIL", "me@example.org")
    assert "mailto:me@example.org" in load_settings(env_file=None).user_agent


def test_embedding_service_defaults_to_chat_service(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
    settings = load_settings(env_file=None)
    assert settings.embed_base_url == "http://localhost:1234/v1"
    assert settings.embed_model == ""
    monkeypatch.setenv("EMBED_BASE_URL", "https://embeddings.example/v1")
    assert load_settings(env_file=None).embed_base_url == "https://embeddings.example/v1"


# --- бюджет агента: локальная модель и облачная ---

def test_agent_budget_depends_on_where_the_model_runs(monkeypatch):
    for name in ("MAX_RESEARCH_CALLS", "MAX_OBSERVATION_CHARS", "ANSWER_HINTS", "READ_WINDOW_CHARS",
                 "LLM_INCEPTION_API_KEY", "INCEPTION_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "lmstudio")
    local = load_settings(env_file=None)
    assert (local.research_calls, local.observation_chars, local.answer_hints) == (6, 6000, False)
    assert local.local_model and local.read_window_chars == 9000

    monkeypatch.setenv("LLM_PROVIDER", "inception")
    cloud = load_settings(env_file=None)
    assert (cloud.research_calls, cloud.observation_chars, cloud.answer_hints) == (12, 14000, True)
    assert not cloud.local_model and cloud.read_window_chars == 60000

    monkeypatch.setenv("MAX_RESEARCH_CALLS", "8")
    monkeypatch.setenv("ANSWER_HINTS", "0")
    tuned = load_settings(env_file=None)
    assert tuned.research_calls == 8 and tuned.answer_hints is False
    monkeypatch.setenv("MAX_OBSERVATION_CHARS", "много")
    with pytest.raises(ValueError, match="MAX_OBSERVATION_CHARS"):
        load_settings(env_file=None)
