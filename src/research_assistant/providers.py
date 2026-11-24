"""Профили сторонних AI-сервисов с OpenAI-совместимым API (chat completions с вызовом инструментов).

Провайдер выбирается переменной LLM_PROVIDER (или флагом --provider). Ключ — под стандартным именем
провайдера (ANTHROPIC_API_KEY, OPENROUTER_API_KEY…) или LLM_<ИМЯ>_API_KEY; модель, адрес и reasoning_effort —
LLM_<ИМЯ>_MODEL, LLM_<ИМЯ>_BASE_URL, LLM_<ИМЯ>_REASONING_EFFORT. Префикс LLM_ нужен, чтобы не пересекаться с
переменными самих SDK и инструментов (OPENAI_BASE_URL, ANTHROPIC_BASE_URL, ANTHROPIC_MODEL у Claude Code).
В .env можно держать сразу несколько провайдеров и переключаться одной строкой. Без LLM_PROVIDER работает
прежняя схема: OPENAI_BASE_URL, OPENAI_API_KEY, MODEL.

Модель по умолчанию указана только там, где её id проверен; иначе <ИМЯ>_MODEL обязателен.
"""

import os
from dataclasses import dataclass

LMSTUDIO_URL = "http://localhost:1234/v1"


@dataclass(frozen=True)
class Provider:
    name: str
    title: str
    base_url: str
    key_env: str | None  # переменная с ключом; None — локальный сервер без ключа
    model: str | None = None
    reasoning_effort: str | None = None
    local_key: str | None = None  # что подставить вместо ключа у локального сервера
    note: str = ""

    @property
    def env_prefix(self) -> str:
        return f"LLM_{self.name.upper()}"

    @property
    def local(self) -> bool:
        return self.key_env is None


PROVIDERS: dict[str, Provider] = {p.name: p for p in [
    Provider("lmstudio", "LM Studio (локально)", LMSTUDIO_URL, None, "qwen3.5-4b", "none", "lm-studio",
             "lms server start; модель загрузить с контекстом 16K: lms load qwen3.5-4b --context-length 16384"),
    Provider("ollama", "Ollama (локально)", "http://localhost:11434/v1", None, None, None, "ollama",
             "ollama serve; модель должна поддерживать вызов инструментов (tools)"),
    Provider("inception", "Inception Labs (Mercury)", "https://api.inceptionlabs.ai/v1", "INCEPTION_API_KEY",
             "mercury-3", None, None, "в настройках платформы выключить «Improve the model for everyone»"),
    Provider("openai", "OpenAI", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    Provider("anthropic", "Anthropic (Claude)", "https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY",
             "claude-sonnet-5-5", None, None, "OpenAI-совместимый эндпоинт Anthropic"),
    Provider("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
             note="id модели вида vendor/model, с поддержкой tools"),
    Provider("deepseek", "DeepSeek", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", "deepseek-chat"),
    Provider("gemini", "Google Gemini", "https://generativelanguage.googleapis.com/v1beta/openai/",
             "GEMINI_API_KEY", note="OpenAI-совместимый эндпоинт Gemini API"),
    Provider("custom", "Любой OpenAI-совместимый сервис", "", "LLM_CUSTOM_API_KEY",
             note="задать LLM_CUSTOM_BASE_URL, LLM_CUSTOM_API_KEY и LLM_CUSTOM_MODEL"),
]}


@dataclass(frozen=True)
class Resolved:
    """Итог выбора провайдера: что передать клиенту OpenAI и как назвать недостающие переменные."""

    provider: Provider | None
    base_url: str | None
    api_key: str | None
    model: str | None
    reasoning_effort: str | None
    key_var: str  # имя переменной ключа — для сообщений об ошибке (значение никогда не печатается)
    model_var: str


def get_provider(name: str) -> Provider:
    key = name.strip().lower()
    if key not in PROVIDERS:
        raise ValueError(f"неизвестный провайдер {name!r}; доступны: {', '.join(PROVIDERS)}")
    return PROVIDERS[key]


def _env(name: str) -> str | None:
    return os.getenv(name) or None


def resolve(name: str | None = None) -> Resolved:
    """Провайдер из аргумента или LLM_PROVIDER; без него — прежние OPENAI_BASE_URL / OPENAI_API_KEY / MODEL."""
    name = name or _env("LLM_PROVIDER")
    if not name:
        return Resolved(None, _env("OPENAI_BASE_URL"), _env("OPENAI_API_KEY"), _env("MODEL"),
                        _env("REASONING_EFFORT"), "OPENAI_API_KEY", "MODEL")
    provider = get_provider(name)
    prefix = provider.env_prefix
    own_key = f"{prefix}_API_KEY"
    key_var = own_key if _env(own_key) or not provider.key_env else provider.key_env
    effort = os.getenv(f"{prefix}_REASONING_EFFORT")
    return Resolved(
        provider=provider,
        base_url=_env(f"{prefix}_BASE_URL") or provider.base_url or None,
        api_key=_env(key_var) or provider.local_key,
        model=_env(f"{prefix}_MODEL") or provider.model,
        # Пустая строка в .env — «не передавать»; переменная не задана — значение профиля.
        reasoning_effort=(effort or None) if effort is not None else provider.reasoning_effort,
        key_var=key_var,
        model_var=f"{prefix}_MODEL",
    )


def describe_all() -> list[dict]:
    """Профили для команды providers: адрес, модель, есть ли ключ (только да/нет, без значения)."""
    active = (_env("LLM_PROVIDER") or "").lower()
    rows = []
    for provider in PROVIDERS.values():
        resolved = resolve(provider.name)
        rows.append({
            "name": provider.name,
            "title": provider.title,
            "active": provider.name == active,
            "base_url": resolved.base_url or f"задайте {provider.env_prefix}_BASE_URL",
            "model": resolved.model or f"задайте {resolved.model_var}",
            "key": "не нужен" if provider.local else ("есть" if _env(resolved.key_var) else f"нет ({resolved.key_var})"),
            "note": provider.note,
        })
    return rows
