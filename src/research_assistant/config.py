"""Настройки проекта из переменных окружения и файла .env."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from research_assistant.providers import LMSTUDIO_URL, resolve


@dataclass(frozen=True)
class Settings:
    base_url: str | None
    api_key: str | None
    model: str | None
    embed_model: str
    data_dir: Path
    runs_dir: Path
    max_steps: int
    contact_email: str | None = None
    openalex_api_key: str | None = None
    embed_base_url: str | None = None
    embed_api_key: str | None = None
    reasoning_effort: str | None = None
    provider: str | None = None  # имя профиля из providers.PROVIDERS; None — прежняя схема OPENAI_*
    key_var: str = "OPENAI_API_KEY"  # откуда брался ключ — для сообщений «не задана переменная»
    model_var: str = "MODEL"
    # Бюджет агента. У локальной 4B-модели (контекст 16K) он маленький: иначе она не останавливается (E5).
    # У облачной модели с большим контекстом — шире, чтобы хватало на вопросы из нескольких частей.
    research_calls: int = 6  # поисков и чтений до принудительного final_answer
    observation_chars: int = 6000  # до скольких символов обрезается одно наблюдение
    answer_hints: bool = False  # подсказки о языке и полноте ответа (agent/prompts.py), не входят в prompt_hash
    read_window_chars: int = 9000  # сколько текста статьи за один вызов читает команда read

    @property
    def local_model(self) -> bool:
        return any(host in (self.base_url or "") for host in ("localhost", "127.0.0.1"))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "library.sqlite"

    @property
    def http_cache_dir(self) -> Path:
        return self.data_dir / "cache" / "http"

    @property
    def fulltext_dir(self) -> Path:
        return self.data_dir / "fulltext"

    @property
    def vectors_dir(self) -> Path:
        return self.data_dir / "vectors"

    @property
    def user_agent(self) -> str:
        contact = f"mailto:{self.contact_email}" if self.contact_email else "student project"
        return f"research-assistant/0.1 ({contact})"

    def missing_llm_vars(self) -> list[str]:
        """Имена переменных, без которых нельзя вызвать модель."""
        missing = []
        if not self.api_key:
            missing.append(self.key_var)
        if not self.model:
            missing.append(self.model_var)
        return missing


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} должно быть целым числом, получено {raw!r}") from None
    if value < 1:
        raise ValueError(f"{name} должно быть не меньше 1, получено {value}")
    return value


def load_settings(env_file: str | Path | None = ".env", provider: str | None = None) -> Settings:
    """Читает настройки. Переменные окружения важнее значений из .env.

    provider (или LLM_PROVIDER) выбирает профиль стороннего AI-сервиса — см. providers.py."""
    if env_file is not None:
        load_dotenv(env_file, override=False)
    chat = resolve(provider)
    if chat.provider is None:
        # Прежняя схема: эмбеддинги по умолчанию у того же сервиса, что и чат-модель.
        embed_url, embed_key = os.getenv("OPENAI_BASE_URL"), os.getenv("OPENAI_API_KEY")
    elif chat.provider.local:
        embed_url, embed_key = chat.base_url, chat.api_key
    else:
        # Облачный провайдер bge-m3 не отдаёт, а индекс библиотеки посчитан ею: эмбеддинги — в LM Studio.
        embed_url, embed_key = LMSTUDIO_URL, "lm-studio"

    raw_steps = os.getenv("MAX_STEPS", "12")
    try:
        max_steps = int(raw_steps)
    except ValueError:
        raise ValueError(f"MAX_STEPS должно быть целым числом, получено {raw_steps!r}") from None
    if max_steps < 1:
        raise ValueError(f"MAX_STEPS должно быть не меньше 1, получено {max_steps}")

    local = chat.provider.local if chat.provider else any(
        host in (chat.base_url or "") for host in ("localhost", "127.0.0.1"))
    hints = os.getenv("ANSWER_HINTS", "").strip().lower()

    return Settings(
        research_calls=_positive_int("MAX_RESEARCH_CALLS", 6 if local else 12),
        observation_chars=_positive_int("MAX_OBSERVATION_CHARS", 6000 if local else 14000),
        answer_hints=(hints in ("1", "true", "yes", "on")) if hints else not local,
        read_window_chars=_positive_int("READ_WINDOW_CHARS", 9000 if local else 60000),
        base_url=chat.base_url,
        api_key=chat.api_key,
        model=chat.model,
        embed_model=os.getenv("EMBED_MODEL", ""),
        data_dir=Path(os.getenv("DATA_DIR", "data")),
        runs_dir=Path(os.getenv("RUNS_DIR", "runs")),
        max_steps=max_steps,
        contact_email=os.getenv("CONTACT_EMAIL") or None,
        openalex_api_key=os.getenv("OPENALEX_API_KEY") or None,
        embed_base_url=os.getenv("EMBED_BASE_URL") or embed_url or None,
        embed_api_key=os.getenv("EMBED_API_KEY") or embed_key or None,
        reasoning_effort=chat.reasoning_effort,
        provider=chat.provider.name if chat.provider else None,
        key_var=chat.key_var,
        model_var=chat.model_var,
    )
