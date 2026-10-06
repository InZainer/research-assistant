"""Настройки проекта из переменных окружения и файла .env."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


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
            missing.append("OPENAI_API_KEY")
        if not self.model:
            missing.append("MODEL")
        return missing


def load_settings(env_file: str | Path | None = ".env") -> Settings:
    """Читает настройки. Переменные окружения важнее значений из .env."""
    if env_file is not None:
        load_dotenv(env_file, override=False)

    raw_steps = os.getenv("MAX_STEPS", "12")
    try:
        max_steps = int(raw_steps)
    except ValueError:
        raise ValueError(f"MAX_STEPS должно быть целым числом, получено {raw_steps!r}") from None
    if max_steps < 1:
        raise ValueError(f"MAX_STEPS должно быть не меньше 1, получено {max_steps}")

    return Settings(
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        api_key=os.getenv("OPENAI_API_KEY") or None,
        model=os.getenv("MODEL") or None,
        embed_model=os.getenv("EMBED_MODEL", ""),
        data_dir=Path(os.getenv("DATA_DIR", "data")),
        runs_dir=Path(os.getenv("RUNS_DIR", "runs")),
        max_steps=max_steps,
        contact_email=os.getenv("CONTACT_EMAIL") or None,
        openalex_api_key=os.getenv("OPENALEX_API_KEY") or None,
        # Эмбеддинги можно брать у другого сервиса, чем чат-модель; по умолчанию — у того же.
        embed_base_url=os.getenv("EMBED_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None,
        embed_api_key=os.getenv("EMBED_API_KEY") or os.getenv("OPENAI_API_KEY") or None,
    )
