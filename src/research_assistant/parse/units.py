"""Общее промежуточное представление документа: строки с признаками оформления.

PDF и HTML приводятся к списку `Unit`, а дальше их обрабатывает одна и та же логика из structure.py.
"""

import re
from dataclasses import dataclass

# Мягкий перенос, пробелы нулевой ширины, BOM.
INVISIBLE_CHARS = "".join(map(chr, (0x00AD, 0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF)))
INVISIBLE_RE = re.compile(f"[{INVISIBLE_CHARS}]")
WORD_RE = re.compile(r"[^\W\d_]+")


@dataclass
class Unit:
    """Строка PDF или абзац HTML.

    block — номер исходного блока: строки одного блока по умолчанию принадлежат одному абзацу.
    """

    text: str
    page: int | None
    block: int
    bold: bool = False
    size: float = 0.0
    y: float | None = None  # верх строки в долях высоты страницы (для сносок); у HTML нет


def clean_text(text: str) -> str:
    """Убирает невидимые символы и схлопывает пробелы. Ничего не переписывает по смыслу."""
    return " ".join(INVISIBLE_RE.sub("", text).replace("\t", " ").split())


def header_key(text: str) -> str | None:
    """Ключ колонтитула: буквенные слова без чисел, по алфавиту.

    «European science № 4 (53) ■ 18» и «19 ■ European science № 4 (53)» дают один ключ.
    Меньше двух слов — не ключ, чтобы не принять за колонтитул «Table 1» и «Table 2».
    """
    words = sorted(w.casefold() for w in WORD_RE.findall(text))
    return " ".join(words) if len(words) >= 2 else None
