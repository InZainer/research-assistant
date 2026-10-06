"""Разбивка на предложения с позициями символов.

Основа — razdel: он хорошо справляется с русским (т.е., и т.д., инициалы, «г.»). Но научные сокращения
он разрывает: «Fig. | 3», «Sec. | 4.2», «табл. | 3». Поверх razdel склеиваем такие разрывы.
"""

import re

from razdel import sentenize

# После этих сокращений предложение не заканчивается никогда (на практике).
STRONG_ABBREVIATIONS = {
    "fig", "figs", "tab", "tbl", "eq", "eqs", "sec", "secs", "cf", "prof", "dr", "mr", "mrs", "ms",
    "vs", "approx", "resp", "app", "ref", "refs", "al", "e.g", "i.e", "algo", "thm", "def",
    "табл", "рис", "гл", "разд", "ур", "прил", "проф", "акад", "им", "напр", "ср", "см", "т.е", "т.к", "т.н",
}
# После этих — может, поэтому склеиваем, только если дальше цифра, строчная буква или скобка.
WEAK_ABBREVIATIONS = {"no", "nos", "vol", "pp", "p", "ch", "n", "с", "стр", "т", "г", "гг", "др", "в", "вв"}

MAX_BRACKET_GLUE_WORDS = 120
# Первое «слово» склеенного куска заканчивает предложение: «3).», «3.», «(2023).» перед заглавной буквой.
NUMBER_THEN_SENTENCE_RE = re.compile(r"\S*?[.!?][\"'»”’)\]]*(?=\s+[A-ZА-ЯЁ])")
LAST_TOKEN_RE = re.compile(r"([^\W\d_]+(?:\.[^\W\d_]+)*)\.$")


def _last_abbreviation(text: str) -> str | None:
    match = LAST_TOKEN_RE.search(text.rstrip())
    return match.group(1).casefold() if match else None


def _glued(left: str, right: str) -> bool:
    # razdel режет по «;» перед заглавной буквой, в том числе внутри ссылок «[Иванов, 2017; Хайкин, 2019]».
    if left.rstrip().endswith(";"):
        return True
    unclosed = left.count("(") > left.count(")") or left.count("[") > left.count("]")
    if unclosed and len(left.split()) < MAX_BRACKET_GLUE_WORDS:
        return True  # скобка ещё не закрыта; лимит — на случай мусорной скобки, которая не закроется никогда
    token = _last_abbreviation(left)
    if token is None:
        return False
    if token in STRONG_ABBREVIATIONS:
        return True
    first = right.lstrip()[:1]
    return token in WEAK_ABBREVIATIONS and (first.isdigit() or first.islower() or first in "([")


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Возвращает (начало, конец) каждого предложения; text[начало:конец] — само предложение."""
    spans: list[tuple[int, int]] = []
    for piece in sentenize(text):
        if not (spans and _glued(text[spans[-1][0]:spans[-1][1]], piece.text)):
            spans.append((piece.start, piece.stop))
            continue
        # razdel не режет «Fig. | 3). Results» и «табл. | 3. Итог»: после склейки граница может
        # стоять сразу за номером — её восстанавливаем.
        boundary = NUMBER_THEN_SENTENCE_RE.match(piece.text)
        if boundary:
            spans[-1] = (spans[-1][0], piece.start + boundary.end())
            rest = piece.text[boundary.end():].lstrip()
            spans.append((piece.stop - len(rest), piece.stop))
        else:
            spans[-1] = (spans[-1][0], piece.stop)
    return spans
