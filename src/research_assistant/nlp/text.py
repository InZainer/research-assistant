"""Анализатор для полнотекстового поиска: токены → стоп-слова → основы (Snowball, английский и русский).

Встроенный токенизатор FTS5 не знает морфологии: «обучения» не нашлось бы по запросу «обучение».
Поэтому в индекс кладётся уже нормализованный текст, а запрос проходит тот же анализатор.
Язык определяется по каждому слову (кириллица или нет), так что смешанный текст тоже обрабатывается.
"""

import re
from functools import lru_cache

import snowballstemmer

TOKEN_RE = re.compile(r"[^\W_]+")
CYRILLIC_RE = re.compile(r"[а-яё]")

STOPWORDS = set(
    """
    a an the and or but if then else of in on at to for from by with without into onto over under about
    as is are was were be been being am do does did done have has had having it its this that these those
    there here we our ours us you your he she they them their i me my not no nor so than too very can could
    may might must shall should will would also such each which who whom whose what when where why how all
    any both few more most other some same only own just via per vs etc e g i et al
    и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее её мне было
    вот от меня еще ещё нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас
    нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем
    была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой совсем
    ним здесь этом один почти мой тем чтобы нее неё сейчас были куда зачем всех никогда можно при наконец
    два об другой хоть после над больше тот через эти нас про всего них какая много разве три эту моя
    впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между
    это также является которые который которая которое которых
    """.split()
)


@lru_cache(maxsize=1)
def _stemmers():
    return snowballstemmer.stemmer("english"), snowballstemmer.stemmer("russian")


@lru_cache(maxsize=200_000)
def stem(token: str) -> str:
    english, russian = _stemmers()
    if token.isdigit():
        return token
    return (russian if CYRILLIC_RE.search(token) else english).stemWord(token)


def analyze(text: str) -> list[str]:
    """Текст → основы слов без стоп-слов. Регистр и «ё» нормализуются."""
    tokens = TOKEN_RE.findall(text.casefold().replace("ё", "е"))
    return [stem(t) for t in tokens if t not in STOPWORDS]


def detect_language(text: str) -> str:
    """Грубое определение языка по доле кириллицы среди букв: ru или en."""
    letters = [c for c in text.casefold() if c.isalpha()]
    if not letters:
        return "en"
    cyrillic = sum(1 for c in letters if "а" <= c <= "я" or c == "ё")
    return "ru" if cyrillic / len(letters) > 0.3 else "en"
