"""Ключевые слова статьи — YAKE: без модели и обучения, работает и для русского."""

import yake

from research_assistant.nlp.text import STOPWORDS, analyze, detect_language

MAX_TEXT_WORDS = 4000


def _is_name(word: str) -> bool:
    """Аббревиатура или название: HSDDW, EAGLE-2, SpecVocab, kNN — заглавные не только в начале или цифры."""
    return any(c.isupper() for c in word[1:]) or any(c.isdigit() for c in word)


def extract_keywords(text: str, top: int = 10, language: str | None = None) -> list[tuple[str, float]]:
    """(фраза, оценка); у YAKE меньше — значимее.

    Одиночные общие слова («model», «tokens») отбрасываются: одно слово остаётся, только если это название
    или аббревиатура. Повторы по основам слов («draft model» / «draft models») схлопываются."""
    words = text.split()
    text = " ".join(words[:MAX_TEXT_WORDS])
    if not text.strip():
        return []
    language = language or detect_language(text)
    extractor = yake.KeywordExtractor(lan=language, n=3, top=top * 3, dedupLim=0.7)
    keywords, seen = [], set()
    for phrase, score in extractor.extract_keywords(text):
        tokens = phrase.casefold().split()
        if any(t in STOPWORDS for t in tokens) or any(len(t) < 2 for t in tokens):
            continue
        if len(tokens) == 1 and not _is_name(phrase):
            continue
        key = " ".join(analyze(phrase))
        if not key or key in seen:
            continue
        seen.add(key)
        keywords.append((phrase, float(score)))
        if len(keywords) == top:
            break
    return keywords
