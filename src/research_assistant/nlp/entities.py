"""Извлечение научных сущностей (метод, датасет, метрика, задача) чат-моделью со структурированным выводом.

Как и с цитатами, модели не верим на слово: сущность остаётся, только если её название дословно
(без учёта регистра и пробелов) встречается в исходном тексте. Выдуманные моделью названия отбрасываются.
"""

import json
import re
from dataclasses import dataclass, field

from research_assistant.nlp.text import analyze

ENTITY_TYPES = ("method", "dataset", "metric", "task")
MAX_ENTITY_WORDS = 8
ABBREVIATION_RE = re.compile(r"^(.+?)\s*\(([^()]{2,15})\)$")

SYSTEM_PROMPT = """You extract named scientific entities from a paper's text (English or Russian).
Return JSON with four lists:
- method: methods, algorithms, architectures, named models and technologies (e.g. "speculative decoding", "EAGLE-2", "kNN", "машинное обучение"). Do not list the umbrella term "artificial intelligence" / "искусственный интеллект".
- dataset: named datasets and benchmarks (e.g. "MT-Bench", "GSM8K", "WMT 2014 De-En").
- metric: evaluation measures (e.g. "speedup", "acceptance rate", "BLEU", "latency", "log-loss").
- task: tasks and applications (e.g. "machine translation", "summarization", "диагностика заболеваний").
Rules: copy each name exactly as it is written in the text, in the text's language; one entry per entity; no explanations; empty list if none."""

SCHEMA = {
    "type": "object",
    "properties": {t: {"type": "array", "items": {"type": "string"}} for t in ENTITY_TYPES},
    "required": list(ENTITY_TYPES),
    "additionalProperties": False,
}
GENERIC = {"model", "models", "method", "methods", "approach", "dataset", "datasets", "metric", "task", "tasks",
           "модель", "метод", "методы", "подход", "задача", "данные"}


def entity_key(name: str) -> str:
    """Ключ сравнения: основы слов без стоп-слов. «Диагностики» и «диагностика» совпадают."""
    return " ".join(analyze(name))


def contains_stems(haystack: list[str], needle: list[str]) -> bool:
    """Идёт ли needle подряд внутри haystack (списки основ слов)."""
    if not needle or len(needle) > len(haystack):
        return False
    first, size = needle[0], len(needle)
    return any(haystack[i:i + size] == needle for i, stem in enumerate(haystack) if stem == first)


def occurs_in(name: str, text: str | list[str]) -> bool:
    """Встречается ли название в тексте с точностью до словоформы: сравниваются основы слов подряд.

    Русский текст склоняется: модель вернёт «диагностика заболеваний», а в тексте «в диагностике заболеваний».
    Поэтому сравнение строк отбросило бы верную сущность. text можно передать уже разобранным analyze().
    """
    stems = analyze(text) if isinstance(text, str) else text
    return contains_stems(stems, analyze(name))


@dataclass
class Extraction:
    entities: dict[str, list[str]] = field(default_factory=lambda: {t: [] for t in ENTITY_TYPES})
    dropped: list[tuple[str, str]] = field(default_factory=list)  # (тип, название) — нет в тексте или мусор
    raw: str = ""
    tokens: int = 0


def split_abbreviation(name: str) -> list[str]:
    """«Approximate Speculative Decoding (ASD)» → полное имя и сокращение по отдельности."""
    match = ABBREVIATION_RE.match(name)
    return [match.group(1).strip(), match.group(2).strip()] if match else [name]


def clean(raw: dict, text: str, grounded: bool = True) -> Extraction:
    """Нормализует ответ модели: «имя (сокращение)» → два имени, дедупликация по ключу, отсев общих слов,
    чисел («30%», «5x») и (если grounded) названий, которых нет в тексте."""
    result = Extraction()
    text_stems = analyze(text)
    for entity_type in ENTITY_TYPES:
        seen = set()
        names = [part for name in raw.get(entity_type) or [] if isinstance(name, str) for part in split_abbreviation(name)]
        for name in names:
            name = " ".join(name.split()).strip(" .,;:\"'«»")
            key = entity_key(name)
            letters = sum(c.isalpha() for c in name)
            if not key or letters < 2 or name.casefold() in GENERIC or len(name.split()) > MAX_ENTITY_WORDS:
                result.dropped.append((entity_type, name))
                continue
            if grounded and not occurs_in(name, text_stems):
                result.dropped.append((entity_type, name))
                continue
            if key in seen:
                continue
            seen.add(key)
            result.entities[entity_type].append(name)
    return result


def parse_json(content: str) -> dict:
    """JSON из ответа; на случай сервиса без structured output — достаёт первый объект из текста."""
    content = re.sub(r"<think>.*?</think>", "", content or "", flags=re.S).strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.S)
        if not match:
            raise ValueError(f"в ответе нет JSON: {content[:200]!r}") from None
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("ответ — не JSON-объект")
    return data


def extract_entities(model, text: str, grounded: bool = True, max_tokens: int = 600) -> Extraction:
    """model — ChatModel; запрос со схемой JSON (structured output)."""
    kwargs = {
        "model": model.model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "entities", "strict": True, "schema": SCHEMA}},
    }
    if model.reasoning_effort:
        kwargs["extra_body"] = {"reasoning_effort": model.reasoning_effort}
    response = model.client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content or ""
    result = clean(parse_json(content), text, grounded=grounded)
    result.raw = content
    result.tokens = (response.usage.prompt_tokens + response.usage.completion_tokens) if response.usage else 0
    return result
