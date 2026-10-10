"""Переранжирование и перевод запроса чат-моделью.

LLMReranker — listwise-подход (RankGPT, Sun et al., 2023): модель видит запрос и пронумерованных кандидатов
и возвращает их порядок. Один вызов модели на запрос, без отдельной модели-реранкера.
QueryTranslator — перевод неанглийского запроса для BM25 по английским статьям (эксперимент E3).
"""

import re

from research_assistant.nlp.entities import parse_json
from research_assistant.nlp.text import detect_language

RERANK_SYSTEM = (
    "You rank passages from scientific papers by how well they answer a search query. "
    "The query and the passages may be in different languages. Return JSON with the passage numbers, "
    "most relevant first."
)
RERANK_SCHEMA = {
    "type": "object",
    "properties": {"ranking": {"type": "array", "items": {"type": "integer"}}},
    "required": ["ranking"],
    "additionalProperties": False,
}
TRANSLATE_SYSTEM = (
    "Translate the user's question into a short English search query for scientific papers. "
    "Keep technical terms and model names. Output only the query, nothing else."
)
PASSAGE_CHARS = 500


def _complete(model, messages: list[dict], max_tokens: int, schema: dict | None = None):
    kwargs = {"model": model.model, "temperature": 0, "max_tokens": max_tokens, "messages": messages}
    if schema:
        kwargs["response_format"] = {"type": "json_schema", "json_schema": {"name": "result", "strict": True, "schema": schema}}
    if model.reasoning_effort:
        kwargs["extra_body"] = {"reasoning_effort": model.reasoning_effort}
    return model.client.chat.completions.create(**kwargs)


class LLMReranker:
    def __init__(self, model, passage_chars: int = PASSAGE_CHARS):
        self.model = model
        self.passage_chars = passage_chars
        self.tokens = 0

    def rerank(self, query: str, candidates: list[tuple[int, str]]) -> list[int]:
        """candidates: (id, текст) в исходном порядке. Возвращает id в новом порядке; кого модель не назвала —
        в конце в исходном порядке, так что сбой модели не теряет кандидатов."""
        if len(candidates) < 2:
            return [i for i, _ in candidates]
        listing = "\n".join(
            f"[{n}] {' '.join(text.split())[:self.passage_chars]}" for n, (_, text) in enumerate(candidates, 1)
        )
        messages = [
            {"role": "system", "content": RERANK_SYSTEM},
            {"role": "user", "content": f"Query: {query}\n\nPassages:\n{listing}\n\n"
                                        f"Return {{\"ranking\": [...]}} with all {len(candidates)} numbers."},
        ]
        try:
            response = _complete(self.model, messages, max_tokens=300, schema=RERANK_SCHEMA)
            if response.usage:
                self.tokens += response.usage.prompt_tokens + response.usage.completion_tokens
            ranking = parse_json(response.choices[0].message.content or "").get("ranking", [])
        except Exception:
            return [i for i, _ in candidates]  # реранкер — улучшение, а не обязательный шаг
        order, seen = [], set()
        for number in ranking:
            if isinstance(number, int) and 1 <= number <= len(candidates) and number not in seen:
                seen.add(number)
                order.append(candidates[number - 1][0])
        order += [i for n, (i, _) in enumerate(candidates, 1) if n not in seen]
        return order


class QueryTranslator:
    def __init__(self, model):
        self.model = model
        self.cache: dict[str, str] = {}
        self.tokens = 0

    def to_english(self, query: str) -> str | None:
        """Перевод для BM25 по английским статьям; для английского запроса — None (переводить нечего)."""
        if detect_language(query) == "en":
            return None
        if query not in self.cache:
            try:
                response = _complete(self.model, [{"role": "system", "content": TRANSLATE_SYSTEM},
                                                  {"role": "user", "content": query}], max_tokens=100)
                if response.usage:
                    self.tokens += response.usage.prompt_tokens + response.usage.completion_tokens
                text = re.sub(r"<think>.*?</think>", "", response.choices[0].message.content or "", flags=re.S)
                lines = [line.strip().strip('"') for line in text.strip().splitlines() if line.strip()]
                self.cache[query] = lines[0] if lines else ""
            except Exception:
                self.cache[query] = ""
        return self.cache[query] or None
