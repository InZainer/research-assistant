"""Эксперимент: перевод неанглийского вопроса чат-моделью перед BM25 (предварительный замер к E3).

Запуск: uv run python eval/translate_queries.py [модель]
Модель и сервис берутся из .env (OPENAI_BASE_URL, OPENAI_API_KEY, MODEL), модель можно передать аргументом.
Сравниваются три варианта запроса для BM25: исходный, переведённый и оба вместе (слияние RRF).
"""

import re
import sys
import time

from openai import OpenAI

from research_assistant.config import load_settings
from research_assistant.evaluation.retrieval import load_questions, mrr_at_k, question_group, recall_at_k, resolve
from research_assistant.nlp.text import detect_language
from research_assistant.retrieval.search import rrf
from research_assistant.store.db import Library

SYSTEM = (
    "Translate the user's question into a short English search query for scientific papers. "
    "Keep technical terms and model names. Output only the query, nothing else."
)
GROUPS = ("en→en", "ru→en", "ru→ru")


def main() -> None:
    settings = load_settings()
    model = sys.argv[1] if len(sys.argv) > 1 else settings.model
    client = OpenAI(base_url=settings.base_url, api_key=settings.api_key)

    def translate(question: str) -> tuple[str, int]:
        # reasoning_effort=none отключает «размышления» у Qwen3.5 в LM Studio: иначе весь лимит
        # токенов уходит на рассуждение и ответ приходит пустым.
        response = client.chat.completions.create(
            model=model, temperature=0, max_tokens=200, extra_body={"reasoning_effort": "none"},
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
        )
        text = re.sub(r"<think>.*?</think>", "", response.choices[0].message.content or "", flags=re.S).strip()
        first_line = text.strip('"').splitlines()[0] if text else ""
        return first_line, response.usage.total_tokens if response.usage else 0

    library = Library(settings.db_path)
    rows: dict[str, list[tuple[str, float, float]]] = {"original": [], "translated": [], "both": []}
    tokens, started = 0, time.perf_counter()
    for q in load_questions("eval/retrieval_gold.jsonl"):
        relevant = resolve(library, q)
        original = [i for i, _ in library.search_fts(q.question, 50)]
        translated = original
        if detect_language(q.question) != "en":
            english, used = translate(q.question)
            tokens += used
            translated = [i for i, _ in library.search_fts(english, 50)] if english else []
            print(f"{q.id}: {english}")
        both = [i for i, _, _ in rrf({"original": original, "translated": translated})]
        for name, ranked in (("original", original), ("translated", translated), ("both", both)):
            rows[name].append((question_group(q), recall_at_k(ranked, relevant, 10), mrr_at_k(ranked, relevant, 10)))

    print(f"\nмодель {model}, токенов {tokens}, время {time.perf_counter() - started:.1f} с")
    print(f"{'запрос':<11} {'группа':<7} {'R@10':>5} {'MRR':>5}")
    for name, data in rows.items():
        for group in GROUPS:
            part = [r for r in data if r[0] == group]
            if part:
                recall = sum(r[1] for r in part) / len(part)
                mrr = sum(r[2] for r in part) / len(part)
                print(f"{name:<11} {group:<7} {recall:>5.2f} {mrr:>5.2f}")


if __name__ == "__main__":
    main()
