"""Оценка поиска на золотом наборе вопросов: Recall@k, MRR@k, nDCG@k по режимам и языкам вопросов.

Золотой ответ задаётся не id фрагмента (id меняются при переразборе), а парой «название статьи +
дословный кусок текста». Релевантны все фрагменты этой статьи, где кусок встречается.
"""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from research_assistant.ingest.models import normalize_title
from research_assistant.nlp.text import detect_language
from research_assistant.store.db import Library


@dataclass
class GoldItem:
    paper: str
    quote: str


@dataclass
class GoldQuestion:
    id: str
    question: str
    language: str
    relevant: list[GoldItem]
    topic: str | None = None
    status: str = "draft"  # draft — набросок, verified — проверен человеком


def load_questions(path: str | Path) -> list[GoldQuestion]:
    questions = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        raw["relevant"] = [GoldItem(**item) for item in raw["relevant"]]
        questions.append(GoldQuestion(**raw))
    return questions


def _squash(text: str) -> str:
    return " ".join(text.split())


def resolve(library: Library, question: GoldQuestion) -> set[int]:
    """id фрагментов, где встречаются золотые цитаты."""
    papers = {normalize_title(r["title"]): r["id"] for r in library.conn.execute("SELECT id, title FROM papers")}
    found = set()
    for item in question.relevant:
        paper_id = papers.get(normalize_title(item.paper))
        if paper_id is None:
            continue
        quote = _squash(item.quote)
        for row in library.conn.execute("SELECT id, text FROM passages WHERE paper_id = ?", (paper_id,)):
            if quote in _squash(row["text"]):
                found.add(row["id"])
    return found


def recall_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def mrr_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    return next((1.0 / place for place, i in enumerate(ranked[:k], 1) if i in relevant), 0.0)


def ndcg_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    dcg = sum(1.0 / math.log2(place + 1) for place, i in enumerate(ranked[:k], 1) if i in relevant)
    ideal = sum(1.0 / math.log2(place + 1) for place in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def question_group(question: GoldQuestion) -> str:
    """«язык вопроса→язык статьи»: en→en, ru→en (межъязыковой поиск), ru→ru."""
    return f"{question.language}→{detect_language(question.relevant[0].quote)}"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate(searcher, questions: list[GoldQuestion], modes: list[str], k: int = 10) -> dict:
    """Прогоняет вопросы по каждому режиму. Метрики — по всем вопросам и по группам «язык вопроса→язык статьи»."""
    resolved = {q.id: resolve(searcher.library, q) for q in questions}
    usable = [q for q in questions if resolved[q.id]]
    report = {
        "k": k,
        "questions": len(questions),
        "unresolved": [q.id for q in questions if not resolved[q.id]],
        "modes": {},
    }
    for mode in modes:
        rows = []
        for q in usable:
            started = time.perf_counter()
            ranked = [hit.passage_id for hit in searcher.search(q.question, k=k, mode=mode)]
            elapsed = time.perf_counter() - started
            relevant = resolved[q.id]
            rows.append({
                "id": q.id, "group": question_group(q),
                "recall@5": recall_at_k(ranked, relevant, 5), f"recall@{k}": recall_at_k(ranked, relevant, k),
                f"mrr@{k}": mrr_at_k(ranked, relevant, k), f"ndcg@{k}": ndcg_at_k(ranked, relevant, k),
                "seconds": elapsed, "ranked": ranked, "relevant": sorted(relevant),
            })
        names = sorted({r["group"] for r in rows})
        groups = {"all": rows, **{name: [r for r in rows if r["group"] == name] for name in names}}
        metrics = ("recall@5", f"recall@{k}", f"mrr@{k}", f"ndcg@{k}", "seconds")
        report["modes"][mode] = {
            "summary": {
                name: {"n": len(group), **{m: _mean([r[m] for r in group]) for m in metrics}}
                for name, group in groups.items() if group
            },
            "per_question": rows,
        }
    return report


def format_report(report: dict) -> str:
    k = report["k"]
    lines = [f"Вопросов: {report['questions']}, без найденных золотых фрагментов: {len(report['unresolved'])}"
             + (f" ({', '.join(report['unresolved'])})" if report["unresolved"] else "")]
    header = f"{'режим':<8} {'вопросы':<9} {'n':>3} {'R@5':>6} {f'R@{k}':>6} {f'MRR@{k}':>7} {f'nDCG@{k}':>8} {'мс':>6}"
    lines += [header, "-" * len(header)]
    for mode, data in report["modes"].items():
        for group, s in data["summary"].items():
            lines.append(
                f"{mode:<8} {group:<9} {s['n']:>3} {s['recall@5']:>6.2f} {s[f'recall@{k}']:>6.2f} "
                f"{s[f'mrr@{k}']:>7.2f} {s[f'ndcg@{k}']:>8.2f} {s['seconds'] * 1000:>6.0f}"
            )
    return "\n".join(lines)
