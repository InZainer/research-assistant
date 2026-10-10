"""Оценка агента на вопросах с золотыми цитатами (eval/retrieval_gold.jsonl).

Два критерия успеха:
- строгий — агент сдал final_answer, не ответил not_found, и хотя бы одна цитата лежит во фрагменте с золотым ответом;
- мягкий — цитата из золотой статьи. Нужен потому, что ответ часто есть в нескольких фрагментах статьи
  (аннотация, результаты, заключение), а золотой набор отмечает один: итеративный агент, который ищет шире,
  находит верный ответ в другом фрагменте и по строгому критерию получает незаслуженный провал.
Для режима quotes (эксперимент E2) ещё считается доля цитат, которые дословно есть в библиотеке.

Повторы (τ-bench): pass@1 — доля успешных ответов по всем прогонам; pass^k — доля вопросов, успешных во всех
k прогонах. Для pass@1 — 95% интервал Вильсона.
"""

import json
import math
import re
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from research_assistant.agent.loop import AgentResult, Claim, run_agent
from research_assistant.agent.tools import FINAL_ANSWER, ToolContext
from research_assistant.answer.citations import load_citations, parse_sentence_id
from research_assistant.llm import Usage
from research_assistant.evaluation.retrieval import GoldQuestion, question_group, resolve
from research_assistant.store.db import Library

SENTENCE_TAG_RE = re.compile(r"\[s\d+\]")

VARIANTS = {
    "agent": {"answer_mode": "ids", "single_shot": False},   # итеративный агент, цитаты по ID
    "single": {"answer_mode": "ids", "single_shot": True},   # E3: один поиск по вопросу и сразу ответ
    "quotes": {"answer_mode": "quotes", "single_shot": False},  # E2: цитаты текстом от модели
    "prefetch": {"answer_mode": "ids", "prefetch": True},  # поиск по вопросу заранее, дальше — как агент
}


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% доверительный интервал Вильсона для доли успехов."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, center - half), min(1.0, center + half)


def passages_with_quote(library: Library, quote: str) -> set[int]:
    """Фрагменты, где цитата встречается дословно (пробелы схлопнуты, остальное — символ в символ).

    Метки ID вида «[s123]» модель часто копирует из выдачи вместе с предложением — их снимаем, это не искажение."""
    quote = " ".join(SENTENCE_TAG_RE.sub(" ", quote).split())
    if not quote:
        return set()
    return {row[0] for row in library.conn.execute("SELECT id FROM passages WHERE instr(text, ?) > 0", (quote,))}


def cited_passages(library: Library, result: AgentResult) -> tuple[set[int], int, int]:
    """(фрагменты, на которые ссылается ответ; сколько цитат; сколько из них дословно найдено в библиотеке)."""
    if result.quotes:
        passages, exact = set(), 0
        for _, quote in result.quotes:
            found = passages_with_quote(library, quote)
            passages |= found
            exact += bool(found)
        return passages, len(result.quotes), exact
    ids = [c.sentence_id for claim in result.claims for c in claim.citations]
    if not ids:
        return set(), 0, 0
    rows = library.conn.execute(
        f"SELECT DISTINCT passage_id FROM sentences WHERE id IN ({','.join('?' * len(ids))})", ids
    ).fetchall()
    return {r[0] for r in rows}, len(ids), len(ids)  # по ID цитата дословна по построению


def papers_of(library: Library, passage_ids: set[int]) -> set[int]:
    if not passage_ids:
        return set()
    ids = list(passage_ids)
    rows = library.conn.execute(f"SELECT DISTINCT paper_id FROM passages WHERE id IN ({','.join('?' * len(ids))})", ids)
    return {r[0] for r in rows}


def score_run(library: Library, relevant: set[int], result: AgentResult) -> dict:
    passages, n_quotes, exact = cited_passages(library, result)
    accepted = result.stop_reason == "final_answer" and not result.not_found
    gold_hit = bool(passages & relevant)
    paper_hit = bool(papers_of(library, passages) & papers_of(library, relevant))
    return {
        "accepted": accepted, "not_found": result.not_found, "stop_reason": result.stop_reason,
        "gold_hit": gold_hit, "success": accepted and gold_hit,
        "paper_hit": paper_hit, "success_paper": accepted and paper_hit,
        "quotes": n_quotes, "exact_quotes": exact,
        "steps": result.steps, "tool_calls": result.tool_calls, "tool_errors": result.tool_errors,
        "tokens": result.usage.total, "seconds": result.seconds,
    }


def _summary(rows: list[dict], runs: int) -> dict:
    n = len(rows)
    successes = sum(r["success"] for r in rows)
    by_question: dict[str, list[bool]] = {}
    for r in rows:
        by_question.setdefault(r["id"], []).append(r["success"])
    quotes = sum(r["quotes"] for r in rows)
    low, high = wilson(successes, n)
    paper_successes = sum(r.get("success_paper", False) for r in rows)
    paper_low, paper_high = wilson(paper_successes, n)
    by_question_paper: dict[str, list[bool]] = {}
    for r in rows:
        by_question_paper.setdefault(r["id"], []).append(r.get("success_paper", False))
    return {
        "n": n,
        "pass@1": successes / n if n else 0.0,
        "pass@1_ci": [low, high],
        f"pass^{runs}": sum(all(v) for v in by_question.values()) / len(by_question) if by_question else 0.0,
        "pass@1_paper": paper_successes / n if n else 0.0,
        "pass@1_paper_ci": [paper_low, paper_high],
        f"pass^{runs}_paper": (sum(all(v) for v in by_question_paper.values()) / len(by_question_paper)
                               if by_question_paper else 0.0),
        "accepted": sum(r["accepted"] for r in rows) / n if n else 0.0,
        "not_found": sum(r["not_found"] for r in rows) / n if n else 0.0,
        "gold_hit": sum(r["gold_hit"] for r in rows) / n if n else 0.0,
        "exact_quote_rate": sum(r["exact_quotes"] for r in rows) / quotes if quotes else None,
        "steps": sum(r["steps"] for r in rows) / n if n else 0.0,
        "tool_errors": sum(r["tool_errors"] for r in rows) / n if n else 0.0,
        "tokens": sum(r["tokens"] for r in rows) / n if n else 0.0,
        "seconds": sum(r["seconds"] for r in rows) / n if n else 0.0,
    }


def evaluate_agent(
    make_context: Callable[[], ToolContext],
    model,
    questions: list[GoldQuestion],
    variant: str = "agent",
    runs: int = 1,
    max_steps: int = 12,
    log_dir: Path | None = None,
    log: Callable[[str], None] = lambda message: None,
) -> dict:
    """make_context — новый ToolContext на каждый прогон (у агента не должно быть памяти о прошлых вопросах)."""
    if variant not in VARIANTS:
        raise ValueError(f"неизвестный вариант {variant}; доступны {', '.join(VARIANTS)}")
    options = VARIANTS[variant]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    rows = []
    started = time.perf_counter()
    for run in range(1, runs + 1):
        for number, q in enumerate(questions, 1):
            ctx = make_context()
            relevant = resolve(ctx.library, q)
            log_path = log_dir / f"eval-agent-{variant}-{stamp}.jsonl" if log_dir else None
            result = run_agent(q.question, ctx, model, max_steps=max_steps, log_path=log_path, **options)
            row = {"id": q.id, "group": question_group(q), "run": run, **score_run(ctx.library, relevant, result)}
            rows.append(row)
            mark = "успех" if row["success"] else ("ответ не по золоту" if row["accepted"] else row["stop_reason"])
            log(f"[прогон {run}/{runs}, {number}/{len(questions)}] {q.id}: {mark}, шагов {row['steps']}, "
                f"{row['seconds']:.0f} с")
    groups = {"all": rows}
    for name in sorted({r["group"] for r in rows}):
        groups[name] = [r for r in rows if r["group"] == name]
    return {
        "variant": variant, "model": model.model, "runs": runs, "questions": len(questions),
        "seconds_total": time.perf_counter() - started,
        "summary": {name: _summary(group, runs) for name, group in groups.items()},
        "rows": rows,
    }


def format_agent_report(report: dict) -> str:
    runs = report["runs"]
    lines = [f"Вариант {report['variant']}, модель {report['model']}, вопросов {report['questions']}, "
             f"прогонов {runs}, всего {report['seconds_total'] / 60:.1f} мин"]
    lines.append("pass@1 — цитата из золотого фрагмента; «статья» — цитата из золотой статьи (мягкий критерий)")
    header = (f"{'группа':<7} {'n':>3} {'pass@1':>6} {'95% ДИ':>11} {'статья':>6} {'95% ДИ':>11} {f'pass^{runs}':>7} "
              f"{'ответ':>6} {'дослов.':>7} {'шагов':>6} {'токенов':>8} {'с':>5}")
    lines += [header, "-" * len(header)]
    for group, s in report["summary"].items():
        exact = f"{s['exact_quote_rate']:.2f}" if s["exact_quote_rate"] is not None else "—"
        lines.append(
            f"{group:<7} {s['n']:>3} {s['pass@1']:>6.2f} {s['pass@1_ci'][0]:>5.2f}–{s['pass@1_ci'][1]:<5.2f} "
            f"{s['pass@1_paper']:>6.2f} {s['pass@1_paper_ci'][0]:>5.2f}–{s['pass@1_paper_ci'][1]:<5.2f} "
            f"{s[f'pass^{runs}']:>7.2f} {s['accepted']:>6.2f} {exact:>7} {s['steps']:>6.1f} "
            f"{s['tokens']:>8.0f} {s['seconds']:>5.1f}"
        )
    return "\n".join(lines)


def results_from_log(library: Library, log_path: Path) -> list[AgentResult]:
    """Восстанавливает результаты прогонов из журнала JSONL — чтобы пересчитать метрики без повторного
    вызова модели (например, после смены критерия успеха)."""
    results: list[AgentResult] = []
    current: AgentResult | None = None
    accepted_args: dict | None = None
    pending_args: dict | None = None
    for line in Path(log_path).read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        kind = event["event"]
        if kind == "start":
            current, accepted_args = AgentResult(question=event["question"], answer=""), None
        elif kind == "tool" and event["name"] == FINAL_ANSWER:
            pending_args = json.loads(event["arguments"] or "{}")
            if event["observation"].startswith("Answer accepted."):
                accepted_args = pending_args
        elif kind == "finish" and current is not None:
            current.stop_reason, current.steps = event["stop_reason"], event["steps"]
            current.tool_calls, current.tool_errors = event["tool_calls"], event["tool_errors"]
            current.not_found, current.answer, current.seconds = event["not_found"], event["answer"], event["seconds"]
            current.usage = Usage(event["usage"]["prompt"], event["usage"]["completion"], event["usage"]["calls"])
            if accepted_args and not current.not_found:
                claims = accepted_args.get("claims") or []
                if any("quote" in c for c in claims if isinstance(c, dict)):
                    current.quotes = [(str(c.get("text", "")), str(c.get("quote", ""))) for c in claims if c.get("quote")]
                else:
                    ids = [parse_sentence_id(i) for c in claims if isinstance(c, dict) for i in c.get("sentence_ids") or []]
                    found = load_citations(library, [i for i in ids if i is not None])
                    current.claims = [Claim("", [found[i] for i in ids if i in found])]
            results.append(current)
            current = None
    return results


def rescore_log(library: Library, questions: list[GoldQuestion], log_path: Path, variant: str, runs: int = 1) -> dict:
    by_text = {q.question: q for q in questions}
    rows = []
    for result in results_from_log(library, log_path):
        q = by_text.get(result.question)
        if q is None:
            continue
        row = {"id": q.id, "group": question_group(q), "run": 1,
               **score_run(library, resolve(library, q), result)}
        rows.append(row)
    groups = {"all": rows}
    for name in sorted({r["group"] for r in rows}):
        groups[name] = [r for r in rows if r["group"] == name]
    return {
        "variant": variant, "model": "(из журнала)", "runs": runs, "questions": len({r["id"] for r in rows}),
        "seconds_total": sum(r["seconds"] for r in rows), "log": str(log_path),
        "summary": {name: _summary(group, runs) for name, group in groups.items()},
        "rows": rows,
    }
