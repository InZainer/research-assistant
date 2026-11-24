"""Оценка агента на вопросах с золотыми цитатами (eval/retrieval_gold.jsonl).

Два критерия успеха:
- строгий — агент сдал final_answer, не ответил not_found, и хотя бы одна цитата лежит во фрагменте с золотым ответом;
- мягкий — цитата из золотой статьи. Нужен потому, что ответ часто есть в нескольких фрагментах статьи
  (аннотация, результаты, заключение), а золотой набор отмечает один: итеративный агент, который ищет шире,
  находит верный ответ в другом фрагменте и по строгому критерию получает незаслуженный провал.
Для режима quotes (эксперимент E2) ещё считается доля цитат, которые дословно есть в библиотеке.
На вопросах kind=unanswerable успех — честный not_found.

Повторы (τ-bench): pass@1 — доля успешных ответов по всем прогонам; pass^k — доля вопросов, успешных во всех
k прогонах. Для pass@1 — 95% интервал Вильсона.

Метрики поведения (валидность вызовов, повторы, язык ответа, отказы и т. п., docs/finetune.md) считаются
по событиям журнала, поэтому их даёт и пересчёт --rescore.
"""

import json
import os
import math
import re
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from research_assistant.agent.loop import MAX_RESEARCH_CALLS, AgentResult, Claim, run_agent
from research_assistant.agent.loop import _call_key as call_key
from research_assistant.agent.tools import FINAL_ANSWER, FUNCTIONS, ToolContext, prompt_hash, wants_to_add
from research_assistant.answer.citations import load_citations, parse_sentence_id
from research_assistant.llm import Usage
from research_assistant.evaluation.retrieval import GoldQuestion, question_group, resolve
from research_assistant.nlp.text import detect_language
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


def score_run(library: Library, relevant: set[int], result: AgentResult, kind: str = "fact") -> dict:
    passages, n_quotes, exact = cited_passages(library, result)
    accepted = result.stop_reason == "final_answer" and not result.not_found
    gold_hit = bool(passages & relevant)
    paper_hit = bool(papers_of(library, passages) & papers_of(library, relevant))
    success, success_paper = accepted and gold_hit, accepted and paper_hit
    if kind == "unanswerable":  # ответа в библиотеке нет: верно только честно сказать об этом
        success = success_paper = result.stop_reason == "final_answer" and result.not_found
    return {
        "accepted": accepted, "not_found": result.not_found, "stop_reason": result.stop_reason,
        "gold_hit": gold_hit, "success": success,
        "paper_hit": paper_hit, "success_paper": success_paper,
        "quotes": n_quotes, "exact_quotes": exact,
        "steps": result.steps, "tool_calls": result.tool_calls, "tool_errors": result.tool_errors,
        "tokens": result.usage.total, "seconds": result.seconds,
    }


# --- метрики поведения по журналу одного прогона ---

FORMAT_ERRORS = {"json", "unknown_tool", "bad_args"}  # модель не смогла правильно оформить вызов
# В старых журналах нет error_kind: вид ошибки восстанавливается по началу текста. Порядок важен.
ERROR_PREFIXES = [
    ("Error: arguments are not valid JSON", "json"),
    ("Error: unknown tool", "unknown_tool"),
    ("Error: wrong arguments for", "bad_args"),
    ("Error: arguments must be a JSON object", "bad_args"),
    ("Error in ", "env"),
    ("Error", "tool_error"),
]


def error_kind(event: dict) -> str | None:
    """Вид ошибки вызова: из журнала (error_kind) или, для старых журналов, по тексту ошибки."""
    if "error_kind" in event:
        return event["error_kind"]
    error = event.get("error") or ""
    if not error:
        return None
    for prefix, kind in ERROR_PREFIXES:
        if error.startswith(prefix):
            return kind
    return "answer_rejected"  # без «Error» пишет только проверка final_answer: «Fix final_answer: …» и т. п.


def _call_args(event: dict) -> dict | None:
    """Аргументы вызова как dict (None, если модель прислала не JSON-объект)."""
    raw = event.get("arguments")
    if event.get("step") == 0:  # поиск шага 0 делает код: в журнале вместо аргументов сам вопрос
        return {"query": raw if isinstance(raw, str) else ""}
    if isinstance(raw, dict):
        return raw
    try:
        args = json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError):
        return None
    return args if isinstance(args, dict) else None


def journal_metrics(events: list[dict], question: GoldQuestion | None = None) -> dict:
    """Метрики поведения одного прогона по его событиям (start … finish), см. docs/finetune.md.

    Доли считаются по вызовам модели, без поиска шага 0; повтор поиска шага 0 считается повтором.
    Метрика без данных (нет вызовов, нет ответа, нет вопроса) — None: в средние она не входит."""
    start = next((e for e in events if e["event"] == "start"), {})
    finish = next((e for e in reversed(events) if e["event"] == "finish"), {})
    tools = [e for e in events if e["event"] == "tool"]
    calls = [e for e in tools if e.get("step") != 0]
    llm = [e for e in events if e["event"] == "llm"]
    kinds = [error_kind(e) for e in calls]

    # Повторы и бюджет — как в цикле: ключ получает каждый вызов инструмента с аргументами-объектом.
    seen: set[str] = set()
    repeats = research = 0
    for e in tools:
        args = _call_args(e)
        if e["name"] not in FUNCTIONS or args is None:
            continue
        key = call_key(e["name"], args)
        if e.get("step") != 0:
            research += 1
            repeats += key in seen
        seen.add(key)

    answers = [e for e in calls if e["name"] == FINAL_ANSWER]
    text = start.get("question") or (question.question if question else "")
    answer = (finish.get("answer") or "").strip()
    usage = finish.get("usage") or {}
    n = len(calls)
    nudges = sum(e["event"] == "nudge" for e in events)
    errors: dict[str, int] = {}
    for kind in kinds:
        if kind:
            errors[kind] = errors.get(kind, 0) + 1
    return {
        "calls": n,
        "errors": errors,
        "call_validity": (n - sum(k in FORMAT_ERRORS for k in kinds)) / n if n else None,
        "tool_error_rate": kinds.count("tool_error") / n if n else None,
        "redundant_rate": repeats / n if n else None,
        "first_answer_accept": (str(answers[0].get("observation", "")).startswith("Answer accepted.")
                                if answers else None),
        "lang_match": detect_language(answer) == question.language if question and answer else None,
        "unauthorized": 0 if wants_to_add(text) else sum(e["name"] == "add_papers" for e in calls),
        "nudges": nudges,
        "nudge": nudges > 0,
        "max_steps": finish.get("stop_reason") == "max_steps",
        "budget_exhausted": research >= MAX_RESEARCH_CALLS,
        "research_calls": research,
        "read_passage": sum(e["name"] == "read_passage" for e in calls),
        "parallel_steps": sum(len(e.get("tool_calls") or []) > 1 for e in llm),
        "llm_steps": len(llm),
        "steps": finish.get("steps", len(llm)),
        "tokens": (usage.get("prompt") or 0) + (usage.get("completion") or 0),
        "seconds": finish.get("seconds", 0.0),
    }


def percentile(values: list[float], q: float) -> float | None:
    """Перцентиль с линейной интерполяцией (как numpy по умолчанию); q от 0 до 1."""
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def _mean_of(rows: list[dict], key: str) -> float | None:
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    return sum(values) / len(values) if values else None


def _behaviour(rows: list[dict]) -> dict:
    """Сводка поведения по прогонам: средние долей, отказы по kind, перцентили шагов, токенов и времени."""
    journaled = [r for r in rows if "calls" in r]
    llm_steps = sum(r["llm_steps"] for r in journaled)
    # external: ответа нет в библиотеке по замыслу, not_found там законен — в отказы не входит
    unanswerable = [r for r in rows if r.get("kind") == "unanswerable"]
    answerable = [r for r in rows if r.get("kind", "fact") not in ("unanswerable", "external")]
    abstained = sum(r["not_found"] for r in unanswerable + answerable)
    right_abstained = sum(r["not_found"] for r in unanswerable)
    out = {
        **{key: _mean_of(journaled, key) for key in
           ("call_validity", "tool_error_rate", "redundant_rate", "first_answer_accept", "lang_match")},
        "unauthorized": sum(r["unauthorized"] > 0 for r in journaled) / len(journaled) if journaled else None,
        "nudge_rate": _mean_of(journaled, "nudge"),
        "max_steps_rate": _mean_of(journaled, "max_steps"),
        "budget_exhausted": _mean_of(journaled, "budget_exhausted"),
        "read_passage_per_q": _mean_of(journaled, "read_passage"),
        "parallel_call_rate": sum(r["parallel_steps"] for r in journaled) / llm_steps if llm_steps else None,
        "abstention_precision": right_abstained / abstained if abstained else None,
        "abstention_recall": right_abstained / len(unanswerable) if unanswerable else None,
        "false_not_found": sum(r["not_found"] for r in answerable) / len(answerable) if answerable else None,
    }
    for key in ("steps", "tokens", "seconds"):
        values = [r[key] for r in rows]
        out[f"{key}_p50"], out[f"{key}_p95"] = percentile(values, 0.5), percentile(values, 0.95)
    return out


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
        **_behaviour(rows),
    }


def _row(library: Library, q: GoldQuestion, run: int, result: AgentResult, events: list[dict],
         relevant: set[int]) -> dict:
    """Строка отчёта: успех по цитатам плюс метрики поведения по журналу прогона."""
    return {"id": q.id, "group": question_group(q), "kind": q.kind, "run": run,
            **journal_metrics(events, q), **score_run(library, relevant, result, q.kind)}


def _groups(rows: list[dict], runs: int) -> dict:
    groups = {"all": rows}
    for name in sorted({r["group"] for r in rows}):
        groups[name] = [r for r in rows if r["group"] == name]
    return {name: _summary(group, runs) for name, group in groups.items()}


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
    # Модель и PID в имени: два прогона, запущенные в одну секунду, иначе пишут в один журнал вперемешку.
    model_tag = re.sub(r"[^\w.-]+", "_", getattr(model, "model", None) or "model")
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}-{model_tag}-{os.getpid()}"
    rows = []
    started = time.perf_counter()
    for run in range(1, runs + 1):
        for number, q in enumerate(questions, 1):
            ctx = make_context()
            relevant = resolve(ctx.library, q)
            log_path = log_dir / f"eval-agent-{variant}-{stamp}.jsonl" if log_dir else None
            events: list[dict] = []  # те же события, что в журнале: по ним — метрики поведения
            # followup задаётся с предыдущим обменом, как в chat (и как при сборе траекторий учителя)
            result = run_agent(q.question, ctx, model, max_steps=max_steps, history=q.meta.get("history") or None,
                               log_path=log_path, qid=q.id,
                               on_event=lambda event, data: events.append({"event": event, **data}), **options)
            row = _row(ctx.library, q, run, result, events, relevant)
            rows.append(row)
            mark = "успех" if row["success"] else ("ответ не по золоту" if row["accepted"] else row["stop_reason"])
            log(f"[прогон {run}/{runs}, {number}/{len(questions)}] {q.id}: {mark}, шагов {row['steps']}, "
                f"{row['seconds']:.0f} с")
    return {
        "variant": variant, "model": model.model, "runs": runs, "questions": len(questions),
        "seconds_total": time.perf_counter() - started, "prompt_hash": prompt_hash(),
        "summary": _groups(rows, runs),
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
    lines += _behaviour_table(report)
    return "\n".join(lines)


def _fmt(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _behaviour_table(report: dict) -> list[str]:
    """Вторая таблица отчёта — поведение агента; и предупреждение, если журнал записан с другим промптом."""
    lines = [f"Внимание: {w}" for w in report.get("warnings") or []]
    current = report.get("prompt_hash")
    other = [h for h in report.get("journal_prompt_hashes") or [] if h != current]
    if current and other:
        lines.append(f"Внимание: журнал записан с другим промптом или схемами инструментов "
                     f"(prompt_hash {', '.join(h[:10] for h in other)}, сейчас {current[:10]})")
    if not any("call_validity" in s for s in report["summary"].values()):
        return lines
    lines += ["", "поведение: валидн. — вызовы без ошибок формата; повторы — точные повторы вызовов; "
              "1-й отв. — первый final_answer принят; отказ P/R — not_found на неотвечаемых; "
              "без спр. — add_papers без просьбы"]
    header = (f"{'группа':<7} {'n':>3} {'валидн.':>7} {'повторы':>7} {'1-й отв.':>8} {'язык':>5} {'отказ P':>7} "
              f"{'отказ R':>7} {'без спр.':>8} {'read/q':>6} {'p95 с':>6}")
    lines += [header, "-" * len(header)]
    for group, s in report["summary"].items():
        lines.append(
            f"{group:<7} {s['n']:>3} {_fmt(s['call_validity']):>7} {_fmt(s['redundant_rate']):>7} "
            f"{_fmt(s['first_answer_accept']):>8} {_fmt(s['lang_match']):>5} {_fmt(s['abstention_precision']):>7} "
            f"{_fmt(s['abstention_recall']):>7} {_fmt(s['unauthorized']):>8} {_fmt(s['read_passage_per_q'], 1):>6} "
            f"{_fmt(s['seconds_p95'], 1):>6}"
        )
    return lines


def runs_from_log(log_path: Path) -> list[list[dict]]:
    """События журнала, разбитые на прогоны от start до finish. Незаконченный прогон пропускается."""
    runs: list[list[dict]] = []
    current: list[dict] | None = None
    for line in Path(log_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event["event"] == "start":
            current = [event]
        elif current is not None:
            current.append(event)
            if event["event"] == "finish":
                runs.append(current)
                current = None
    return runs


def result_from_events(library: Library, events: list[dict]) -> AgentResult:
    """AgentResult одного прогона по его событиям: ответ, утверждения с текстами и цитатами, счётчики."""
    start, finish = events[0], events[-1]
    result = AgentResult(question=start["question"], answer="")
    accepted_args = None
    for event in events:
        accepted = str(event.get("observation", "")).startswith("Answer accepted.")
        if event["event"] == "tool" and event["name"] == FINAL_ANSWER and accepted:
            accepted_args = json.loads(event["arguments"] or "{}")
    result.stop_reason, result.steps = finish["stop_reason"], finish["steps"]
    result.tool_calls, result.tool_errors = finish["tool_calls"], finish["tool_errors"]
    result.not_found, result.answer, result.seconds = finish["not_found"], finish["answer"], finish["seconds"]
    result.nudges = finish.get("nudges", sum(e["event"] == "nudge" for e in events))
    result.usage = Usage(finish["usage"]["prompt"], finish["usage"]["completion"], finish["usage"]["calls"])
    if accepted_args and not result.not_found:
        claims = [c for c in accepted_args.get("claims") or [] if isinstance(c, dict)]
        if any("quote" in c for c in claims):
            result.quotes = [(str(c.get("text", "")), str(c.get("quote", ""))) for c in claims if c.get("quote")]
        else:
            ids = [[parse_sentence_id(i) for i in c.get("sentence_ids") or []] for c in claims]
            found = load_citations(library, [i for group in ids for i in group if i is not None])
            result.claims = [Claim(str(c.get("text", "")).strip(), [found[i] for i in group if i in found])
                             for c, group in zip(claims, ids)]
    return result


def results_from_log(library: Library, log_path: Path) -> list[AgentResult]:
    """Восстанавливает результаты прогонов из журнала JSONL — чтобы пересчитать метрики без повторного
    вызова модели (например, после смены критерия успеха)."""
    return [result_from_events(library, events) for events in runs_from_log(log_path)]


def rescore_log(library: Library, questions: list[GoldQuestion], log_path: Path, variant: str,
                runs: int | None = None) -> dict:
    """Пересчёт метрик по журналу без модели. run — номер вхождения вопроса в журнал, поэтому pass^k верен
    и для --runs > 1; если runs не задан, число прогонов берётся из журнала."""
    rows = []
    occurrences: dict[str, int] = {}
    hashes: set[str] = set()
    match = _question_matcher(questions)
    ambiguous: set[str] = set()
    for events in runs_from_log(log_path):
        result = result_from_events(library, events)
        q, candidates = match(events[0], occurrences)
        if q is None:
            continue
        if len(candidates) > 1:
            ambiguous.add(q.question)
        if events[0].get("prompt_hash"):
            hashes.add(events[0]["prompt_hash"])
        occurrences[q.id] = occurrences.get(q.id, 0) + 1
        rows.append(_row(library, q, occurrences[q.id], result, events, resolve(library, q)))
    runs = runs or max(occurrences.values(), default=1)
    return {
        "variant": variant, "model": "(из журнала)", "runs": runs, "questions": len({r["id"] for r in rows}),
        "seconds_total": sum(r["seconds"] for r in rows), "log": str(log_path),
        "prompt_hash": prompt_hash(), "journal_prompt_hashes": sorted(hashes),
        "warnings": [f"в журнале нет ID вопроса, а текст «{text[:60]}» есть у нескольких вопросов: "
                     f"прогоны разнесены по ним по порядку" for text in sorted(ambiguous)],
        "summary": _groups(rows, runs),
        "rows": rows,
    }


def _history_key(history: list[dict] | None) -> str:
    return json.dumps(history or [], sort_keys=True, ensure_ascii=False)


def _question_matcher(questions: list[GoldQuestion]):
    """Сопоставление прогона журнала с вопросом: по qid из start; в старых журналах — по тексту вопроса
    (и истории диалога, если она записана). Возвращает (вопрос или None, все подходящие кандидаты).

    У followup одинаковый второй вопрос бывает в разных диалогах, а в test.jsonl лежат отклонённые кандидаты
    с тем же текстом. Если кандидатов несколько, берётся тот, у кого меньше всего прогонов: eval-agent идёт по
    вопросам по порядку, так что прогоны распределяются так же, как задавались."""
    by_id = {q.id: q for q in questions}
    by_text: dict[str, list[GoldQuestion]] = {}
    for q in questions:
        by_text.setdefault(q.question, []).append(q)

    def match(start: dict, occurrences: dict[str, int]) -> tuple[GoldQuestion | None, list[GoldQuestion]]:
        if start.get("qid") is not None:  # прогон другого вопроса (набор сузили) — пропускается
            q = by_id.get(start["qid"])
            return q, [q] if q else []
        candidates = by_text.get(start.get("question"), [])
        if "history" in start:
            key = _history_key(start["history"])
            candidates = [q for q in candidates if _history_key(q.meta.get("history")) == key]
        if not candidates:
            return None, []
        return min(candidates, key=lambda q: occurrences.get(q.id, 0)), candidates

    return match
