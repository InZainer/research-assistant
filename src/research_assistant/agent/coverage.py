"""Проверка «чего в литературе нет»: один и тот же вопрос задаётся каждой статье темы.

Зачем. Обычный ответ агента строится по лучшим фрагментам на один запрос, поэтому вывод «в библиотеке этого
нет» на деле значит «по моей формулировке ничего не нашлось». Для утверждений о новизне этого мало: нужно
знать, какие статьи проверены, по каким формулировкам и что именно в них нашлось рядом.

Как. 1) Модель один раз переформулирует аспект: синонимы, более узкие и более широкие термины, русский и
английский варианты — чтобы результат не зависел от одной удачной или неудачной формулировки.
2) Статьи темы упорядочиваются по близости к этим формулировкам (BM25 и, если есть, эмбеддинги).
3) По каждой статье из первых `limit` модель получает её лучшие фрагменты и выносит вердикт: аспект
рассмотрен (yes), рассмотрен частично или в другой постановке (partial), не рассмотрен (no). Вердикты yes и
partial обязаны ссылаться на ID предложений; текст цитаты подставляет код.
4) Отчёт честно перечисляет границы: сколько статей проверено моделью, сколько не проверялось, у скольких
в библиотеке только аннотация (по аннотации отсутствие не доказать).

Это проверка по фрагментам, а не чтение: ближайшие аналоги (yes и partial) стоит прочитать целиком командой read.
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from research_assistant.agent.loop import RunLog
from research_assistant.answer.citations import Citation, load_citations, parse_sentence_id
from research_assistant.llm import ChatModel, Usage
from research_assistant.nlp.text import detect_language
from research_assistant.retrieval.search import RRF_K, Searcher
from research_assistant.store.db import Library, PassageFilter

RANK_DEPTH = 400  # сколько фрагментов на формулировку берём, чтобы упорядочить статьи
PER_QUERY = 2  # фрагментов статьи на одну формулировку
MAX_PHRASINGS = 6
VERDICTS = ("yes", "partial", "no")

EXPAND_SYSTEM = (
    "You help search a library of scientific papers (English and Russian) for one aspect: a problem setting, a "
    "method, a kind of data or an evaluation. Give alternative search phrasings of the aspect so that a paper "
    "that addresses it in other words is still found: synonyms, narrower and broader terms, typical names of "
    "the task in the literature. Include both English and Russian phrasings. Each phrasing is a short search "
    "query (3-8 words), not a sentence."
)
EXPAND_TOOL = {
    "type": "function",
    "function": {
        "name": "search_phrasings",
        "description": "Alternative search phrasings of the aspect.",
        "parameters": {
            "type": "object",
            "properties": {"phrasings": {"type": "array", "items": {"type": "string"},
                                         "description": f"{MAX_PHRASINGS - 2}-{MAX_PHRASINGS} phrasings."}},
            "required": ["phrasings"],
        },
    },
}
JUDGE_SYSTEM = """You check whether one scientific paper addresses a given aspect. You see the passages of the \
paper that are closest to the aspect; every sentence has an id like [s123]. Lines marked [table] are table rows.

Decide from these passages only, without outside knowledge:
- "yes": a sentence states that the paper does, studies, measures or proposes exactly this aspect.
- "partial": the paper is close but differs in an essential way (another outcome variable, another setting, \
only mentions it as related or future work, only a special case). Say in the note exactly what differs.
- "no": none of the passages addresses the aspect.
For "yes" and "partial" cite 1-3 sentence ids that show it. Never invent ids. Be strict: a shared keyword is \
not enough for "yes". Write the note in the language of the aspect, one or two sentences."""
VERDICT_TOOL = {
    "type": "function",
    "function": {
        "name": "verdict",
        "description": "Verdict on whether the paper addresses the aspect.",
        "parameters": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": list(VERDICTS)},
                "note": {"type": "string", "description": "What the paper does about the aspect, or what differs."},
                "sentence_ids": {"type": "array", "items": {"type": "string"},
                                 "description": "Supporting sentence ids for yes/partial; empty for no."},
            },
            "required": ["verdict", "note", "sentence_ids"],
        },
    },
}


@dataclass
class PaperVerdict:
    paper_id: int
    title: str
    year: int | None
    verdict: str  # yes | partial | no | unverified (модель сказала yes/partial без верных ID) | error
    note: str = ""
    citations: list[Citation] = field(default_factory=list)
    fulltext: bool = True  # False — в библиотеке только аннотация


@dataclass
class CoverageReport:
    aspect: str
    topic: str | None
    phrasings: list[str]
    verdicts: list[PaperVerdict] = field(default_factory=list)
    scope: int = 0  # статей в теме с текстом
    unchecked: list[tuple[int, str]] = field(default_factory=list)  # (id, название): не дошли до модели
    no_hits: int = 0  # из непроверенных — ни одного близкого фрагмента ни по одной формулировке
    usage: Usage = field(default_factory=Usage)
    seconds: float = 0.0
    log_path: str | None = None

    def of(self, verdict: str) -> list[PaperVerdict]:
        return [v for v in self.verdicts if v.verdict == verdict]


def _tool_args(model: ChatModel, messages: list[dict], tool: dict, usage: Usage, log: RunLog, step: str) -> dict | None:
    response = model.complete(messages, [tool])
    usage.add(response.usage)
    message = response.choices[0].message
    calls = message.tool_calls or []
    log.write("llm", step=step, content=message.content,
              tool_calls=[{"name": c.function.name, "arguments": c.function.arguments} for c in calls],
              usage={"prompt": getattr(response.usage, "prompt_tokens", None),
                     "completion": getattr(response.usage, "completion_tokens", None)})
    for call in calls:
        if call.function.name == tool["function"]["name"]:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                return None
            return args if isinstance(args, dict) else None
    return None


def expand_aspect(model: ChatModel, aspect: str, usage: Usage, log: RunLog) -> list[str]:
    """Формулировки аспекта для поиска: сам аспект первым. Сбой модели — остаётся одна формулировка."""
    try:
        args = _tool_args(model, [{"role": "system", "content": EXPAND_SYSTEM},
                                  {"role": "user", "content": f"Aspect: {aspect}"}],
                          EXPAND_TOOL, usage, log, step="expand")
    except Exception as exc:  # расширение — улучшение, а не обязательный шаг
        log.write("error", step="expand", error=f"{type(exc).__name__}: {exc}")
        args = None
    extra = [" ".join(str(p).split()) for p in (args or {}).get("phrasings") or [] if str(p).strip()]
    return list(dict.fromkeys([aspect, *extra]))[:MAX_PHRASINGS + 1]


def rank_papers(searcher: Searcher, phrasings: list[str], where: PassageFilter) -> dict[int, float]:
    """Оценка близости статьи: сумма по спискам 1 / (60 + место её лучшего фрагмента), как в RRF."""
    library = searcher.library
    scores: dict[int, float] = {}
    for phrasing in phrasings:
        rankings = [[i for i, _ in searcher.bm25(phrasing, RANK_DEPTH, where)]]
        if searcher.dense_ready and not searcher.dense_failed:
            try:
                rankings.append([i for i, _ in searcher.dense(phrasing, RANK_DEPTH, where)])
            except Exception as exc:
                searcher.dense_failed = f"плотный поиск недоступен ({type(exc).__name__}), только BM25"
        for ranking in rankings:
            papers = library.passages_by_ids(ranking)
            best: dict[int, int] = {}
            for place, passage_id in enumerate(ranking, 1):
                paper_id = papers.get(passage_id, {}).get("paper_id")
                if paper_id is not None and paper_id not in best:
                    best[paper_id] = place
            for paper_id, place in best.items():
                scores[paper_id] = scores.get(paper_id, 0.0) + 1.0 / (RRF_K + place)
    return scores


def paper_evidence(searcher: Searcher, paper_id: int, phrasings: list[str], per_paper: int) -> list[dict]:
    """Лучшие фрагменты одной статьи по всем формулировкам: по очереди от каждой, без повторов."""
    where = PassageFilter(paper_ids=[paper_id])
    lists = [[h.passage for h in searcher.search(p, k=PER_QUERY, mode="hybrid", where=where)] for p in phrasings]
    chosen: dict[int, dict] = {}
    for depth in range(PER_QUERY):
        for hits in lists:
            if depth < len(hits) and hits[depth].get("id") is not None and len(chosen) < per_paper:
                chosen.setdefault(hits[depth]["id"], hits[depth])
    return list(chosen.values())


def _evidence_text(library: Library, passages: list[dict]) -> tuple[str, set[int]]:
    lines, ids = [], set()
    for passage in passages:
        where = f"page {passage['page']}" if passage.get("page") else (
            "abstract from metadata" if passage.get("source") == "metadata" else "no page")
        lines.append(f"passage p{passage['id']} | section: {passage['section'] or '-'} "
                     f"({passage['section_kind']}) | {where}")
        mark = "[table] " if passage.get("kind") == "table" else ""
        for row in library.conn.execute(
                "SELECT id, text FROM sentences WHERE passage_id = ? ORDER BY order_idx", (passage["id"],)):
            ids.add(row["id"])
            lines.append(f"  {mark}[s{row['id']}] {row['text']}")
    return "\n".join(lines), ids


def check_coverage(
    library: Library,
    searcher: Searcher,
    model: ChatModel,
    aspect: str,
    topic: str | None = None,
    paper_ids: list[int] | None = None,
    limit: int = 60,
    per_paper: int = 5,
    expand: bool = True,
    log_path: Path | None = None,
    progress: Callable[[str], None] = lambda message: None,
) -> CoverageReport:
    started = time.perf_counter()
    aspect = " ".join(aspect.split())
    if not aspect:
        raise ValueError("пустой аспект")
    log = RunLog(log_path)
    report = CoverageReport(aspect=aspect, topic=topic, phrasings=[aspect],
                            log_path=str(log_path) if log_path else None)
    papers = {p["id"]: p for p in library.papers_where(topic)}
    if paper_ids:
        papers = {i: p for i, p in papers.items() if i in set(paper_ids)}
    with_text = {r[0] for r in library.conn.execute("SELECT DISTINCT paper_id FROM passages")}
    papers = {i: p for i, p in papers.items() if i in with_text}
    report.scope = len(papers)
    log.write("start", aspect=aspect, topic=topic, paper_ids=paper_ids, scope=report.scope, limit=limit,
              model=model.model)
    if not papers:
        log.write("finish", scope=0)
        return report

    if expand:
        report.phrasings = expand_aspect(model, aspect, report.usage, log)
    progress(f"Формулировки поиска: {'; '.join(report.phrasings)}")
    where = PassageFilter(topic=topic, paper_ids=list(papers) if paper_ids else None)
    scores = rank_papers(searcher, report.phrasings, where)
    scores = {i: s for i, s in scores.items() if i in papers}
    ordered = sorted(scores, key=lambda i: -scores[i])
    selected = ordered if paper_ids else ordered[:max(1, limit)]
    chosen = set(selected)
    report.unchecked = [(i, papers[i]["title"]) for i in papers if i not in chosen]
    report.no_hits = sum(1 for i in papers if i not in scores)
    fulltext = {r[0] for r in library.conn.execute("SELECT paper_id FROM parses WHERE status = 'ok'")}
    language = "Russian" if detect_language(aspect) == "ru" else "English"

    for number, paper_id in enumerate(selected, 1):
        paper = papers[paper_id]
        verdict = PaperVerdict(paper_id, paper["title"], paper["year"], "error", fulltext=paper_id in fulltext)
        try:
            passages = paper_evidence(searcher, paper_id, report.phrasings, per_paper)
            text, allowed = _evidence_text(library, passages)
            basis = ("the full text of the paper is in the library" if verdict.fulltext else
                     "ONLY THE ABSTRACT of this paper is in the library")
            user = (f"Aspect: {aspect}\n\nPaper {paper_id}: \"{paper['title']}\" ({paper['year'] or 'n.d.'}); {basis}.\n"
                    f"Write the note in {language}.\n\nClosest passages of this paper:\n{text or '(none)'}")
            args = _tool_args(model, [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
                              VERDICT_TOOL, report.usage, log, step=f"paper {paper_id}") if passages else \
                {"verdict": "no", "note": "", "sentence_ids": []}
            if args is not None and args.get("verdict") in VERDICTS:
                raw = args.get("sentence_ids") if isinstance(args.get("sentence_ids"), list) else []
                ids = [i for i in dict.fromkeys(parse_sentence_id(v) for v in raw) if i in allowed]
                verdict.verdict, verdict.note = args["verdict"], str(args.get("note") or "").strip()
                if verdict.verdict != "no":
                    if ids:
                        found = load_citations(library, ids)
                        verdict.citations = [found[i] for i in ids if i in found]
                    else:
                        verdict.verdict = "unverified"  # утверждение без цитаты в отчёт как факт не идёт
        except Exception as exc:  # сбой на одной статье не роняет всю проверку
            verdict.note = f"{type(exc).__name__}: {exc}"
        log.write("verdict", paper_id=paper_id, title=paper["title"], verdict=verdict.verdict, note=verdict.note,
                  ids=[c.sentence_id for c in verdict.citations], fulltext=verdict.fulltext)
        progress(f"[{number}/{len(selected)}] {verdict.verdict}: #{paper_id} {paper['title'][:70]}")
        report.verdicts.append(verdict)

    report.seconds = time.perf_counter() - started
    log.write("finish", scope=report.scope, checked=len(report.verdicts),
              counts={v: len(report.of(v)) for v in (*VERDICTS, "unverified", "error")},
              unchecked=len(report.unchecked), no_hits=report.no_hits, phrasings=report.phrasings,
              usage={"prompt": report.usage.prompt_tokens, "completion": report.usage.completion_tokens,
                     "calls": report.usage.calls},
              seconds=round(report.seconds, 2))
    return report


def format_coverage(report: CoverageReport, show_no: int = 15) -> str:
    """Отчёт для терминала: сначала границы проверки, затем статьи по вердиктам."""
    from research_assistant.answer.citations import format_citation

    scope = f"теме «{report.topic}»" if report.topic else "библиотеке"
    lines = [f"Аспект: {report.aspect}"]
    if not report.scope:
        return "\n".join(lines + [f"В {scope} нет статей с текстом."])
    checked = report.verdicts
    abstract_only = [v for v in checked if not v.fulltext]
    lines += [
        f"Формулировки поиска: {'; '.join(report.phrasings)}",
        f"Проверено моделью: {len(checked)} из {report.scope} статей в {scope}"
        + (f" (у {len(abstract_only)} из них в библиотеке только аннотация)" if abstract_only else "") + ".",
        f"Рассматривают: {len(report.of('yes'))}; частично или в другой постановке: {len(report.of('partial'))}; "
        f"не рассматривают: {len(report.of('no'))}"
        + (f"; без подтверждающей цитаты: {len(report.of('unverified'))}" if report.of("unverified") else "")
        + (f"; ошибок: {len(report.of('error'))}" if report.of("error") else "") + ".",
    ]
    if report.unchecked:
        lines.append(f"Не проверялись: {len(report.unchecked)} статей, наименее близких к формулировкам "
                     f"(из них без единого близкого фрагмента — {report.no_hits}). Проверить все: --limit {report.scope}.")
    for verdict, title in (("yes", "Рассматривают"), ("partial", "Частично или в другой постановке"),
                           ("unverified", "Модель сочла близкими, но не привела цитату (проверьте сами)")):
        group = report.of(verdict)
        if not group:
            continue
        lines += ["", f"{title}:"]
        for v in group:
            mark = "" if v.fulltext else " [только аннотация]"
            lines.append(f"  #{v.paper_id} {v.title} ({v.year or '—'}){mark}")
            if v.note:
                lines.append(f"      {v.note}")
            lines += [f"      {format_citation(c)}" for c in v.citations]
    negative = report.of("no")
    if negative:
        lines += ["", f"Не рассматривают ({len(negative)}):"]
        for v in negative[:show_no]:
            lines.append(f"  #{v.paper_id} {v.title[:90]} ({v.year or '—'})" + ("" if v.fulltext else " [только аннотация]"))
        if len(negative) > show_no:
            lines.append(f"  … и ещё {len(negative) - show_no}")
    for v in report.of("error"):
        lines.append(f"  [!] #{v.paper_id} {v.title[:70]}: {v.note}")

    lines += ["", "Что из этого следует:"]
    if report.of("yes"):
        lines.append("  Аспект в литературе есть — см. статьи выше; утверждать его новизну нельзя.")
    else:
        strength = "по фрагментам проверенных статей" + (
            f"; {len(report.unchecked)} статей не проверялись" if report.unchecked else "")
        lines.append(f"  Прямого рассмотрения аспекта не найдено ({strength}). Это вывод о библиотеке, а не о всей "
                     f"литературе, и о лучших фрагментах, а не о полных текстах.")
        if abstract_only:
            lines.append(f"  У {len(abstract_only)} статей в библиотеке только аннотация: их отсутствие не доказано.")
    close = [v for v in checked if v.verdict in ("yes", "partial", "unverified")]
    if close:
        ids = " ".join(str(v.paper_id) for v in close[:10])
        lines.append(f"  Ближайшие аналоги стоит прочитать целиком: research-assistant read <id> (id: {ids}).")
    usage = report.usage
    lines += ["", f"Вызовов модели {usage.calls}, токенов {usage.total}, {report.seconds:.1f} с."
                  + (f" Журнал: {report.log_path}" if report.log_path else "")]
    return "\n".join(lines)
