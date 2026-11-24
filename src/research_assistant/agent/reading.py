"""Чтение статьи целиком: модель получает весь текст по порядку, а не лучшие фрагменты по запросу.

Зачем. Поиск по фрагментам хорошо отвечает на вопрос «что сделано в статье» и плохо — на «чего в ней нет»:
отрицательный вывод держится на том, о чём догадались спросить; ход рассуждения рвётся на куски; противоречие
между текстом и таблицей не видно, потому что они приходят разными фрагментами. Для ближайших аналогов, от
которых зависят формулировки новизны, статью нужно прочитать целиком.

Как. Текст статьи выдаётся модели по разделам, каждое предложение со своим ID. Статья, которая помещается в
одно окно (Settings.read_window_chars), читается за один вызов. Длинная — окнами по порядку: из каждого окна
модель выписывает заметки (числа, определения, допущения, ограничения) с ID предложений и видит заметки
предыдущих окон, поэтому может заметить расхождение между частями. Итоговый отчёт пишется по заметкам.

Цитаты — как у агента: модель называет только ID предложений, текст подставляет код. ID, которого не было
в показанном тексте, отбрасывается, а утверждение без единого верного ID в отчёт не попадает.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from research_assistant.agent.loop import RunLog
from research_assistant.answer.citations import Citation, load_citations, parse_sentence_id
from research_assistant.llm import ChatModel, Usage
from research_assistant.nlp.text import detect_language
from research_assistant.retrieval.papers import authors_of
from research_assistant.store.db import Library

DEFAULT_WINDOW_CHARS = 9000
ASPECTS = ("task", "data", "method", "metric", "result", "limitation", "assumption", "other")
MAX_RETRIES = 1  # сколько раз просим модель переделать вызов (нет вызова инструмента, все ID неверные)

SYSTEM = """You read one scientific paper in full for a researcher who must position their own work against it.
You are given the whole text of the paper in reading order. Every sentence starts with its id like [s123].
Lines marked [table] are rows of a table (cells separated by ' | '), lines marked [caption] are figure or \
table captions.

Rules:
1. Use only the text you are given. No outside knowledge about this paper, its authors or follow-up work.
2. Every finding must cite the ids of the sentences that state it. Never invent ids. Prefer the sentence that \
states the fact most literally; for numbers cite the sentence or table row that contains the number.
3. Follow the argument across sections: check that the abstract, the method, the tables and the conclusion \
agree. If two places contradict each other (a number in the text differs from the table, a claim in the \
abstract is not supported by the results, a formula does not match its description), report it as an \
inconsistency and cite both places.
4. Absence matters as much as presence. For every aspect the reader asks about, decide: the paper addresses \
it (finding with ids), or it does not (list it in not_covered). List an aspect in not_covered only if no \
sentence of the text addresses it; if the paper touches it partially, write a finding that says exactly what \
is and is not done.
5. If tables or formulas look garbled (numbers run together, columns lost), do not guess the values: say so \
in the finding.
6. Write in the language of the reader's request."""

FINDING_SCHEMA = {
    "type": "object",
    "properties": {
        "aspect": {"type": "string", "enum": list(ASPECTS)},
        "text": {"type": "string", "description": "One finding, precise: what exactly the paper does or reports."},
        "sentence_ids": {"type": "array", "items": {"type": "string"},
                         "description": "Supporting sentence ids from the text, e.g. ['s123']."},
    },
    "required": ["aspect", "text", "sentence_ids"],
}
CITED_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "sentence_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["text", "sentence_ids"],
}
REPORT_TOOL = {
    "type": "function",
    "function": {
        "name": "paper_report",
        "description": "Final report on the paper after reading all of it.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string",
                            "description": "Direct answer to the reader's request (or what the paper does, to whom "
                                           "it matters and how strong the evidence is), 3-8 sentences."},
                "findings": {"type": "array", "items": FINDING_SCHEMA,
                             "description": "What the paper states: task, data, method, metrics, results with "
                                            "numbers, limitations, assumptions. 6-20 findings."},
                "not_covered": {"type": "array", "items": {"type": "string"},
                                "description": "What the paper does NOT contain: aspects the reader asked about, "
                                               "and things a reader would expect here, that no sentence addresses."},
                "inconsistencies": {"type": "array", "items": CITED_SCHEMA,
                                    "description": "Contradictions between parts of the paper, citing both places. "
                                                   "Empty if none."},
            },
            "required": ["summary", "findings", "not_covered", "inconsistencies"],
        },
    },
}
NOTES_TOOL = {
    "type": "function",
    "function": {
        "name": "take_notes",
        "description": "Notes on the current part of the paper; you will write the final report from these notes.",
        "parameters": {
            "type": "object",
            "properties": {
                "notes": {
                    "type": "array", "items": FINDING_SCHEMA,
                    "description": "Everything from this part that the report or a later consistency check may "
                                   "need: task, data, definitions, assumptions, method steps, every reported "
                                   "number with what it measures, limitations. If this part contradicts an "
                                   "earlier note, add a note that starts with 'INCONSISTENCY:' and cite both ids.",
                },
            },
            "required": ["notes"],
        },
    },
}


@dataclass
class Finding:
    aspect: str
    text: str
    citations: list[Citation]


@dataclass
class PaperReading:
    paper_id: int
    title: str
    request: str
    summary: str = ""
    findings: list[Finding] = field(default_factory=list)
    not_covered: list[str] = field(default_factory=list)
    inconsistencies: list[Finding] = field(default_factory=list)
    source: str = "fulltext"  # fulltext | metadata: по одной аннотации выводы «чего нет» недействительны
    sentences: int = 0
    windows: int = 0
    dropped: int = 0  # утверждений без единого верного ID — в отчёт не попали
    stop_reason: str = "report"  # report | no_report | empty
    usage: Usage = field(default_factory=Usage)
    seconds: float = 0.0
    log_path: str | None = None


@dataclass
class Window:
    text: str
    sentence_ids: set[int]


def paper_windows(library: Library, paper_id: int, window_chars: int = DEFAULT_WINDOW_CHARS) -> list[Window]:
    """Текст статьи по порядку, нарезанный на окна по границам фрагментов. Заголовок раздела повторяется
    в начале окна, чтобы модель знала, где находится."""
    windows: list[Window] = []
    lines: list[str] = []
    ids: set[int] = set()
    size = 0

    def flush() -> None:
        nonlocal lines, ids, size
        if ids:
            windows.append(Window("\n".join(lines), ids))
        lines, ids, size = [], set(), 0

    for section in library.paper_structure(paper_id):
        head = f"## {section['title'] or '(no heading)'} [{section['kind']}]"
        head_shown = False
        for passage in section["passages"]:
            mark = {"table": "[table] ", "caption": "[caption] "}.get(passage["kind"], "")
            where = f" (page {passage['page']})" if passage["page"] else ""
            block = [f"{mark}[s{s['id']}] {s['text']}" for s in passage["sentences"]]
            if not block:
                continue
            block_size = sum(len(line) + 1 for line in block)
            if ids and size + block_size > window_chars:
                flush()
                head_shown = False
            if not head_shown:
                lines.append(head + where)
                head_shown = True
            lines.extend(block)
            ids.update(s["id"] for s in passage["sentences"])
            size += block_size
    flush()
    return windows


def _request_text(question: str | None, aspects: list[str] | None) -> str:
    parts = []
    if question and question.strip():
        parts.append(f"Reader's question: {question.strip()}")
    if aspects:
        parts.append("Aspects to check one by one (for each: addressed, with ids — or not covered):\n"
                     + "\n".join(f"- {a}" for a in aspects))
    if not parts:
        parts.append("Reader's request: a full reading report — what the paper does, on what data, with which "
                     "method and metrics, the main numbers, stated limitations and assumptions, and what a "
                     "reader would expect here but the paper does not contain.")
    return "\n".join(parts)


def _notes_text(notes: list[dict]) -> str:
    return "\n".join(
        f"- ({n['aspect']}) {n['text']} [{', '.join('s%d' % i for i in n['ids'])}]" for n in notes
    )


def _valid_ids(raw, allowed: set[int]) -> list[int]:
    ids = [parse_sentence_id(value) for value in raw or []] if isinstance(raw, list) else []
    return list(dict.fromkeys(i for i in ids if i is not None and i in allowed))


def _call(model: ChatModel, messages: list[dict], tool: dict, log: RunLog, usage: Usage, step: str) -> dict | None:
    """Один вызов модели с единственным инструментом. Возвращает его аргументы или None."""
    name = tool["function"]["name"]
    for attempt in range(MAX_RETRIES + 1):
        response = model.complete(messages, [tool])
        usage.add(response.usage)
        message = response.choices[0].message
        calls = message.tool_calls or []
        log.write("llm", step=step, attempt=attempt, content=message.content,
                  tool_calls=[{"name": c.function.name, "arguments": c.function.arguments} for c in calls],
                  usage={"prompt": getattr(response.usage, "prompt_tokens", None),
                         "completion": getattr(response.usage, "completion_tokens", None)},
                  finish_reason=response.choices[0].finish_reason)
        for call in calls:
            if call.function.name != name:
                continue
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                continue
            if isinstance(args, dict):
                return args
        messages = [*messages, {"role": "assistant", "content": message.content or ""},
                    {"role": "user", "content": f"Call the {name} tool with valid JSON arguments."}]
    return None


def _cited(items, allowed: set[int], default_aspect: str = "other") -> tuple[list[dict], int]:
    """Утверждения модели → [{aspect, text, ids}] с проверенными ID; второе значение — сколько отброшено."""
    kept, dropped = [], 0
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or not str(item.get("text", "")).strip():
            continue
        ids = _valid_ids(item.get("sentence_ids"), allowed)
        if not ids:
            dropped += 1
            continue
        aspect = item.get("aspect") if item.get("aspect") in ASPECTS else default_aspect
        kept.append({"aspect": aspect, "text": str(item["text"]).strip(), "ids": ids})
    return kept, dropped


def read_paper(
    library: Library,
    model: ChatModel,
    paper_id: int,
    question: str | None = None,
    aspects: list[str] | None = None,
    window_chars: int = DEFAULT_WINDOW_CHARS,
    log_path: Path | None = None,
) -> PaperReading:
    started = time.perf_counter()
    paper = library.get_paper(paper_id)
    if paper is None:
        raise ValueError(f"статьи #{paper_id} нет в библиотеке")
    request = _request_text(question, aspects)
    reading = PaperReading(paper_id=paper_id, title=paper["title"], request=request,
                           log_path=str(log_path) if log_path else None)
    log = RunLog(log_path)
    windows = paper_windows(library, paper_id, window_chars)
    parse = library.get_parse(paper_id)
    reading.source = "fulltext" if parse is not None and parse["status"] == "ok" else "metadata"
    reading.windows = len(windows)
    reading.sentences = sum(len(w.sentence_ids) for w in windows)
    log.write("start", paper_id=paper_id, title=paper["title"], model=model.model, request=request,
              windows=len(windows), sentences=reading.sentences, source=reading.source, window_chars=window_chars)
    if not windows:
        reading.stop_reason = "empty"
        log.write("finish", stop_reason="empty")
        return reading

    authors = authors_of(paper)
    head = (f"Paper: \"{paper['title']}\" ({paper['year'] or 'n.d.'})"
            + (f" by {', '.join(authors[:4])}{' et al.' if len(authors) > 4 else ''}" if authors else ""))
    language = ("Write the report in Russian." if detect_language(question or " ".join(aspects or []) or "ru") == "ru"
                else "Write the report in English.")
    if reading.source == "metadata":
        head += ("\nNOTE: only the abstract of this paper is in the library. Do not list anything in not_covered: "
                 "absence cannot be judged from an abstract. Say this in the summary.")
    allowed: set[int] = set()
    notes: list[dict] = []

    if len(windows) == 1:
        allowed = set(windows[0].sentence_ids)
        final_user = f"{head}\n{request}\n{language}\n\nFull text of the paper:\n{windows[0].text}"
    else:
        for number, window in enumerate(windows, 1):
            allowed |= window.sentence_ids
            earlier = f"Your notes on the earlier parts:\n{_notes_text(notes)}\n\n" if notes else ""
            user = (f"{head}\n{request}\n{language}\n\nThis is part {number} of {len(windows)} of the paper. "
                    f"Take notes with the take_notes tool.\n\n{earlier}Part {number}:\n{window.text}")
            args = _call(model, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                         NOTES_TOOL, log, reading.usage, step=f"notes {number}/{len(windows)}")
            kept, dropped = _cited((args or {}).get("notes"), allowed)
            notes.extend(kept)
            reading.dropped += dropped
        final_user = (f"{head}\n{request}\n{language}\n\nYou have read the whole paper in {len(windows)} parts. "
                      f"Write the final report with the paper_report tool from your notes; cite the same ids. "
                      f"Notes that start with 'INCONSISTENCY:' go to inconsistencies.\n\n"
                      f"Your notes:\n{_notes_text(notes)}")
        allowed = {i for note in notes for i in note["ids"]}  # в отчёте — только ID из собственных заметок

    args = _call(model, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": final_user}],
                 REPORT_TOOL, log, reading.usage, step="report")
    if args is None:
        reading.stop_reason = "no_report"
    else:
        findings, dropped_a = _cited(args.get("findings"), allowed)
        problems, dropped_b = _cited(args.get("inconsistencies"), allowed)
        reading.dropped += dropped_a + dropped_b
        citations = load_citations(library, [i for item in findings + problems for i in item["ids"]])
        reading.summary = str(args.get("summary") or "").strip()
        reading.findings = [Finding(f["aspect"], f["text"], [citations[i] for i in f["ids"] if i in citations])
                            for f in findings]
        reading.inconsistencies = [Finding("other", f["text"], [citations[i] for i in f["ids"] if i in citations])
                                   for f in problems]
        if reading.source == "fulltext":
            reading.not_covered = [str(x).strip() for x in args.get("not_covered") or [] if str(x).strip()]
    reading.seconds = time.perf_counter() - started
    log.write("finish", stop_reason=reading.stop_reason, summary=reading.summary,
              findings=[{"aspect": f.aspect, "text": f.text, "ids": [c.sentence_id for c in f.citations]}
                        for f in reading.findings],
              not_covered=reading.not_covered,
              inconsistencies=[{"text": f.text, "ids": [c.sentence_id for c in f.citations]}
                               for f in reading.inconsistencies],
              dropped=reading.dropped, windows=reading.windows, sentences=reading.sentences,
              usage={"prompt": reading.usage.prompt_tokens, "completion": reading.usage.completion_tokens,
                     "calls": reading.usage.calls},
              seconds=round(reading.seconds, 2))
    return reading


ASPECT_TITLES = {
    "task": "Задача", "data": "Данные", "method": "Метод", "metric": "Метрики", "result": "Результаты",
    "limitation": "Ограничения", "assumption": "Допущения", "other": "Прочее",
}


def format_reading(reading: PaperReading) -> str:
    """Отчёт для терминала."""
    from research_assistant.answer.citations import format_citation

    lines = [f"#{reading.paper_id} {reading.title}"]
    if reading.stop_reason == "empty":
        return "\n".join(lines + ["В библиотеке нет текста этой статьи: выполните parse."])
    basis = (f"прочитано целиком: {reading.sentences} предложений, окон {reading.windows}"
             if reading.source == "fulltext" else
             f"в библиотеке только аннотация ({reading.sentences} предл.) — выводы «чего в статье нет» по ней "
             f"делать нельзя")
    lines.append(f"Основание: {basis}.")
    if reading.stop_reason == "no_report":
        return "\n".join(lines + ["[!] Модель не оформила отчёт (нет вызова paper_report)."])
    lines += ["", reading.summary or "(нет резюме)"]
    for aspect in ASPECTS:
        group = [f for f in reading.findings if f.aspect == aspect]
        if not group:
            continue
        lines += ["", f"{ASPECT_TITLES[aspect]}:"]
        for finding in group:
            lines.append(f"  • {finding.text}")
            lines += [f"      {format_citation(c)}" for c in finding.citations]
    if reading.inconsistencies:
        lines += ["", "Расхождения внутри статьи:"]
        for finding in reading.inconsistencies:
            lines.append(f"  • {finding.text}")
            lines += [f"      {format_citation(c)}" for c in finding.citations]
    if reading.source == "fulltext":
        lines += ["", "Чего в статье нет (по полному тексту):"]
        lines += [f"  – {item}" for item in reading.not_covered] or ["  (модель ничего не отметила)"]
    if reading.dropped:
        lines += ["", f"Отброшено утверждений без верных ID предложений: {reading.dropped}."]
    usage = reading.usage
    lines += ["", f"Вызовов модели {usage.calls}, токенов {usage.total} (prompt {usage.prompt_tokens}, "
                  f"completion {usage.completion_tokens}), {reading.seconds:.1f} с."
                  + (f" Журнал: {reading.log_path}" if reading.log_path else "")]
    return "\n".join(lines)
