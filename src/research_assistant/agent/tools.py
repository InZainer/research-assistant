"""Инструменты агента: схемы для модели и реализации.

Каждый инструмент возвращает сжатый текст (не весь ответ базы): найденные фрагменты с ID предложений,
по которым модель потом цитирует. Ошибка инструмента — исключение ToolError: цикл вернёт её модели как
наблюдение, а не упадёт.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from research_assistant.ingest.models import PaperRecord
from research_assistant.ingest.pipeline import search_sources
from research_assistant.retrieval.papers import authors_of
from research_assistant.retrieval.papers import find_papers as find_papers_in
from research_assistant.retrieval.search import Searcher
from research_assistant.store.db import Library, PassageFilter

MAX_K = 8
MAX_ADD = 5
SECTION_KINDS = [
    "abstract", "introduction", "related_work", "method", "experiments", "results", "discussion",
    "conclusion", "limitations", "appendix", "other", "body",
]


class ToolError(Exception):
    pass


@dataclass
class ToolContext:
    library: Library
    searcher: Searcher
    sources: list | None = None  # внешние источники для search_external
    add_papers_fn: Callable[[list[PaperRecord], str], str] | None = None
    candidates: dict[str, PaperRecord] = field(default_factory=dict)  # последние найденные во внешних источниках
    seen_sentences: set[int] = field(default_factory=set)  # что агент видел: цитировать можно только это
    allow_add: bool = True  # add_papers только по явной просьбе пользователя (выставляет run_agent)


# Глаголы просьбы добавить статьи. Маленькая модель на «найди статьи» сама вызывала add_papers и качала
# десяток статей в библиотеку, поэтому без такой просьбы инструмента у неё просто нет.
ADD_REQUEST_RE = re.compile(r"добав|загруз|скача|сохран|занес|\badd\b|download|\bimport\b|\bsave\b", re.I)


def wants_to_add(question: str) -> bool:
    return bool(ADD_REQUEST_RE.search(question or ""))


def _sentences(library: Library, passage_ids: list[int]) -> dict[int, list]:
    if not passage_ids:
        return {}
    rows = library.conn.execute(
        f"""SELECT id, passage_id, text FROM sentences WHERE passage_id IN ({','.join('?' * len(passage_ids))})
            ORDER BY passage_id, order_idx""",
        passage_ids,
    ).fetchall()
    grouped: dict[int, list] = {}
    for row in rows:
        grouped.setdefault(row["passage_id"], []).append(row)
    return grouped


def _format_passages(ctx: ToolContext, passages: list[dict], numbered: bool = True) -> str:
    sentences = _sentences(ctx.library, [p["id"] for p in passages])
    blocks = []
    for number, p in enumerate(passages, 1):
        where = f"page {p['page']}" if p.get("page") else ("abstract from metadata" if p.get("source") == "metadata" else "no page")
        head = f"[{number}] " if numbered else ""
        blocks.append(
            f"{head}passage p{p['id']} | paper {p['paper_id']}: \"{p['title']}\" ({p['year']}) | "
            f"section: {p['section'] or '-'} ({p['section_kind']}) | {where}"
        )
        for sentence in sentences.get(p["id"], []):
            ctx.seen_sentences.add(sentence["id"])
            blocks.append(f"  [s{sentence['id']}] {sentence['text']}")
    return "\n".join(blocks)


def search_library(ctx: ToolContext, query: str, k: int = 4, topic: str | None = None,
                   section_kinds: list[str] | None = None, year_from: int | None = None,
                   year_to: int | None = None, language: str | None = None, entity: str | None = None,
                   paper_ids: list[int] | None = None) -> str:
    if not query or not query.strip():
        raise ToolError("query is empty")
    k = max(1, min(int(k), MAX_K))
    unknown = [s for s in section_kinds or [] if s not in SECTION_KINDS]
    if unknown:
        raise ToolError(f"unknown section_kinds {unknown}; allowed: {', '.join(SECTION_KINDS)}")
    if section_kinds:
        # У статей без заголовков весь текст — раздел body: тип неизвестен, поэтому фильтр его не отсекает
        # (иначе ответ из такой статьи не находится — так было в первых прогонах).
        section_kinds = list(dict.fromkeys([*section_kinds, "body"]))
    where = PassageFilter(topic=topic, year_from=year_from, year_to=year_to,
                          section_kinds=section_kinds or None, language=language, entity=entity,
                          paper_ids=[int(i) for i in paper_ids] if paper_ids else None)
    hits = ctx.searcher.search(query, k=k, mode="hybrid", where=where)
    if not hits:
        return f'No passages found for "{query}". Try other words, English terms, or fewer filters.'
    return f'Found {len(hits)} passages for "{query}":\n' + _format_passages(ctx, [h.passage for h in hits])


def _parse_id(raw, prefix: str) -> tuple[int, str | None]:
    """«p123», «s123», «123» → (123, префикс или None)."""
    match = re.fullmatch(r"\s*\[?([ps])?(\d+)\]?\s*", str(raw), re.I)
    if not match:
        raise ToolError(f"bad id {raw!r}: expected something like {prefix}123")
    return int(match.group(2)), (match.group(1) or "").lower() or None


def read_passage(ctx: ToolContext, passage_id, neighbors: int = 1) -> str:
    number, kind = _parse_id(passage_id, "p")
    note = ""
    row = None
    if kind != "s":
        row = ctx.library.conn.execute("SELECT id, paper_id, order_idx FROM passages WHERE id = ?", (number,)).fetchone()
    if row is None:
        # Частая ошибка маленьких моделей — передать ID предложения вместо ID фрагмента.
        row = ctx.library.conn.execute(
            """SELECT p.id, p.paper_id, p.order_idx FROM sentences s JOIN passages p ON p.id = s.passage_id
               WHERE s.id = ?""", (number,)).fetchone()
        if row is None:
            raise ToolError(f"no passage or sentence with id {number}; passage ids look like p123")
        note = f"(s{number} is a sentence id; showing its passage p{row['id']})\n"
    neighbors = max(0, min(int(neighbors), 3))
    ids = [r["id"] for r in ctx.library.conn.execute(
        "SELECT id FROM passages WHERE paper_id = ? AND order_idx BETWEEN ? AND ? ORDER BY order_idx",
        (row["paper_id"], row["order_idx"] - neighbors, row["order_idx"] + neighbors),
    )]
    found = ctx.library.passages_by_ids(ids)
    return note + _format_passages(ctx, [found[i] for i in ids if i in found], numbered=False)


ENTITY_TYPES = ("method", "dataset", "metric", "task")


def list_entities(ctx: ToolContext, type: str, topic: str | None = None, limit: int = 15) -> str:
    if type not in ENTITY_TYPES:
        raise ToolError(f"unknown type {type!r}; allowed: {', '.join(ENTITY_TYPES)}")
    rows = ctx.library.entity_stats(type, topic, limit=max(1, min(int(limit), 30)))
    if not rows:
        return f"No {type} entities found" + (f' for topic "{topic}"' if topic else "") + ". Papers may be not annotated yet."
    scope = f' in topic "{topic}"' if topic else ""
    lines = [f"Most frequent {type} entities{scope} (by number of papers):"]
    for row in rows:
        lines.append(f"- {row['name']}: {row['papers']} papers (paper ids {row['paper_ids']}), {row['mentions']} passages")
    lines.append("To get citable sentences, call search_library with entity=<name>.")
    return "\n".join(lines)


MAX_COMPARE = 4


def compare_papers(ctx: ToolContext, paper_ids: list[int], aspect: str, k: int = 2) -> str:
    """По каждой статье: её разметка (датасеты, метрики, методы) и лучшие фрагменты по аспекту сравнения."""
    if not isinstance(paper_ids, list) or not 2 <= len(paper_ids) <= MAX_COMPARE:
        raise ToolError(f"give 2-{MAX_COMPARE} paper ids (see library_overview or search results)")
    if not aspect or not aspect.strip():
        raise ToolError("aspect is empty, e.g. 'speedup and datasets'")
    k = max(1, min(int(k), 3))
    blocks = []
    for paper_id in paper_ids:
        paper = ctx.library.get_paper(int(paper_id))
        if paper is None:
            raise ToolError(f"paper {paper_id} does not exist")
        entities = ctx.library.paper_entities(paper["id"])
        lines = [f"=== paper {paper['id']}: \"{paper['title']}\" ({paper['year']})"]
        for entity_type in ("dataset", "metric", "method"):
            if entities.get(entity_type):
                lines.append(f"{entity_type}s: {', '.join(entities[entity_type][:6])}")
        hits = ctx.searcher.search(aspect, k=k, mode="hybrid", where=PassageFilter(paper_ids=[paper["id"]]))
        lines.append(_format_passages(ctx, [h.passage for h in hits]) if hits else "(no passages on this aspect)")
        blocks.append("\n".join(lines))
    return f'Comparison on "{aspect}":\n\n' + "\n\n".join(blocks)


MAX_FIND = 10


def find_papers(ctx: ToolContext, query: str | None = None, author: str | None = None, topic: str | None = None,
                year_from: int | None = None, year_to: int | None = None) -> str:
    """Статьи библиотеки по автору и/или словам из названия и аннотации. Цитат не даёт: только id статей."""
    if not (query and query.strip()) and not (author and author.strip()):
        raise ToolError("give query (words from title or abstract) and/or author")
    hits = find_papers_in(ctx.library, query, author, topic, year_from, year_to, searcher=ctx.searcher, limit=MAX_FIND)
    if not hits:
        return "No papers found. Try another spelling of the author (Latin or Cyrillic) or fewer words."
    lines = [f"Found {len(hits)} papers:"]
    for hit in hits:
        p = hit.paper
        authors = authors_of(p)
        names = ", ".join(authors[:4]) + (" et al." if len(authors) > 4 else "")
        how = "similar by meaning" if hit.matched == ["semantic"] else "matched in " + " and ".join(hit.matched)
        lines.append(f"- paper {p['id']}: \"{p['title']}\" ({p['year']}, {p['language'] or '?'}) by {names or '?'} [{how}]")
    lines.append("To quote these papers, call search_library with paper_ids=[...].")
    return "\n".join(lines)


def library_overview(ctx: ToolContext, topic: str | None = None) -> str:
    if topic is None:
        rows = ctx.library.topic_summary()
        if not rows:
            return "The library is empty."
        return "Topics in the library:\n" + "\n".join(
            f"- {r['name']}: {r['papers']} papers ({r['fulltext']} with full text, {r['russian']} in Russian)" for r in rows
        )
    papers = ctx.library.list_papers(topic, limit=40)
    if not papers:
        raise ToolError(f'no topic "{topic}"; call library_overview without topic to list topics')
    lines = [f'Papers on "{topic}":']
    for p in papers:
        keywords = ctx.library.paper_keywords(p["id"], limit=3)
        tail = f" — keywords: {', '.join(keywords)}" if keywords else ""
        lines.append(f"- paper {p['id']}: {p['title']} ({p['year']}, {p['language'] or '?'}){tail}")
    return "\n".join(lines)


def search_external(ctx: ToolContext, query: str | None = None, limit: int = 8, year_from: int | None = None,
                    language: str | None = None, author: str | None = None) -> str:
    if not ctx.sources:
        raise ToolError("external search is not available in this session")
    if not (query and query.strip()) and not (author and author.strip()):
        raise ToolError("give query and/or author")
    result = search_sources(ctx.sources, query or "", limit=max(1, min(int(limit), 10)), year_from=year_from,
                            language=language, author=author)
    what = (f' for "{query}"' if query else "") + (f" by {author}" if author else "")
    if not result.records:
        errors = "; ".join(f"{k}: {v}" for k, v in result.errors.items())
        return f"Nothing found{what}." + (f" Source errors: {errors}" if errors else "")
    lines = [f"Found {len(result.records)} candidate papers{what} (not in the library yet unless marked):"]
    for record in result.records[:10]:
        ctx.candidates[record.external_id] = record
        known = " [already in library]" if ctx.library.find_paper(record) else ""
        abstract = (record.abstract or "")[:200]
        authors = ", ".join(record.authors[:3]) + (" et al." if len(record.authors) > 3 else "")
        lines.append(f"- {record.external_id}: {record.title} ({record.year}, {record.language or '?'}) by "
                     f"{authors or '?'}{known}. {abstract}")
    return "\n".join(lines)


def add_papers(ctx: ToolContext, ids: list[str], topic: str) -> str:
    if ctx.add_papers_fn is None:
        raise ToolError("adding papers is not available in this session")
    if not ctx.allow_add:
        raise ToolError("the user did not ask to add papers: list the candidates in final_answer instead")
    if not ids:
        raise ToolError("ids is empty")
    if len(ids) > MAX_ADD:
        raise ToolError(f"at most {MAX_ADD} papers per call")
    unknown = [i for i in ids if i not in ctx.candidates]
    if unknown:
        raise ToolError(f"unknown ids {unknown}: use ids returned by search_external in this session")
    return ctx.add_papers_fn([ctx.candidates[i] for i in ids], topic)


FINAL_ANSWER = "final_answer"

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_library",
            "description": (
                "Search the local library of scientific papers (English and Russian) and return the best passages. "
                "Every sentence comes with an id like [s123]; cite only these ids. Use it before answering any "
                "question about the papers. For a Russian question about English-language papers, search a second "
                "time with an English query. Returns at most k passages (default 4)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to find, e.g. 'draft model latency bottleneck'."},
                    "k": {"type": "integer", "description": "Number of passages, 1-8. Default 4."},
                    "topic": {"type": "string", "description": "Restrict to a library topic (see library_overview)."},
                    "section_kinds": {
                        "type": "array", "items": {"type": "string", "enum": SECTION_KINDS},
                        "description": "Restrict to section types, e.g. ['limitations'] for drawbacks, ['results'] for numbers.",
                    },
                    "year_from": {"type": "integer"},
                    "year_to": {"type": "integer"},
                    "language": {"type": "string", "enum": ["en", "ru"], "description": "Language of the papers."},
                    "entity": {
                        "type": "string",
                        "description": "Only passages that mention this method/dataset/metric/task, e.g. 'MT-Bench' or 'EAGLE'.",
                    },
                    "paper_ids": {
                        "type": "array", "items": {"type": "integer"},
                        "description": "Only passages from these papers (ids from find_papers or earlier results).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_entities",
            "description": (
                "List the most frequent annotated entities of one type (method, dataset, metric, task) across the "
                "library or a topic, with paper counts. Use it for questions like 'which datasets are used to evaluate "
                "X' or 'which methods exist'; then call search_library with entity=<name> to get citable sentences."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(ENTITY_TYPES)},
                    "topic": {"type": "string", "description": "Library topic; omit for the whole library."},
                    "limit": {"type": "integer", "description": "How many entities, 1-30. Default 15."},
                },
                "required": ["type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_papers",
            "description": (
                f"Compare 2-{MAX_COMPARE} papers on one aspect (e.g. 'speedup and datasets', 'limitations'): returns each "
                "paper's annotated datasets, metrics and methods and its best passages on the aspect, with sentence ids. "
                "Use paper ids from search results (paper N) or library_overview."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_ids": {"type": "array", "items": {"type": "integer"}},
                    "aspect": {"type": "string", "description": "What to compare, e.g. 'speedup on MT-Bench'."},
                    "k": {"type": "integer", "description": "Passages per paper, 1-3. Default 2."},
                },
                "required": ["paper_ids", "aspect"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_passage",
            "description": (
                "Read a passage together with its neighbouring passages from the same paper, when a search hit is "
                "cut short or you need the surrounding context. Sentences come with ids like [s123]."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "passage_id": {"type": "string", "description": "Passage id from search_library, e.g. 'p123'."},
                    "neighbors": {"type": "integer", "description": "Passages before and after, 0-3. Default 1."},
                },
                "required": ["passage_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_papers",
            "description": (
                "Find papers in the library by author and/or by words from the title or abstract. Use it for questions "
                "like 'papers by Tri Dao', 'is there a paper about X', 'what did <author> write'. Returns paper ids, "
                "titles, authors and years, without citable sentences."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Words from the title or abstract."},
                    "author": {"type": "string", "description": "Author name or surname, e.g. 'Leviathan' or 'Барсуков'."},
                    "topic": {"type": "string"},
                    "year_from": {"type": "integer"},
                    "year_to": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "library_overview",
            "description": "List the topics in the library, or the papers of one topic. Use it when asked what is in the library.",
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string", "description": "Topic name; omit to list all topics."}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_external",
            "description": (
                "Search OpenAlex and arXiv for new papers that are NOT yet in the library. Use only when the user asks "
                "to collect or find new papers. For papers written by a person, pass author (the query may be "
                "omitted). Returns candidate ids like arxiv:2402.01528 for add_papers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Topic words; optional when author is given."},
                    "author": {"type": "string", "description": "Author name, e.g. 'Geoffrey Hinton'."},
                    "limit": {"type": "integer", "description": "Candidates to return, 1-10. Default 8."},
                    "year_from": {"type": "integer"},
                    "language": {"type": "string", "enum": ["en", "ru"]},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_papers",
            "description": (
                "Download, parse and index candidate papers from search_external into the library under a topic. "
                f"At most {MAX_ADD} per call; slow (tens of seconds per paper). Use only when the user asks to add papers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ids": {"type": "array", "items": {"type": "string"}, "description": "Ids from search_external."},
                    "topic": {"type": "string", "description": "Library topic to file the papers under."},
                },
                "required": ["ids", "topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": FINAL_ANSWER,
            "description": (
                "Finish with the answer. Every claim must cite sentence ids that you saw in tool results and that "
                "directly support it; the quotes are inserted automatically from these ids. If the library has no "
                "supporting sentences, set not_found to true instead of answering from memory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "Short answer in the language of the user's question."},
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string", "description": "One claim, in the user's language."},
                                "sentence_ids": {
                                    "type": "array", "items": {"type": "string"},
                                    "description": "Supporting sentence ids, e.g. ['s123', 's124'].",
                                },
                            },
                            "required": ["text", "sentence_ids"],
                        },
                    },
                    "not_found": {"type": "boolean", "description": "True if the library does not answer the question."},
                },
                "required": ["answer", "claims"],
            },
        },
    },
]

FUNCTIONS: dict[str, Callable] = {
    "search_library": search_library,
    "read_passage": read_passage,
    "library_overview": library_overview,
    "find_papers": find_papers,
    "list_entities": list_entities,
    "compare_papers": compare_papers,
    "search_external": search_external,
    "add_papers": add_papers,
}


# Вариант для эксперимента E2: модель пишет цитату текстом сама, без ID. Ответ не проверяется,
# дословность считается после прогона — так видно, сколько цитат модель искажает.
FINAL_ANSWER_QUOTES = {
    "type": "function",
    "function": {
        "name": FINAL_ANSWER,
        "description": (
            "Finish with the answer. Every claim must include a verbatim quote copied exactly, character by character, "
            "from the passages in tool results. If the library has no supporting text, set not_found to true."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "Short answer in the language of the user's question."},
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "quote": {"type": "string", "description": "Exact sentence copied from a passage."},
                        },
                        "required": ["text", "quote"],
                    },
                },
                "not_found": {"type": "boolean"},
            },
            "required": ["answer", "claims"],
        },
    },
}


def available_schemas(ctx: ToolContext) -> list[dict]:
    """Инструменты, которые реально работают в этой сессии: без источников не показываем внешний поиск."""
    hidden = set()
    if not ctx.sources:
        hidden.add("search_external")
    if ctx.add_papers_fn is None or not ctx.allow_add:
        hidden.add("add_papers")
    return [s for s in SCHEMAS if s["function"]["name"] not in hidden]
