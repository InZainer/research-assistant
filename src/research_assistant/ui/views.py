"""Данные для страниц веб-интерфейса: карточки цитат, статей и фрагментов, подписи к шагам агента, тексты ошибок.

Здесь нет Streamlit, поэтому всё проверяется обычными тестами; app.py только раскладывает это по виджетам.
"""

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

from research_assistant.agent.loop import AgentResult
from research_assistant.answer.citations import Citation, citation_place
from research_assistant.ingest.models import PaperRecord
from research_assistant.nlp.text import analyze
from research_assistant.retrieval.papers import PaperHit, authors_of
from research_assistant.store.db import Library

MARKDOWN_SPECIAL = set("\\`*_{}[]<>()#+-.!|$~")

SECTION_LABELS = {
    "abstract": "Аннотация", "introduction": "Введение", "related_work": "Обзор литературы", "method": "Метод",
    "experiments": "Эксперименты", "results": "Результаты", "discussion": "Обсуждение", "conclusion": "Заключение",
    "limitations": "Ограничения", "appendix": "Приложение", "other": "Другое", "body": "Текст без разделов",
}
ENTITY_LABELS = {"method": "методы", "dataset": "датасеты", "metric": "метрики", "task": "задачи"}
LANGUAGE_LABELS = {"en": "английский", "ru": "русский"}


def md_escape(text: str) -> str:
    """Текст статьи как есть: без экранирования «$…$» стал бы формулой, а «*» — курсивом."""
    return "".join("\\" + ch if ch in MARKDOWN_SPECIAL else ch for ch in text)


def highlight(text: str, query: str | None) -> str:
    """Markdown с подсвеченными словами запроса (фон, а не жирный: так видно и внутри заголовка);
    слова сравниваются по основам, так что «модели» находит «модель»."""
    stems = set(analyze(query or ""))
    if not stems:
        return md_escape(text)
    parts = re.split(r"(\w+)", text)
    return "".join(
        f":orange-background[{md_escape(part)}]" if i % 2 and stems & set(analyze(part)) else md_escape(part)
        for i, part in enumerate(parts)
    )


def source_link(paper: Mapping, page: int | None = None) -> str | None:
    """Ссылка на источник; для PDF — сразу на страницу (#page=N понимают браузеры и просмотрщики PDF)."""
    anchor = f"#page={page}" if page else ""
    if paper.get("pdf_url"):
        return paper["pdf_url"] + anchor
    if paper.get("arxiv_id"):
        return f"https://arxiv.org/pdf/{paper['arxiv_id']}{anchor}"
    if paper.get("landing_url"):
        return paper["landing_url"]
    if paper.get("doi"):
        return f"https://doi.org/{paper['doi']}"
    return None


def link_label(link: str, page: int | None) -> str:
    return f"Открыть PDF на с. {page}" if page and "#page=" in link else "Открыть источник"


def short_authors(authors: list[str], limit: int = 3) -> str:
    if not authors:
        return "авторы не указаны"
    return ", ".join(authors[:limit]) + (" и др." if len(authors) > limit else "")


def sentence_context(library: Library, sentence_id: int) -> tuple[str, str, str] | None:
    """Фрагмент вокруг цитаты: (текст до, само предложение, текст после) — для подсветки в контексте."""
    row = library.conn.execute(
        """SELECT se.char_start, se.char_end, p.text FROM sentences se JOIN passages p ON p.id = se.passage_id
           WHERE se.id = ?""",
        (sentence_id,),
    ).fetchone()
    if row is None:
        return None
    text, start, end = row["text"], row["char_start"], row["char_end"]
    return text[:start], text[start:end], text[end:]


# --- ответ агента ---

@dataclass
class CitationCard:
    sentence_id: int
    quote: str
    where: str  # «Название (год), раздел «…», с. N»
    link: str | None
    context: tuple[str, str, str] | None
    page: int | None = None


def citation_card(library: Library, citation: Citation) -> CitationCard:
    paper = library.get_paper(citation.paper_id)
    return CitationCard(
        sentence_id=citation.sentence_id,
        quote=citation.text,
        where=citation_place(citation),
        link=source_link(dict(paper), citation.page) if paper else None,
        context=sentence_context(library, citation.sentence_id),
        page=citation.page,
    )


@dataclass
class AnswerView:
    question: str
    answer: str
    verified: bool  # ответ сдан через final_answer и его цитаты проверены
    not_found: bool
    claims: list[tuple[str, list[CitationCard]]]
    stats: str
    steps: list[str] = field(default_factory=list)  # что делал агент, человеческими словами
    seconds: float = 0.0
    found_papers: list[dict] = field(default_factory=list)  # статьи из search_external (PaperRecord как dict)


def answer_view(library: Library, result: AgentResult, steps: list[str] | None = None,
                found: list[PaperRecord] | None = None) -> AnswerView:
    """found — кандидаты из интернета, которых агент нашёл по просьбе пользователя: их показывают под ответом
    с кнопкой «Добавить», потому что сам агент добавлять статьи без явной просьбы не может."""
    usage = result.usage
    stats = f"шагов {result.steps}, токенов {usage.total}, {result.seconds:.0f} с"
    return AnswerView(
        question=result.question,
        answer=result.answer or "(ассистент не дал ответа)",
        verified=result.stop_reason == "final_answer",
        not_found=result.not_found,
        claims=[(claim.text, [citation_card(library, c) for c in claim.citations]) for claim in result.claims],
        stats=stats,
        steps=list(steps or []),
        seconds=result.seconds,
        found_papers=[asdict(r) for r in found or []],
    )


def _arguments(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {"query": str(raw)}  # поиск по самому вопросу (prefetch) пишется в журнал строкой
    return parsed if isinstance(parsed, dict) else {}


def describe_event(event: str, data: dict) -> str | None:
    """Шаг агента для пользователя: «Ищу в библиотеке: …» вместо имени инструмента и JSON."""
    if event == "nudge":
        return "Прошу модель оформить ответ с цитатами"
    if event != "tool":
        return None
    name, args, error = data.get("name"), _arguments(data.get("arguments")), data.get("error")
    if name == "final_answer":
        return "Ответ отклонён: не все цитаты подтверждены, модель исправляет" if error else "Ответ готов, цитаты проверены"
    if name == "search_library":
        text = f"Ищу в библиотеке: «{args.get('query', '')}»"
        if args.get("paper_ids"):
            text += " в выбранных статьях"
    elif name == "read_passage":
        text = "Читаю абзац целиком, вместе с соседними"
    elif name == "find_papers":
        parts = []
        if args.get("author"):
            parts.append(f"автор {args['author']}")
        if args.get("query"):
            parts.append(f"«{args['query']}»")
        text = "Ищу статьи: " + ", ".join(parts) if parts else "Ищу статьи"
    elif name == "library_overview":
        text = "Смотрю, что есть в библиотеке"
    elif name == "list_entities":
        text = f"Смотрю, какие {ENTITY_LABELS.get(args.get('type'), 'сущности')} встречаются в статьях"
    elif name == "compare_papers":
        text = f"Сравниваю статьи по аспекту «{args.get('aspect', '')}»"
    elif name == "search_external":
        parts = []
        if args.get("author"):
            parts.append(f"автор {args['author']}")
        if args.get("query"):
            parts.append(f"«{args['query']}»")
        text = "Ищу новые статьи в OpenAlex и arXiv: " + ", ".join(parts) if parts else "Ищу новые статьи в интернете"
    elif name == "add_papers":
        text = "Скачиваю и разбираю новые статьи — это может занять несколько минут"
    else:
        text = f"Вызываю {name}"
    return text + (" — не получилось, модель попробует иначе" if error else "")


def friendly_error(error: Exception, base_url: str | None, model: str | None) -> str:
    """Что сказать пользователю, если агент упал: чаще всего не запущен сервер модели."""
    kind = type(error).__name__
    if kind in ("APIConnectionError", "ConnectError", "ConnectionError", "APITimeoutError"):
        return (f"Не удалось связаться с моделью по адресу {base_url}. Запустите LM Studio, включите в нём сервер "
                f"и загрузите модель {model}, затем задайте вопрос ещё раз.")
    if kind == "NotFoundError":
        return f"Сервер модели работает, но модель {model} не найдена. Загрузите её в LM Studio или поправьте MODEL в .env."
    if kind == "BadRequestError" and "context" in str(error).lower():
        return ("Модели не хватило контекста. Загрузите её с контекстом побольше: "
                "lms load qwen3.5-4b --context-length 16384, либо начните новый чат.")
    return f"Ассистент не смог ответить: {kind}: {error}"


# --- поиск ---

def hit_rows(library: Library, hits) -> list[dict]:
    """Результаты поиска фрагментов для карточек: статья, раздел, страница, ссылка, текст."""
    rows = []
    for place, hit in enumerate(hits, 1):
        p = hit.passage
        paper = library.get_paper(p["paper_id"])
        rows.append({
            "place": place,
            "passage_id": hit.passage_id,
            "paper_id": p["paper_id"],
            "title": p["title"],
            "year": p["year"],
            "section": p["section"] or SECTION_LABELS.get(p["section_kind"], ""),
            "section_kind": p["section_kind"],
            "page": p["page"],
            "link": source_link(dict(paper), p["page"]) if paper else None,
            "text": p["text"],
        })
    return rows


MATCH_LABELS = {"title": "совпадение в названии", "abstract": "совпадение в аннотации", "semantic": "похожа по смыслу"}


def paper_card(library: Library, hit: PaperHit) -> dict:
    """Карточка статьи для раздела «Статьи»: авторы, год, язык, есть ли полный текст, где совпал запрос."""
    p = hit.paper
    parse = library.get_parse(p["id"])
    return {
        "id": p["id"],
        "title": p["title"],
        "authors": short_authors(authors_of(p)),
        "all_authors": authors_of(p),
        "year": p["year"],
        "venue": p["venue"],
        "language": LANGUAGE_LABELS.get(p["language"], ""),
        "abstract": p["abstract"] or "",
        "fulltext": bool(parse and parse["status"] == "ok"),
        "topics": library.paper_topics(p["id"]),
        "matched": [MATCH_LABELS[m] for m in hit.matched],
        "link": source_link(p),
    }


def external_card(library: Library, record: PaperRecord) -> dict:
    """Карточка статьи из OpenAlex/arXiv: ещё не в библиотеке (или уже есть — тогда добавлять незачем)."""
    return {
        "id": record.external_id,
        "title": record.title,
        "authors": short_authors(record.authors, limit=4),
        "year": record.year,
        "venue": record.venue,
        "language": LANGUAGE_LABELS.get(record.language, ""),
        "abstract": record.abstract or "",
        "in_library": library.find_paper(record) is not None,
        "has_pdf": bool(record.pdf_url or record.arxiv_id),
        "link": source_link({"pdf_url": record.pdf_url, "arxiv_id": record.arxiv_id,
                             "landing_url": record.landing_url, "doi": record.doi}),
    }
