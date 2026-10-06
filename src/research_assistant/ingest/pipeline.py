"""Конвейер сбора: поиск по источникам → слияние дублей → запись в библиотеку → полные тексты."""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from research_assistant.ingest.dedup import dedupe
from research_assistant.ingest.download import FulltextFetcher
from research_assistant.ingest.models import PaperRecord
from research_assistant.ingest.sources import Source
from research_assistant.store.db import Library


@dataclass
class SearchResult:
    records: list[PaperRecord]
    found: dict[str, int] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


@dataclass
class IngestReport:
    topic: str
    query: str
    found: dict[str, int]
    errors: dict[str, str]
    unique: int = 0
    added: int = 0
    already_in_library: int = 0
    fulltext: Counter = field(default_factory=Counter)


def search_sources(
    sources: list[Source],
    query: str,
    limit: int = 20,
    year_from: int | None = None,
    year_to: int | None = None,
    language: str | None = None,
) -> SearchResult:
    """Ищет во всех источниках. Сбой одного источника не останавливает остальные."""
    result = SearchResult(records=[])
    collected: list[PaperRecord] = []
    for source in sources:
        try:
            records = source.search(query, limit=limit, year_from=year_from, year_to=year_to, language=language)
        except Exception as exc:  # источник недоступен — фиксируем и идём дальше
            result.errors[source.name] = f"{type(exc).__name__}: {exc}"
            continue
        result.found[source.name] = len(records)
        collected.extend(records)
    result.records = dedupe(collected)
    return result


def ingest_topic(
    library: Library,
    sources: list[Source],
    topic: str,
    query: str | None = None,
    limit: int = 20,
    year_from: int | None = None,
    year_to: int | None = None,
    language: str | None = None,
    fetcher: FulltextFetcher | None = None,
    retry_failed: bool = False,
    log: Callable[[str], None] = lambda message: None,
) -> IngestReport:
    query = query or topic
    found = search_sources(sources, query, limit, year_from, year_to, language)
    report = IngestReport(topic=topic, query=query, found=found.found, errors=found.errors, unique=len(found.records))
    log(f"Найдено: {found.found or '{}'}, после слияния дублей: {len(found.records)}")
    for name, error in found.errors.items():
        log(f"Источник {name} недоступен: {error}")

    topic_id = library.get_or_create_topic(topic)
    for record in found.records:
        paper_id, created = library.upsert_paper(record)
        library.link_topic(paper_id, topic_id, query)
        if created:
            report.added += 1
        else:
            report.already_in_library += 1

    if fetcher is None:
        return report
    statuses = ("pending", "failed") if retry_failed else ("pending",)
    papers = library.papers_for_fulltext(topic_id, statuses)
    for number, paper in enumerate(papers, 1):
        result = fetcher.fetch(paper)
        library.set_fulltext(paper["id"], result.status, result.path, result.kind, result.error)
        report.fulltext[result.status] += 1
        detail = result.kind if result.status == "ok" else result.error
        log(f"[{number}/{len(papers)}] {result.status}: {paper['title'][:70]} ({detail})")
    return report
