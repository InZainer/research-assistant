"""Конвейер сбора: поиск по источникам → слияние дублей → запись в библиотеку → полные тексты."""

import queue
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from research_assistant.ingest.dedup import dedupe
from urllib.parse import urlsplit

from research_assistant.ingest.download import (
    FulltextFetcher,
    FulltextResult,
    fulltext_candidates,
    landing_page,
)
from research_assistant.ingest.models import PaperRecord, author_matches
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
    off_topic: int = 0  # отброшено фильтром релевантности (ingest/relevance.py)
    paper_ids: list[int] = field(default_factory=list)  # что добавила команда add
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
    author: str | None = None,
) -> SearchResult:
    """Ищет во всех источниках. Сбой одного источника не останавливает остальные.
    С author остаются только записи, где этот человек среди авторов: источники ищут по имени нечётко."""
    result = SearchResult(records=[])
    collected: list[PaperRecord] = []
    extra = {"author": author} if author and author.strip() else {}
    for source in sources:
        try:
            records = source.search(query, limit=limit, year_from=year_from, year_to=year_to, language=language,
                                    **extra)
        except Exception as exc:  # источник недоступен — фиксируем и идём дальше
            result.errors[source.name] = f"{type(exc).__name__}: {exc}"
            continue
        if extra:
            records = [r for r in records if author_matches(r.authors, author)]
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
    queries: list[str] | None = None,
    relevance=None,
    workers: int = 1,
) -> IngestReport:
    """queries — несколько формулировок одной темы (синонимы, русский и английский варианты): каждая ищется
    отдельно, limit действует на каждую, найденное складывается в одну тему. Одна формулировка находит
    только статьи с её словами, поэтому широкую тему собирают набором запросов.
    relevance — TermFilter: записи без слов темы в названии и аннотации в библиотеку не попадают."""
    wanted = [q.strip() for q in (queries or []) if q and q.strip()]
    if query and query.strip():
        wanted.insert(0, query.strip())
    wanted = list(dict.fromkeys(wanted)) or [topic]
    report = IngestReport(topic=topic, query="; ".join(wanted), found={}, errors={})
    topic_id = library.get_or_create_topic(topic)
    seen: set[int] = set()
    for number, one in enumerate(wanted, 1):
        found = search_sources(sources, one, limit, year_from, year_to, language)
        records = found.records
        if relevance is not None:
            records = [r for r in records if relevance.matches(r)]
            report.off_topic += len(found.records) - len(records)
        prefix = f"[запрос {number}/{len(wanted)}] «{one}»: " if len(wanted) > 1 else ""
        log(f"{prefix}Найдено: {found.found or '{}'}, после слияния дублей: {len(found.records)}"
            + (f", по теме: {len(records)}" if relevance is not None else ""))
        for name, error in found.errors.items():
            log(f"Источник {name} недоступен: {error}")
            report.errors[name] = error
        for name, count in found.found.items():
            report.found[name] = report.found.get(name, 0) + count
        for record in records:
            paper_id, created = library.upsert_paper(record)
            library.link_topic(paper_id, topic_id, one)
            if paper_id in seen:
                continue  # эту статью уже нашёл предыдущий запрос
            seen.add(paper_id)
            report.unique += 1
            if created:
                report.added += 1
            else:
                report.already_in_library += 1

    if fetcher is None:
        return report
    statuses = ("pending", "failed") if retry_failed else ("pending",)
    fetch_fulltexts(library, topic_id, fetcher, statuses, report.fulltext, log, workers=workers)
    return report


def _site(paper) -> str:
    """С какого сайта качается статья. У статей без прямой ссылки это издатель из префикса DOI:
    doi.org — только перенаправление, и держать все такие статьи в одной очереди незачем."""
    candidates = fulltext_candidates(paper)
    url = candidates[0][0] if candidates else (landing_page(paper) or "")
    host = urlsplit(url).netloc
    if host == "doi.org":
        return "doi:" + urlsplit(url).path.strip("/").split("/")[0]
    return host


def interleave_by_host(papers: list) -> list:
    """Переставляет статьи так, чтобы соседние качались с разных сайтов. Пауза между запросами считается
    на сайт (у arxiv.org это 15 секунд по robots.txt): пока она идёт, качаем с других сайтов, а не ждём."""
    groups: dict[str, list] = {}
    for paper in papers:
        groups.setdefault(_site(paper), []).append(paper)
    ordered = []
    queues = sorted(groups.values(), key=len, reverse=True)
    while queues:
        ordered.extend(queue.pop(0) for queue in queues)
        queues = [queue for queue in queues if queue]
    return ordered


def split_by_host(papers: list, workers: int) -> list[list]:
    """Делит статьи между потоками так, чтобы один сайт целиком доставался одному потоку (вежливость к сайту
    сохраняется), а нагрузка была примерно равной: самые большие сайты раздаются первыми."""
    groups: dict[str, list] = {}
    for paper in papers:
        groups.setdefault(_site(paper), []).append(paper)
    buckets: list[list] = [[] for _ in range(max(1, workers))]
    for group in sorted(groups.values(), key=len, reverse=True):
        min(buckets, key=len).extend(group)
    return [interleave_by_host(bucket) for bucket in buckets if bucket]


def fetch_fulltexts(library: Library, topic_id: int | None, fetcher: FulltextFetcher, statuses: tuple[str, ...],
                    counter: Counter, log: Callable[[str], None] = lambda message: None,
                    paper_ids: list[int] | None = None, workers: int = 1) -> None:
    """paper_ids — качать только эти статьи темы (команда add не должна трогать остальные ожидающие статьи:
    их может в это время качать ingest). workers — число потоков; в базу пишет только вызывающий поток."""
    papers = library.papers_for_fulltext(topic_id, statuses)
    if paper_ids is not None:
        papers = [p for p in papers if p["id"] in set(paper_ids)]
    total = len(papers)

    def save(number: int, paper, result) -> None:
        library.set_fulltext(paper["id"], result.status, result.path, result.kind, result.error)
        counter[result.status] += 1
        detail = result.kind if result.status == "ok" else result.error
        log(f"[{number}/{total}] {result.status}: {paper['title'][:70]} ({detail})")

    if workers <= 1 or total < 2:
        for number, paper in enumerate(interleave_by_host(papers), 1):
            save(number, paper, fetcher.fetch(paper))
        return

    done: queue.Queue = queue.Queue()

    def run(bucket: list) -> None:
        for paper in bucket:
            try:
                result = fetcher.fetch(paper)
            except Exception as exc:  # сбой на одной статье не останавливает поток
                result = FulltextResult("failed", error=f"{type(exc).__name__}: {exc}")
            done.put((paper, result))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for bucket in split_by_host(papers, workers):
            pool.submit(run, bucket)
        for number in range(1, total + 1):
            save(number, *done.get())


def add_references(
    library: Library,
    references: list[str],
    topic: str,
    resolve: Callable[[str], PaperRecord],
    fetcher: FulltextFetcher | None = None,
    log: Callable[[str], None] = lambda message: None,
) -> IngestReport:
    """Добавляет в тему статьи по ссылкам (DOI, arXiv, адрес страницы). resolve — ingest.resolve.resolve_reference
    с уже подставленными клиентами; ссылка, которую не удалось разобрать, попадает в report.errors."""
    report = IngestReport(topic=topic, query="add", found={}, errors={})
    topic_id = library.get_or_create_topic(topic)
    added: list[int] = []
    for ref in dict.fromkeys(r.strip() for r in references if r.strip()):
        try:
            record = resolve(ref)
        except Exception as exc:
            report.errors[ref] = f"{type(exc).__name__}: {exc}"
            log(f"не добавлена {ref}: {exc}")
            continue
        paper_id, created = library.upsert_paper(record)
        library.link_topic(paper_id, topic_id, "add")
        added.append(paper_id)
        report.unique += 1
        report.added += created
        report.already_in_library += not created
        log(f"{'новая' if created else 'уже была'} #{paper_id}: {record.title[:80]}")
    if fetcher is not None:
        fetch_fulltexts(library, topic_id, fetcher, ("pending", "failed"), report.fulltext, log, paper_ids=added)
    report.paper_ids = added
    return report
