"""Разбор статей библиотеки: полный текст → разделы → фрагменты → предложения → SQLite."""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from research_assistant.parse.html import cyberleninka_units
from research_assistant.parse.passages import SectionOut, document_passages, paragraph_passages
from research_assistant.parse.pdf import pdf_units
from research_assistant.parse.structure import Document, Paragraph, build_document
from research_assistant.store.db import Library

# Увеличивать при изменении правил разбора: тогда `parse` переразберёт старые статьи.
PARSER_VERSION = 1


@dataclass
class ParseOutcome:
    status: str  # ok | metadata | empty | failed
    parser: str | None
    sections: int = 0
    passages: int = 0
    sentences: int = 0
    error: str | None = None


def parse_fulltext(path: str | Path, kind: str, title: str | None) -> tuple[Document, int | None]:
    if kind == "pdf":
        units, n_pages = pdf_units(path)
    elif kind == "html":
        units, n_pages = cyberleninka_units(Path(path).read_text(encoding="utf-8")), None
    else:
        raise ValueError(f"неизвестный формат полного текста: {kind}")
    return build_document(units, title=title), n_pages


def metadata_sections(abstract: str | None) -> list[SectionOut]:
    """Аннотация из метаданных: чтобы статья без полного текста тоже находилась поиском."""
    if not abstract or not abstract.strip():
        return []
    section = SectionOut("Abstract", "abstract", 1)
    section.passages = paragraph_passages(Paragraph(abstract.strip(), [(0, None)]))
    return [section]


def parse_paper(library: Library, paper) -> ParseOutcome:
    error = None
    if paper["fulltext_status"] == "ok" and paper["fulltext_path"]:
        try:
            document, n_pages = parse_fulltext(paper["fulltext_path"], paper["fulltext_kind"], paper["title"])
            sections = document_passages(document)
            if any(s.passages for s in sections):
                counts = library.save_structure(
                    paper["id"], sections, status="ok", parser=paper["fulltext_kind"], version=PARSER_VERSION,
                    has_abstract=document.has_abstract, n_pages=n_pages,
                )
                return ParseOutcome("ok", paper["fulltext_kind"], **counts)
        except Exception as exc:  # битый PDF не должен останавливать разбор остальных
            error = f"{type(exc).__name__}: {exc}"

    sections = metadata_sections(paper["abstract"])
    status = "failed" if error else ("metadata" if sections else "empty")
    counts = library.save_structure(
        paper["id"], sections, status=status, parser="metadata" if sections else None, version=PARSER_VERSION,
        source="metadata", has_abstract=bool(sections), error=error,
    )
    return ParseOutcome(status, "metadata" if sections else None, error=error, **counts)


def parse_papers(
    library: Library,
    topic_id: int | None = None,
    paper_ids: list[int] | None = None,
    force: bool = False,
    log: Callable[[str], None] = lambda message: None,
) -> Counter:
    papers = library.papers_to_parse(PARSER_VERSION, topic_id=topic_id, paper_ids=paper_ids, force=force)
    statuses = Counter()
    for number, paper in enumerate(papers, 1):
        outcome = parse_paper(library, paper)
        statuses[outcome.status] += 1
        detail = outcome.error or (
            f"{outcome.sections} разд., {outcome.passages} фрагм., {outcome.sentences} предл."
        )
        log(f"[{number}/{len(papers)}] {outcome.status} ({outcome.parser or '-'}): {paper['title'][:60]} — {detail}")
    return statuses
