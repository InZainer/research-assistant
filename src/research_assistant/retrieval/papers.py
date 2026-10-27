"""Поиск статей целиком (а не фрагментов): по автору, по словам из названия и аннотации, по смыслу аннотации.

Слова сравниваются по основам (nlp/text.analyze), поэтому «декодирования» находит «декодирование».
Поиск по смыслу идёт по фрагментам-аннотациям через эмбеддинги и находит статьи на другом языке;
такие статьи идут после совпавших по словам и помечаются отдельно, потому что плотный поиск
всегда что-то возвращает, даже когда подходящих статей нет.
"""

import json
from dataclasses import dataclass, field

from research_assistant.ingest.models import author_matches
from research_assistant.nlp.text import analyze
from research_assistant.store.db import Library, PassageFilter

TITLE_WEIGHT = 2.0
SEMANTIC_PAPERS = 5  # сколько «похожих по смыслу» статей добавлять после совпавших по словам


@dataclass
class PaperHit:
    paper: dict
    score: float
    matched: list[str] = field(default_factory=list)  # title | abstract | semantic


def authors_of(paper) -> list[str]:
    try:
        authors = json.loads(paper["authors"] or "[]")
    except (TypeError, ValueError):
        return []
    return [a for a in authors if isinstance(a, str)]


def find_papers(
    library: Library,
    query: str | None = None,
    author: str | None = None,
    topic: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    language: str | None = None,
    searcher=None,
    limit: int = 30,
) -> list[PaperHit]:
    """Без query — все статьи под фильтрами (новые первыми). С query — статьи, где есть все слова запроса
    (в названии вес выше, чем в аннотации), затем до SEMANTIC_PAPERS похожих по смыслу, если у searcher
    есть эмбеддинги."""
    candidates = [dict(p) for p in library.papers_where(topic, year_from, year_to, language)]
    if author and author.strip():
        candidates = [p for p in candidates if author_matches(authors_of(p), author)]
    terms = list(dict.fromkeys(analyze(query or "")))
    if not terms:
        return [PaperHit(p, 0.0) for p in candidates[:limit]]

    hits = []
    for paper in candidates:
        title_stems = set(analyze(paper["title"] or ""))
        abstract_stems = set(analyze(paper["abstract"] or ""))
        # Нужны все слова запроса: иначе общее слово («модель») тянет наверх статьи на чужую тему.
        if not all(t in title_stems or t in abstract_stems for t in terms):
            continue
        in_title = sum(t in title_stems for t in terms)
        in_abstract = sum(t in abstract_stems for t in terms)
        matched = (["title"] if in_title else []) + (["abstract"] if in_abstract else [])
        hits.append(PaperHit(paper, (TITLE_WEIGHT * in_title + in_abstract) / len(terms), matched))
    hits.sort(key=lambda h: (-h.score, -(h.paper["year"] or 0)))
    hits = hits[:limit]

    if searcher is not None and searcher.dense_ready and candidates and len(hits) < limit:
        found = {h.paper["id"] for h in hits}
        by_id = {p["id"]: p for p in candidates}
        where = PassageFilter(section_kinds=["abstract"], paper_ids=list(by_id))
        added = 0
        for hit in searcher.search(query, k=SEMANTIC_PAPERS * 4, mode="dense", where=where):
            paper_id = hit.passage["paper_id"]
            if paper_id in found:
                continue
            found.add(paper_id)
            hits.append(PaperHit(by_id[paper_id], hit.score, ["semantic"]))
            added += 1
            if added >= SEMANTIC_PAPERS or len(hits) >= limit:
                break
    return hits
