"""Цитаты по ID предложений: модель называет только ID, текст цитаты берётся из базы.

Поэтому цитата всегда дословная. Проверяется другое: что предложение существует и что агент действительно
видел его в результатах инструментов, а не угадал номер.
"""

import re
from dataclasses import dataclass

from research_assistant.store.db import Library

SENTENCE_ID_RE = re.compile(r"^\s*\[?s?(\d+)\]?\s*$", re.I)


@dataclass
class Citation:
    sentence_id: int
    text: str
    paper_id: int
    title: str
    year: int | None
    section: str
    page: int | None
    source: str  # fulltext | metadata


def parse_sentence_id(raw) -> int | None:
    """Принимает 52314, "52314", "s52314", "[s52314]"."""
    if isinstance(raw, int):
        return raw
    match = SENTENCE_ID_RE.match(str(raw))
    return int(match.group(1)) if match else None


def load_citations(library: Library, sentence_ids: list[int]) -> dict[int, Citation]:
    if not sentence_ids:
        return {}
    rows = library.conn.execute(
        f"""SELECT se.id, se.text, se.page, p.paper_id, p.source, s.title AS section, pa.title, pa.year
            FROM sentences se JOIN passages p ON p.id = se.passage_id
            JOIN sections s ON s.id = p.section_id JOIN papers pa ON pa.id = p.paper_id
            WHERE se.id IN ({','.join('?' * len(sentence_ids))})""",
        sentence_ids,
    ).fetchall()
    return {
        r["id"]: Citation(r["id"], r["text"], r["paper_id"], r["title"], r["year"], r["section"] or "",
                          r["page"], r["source"])
        for r in rows
    }


def citation_place(citation: Citation) -> str:
    """Откуда цитата: «Название (год), раздел «…», с. N»."""
    where = [f"{citation.title} ({citation.year})" if citation.year else citation.title]
    if citation.section:
        where.append(f"раздел «{citation.section}»")
    if citation.page:
        where.append(f"с. {citation.page}")
    elif citation.source == "metadata":
        where.append("аннотация из метаданных")
    return ", ".join(where)


def format_citation(citation: Citation) -> str:
    return f"«{citation.text}» — {citation_place(citation)} [s{citation.sentence_id}]"
