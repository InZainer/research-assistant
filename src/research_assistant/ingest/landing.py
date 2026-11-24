"""Страница статьи на сайте журнала: метаданные и ссылка на PDF из метатегов.

Почти все научные сайты (OJS-журналы, CyberLeninka, arXiv, издательства) размечают страницу статьи метатегами
Highwire Press для Google Scholar: citation_title, citation_author, citation_pdf_url… По ним находим PDF,
когда источник дал только адрес страницы, и собираем запись о статье, которой нет в OpenAlex.
"""

import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from research_assistant.ingest.models import PaperRecord
from research_assistant.nlp.text import detect_language

YEAR_RE = re.compile(r"(?:19|20)\d{2}")
ABSTRACT_TAGS = ("citation_abstract", "dc.description", "description", "og:description")
MIN_ABSTRACT_CHARS = 80  # короче — не аннотация, а подпись сайта


def _meta(soup: BeautifulSoup, *names: str) -> list[str]:
    """Значения метатегов с этими именами (name или property, без учёта регистра), в порядке страницы."""
    wanted = {n.lower() for n in names}
    values = []
    for tag in soup.find_all("meta"):
        key = (tag.get("name") or tag.get("property") or "").strip().lower()
        content = " ".join((tag.get("content") or "").split())
        if key in wanted and content:
            values.append(content)
    return values


def pdf_links(html: str, base_url: str) -> list[str]:
    """Ссылки на PDF из citation_pdf_url, абсолютные, без повторов."""
    soup = BeautifulSoup(html, "html.parser")
    links = [urljoin(base_url, value) for value in _meta(soup, "citation_pdf_url")]
    return [link for link in dict.fromkeys(links) if link.lower().startswith(("http://", "https://"))]


def record_from_page(html: str, url: str) -> PaperRecord | None:
    """Запись о статье по метатегам страницы; None — на странице нет citation_title."""
    soup = BeautifulSoup(html, "html.parser")
    titles = _meta(soup, "citation_title")
    if not titles:
        return None
    authors = []
    for value in _meta(soup, "citation_author", "citation_authors"):
        authors.extend(a.strip() for a in value.split(";") if a.strip())
    dates = _meta(soup, "citation_publication_date", "citation_date", "citation_online_date", "citation_year")
    year = next((int(m.group()) for m in (YEAR_RE.search(d) for d in dates) if m), None)
    abstract = None
    for tag in ABSTRACT_TAGS:  # по убыванию надёжности: og:description бывает общим для всего сайта
        found = [v for v in _meta(soup, tag) if len(v) >= MIN_ABSTRACT_CHARS]
        if found:
            abstract = max(found, key=len)
            break
    languages = [v[:2].lower() for v in _meta(soup, "citation_language", "dc.language") if v[:2].isalpha()]
    pdfs = pdf_links(html, url)
    return PaperRecord(
        title=titles[0],
        sources=["page"],
        authors=list(dict.fromkeys(authors)),
        year=year,
        abstract=abstract,
        doi=next(iter(_meta(soup, "citation_doi")), None),
        arxiv_id=next(iter(_meta(soup, "citation_arxiv_id")), None),
        venue=next(iter(_meta(soup, "citation_journal_title", "citation_conference_title")), None),
        language=languages[0] if languages else detect_language(f"{titles[0]} {abstract or ''}"),
        landing_url=url,
        pdf_url=pdfs[0] if pdfs else None,
    )
