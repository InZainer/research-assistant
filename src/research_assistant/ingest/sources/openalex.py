"""Поиск статей через OpenAlex API.

Документация: https://docs.openalex.org. Через OpenAlex находим и русскоязычные статьи (фильтр language:ru),
в том числе размещённые на CyberLeninka: её собственный поиск и API закрыты для роботов в robots.txt.
"""

import re

from research_assistant.ingest.http import HttpClient
from research_assistant.ingest.models import PaperRecord, normalize_arxiv_id

API_URL = "https://api.openalex.org/works"
API_HOST = "api.openalex.org"
MIN_INTERVAL = 0.2
SELECT = (
    "id,doi,display_name,publication_year,language,type,authorships,"
    "abstract_inverted_index,primary_location,best_oa_location,locations"
)
MAX_PER_PAGE = 200


def build_params(
    query: str,
    limit: int,
    year_from: int | None = None,
    year_to: int | None = None,
    language: str | None = None,
) -> dict:
    """Только статьи в открытом доступе: без полного текста они проекту не нужны."""
    filters = ["is_oa:true"]
    if year_from and year_to:
        filters.append(f"publication_year:{year_from}-{year_to}")
    elif year_from:
        filters.append(f"publication_year:>{year_from - 1}")
    elif year_to:
        filters.append(f"publication_year:<{year_to + 1}")
    if language:
        filters.append(f"language:{language}")
    return {"search": query, "filter": ",".join(filters), "per_page": min(limit, MAX_PER_PAGE), "select": SELECT}


def rebuild_abstract(inverted: dict[str, list[int]] | None) -> str | None:
    """OpenAlex хранит аннотацию как «слово → позиции»; собираем обратно в текст."""
    if not inverted:
        return None
    positions = {pos: word for word, places in inverted.items() for pos in places}
    return " ".join(positions[i] for i in sorted(positions)) or None


def parse_work(work: dict) -> PaperRecord | None:
    title = re.sub(r"<[^>]+>", "", work.get("display_name") or "").strip()
    if not title:
        return None
    best = work.get("best_oa_location") or {}
    primary = work.get("primary_location") or {}
    locations = work.get("locations") or []

    pdf_url = best.get("pdf_url") or next((loc["pdf_url"] for loc in locations if loc.get("pdf_url")), None)
    arxiv_id = None
    for loc in [best, primary, *locations]:
        arxiv_id = normalize_arxiv_id(loc.get("landing_page_url")) or normalize_arxiv_id(loc.get("pdf_url"))
        if arxiv_id:
            break

    return PaperRecord(
        title=title,
        sources=["openalex"],
        authors=[a["author"]["display_name"] for a in work.get("authorships") or [] if a.get("author")],
        year=work.get("publication_year"),
        abstract=rebuild_abstract(work.get("abstract_inverted_index")),
        doi=work.get("doi"),
        arxiv_id=arxiv_id,
        openalex_id=(work.get("id") or "").rsplit("/", 1)[-1] or None,
        venue=(primary.get("source") or {}).get("display_name"),
        language=work.get("language"),
        landing_url=best.get("landing_page_url") or primary.get("landing_page_url"),
        pdf_url=pdf_url,
    )


class OpenAlexSource:
    name = "openalex"

    def __init__(self, http: HttpClient, mailto: str | None = None, api_key: str | None = None):
        self.http = http
        self.mailto = mailto
        self.api_key = api_key
        http.set_interval(API_HOST, MIN_INTERVAL)

    def search(
        self,
        query: str,
        limit: int = 20,
        year_from: int | None = None,
        year_to: int | None = None,
        language: str | None = None,
    ) -> list[PaperRecord]:
        params = build_params(query, limit, year_from, year_to, language)
        if self.mailto:
            params["mailto"] = self.mailto
        if self.api_key:
            params["api_key"] = self.api_key
        data = self.http.get_json(API_URL, params)
        return [r for r in (parse_work(w) for w in data.get("results", [])) if r is not None]
