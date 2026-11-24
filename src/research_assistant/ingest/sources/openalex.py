"""Поиск статей через OpenAlex API.

Документация: https://docs.openalex.org. Через OpenAlex находим и русскоязычные статьи (фильтр language:ru),
в том числе размещённые на CyberLeninka: её собственный поиск и API закрыты для роботов в robots.txt.
"""

import re

from research_assistant.ingest.http import HttpClient
from research_assistant.ingest.models import PaperRecord, author_matches, normalize_arxiv_id

API_URL = "https://api.openalex.org/works"
AUTHORS_URL = "https://api.openalex.org/authors"
API_HOST = "api.openalex.org"
MIN_INTERVAL = 0.2
SELECT = (
    "id,doi,display_name,publication_year,language,type,authorships,"
    "abstract_inverted_index,primary_location,best_oa_location,locations"
)
MAX_PER_PAGE = 200
# Потолок на один запрос: глубже 2000 результатов поиск OpenAlex отдаёт в основном шум.
MAX_LIMIT = 2000
# Один человек в OpenAlex часто разбит на несколько профилей («Geoffrey E. Hinton», «Hinton Geoffrey»).
MAX_AUTHOR_PROFILES = 5


def build_params(
    query: str,
    limit: int,
    year_from: int | None = None,
    year_to: int | None = None,
    language: str | None = None,
    author_ids: list[str] | None = None,
    open_access_only: bool = True,
    country: str | None = None,
) -> dict:
    """По умолчанию только статьи в открытом доступе: у них есть полный текст. open_access_only=False —
    ещё и закрытые: в библиотеку попадёт их аннотация (см. OpenAlexSource.search).
    С author_ids — работы этих профилей авторов; без запроса они сортируются по цитированиям.
    country — двухбуквенный код страны организаций авторов (RU): так находятся работы российских авторов,
    у которых название и аннотация в OpenAlex записаны по-английски и русским запросом не ищутся."""
    filters = ["is_oa:true"] if open_access_only else []
    if country:
        filters.append(f"authorships.institutions.country_code:{country.upper()}")
    if author_ids:
        filters.append("author.id:" + "|".join(author_ids))
    if year_from and year_to:
        filters.append(f"publication_year:{year_from}-{year_to}")
    elif year_from:
        filters.append(f"publication_year:>{year_from - 1}")
    elif year_to:
        filters.append(f"publication_year:<{year_to + 1}")
    if language:
        filters.append(f"language:{language}")
    params = {"per_page": min(limit, MAX_PER_PAGE), "select": SELECT}
    if filters:
        params["filter"] = ",".join(filters)
    if query and query.strip():
        params["search"] = query
    else:
        params["sort"] = "cited_by_count:desc"
    return params


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

    def __init__(self, http: HttpClient, mailto: str | None = None, api_key: str | None = None,
                 open_access_only: bool = True, country: str | None = None):
        self.http = http
        self.mailto = mailto
        self.api_key = api_key
        self.open_access_only = open_access_only
        self.country = country
        http.set_interval(API_HOST, MIN_INTERVAL)

    def search(
        self,
        query: str,
        limit: int = 20,
        year_from: int | None = None,
        year_to: int | None = None,
        language: str | None = None,
        author: str | None = None,
    ) -> list[PaperRecord]:
        author_ids = None
        if author and author.strip():
            author_ids = self.author_ids(author)
            if not author_ids:
                return []
        limit = max(1, min(int(limit), MAX_LIMIT))
        params = self._auth(build_params(query, limit, year_from, year_to, language, author_ids,
                                         self.open_access_only, self.country))
        if limit <= MAX_PER_PAGE:
            works = self.http.get_json(API_URL, params).get("results", [])
        else:
            works = self._all_pages(params, limit)
        records = [r for r in (parse_work(w) for w in works) if r is not None]
        if not self.open_access_only:
            # Закрытая статья без аннотации — одно название: искать и цитировать в ней нечего.
            records = [r for r in records if r.abstract or r.pdf_url or r.arxiv_id]
        return records

    def _all_pages(self, params: dict, limit: int) -> list[dict]:
        """Больше 200 результатов — постранично по курсору (cursor=* → meta.next_cursor)."""
        works: list[dict] = []
        cursor = "*"
        while cursor and len(works) < limit:
            data = self.http.get_json(API_URL, {**params, "per_page": MAX_PER_PAGE, "cursor": cursor})
            page = data.get("results", [])
            works.extend(page)
            cursor = (data.get("meta") or {}).get("next_cursor")
            if not page:
                break
        return works[:limit]

    def by_doi(self, doi: str) -> PaperRecord | None:
        """Одна работа по DOI (для команды add); None — OpenAlex её не знает."""
        params = self._auth({"filter": f"doi:{doi}", "per_page": 1, "select": SELECT})
        works = self.http.get_json(API_URL, params).get("results", [])
        return parse_work(works[0]) if works else None

    def author_ids(self, name: str) -> list[str]:
        """Профили авторов OpenAlex с этим именем. Поиск авторов нечёткий («Geoffrey Hinton» находит и Yoshua
        Bengio), поэтому имя профиля сверяется с запросом."""
        params = self._auth({"search": name, "per_page": 25, "select": "id,display_name,works_count"})
        data = self.http.get_json(AUTHORS_URL, params)
        profiles = [a for a in data.get("results", []) if author_matches([a.get("display_name") or ""], name)]
        profiles.sort(key=lambda a: -(a.get("works_count") or 0))
        return [a["id"].rsplit("/", 1)[-1] for a in profiles[:MAX_AUTHOR_PROFILES] if a.get("id")]

    def _auth(self, params: dict) -> dict:
        if self.mailto:
            params["mailto"] = self.mailto
        if self.api_key:
            params["api_key"] = self.api_key
        return params
