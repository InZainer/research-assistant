"""Статья по ссылке: DOI, идентификатор arXiv или адрес страницы статьи → PaperRecord.

Нужно для статей, которых поиск по источникам не находит: CyberLeninka закрывает свой поиск в robots.txt
(страницы статей при этом открыты), а многие русскоязычные журналы плохо ищутся в OpenAlex по теме.
Такую статью находят вручную и добавляют командой `add`.
"""

from research_assistant.ingest.http import FetchError, HttpClient
from research_assistant.ingest.landing import record_from_page
from research_assistant.ingest.models import PaperRecord, normalize_arxiv_id, normalize_doi
from research_assistant.ingest.robots import RobotsPolicy


def _page_record(url: str, http: HttpClient, robots: RobotsPolicy) -> PaperRecord:
    if not robots.allowed(url):
        raise FetchError(f"robots.txt запрещает {url}")
    response = http.get(url)
    if response.status_code != 200:
        raise FetchError(f"HTTP {response.status_code}: {url}")
    final_url = str(response.url)
    if final_url != url and not robots.allowed(final_url):
        raise FetchError(f"robots.txt запрещает {final_url} (редирект с {url})")
    record = record_from_page(response.text, final_url.split("#")[0])
    if record is None:
        raise FetchError(f"на странице нет метаданных статьи (citation_title): {url}")
    return record


def resolve_reference(ref: str, http: HttpClient, robots: RobotsPolicy, openalex=None) -> PaperRecord:
    """ref — DOI («10.1234/x», «https://doi.org/…»), arXiv («2402.01528», ссылка arxiv.org) или адрес страницы
    статьи. Сначала спрашиваем OpenAlex (у него готовые аннотация и ссылки открытого доступа), иначе читаем
    метатеги страницы. Не получилось — FetchError с причиной."""
    ref = ref.strip()
    if not ref:
        raise FetchError("пустая ссылка")
    is_url = ref.lower().startswith(("http://", "https://"))
    arxiv_id = normalize_arxiv_id(ref)
    doi = f"10.48550/arxiv.{arxiv_id}" if arxiv_id else normalize_doi(ref)
    if doi and openalex is not None:
        try:
            record = openalex.by_doi(doi)
        except Exception:  # OpenAlex недоступен — остаётся страница статьи
            record = None
        if record is not None:
            return record
    if arxiv_id:
        return _page_record(f"https://arxiv.org/abs/{arxiv_id}", http, robots)
    if doi:
        return _page_record(f"https://doi.org/{doi}", http, robots)
    if is_url:
        return _page_record(ref, http, robots)
    raise FetchError(f"не похоже на DOI, идентификатор arXiv или адрес страницы: {ref}")
