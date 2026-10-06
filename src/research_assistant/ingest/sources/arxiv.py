"""Поиск статей через arXiv API (Atom).

Документация: https://info.arxiv.org/help/api/user-manual.html. arXiv просит не чаще одного запроса
в три секунды и при превышении отвечает 429, поэтому пауза для хоста задаётся в HttpClient.

На практике API отвечает 429 и таймаутами и при редких запросах (общий лимит на IP), поэтому повторов
немного: метаданные тех же статей приходят и через OpenAlex, а PDF качаются с arxiv.org, не через API.
"""

import re
import xml.etree.ElementTree as ET

from research_assistant.ingest.http import FetchError, HttpClient
from research_assistant.ingest.models import PaperRecord, normalize_arxiv_id

API_URL = "https://export.arxiv.org/api/query"
API_HOST = "export.arxiv.org"
MIN_INTERVAL = 3.5
BACKOFF = (20,)
TIMEOUT = 20
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def build_query(query: str, year_from: int | None = None, year_to: int | None = None) -> str:
    """Слова через AND по всем полям; фраза в кавычках ищется целиком."""
    parts = []
    for token in re.findall(r'"[^"]+"|\S+', query):
        phrase = token.startswith('"')
        text = re.sub(r'[():\[\]"]', " ", token).strip()
        if not text:
            continue
        parts.append(f'all:"{text}"' if phrase or " " in text else f"all:{text}")
    if not parts:
        raise ValueError("пустой запрос")
    search = " AND ".join(parts)
    if year_from or year_to:
        start = f"{year_from or 1991}01010000"
        end = f"{year_to or 2100}12312359"
        search += f" AND submittedDate:[{start} TO {end}]"
    return search


def _text(element: ET.Element, path: str) -> str | None:
    found = element.find(path, NS)
    if found is None or not found.text:
        return None
    return " ".join(found.text.split())


def parse_feed(xml_text: str) -> list[PaperRecord]:
    root = ET.fromstring(xml_text)
    records = []
    for entry in root.findall("a:entry", NS):
        entry_id = _text(entry, "a:id") or ""
        if "api/errors" in entry_id:
            raise FetchError(f"arXiv вернул ошибку запроса: {_text(entry, 'a:summary')}")
        arxiv_id = normalize_arxiv_id(entry_id)
        title = _text(entry, "a:title")
        if not arxiv_id or not title:
            continue
        published = _text(entry, "a:published") or ""
        records.append(
            PaperRecord(
                title=title,
                sources=["arxiv"],
                authors=[a for a in (_text(x, "a:name") for x in entry.findall("a:author", NS)) if a],
                year=int(published[:4]) if published[:4].isdigit() else None,
                abstract=_text(entry, "a:summary"),
                doi=_text(entry, "arxiv:doi"),
                arxiv_id=arxiv_id,
                venue=_text(entry, "arxiv:journal_ref") or "arXiv",
                language="en",
                landing_url=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
            )
        )
    return records


class ArxivSource:
    name = "arxiv"

    def __init__(self, http: HttpClient):
        self.http = http
        http.set_interval(API_HOST, MIN_INTERVAL)
        http.set_backoff(API_HOST, BACKOFF)

    def search(
        self,
        query: str,
        limit: int = 20,
        year_from: int | None = None,
        year_to: int | None = None,
        language: str | None = None,
    ) -> list[PaperRecord]:
        if language not in (None, "en"):
            return []  # arXiv почти целиком англоязычный и не фильтрует по языку
        params = {
            "search_query": build_query(query, year_from, year_to),
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        return parse_feed(self.http.get_text(API_URL, params, timeout=TIMEOUT))
