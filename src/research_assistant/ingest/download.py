"""Скачивание полных текстов: PDF или HTML-страница статьи CyberLeninka.

Если прямой ссылки на PDF нет или она не сработала, открывается страница статьи (адрес из источника или
https://doi.org/<DOI>) и PDF берётся из метатега citation_pdf_url — так находятся тексты в журналах на OJS
и у издательств, для которых OpenAlex знает только страницу.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from research_assistant.ingest.http import FetchError, HttpClient
from research_assistant.ingest.landing import pdf_links
from research_assistant.ingest.robots import RobotsPolicy

# Страница статьи CyberLeninka содержит полный текст в <div class="ocr">; сама страница robots.txt разрешена,
# а /pdf и /api — нет.
CYBERLENINKA_ARTICLE_RE = re.compile(r"^https?://cyberleninka\.ru/article/n/[^/?#]+/?$")
CYBERLENINKA_MARKER = 'class="ocr"'
MAX_BYTES = 50 * 1024 * 1024
MAX_PAGE_PDF_LINKS = 2  # сколько ссылок citation_pdf_url пробовать с одной страницы


@dataclass
class FulltextResult:
    status: str  # ok | failed | unavailable
    path: str | None = None
    kind: str | None = None
    error: str | None = None


def fulltext_candidates(paper) -> list[tuple[str, str]]:
    """Адреса полного текста в порядке предпочтения: arXiv PDF, PDF из источника, страница CyberLeninka."""
    candidates = []
    if paper["arxiv_id"]:
        candidates.append((f"https://arxiv.org/pdf/{paper['arxiv_id']}", "pdf"))
    if paper["pdf_url"]:
        candidates.append((paper["pdf_url"], "pdf"))
    if paper["landing_url"] and CYBERLENINKA_ARTICLE_RE.match(paper["landing_url"]):
        candidates.append((paper["landing_url"].rstrip("/"), "html"))
    return list(dict.fromkeys(candidates))


def _field(paper, name: str):
    """Поле записи статьи: sqlite3.Row или dict, в котором поля может не быть."""
    try:
        return paper[name]
    except (KeyError, IndexError):
        return None


def landing_page(paper) -> str | None:
    """Страница статьи, на которой можно поискать ссылку на PDF: адрес из источника, иначе DOI."""
    landing = _field(paper, "landing_url")
    if landing and landing.lower().startswith(("http://", "https://")):
        return landing
    doi = _field(paper, "doi")
    return f"https://doi.org/{doi}" if doi else None


class FulltextFetcher:
    def __init__(self, http: HttpClient, robots: RobotsPolicy, out_dir: Path, max_bytes: int = MAX_BYTES,
                 discover: bool = True):
        self.http = http
        self.robots = robots
        self.out_dir = out_dir
        self.max_bytes = max_bytes
        self.discover = discover  # искать PDF на странице статьи, если прямые ссылки не сработали

    def fetch(self, paper) -> FulltextResult:
        candidates = fulltext_candidates(paper)
        errors = []
        for url, kind in candidates:
            try:
                content = self._download(url, kind)
            except FetchError as exc:
                errors.append(str(exc))
                continue
            return self._save(paper, content, kind)
        page = landing_page(paper) if self.discover else None
        if page and (page.rstrip("/"), "html") not in candidates:
            try:
                found = self._from_page(page, tried={url for url, _ in candidates})
            except FetchError as exc:
                errors.append(str(exc))
            else:
                if found is not None:
                    return self._save(paper, *found)
                if not candidates:
                    return FulltextResult("unavailable", error=f"на странице статьи нет ссылки на PDF: {page}")
        if not candidates and not errors:
            return FulltextResult("unavailable", error="нет ссылок на полный текст")
        return FulltextResult("failed", error="; ".join(errors))

    def _save(self, paper, content: bytes, kind: str) -> FulltextResult:
        path = self.out_dir / f"{paper['id']}.{kind}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return FulltextResult("ok", path=str(path), kind=kind)

    def _from_page(self, page: str, tried: set[str]) -> tuple[bytes, str] | None:
        """Открывает страницу статьи. Возвращает (содержимое, вид) или None, если текста и ссылок на ней нет.
        Бросает FetchError, если ссылки были, но ни одна не скачалась."""
        response = self._get(page)
        content, final_url = response.content, str(response.url)
        if content.startswith(b"%PDF-"):  # DOI иногда ведёт сразу на файл
            return self._checked(content, "pdf", final_url), "pdf"
        if CYBERLENINKA_ARTICLE_RE.match(final_url) and CYBERLENINKA_MARKER.encode() in content:
            return content, "html"
        links = [link for link in pdf_links(response.text, final_url) if link not in tried][:MAX_PAGE_PDF_LINKS]
        errors = []
        for link in links:
            try:
                return self._download(link, "pdf"), "pdf"
            except FetchError as exc:
                errors.append(str(exc))
        if errors:
            raise FetchError("; ".join(errors))
        return None

    def _get(self, url: str):
        """GET с проверкой robots.txt до запроса и после редиректа; не 200 — FetchError."""
        if not self.robots.allowed(url):
            raise FetchError(f"robots.txt запрещает {url}")
        delay = self.robots.crawl_delay(url)
        if delay:
            self.http.set_interval(urlsplit(url).netloc, delay)

        response = self.http.get(url)
        final_url = str(response.url)
        if final_url != url and not self.robots.allowed(final_url):
            raise FetchError(f"robots.txt запрещает {final_url} (редирект с {url})")
        if response.status_code != 200:
            raise FetchError(f"HTTP {response.status_code}: {url}")
        return response

    def _checked(self, content: bytes, kind: str, url: str, content_type: str = "?") -> bytes:
        if len(content) > self.max_bytes:
            raise FetchError(f"файл больше {self.max_bytes // 2**20} МБ: {url}")
        if kind == "pdf" and not content.startswith(b"%PDF-"):
            raise FetchError(f"по ссылке не PDF ({content_type}): {url}")
        if kind == "html" and CYBERLENINKA_MARKER.encode() not in content:
            raise FetchError(f"на странице нет текста статьи: {url}")
        return content

    def _download(self, url: str, kind: str) -> bytes:
        response = self._get(url)
        return self._checked(response.content, kind, url, response.headers.get("content-type", "?"))
