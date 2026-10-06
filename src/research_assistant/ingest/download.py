"""Скачивание полных текстов: PDF или HTML-страница статьи CyberLeninka."""

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from research_assistant.ingest.http import FetchError, HttpClient
from research_assistant.ingest.robots import RobotsPolicy

# Страница статьи CyberLeninka содержит полный текст в <div class="ocr">; сама страница robots.txt разрешена,
# а /pdf и /api — нет.
CYBERLENINKA_ARTICLE_RE = re.compile(r"^https?://cyberleninka\.ru/article/n/[^/?#]+/?$")
CYBERLENINKA_MARKER = 'class="ocr"'
MAX_BYTES = 50 * 1024 * 1024


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


class FulltextFetcher:
    def __init__(self, http: HttpClient, robots: RobotsPolicy, out_dir: Path, max_bytes: int = MAX_BYTES):
        self.http = http
        self.robots = robots
        self.out_dir = out_dir
        self.max_bytes = max_bytes

    def fetch(self, paper) -> FulltextResult:
        candidates = fulltext_candidates(paper)
        if not candidates:
            return FulltextResult("unavailable", error="нет ссылок на полный текст")
        errors = []
        for url, kind in candidates:
            try:
                content = self._download(url, kind)
            except FetchError as exc:
                errors.append(str(exc))
                continue
            path = self.out_dir / f"{paper['id']}.{kind}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            return FulltextResult("ok", path=str(path), kind=kind)
        return FulltextResult("failed", error="; ".join(errors))

    def _download(self, url: str, kind: str) -> bytes:
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
        content = response.content
        if len(content) > self.max_bytes:
            raise FetchError(f"файл больше {self.max_bytes // 2**20} МБ: {url}")
        if kind == "pdf" and not content.startswith(b"%PDF-"):
            raise FetchError(f"по ссылке не PDF ({response.headers.get('content-type', '?')}): {url}")
        if kind == "html" and CYBERLENINKA_MARKER.encode() not in content:
            raise FetchError(f"на странице нет текста статьи: {url}")
        return content
