"""Проверка robots.txt перед скачиванием полных текстов.

Стандартный urllib.robotparser не понимает шаблоны `*` и `$` (например, `Disallow: /article/n/*/pdf`
у CyberLeninka), поэтому используется protego — парсер robots.txt из Scrapy.
"""

from urllib.parse import urlsplit

from protego import Protego

from research_assistant.ingest.http import FetchError, HttpClient

ROBOTS_AGENT = "research-assistant"


class RobotsPolicy:
    """Кэширует robots.txt по хостам.

    404 и другие 4xx — ограничений нет. 5xx и сетевая ошибка — скачивать с хоста нельзя:
    лучше пропустить статью, чем нарушить правила сайта.
    """

    def __init__(self, http: HttpClient, agent: str = ROBOTS_AGENT):
        self.http = http
        self.agent = agent
        self._rules: dict[str, Protego | None] = {}

    def _rules_for(self, url: str) -> Protego | None:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._rules:
            self._rules[origin] = self._load(origin)
        return self._rules[origin]

    def _load(self, origin: str) -> Protego | None:
        try:
            response = self.http.get(f"{origin}/robots.txt")
        except FetchError:
            return None
        if response.status_code == 200:
            return Protego.parse(response.text)
        if 400 <= response.status_code < 500:
            return Protego.parse("")
        return None

    def allowed(self, url: str) -> bool:
        rules = self._rules_for(url)
        return rules is not None and rules.can_fetch(url, self.agent)

    def crawl_delay(self, url: str) -> float | None:
        rules = self._rules_for(url)
        if rules is None:
            return None
        delay = rules.crawl_delay(self.agent)
        return float(delay) if delay is not None else None
