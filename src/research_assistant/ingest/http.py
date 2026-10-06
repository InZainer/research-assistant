"""HTTP-клиент для внешних источников: паузы между запросами, повторы и дисковый кэш."""

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlencode, urlsplit

import httpx

RETRY_STATUSES = {429, 500, 502, 503, 504}


class FetchError(Exception):
    pass


class HttpClient:
    def __init__(
        self,
        user_agent: str,
        cache_dir: Path | None = None,
        intervals: dict[str, float] | None = None,
        default_interval: float = 1.0,
        backoff: tuple[float, ...] = (10, 30, 60),
        timeout: float = 60,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.user_agent = user_agent
        self.cache_dir = cache_dir
        self.intervals = dict(intervals or {})
        self.default_interval = default_interval
        self.backoff = backoff
        self.backoffs: dict[str, tuple[float, ...]] = {}
        self._sleep = sleep
        self._clock = clock
        self._last_request: dict[str, float] = {}
        self._client = httpx.Client(
            headers={"User-Agent": user_agent}, follow_redirects=True, timeout=timeout, transport=transport
        )

    def close(self) -> None:
        self._client.close()

    def set_interval(self, host: str, seconds: float) -> None:
        """Увеличивает паузу для хоста (например, по Crawl-delay из robots.txt); уменьшить нельзя."""
        self.intervals[host] = max(self.intervals.get(host, self.default_interval), seconds)

    def set_backoff(self, host: str, backoff: tuple[float, ...]) -> None:
        """Свои паузы между повторами для хоста: для нестабильного источника лучше быстро сдаться."""
        self.backoffs[host] = backoff

    def _throttle(self, host: str) -> None:
        interval = self.intervals.get(host, self.default_interval)
        last = self._last_request.get(host)
        if last is not None:
            wait = interval - (self._clock() - last)
            if wait > 0:
                self._sleep(wait)
        self._last_request[host] = self._clock()

    def get(self, url: str, params: dict | None = None, timeout: float | None = None) -> httpx.Response:
        """GET с паузой для хоста и повторами на 429/5xx и сетевых ошибках. Остальные коды возвращает как есть."""
        host = urlsplit(url).netloc
        backoff = self.backoffs.get(host, self.backoff)
        for attempt in range(len(backoff) + 1):
            self._throttle(host)
            try:
                response = self._client.get(
                    url, params=params, timeout=timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT
                )
            except httpx.TransportError as exc:
                if attempt == len(backoff):
                    raise FetchError(f"сеть недоступна: {url}: {type(exc).__name__}") from exc
                self._sleep(backoff[attempt])
                continue
            if response.status_code not in RETRY_STATUSES:
                return response
            if attempt == len(backoff):
                raise FetchError(f"HTTP {response.status_code} после {attempt + 1} попыток: {url}")
            self._sleep(_retry_after(response) or backoff[attempt])
        raise AssertionError("недостижимо")

    def get_text(
        self, url: str, params: dict | None = None, use_cache: bool = True, timeout: float | None = None
    ) -> str:
        cache_file = self._cache_file(url, params) if use_cache else None
        if cache_file and cache_file.exists():
            return cache_file.read_text(encoding="utf-8")
        response = self.get(url, params, timeout)
        if response.status_code != 200:
            raise FetchError(f"HTTP {response.status_code}: {url}: {response.text[:200]}")
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(response.text, encoding="utf-8")
        return response.text

    def get_json(self, url: str, params: dict | None = None, use_cache: bool = True, timeout: float | None = None):
        return json.loads(self.get_text(url, params, use_cache, timeout))

    def _cache_file(self, url: str, params: dict | None) -> Path | None:
        if self.cache_dir is None:
            return None
        # Ключ API в имени кэша не участвует: он не меняет ответ.
        clean = {k: v for k, v in sorted((params or {}).items()) if k != "api_key"}
        key = hashlib.sha256(f"{url}?{urlencode(clean)}".encode()).hexdigest()[:32]
        return self.cache_dir / f"{key}.txt"


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After", "")
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if seconds > 0 else None
