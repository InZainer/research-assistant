import httpx
import pytest

from research_assistant.ingest.http import FetchError, HttpClient
from research_assistant.ingest.robots import RobotsPolicy


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def make_client(handler, clock=None, cache_dir=None, **kwargs):
    clock = clock or FakeClock()
    return HttpClient(
        "test-agent", cache_dir=cache_dir, transport=httpx.MockTransport(handler),
        sleep=clock.sleep, clock=clock.time, **kwargs,
    )


def test_retries_on_429_with_backoff():
    statuses = iter([429, 503, 200])

    def handler(request):
        return httpx.Response(next(statuses), text="ok")

    clock = FakeClock()
    client = make_client(handler, clock, backoff=(5, 7, 9), default_interval=0)
    assert client.get("https://example.org/a").status_code == 200
    assert clock.sleeps == [5, 7]


def test_retry_after_header_wins():
    statuses = iter([(429, {"Retry-After": "42"}), (200, {})])

    def handler(request):
        status, headers = next(statuses)
        return httpx.Response(status, headers=headers)

    clock = FakeClock()
    make_client(handler, clock, backoff=(5,), default_interval=0).get("https://example.org/a")
    assert clock.sleeps == [42]


def test_gives_up_after_all_retries():
    client = make_client(lambda r: httpx.Response(429), backoff=(1, 1), default_interval=0)
    with pytest.raises(FetchError, match="429"):
        client.get("https://example.org/a")


def test_non_retry_status_is_returned():
    client = make_client(lambda r: httpx.Response(404), default_interval=0)
    assert client.get("https://example.org/a").status_code == 404


def test_throttles_requests_to_same_host():
    clock = FakeClock()
    client = make_client(lambda r: httpx.Response(200), clock, default_interval=3)
    client.get("https://example.org/a")
    client.get("https://example.org/b")
    client.get("https://other.org/c")
    assert clock.sleeps == [3]


def test_set_interval_only_increases():
    client = make_client(lambda r: httpx.Response(200), default_interval=1)
    client.set_interval("h", 15)
    client.set_interval("h", 2)
    assert client.intervals["h"] == 15


def test_cache_avoids_second_request_and_ignores_api_key(tmp_path):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, text="payload")

    client = make_client(handler, cache_dir=tmp_path, default_interval=0)
    assert client.get_text("https://example.org/api", {"q": "x", "api_key": "1"}) == "payload"
    assert client.get_text("https://example.org/api", {"api_key": "2", "q": "x"}) == "payload"
    assert len(calls) == 1
    assert client.get_text("https://example.org/api", {"q": "x"}, use_cache=False) == "payload"
    assert len(calls) == 2


def test_get_text_raises_on_error_status(tmp_path):
    client = make_client(lambda r: httpx.Response(400, text="bad filter"), cache_dir=tmp_path, default_interval=0)
    with pytest.raises(FetchError, match="bad filter"):
        client.get_text("https://example.org/api")
    assert list(tmp_path.iterdir()) == []


# --- robots.txt ---

ROBOTS = "User-Agent: *\nCrawl-delay: 15\nDisallow: /api/\nDisallow: /article/n/*/pdf\n"


def robots_client(status=200, body=ROBOTS):
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(status, text=body)
        return httpx.Response(200)

    return make_client(handler, default_interval=0, backoff=())


def test_robots_rules_and_crawl_delay():
    policy = RobotsPolicy(robots_client())
    assert policy.allowed("https://site.org/article/n/some-article")
    assert not policy.allowed("https://site.org/api/search")
    assert policy.crawl_delay("https://site.org/x") == 15


def test_missing_robots_allows_everything():
    policy = RobotsPolicy(robots_client(status=404))
    assert policy.allowed("https://site.org/api/search")


def test_server_error_on_robots_forbids():
    policy = RobotsPolicy(robots_client(status=500))
    assert not policy.allowed("https://site.org/anything")


def test_robots_wildcards_are_honoured():
    policy = RobotsPolicy(robots_client())
    assert not policy.allowed("https://site.org/article/n/some-article/pdf")


def test_per_host_backoff():
    clock = FakeClock()
    client = make_client(lambda r: httpx.Response(429), clock, backoff=(10, 30, 60), default_interval=0)
    client.set_backoff("flaky.org", (2,))
    with pytest.raises(FetchError):
        client.get("https://flaky.org/api")
    assert clock.sleeps == [2]


def test_timeout_error_is_retried_then_reported():
    def handler(request):
        assert request.extensions["timeout"]["read"] == 3
        raise httpx.ReadTimeout("slow", request=request)

    clock = FakeClock()
    client = make_client(handler, clock, backoff=(1,), default_interval=0)
    with pytest.raises(FetchError, match="ReadTimeout"):
        client.get("https://slow.org/api", timeout=3)
    assert clock.sleeps == [1]
