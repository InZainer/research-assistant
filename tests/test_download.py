import httpx

from research_assistant.ingest.download import FulltextFetcher, fulltext_candidates
from research_assistant.ingest.http import HttpClient
from research_assistant.ingest.robots import RobotsPolicy

PDF = b"%PDF-1.7 fake pdf body"
CYBERLENINKA_ROBOTS = "User-Agent: *\nDisallow: /api/\nDisallow: /article/n/*/pdf\n"


def paper(**overrides):
    values = {"id": 7, "arxiv_id": None, "pdf_url": None, "landing_url": None}
    values.update(overrides)
    return values


def make_fetcher(tmp_path, routes):
    """routes: {(host, path): httpx.Response}; robots.txt без записи — 404."""
    requested = []

    def handler(request):
        requested.append(f"{request.url.host}{request.url.path}")
        response = routes.get((request.url.host, request.url.path))
        return response if response is not None else httpx.Response(404)

    http = HttpClient("test", transport=httpx.MockTransport(handler), default_interval=0, backoff=(),
                      sleep=lambda s: None)
    return FulltextFetcher(http, RobotsPolicy(http), tmp_path), requested


def test_candidates_order_and_dedup():
    row = paper(arxiv_id="2402.01528", pdf_url="https://arxiv.org/pdf/2402.01528",
                landing_url="https://cyberleninka.ru/article/n/slug")
    assert fulltext_candidates(row) == [
        ("https://arxiv.org/pdf/2402.01528", "pdf"),
        ("https://cyberleninka.ru/article/n/slug", "html"),
    ]
    assert fulltext_candidates(paper(landing_url="https://journal.org/article/1")) == []


def test_downloads_pdf(tmp_path):
    fetcher, _ = make_fetcher(tmp_path, {("arxiv.org", "/pdf/2402.01528"): httpx.Response(200, content=PDF)})
    result = fetcher.fetch(paper(arxiv_id="2402.01528"))
    assert result.status == "ok" and result.kind == "pdf"
    assert (tmp_path / "7.pdf").read_bytes() == PDF


def test_html_instead_of_pdf_falls_back_to_next_candidate(tmp_path):
    routes = {
        ("journal.org", "/download/1"): httpx.Response(200, content=b"<html>login</html>"),
        ("cyberleninka.ru", "/robots.txt"): httpx.Response(200, text=CYBERLENINKA_ROBOTS),
        ("cyberleninka.ru", "/article/n/slug"): httpx.Response(200, content=b'<div class="ocr"><p>text</p></div>'),
    }
    fetcher, _ = make_fetcher(tmp_path, routes)
    result = fetcher.fetch(paper(pdf_url="https://journal.org/download/1",
                                 landing_url="https://cyberleninka.ru/article/n/slug"))
    assert result.status == "ok" and result.kind == "html"
    assert (tmp_path / "7.html").exists()


def test_robots_disallowed_url_is_never_requested(tmp_path):
    routes = {
        ("cyberleninka.ru", "/robots.txt"): httpx.Response(200, text=CYBERLENINKA_ROBOTS),
        ("cyberleninka.ru", "/article/n/slug/pdf"): httpx.Response(200, content=PDF),
    }
    fetcher, requested = make_fetcher(tmp_path, routes)
    result = fetcher.fetch(paper(pdf_url="https://cyberleninka.ru/article/n/slug/pdf"))
    assert result.status == "failed"
    assert "robots.txt" in result.error
    assert "cyberleninka.ru/article/n/slug/pdf" not in requested


def test_crawl_delay_raises_host_interval(tmp_path):
    routes = {
        ("arxiv.org", "/robots.txt"): httpx.Response(200, text="User-agent: *\nCrawl-delay: 15\nAllow: /pdf\n"),
        ("arxiv.org", "/pdf/2402.01528"): httpx.Response(200, content=PDF),
    }
    fetcher, _ = make_fetcher(tmp_path, routes)
    fetcher.fetch(paper(arxiv_id="2402.01528"))
    assert fetcher.http.intervals["arxiv.org"] == 15


def test_no_candidates_is_unavailable(tmp_path):
    fetcher, _ = make_fetcher(tmp_path, {})
    assert fetcher.fetch(paper()).status == "unavailable"
