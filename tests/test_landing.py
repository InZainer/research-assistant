import httpx
import pytest

from research_assistant.ingest.download import FulltextFetcher
from research_assistant.ingest.http import FetchError, HttpClient
from research_assistant.ingest.landing import pdf_links, record_from_page
from research_assistant.ingest.models import PaperRecord
from research_assistant.ingest.resolve import resolve_reference
from research_assistant.ingest.robots import RobotsPolicy

PDF = b"%PDF-1.7 fake pdf body"
ABSTRACT = ("Рассмотрены цифровые технологии в ресторанном бизнесе России: чат-боты, прогнозирование спроса "
            "и роботизация кухни на примере сетевых заведений.")
PAGE = f"""<html><head>
<meta name="citation_title" content="Цифровизация  ресторанного бизнеса">
<meta name="citation_author" content="Иванов И. И.">
<meta name="citation_author" content="Петров П. П.">
<meta name="citation_publication_date" content="2023/05/10">
<meta name="citation_journal_title" content="Вестник сервиса">
<meta name="citation_doi" content="10.1234/vs.2023.5">
<meta name="citation_pdf_url" content="/files/article-17.pdf">
<meta name="citation_pdf_url" content="/files/article-17.pdf">
<meta name="description" content="{ABSTRACT}">
<meta property="og:description" content="Журнал о сервисе">
</head><body>страница статьи</body></html>"""


def make_http(routes):
    requested = []

    def handler(request):
        requested.append(f"{request.url.host}{request.url.path}")
        response = routes.get((request.url.host, request.url.path))
        return response if response is not None else httpx.Response(404)

    http = HttpClient("test", transport=httpx.MockTransport(handler), default_interval=0, backoff=(),
                      sleep=lambda s: None)
    return http, requested


# --- метатеги страницы ---

def test_record_from_page_reads_highwire_meta():
    record = record_from_page(PAGE, "https://journal.ru/article/17")
    assert record.title == "Цифровизация ресторанного бизнеса"
    assert record.authors == ["Иванов И. И.", "Петров П. П."]
    assert record.year == 2023 and record.venue == "Вестник сервиса"
    assert record.doi == "10.1234/vs.2023.5"
    assert record.language == "ru"  # метатега языка нет — определяется по тексту
    assert record.abstract == ABSTRACT  # короткий og:description — подпись сайта, а не аннотация
    assert record.landing_url == "https://journal.ru/article/17"
    assert record.pdf_url == "https://journal.ru/files/article-17.pdf"


def test_page_without_citation_title_is_not_a_paper():
    assert record_from_page("<html><head><title>Вход</title></head></html>", "https://journal.ru/login") is None


def test_pdf_links_are_absolute_and_unique():
    assert pdf_links(PAGE, "https://journal.ru/article/17") == ["https://journal.ru/files/article-17.pdf"]
    assert pdf_links("<meta name='citation_pdf_url' content='javascript:void(0)'>", "https://j.ru/a") == []


# --- полный текст через страницу статьи ---

def paper(**overrides):
    values = {"id": 7, "arxiv_id": None, "pdf_url": None, "landing_url": None, "doi": None}
    values.update(overrides)
    return values


def test_fetcher_finds_pdf_on_the_landing_page(tmp_path):
    http, requested = make_http({
        ("journal.ru", "/article/17"): httpx.Response(200, text=PAGE),
        ("journal.ru", "/files/article-17.pdf"): httpx.Response(200, content=PDF),
    })
    result = FulltextFetcher(http, RobotsPolicy(http), tmp_path).fetch(paper(landing_url="https://journal.ru/article/17"))
    assert result.status == "ok" and result.kind == "pdf"
    assert (tmp_path / "7.pdf").read_bytes() == PDF
    assert "journal.ru/files/article-17.pdf" in requested


def test_fetcher_uses_doi_when_there_is_no_landing_url(tmp_path):
    http, _ = make_http({
        ("doi.org", "/10.1234/vs.2023.5"): httpx.Response(200, text=PAGE.replace("/files/", "https://cdn.ru/")),
        ("cdn.ru", "/article-17.pdf"): httpx.Response(200, content=PDF),
    })
    result = FulltextFetcher(http, RobotsPolicy(http), tmp_path).fetch(paper(doi="10.1234/vs.2023.5"))
    assert result.status == "ok"


def test_broken_direct_link_falls_back_to_the_landing_page(tmp_path):
    http, _ = make_http({
        ("journal.ru", "/old.pdf"): httpx.Response(404),
        ("journal.ru", "/article/17"): httpx.Response(200, text=PAGE),
        ("journal.ru", "/files/article-17.pdf"): httpx.Response(200, content=PDF),
    })
    fetcher = FulltextFetcher(http, RobotsPolicy(http), tmp_path)
    row = paper(pdf_url="https://journal.ru/old.pdf", landing_url="https://journal.ru/article/17")
    assert fetcher.fetch(row).status == "ok"


def test_landing_page_without_pdf_link(tmp_path):
    http, _ = make_http({("journal.ru", "/article/1"): httpx.Response(200, text="<html>paywall</html>")})
    result = FulltextFetcher(http, RobotsPolicy(http), tmp_path).fetch(paper(landing_url="https://journal.ru/article/1"))
    assert result.status == "unavailable" and "нет ссылки на PDF" in result.error
    # ссылка на странице есть, но не скачалась — это сбой, его можно повторить (--retry-failed)
    http, _ = make_http({("journal.ru", "/article/17"): httpx.Response(200, text=PAGE)})
    result = FulltextFetcher(http, RobotsPolicy(http), tmp_path).fetch(paper(landing_url="https://journal.ru/article/17"))
    assert result.status == "failed" and "404" in result.error


def test_landing_page_respects_robots_and_can_be_switched_off(tmp_path):
    http, requested = make_http({
        ("journal.ru", "/robots.txt"): httpx.Response(200, text="User-agent: *\nDisallow: /article/\n"),
        ("journal.ru", "/article/17"): httpx.Response(200, text=PAGE),
    })
    result = FulltextFetcher(http, RobotsPolicy(http), tmp_path).fetch(paper(landing_url="https://journal.ru/article/17"))
    assert result.status == "failed" and "robots.txt" in result.error
    assert "journal.ru/article/17" not in requested
    off = FulltextFetcher(http, RobotsPolicy(http), tmp_path, discover=False)
    assert off.fetch(paper(landing_url="https://journal.ru/article/17")).status == "unavailable"


# --- статья по ссылке ---

class FakeOpenAlex:
    def __init__(self, record=None):
        self.record = record
        self.asked = []

    def by_doi(self, doi):
        self.asked.append(doi)
        return self.record


def test_resolve_prefers_openalex_for_doi_and_arxiv():
    http, requested = make_http({})
    known = PaperRecord(title="Decoding Speculative Decoding", sources=["openalex"], openalex_id="W1")
    openalex = FakeOpenAlex(known)
    assert resolve_reference("https://doi.org/10.1234/X", http, RobotsPolicy(http), openalex) is known
    assert resolve_reference("arXiv:2402.01528v2", http, RobotsPolicy(http), openalex) is known
    assert openalex.asked == ["10.1234/x", "10.48550/arxiv.2402.01528"]
    assert requested == []  # страницу не открывали


def test_resolve_reads_the_page_when_openalex_does_not_know_the_paper():
    http, _ = make_http({
        ("cyberleninka.ru", "/article/n/slug"): httpx.Response(200, text=PAGE),
        ("arxiv.org", "/abs/2402.01528"): httpx.Response(200, text=PAGE),
    })
    robots = RobotsPolicy(http)
    record = resolve_reference("https://cyberleninka.ru/article/n/slug", http, robots, FakeOpenAlex(None))
    assert record.title == "Цифровизация ресторанного бизнеса" and record.sources == ["page"]
    assert record.landing_url == "https://cyberleninka.ru/article/n/slug"
    assert resolve_reference("2402.01528", http, robots, openalex=None).landing_url == "https://arxiv.org/abs/2402.01528"


def test_resolve_reports_why_it_failed():
    http, _ = make_http({
        ("closed.ru", "/robots.txt"): httpx.Response(200, text="User-agent: *\nDisallow: /\n"),
        ("journal.ru", "/login"): httpx.Response(200, text="<html>вход</html>"),
    })
    robots = RobotsPolicy(http)
    with pytest.raises(FetchError, match="robots.txt"):
        resolve_reference("https://closed.ru/article/1", http, robots)
    with pytest.raises(FetchError, match="citation_title"):
        resolve_reference("https://journal.ru/login", http, robots)
    with pytest.raises(FetchError, match="HTTP 404"):
        resolve_reference("https://journal.ru/missing", http, robots)
    with pytest.raises(FetchError, match="не похоже"):
        resolve_reference("какая-то статья про рестораны", http, robots)
