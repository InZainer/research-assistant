import json
from pathlib import Path

import pytest

from research_assistant.ingest.http import FetchError
from research_assistant.ingest.sources.arxiv import ArxivSource, build_query, parse_feed
from research_assistant.ingest.sources.openalex import OpenAlexSource, build_params, parse_work, rebuild_abstract

FIXTURES = Path(__file__).parent / "fixtures"


class FakeHttp:
    def __init__(self, text: str = ""):
        self.text = text
        self.calls = []
        self.intervals = {}
        self.backoffs = {}

    def set_interval(self, host, seconds):
        self.intervals[host] = seconds

    def set_backoff(self, host, backoff):
        self.backoffs[host] = backoff

    def get_text(self, url, params=None, use_cache=True, timeout=None):
        self.calls.append((url, params))
        self.timeout = timeout
        return self.text

    def get_json(self, url, params=None, use_cache=True, timeout=None):
        return json.loads(self.get_text(url, params, use_cache, timeout))


# --- arXiv ---

def test_build_query_words_phrases_and_years():
    assert build_query("speculative decoding") == "all:speculative AND all:decoding"
    assert build_query('"speculative decoding" LLM') == 'all:"speculative decoding" AND all:LLM'
    assert build_query("rag", 2023, 2026).endswith("AND submittedDate:[202301010000 TO 202612312359]")
    assert build_query("rag (survey)") == "all:rag AND all:survey"


def test_build_query_rejects_empty():
    with pytest.raises(ValueError):
        build_query('  "" ()  ')


def test_parse_arxiv_feed():
    records = parse_feed((FIXTURES / "arxiv_feed.xml").read_text(encoding="utf-8"))
    assert len(records) == 2
    first = records[0]
    assert first.arxiv_id == "2203.16487"
    assert first.title.startswith("Speculative Decoding: Exploiting Speculative Execution")
    assert first.year == 2022
    assert first.authors[0] == "Heming Xia" and len(first.authors) == 6
    assert first.pdf_url == "https://arxiv.org/pdf/2203.16487"
    assert first.sources == ["arxiv"]
    assert "\n" not in first.abstract


def test_arxiv_error_feed_raises():
    feed = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <id>http://arxiv.org/api/errors#incorrect_id_format</id><title>Error</title>
      <summary>incorrect id format</summary></entry></feed>"""
    with pytest.raises(FetchError, match="incorrect id format"):
        parse_feed(feed)


def test_arxiv_source_sets_interval_and_skips_non_english():
    http = FakeHttp((FIXTURES / "arxiv_feed.xml").read_text(encoding="utf-8"))
    source = ArxivSource(http)
    assert http.intervals["export.arxiv.org"] >= 3
    assert sum(http.backoffs["export.arxiv.org"]) <= 30
    assert source.search("машинное обучение", language="ru") == []
    assert http.calls == []
    assert len(source.search("speculative decoding", limit=2)) == 2
    assert http.calls[0][1]["max_results"] == 2
    assert http.timeout is not None and http.timeout <= 30


# --- OpenAlex ---

def openalex_results():
    return json.loads((FIXTURES / "openalex_works.json").read_text(encoding="utf-8"))["results"]


def test_build_params_filters():
    params = build_params("rag", 10, 2023, 2026, "ru")
    assert params["filter"] == "is_oa:true,publication_year:2023-2026,language:ru"
    assert build_params("rag", 10, year_from=2023)["filter"] == "is_oa:true,publication_year:>2022"
    assert build_params("rag", 500)["per_page"] == 200


def test_rebuild_abstract():
    assert rebuild_abstract({"world": [1], "hello": [0], "again": [2]}) == "hello world again"
    assert rebuild_abstract(None) is None


def test_parse_openalex_arxiv_preprint():
    record = parse_work(openalex_results()[0])
    assert record.title == "Speculative Speculative Decoding"
    assert record.openalex_id == "W7133563540"
    assert record.arxiv_id == "2603.03251"
    assert record.pdf_url == "https://arxiv.org/pdf/2603.03251"
    assert record.abstract.startswith("Autoregressive decoding")


def test_parse_openalex_russian_cyberleninka():
    works = {w["display_name"]: w for w in openalex_results()}
    record = parse_work(works["Машинное обучение в бизнесе"])
    assert record.language == "ru"
    assert record.landing_url == "https://cyberleninka.ru/article/n/mashinnoe-obuchenie-v-biznese"
    assert record.pdf_url is None
    assert record.arxiv_id is None


def test_parse_openalex_strips_html_and_skips_untitled():
    assert parse_work({"display_name": "On <i>ab initio</i> models", "id": "https://openalex.org/W1"}).title == \
        "On ab initio models"
    assert parse_work({"display_name": None, "id": "https://openalex.org/W2"}) is None


def test_openalex_source_passes_contact_and_key():
    http = FakeHttp((FIXTURES / "openalex_works.json").read_text(encoding="utf-8"))
    source = OpenAlexSource(http, mailto="me@example.org", api_key="k")
    records = source.search("speculative decoding", limit=5, language="ru")
    assert len(records) == 6
    params = http.calls[0][1]
    assert params["mailto"] == "me@example.org" and params["api_key"] == "k"
    assert "language:ru" in params["filter"]


# --- поиск по автору ---

def test_arxiv_author_query():
    assert build_query(None, author="Geoffrey E. Hinton") == "au:geoffrey AND au:hinton"
    assert build_query("capsules", author="Hinton") == "au:hinton AND all:capsules"


def test_openalex_author_params():
    params = build_params("", 10, author_ids=["A1", "A2"])
    assert params["filter"] == "is_oa:true,author.id:A1|A2"
    assert "search" not in params and params["sort"] == "cited_by_count:desc"  # без запроса — самые цитируемые
    assert build_params("capsules", 10, author_ids=["A1"])["search"] == "capsules"


class RoutedHttp(FakeHttp):
    """Ответ зависит от адреса: каталог авторов или работы."""

    def __init__(self, routes: dict[str, dict]):
        super().__init__()
        self.routes = routes

    def get_text(self, url, params=None, use_cache=True, timeout=None):
        self.calls.append((url, params))
        return json.dumps(self.routes[url])


def test_openalex_author_search_checks_profile_names():
    authors = {"results": [
        {"id": "https://openalex.org/A1", "display_name": "Geoffrey E. Hinton", "works_count": 562},
        {"id": "https://openalex.org/A2", "display_name": "Yoshua Bengio", "works_count": 1579},  # нечёткий поиск
        {"id": "https://openalex.org/A3", "display_name": "Hinton Geoffrey", "works_count": 1},
    ]}
    http = RoutedHttp({"https://api.openalex.org/authors": authors,
                       "https://api.openalex.org/works": {"results": openalex_results()[:1]}})
    source = OpenAlexSource(http)
    assert source.author_ids("Geoffrey Hinton") == ["A1", "A3"]
    source.search("", author="Geoffrey Hinton")
    assert "author.id:A1|A3" in http.calls[-1][1]["filter"]

    http = RoutedHttp({"https://api.openalex.org/authors": {"results": []}})
    assert OpenAlexSource(http).search("", author="Nobody Atall") == []
    assert [url for url, _ in http.calls] == ["https://api.openalex.org/authors"]  # работы без автора не ищем
