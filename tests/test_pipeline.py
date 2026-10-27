from research_assistant.ingest.download import FulltextResult
from research_assistant.ingest.models import PaperRecord
from research_assistant.ingest.pipeline import ingest_topic, search_sources
from research_assistant.store.db import Library


class FakeSource:
    def __init__(self, name, records=None, error=None):
        self.name = name
        self.records = records or []
        self.error = error
        self.calls = []

    def search(self, query, limit=20, year_from=None, year_to=None, language=None):
        self.calls.append((query, limit, year_from, year_to, language))
        if self.error:
            raise self.error
        return self.records


class FakeFetcher:
    def __init__(self):
        self.fetched = []

    def fetch(self, paper):
        self.fetched.append(paper["id"])
        return FulltextResult("ok", path=f"{paper['id']}.pdf", kind="pdf") if paper["arxiv_id"] else \
            FulltextResult("unavailable", error="нет ссылок")


ARXIV = [
    PaperRecord(title="Decoding Speculative Decoding", sources=["arxiv"], arxiv_id="2402.01528"),
    PaperRecord(title="Speculative Decoding with Big Little Decoder", sources=["arxiv"], arxiv_id="2302.07863"),
]
OPENALEX = [
    PaperRecord(title="Decoding speculative decoding", sources=["openalex"], openalex_id="W4391556558",
                doi="10.48550/arxiv.2402.01528"),
    PaperRecord(title="Машинное обучение в бизнесе и промышленности", sources=["openalex"], openalex_id="W2",
                language="ru"),
]


def test_search_survives_failing_source():
    result = search_sources([FakeSource("arxiv", ARXIV), FakeSource("openalex", error=RuntimeError("down"))], "q")
    assert result.found == {"arxiv": 2}
    assert "RuntimeError: down" in result.errors["openalex"]
    assert len(result.records) == 2


def test_ingest_dedupes_links_topic_and_fetches_fulltext():
    fetcher = FakeFetcher()
    with Library(":memory:") as library:
        report = ingest_topic(library, [FakeSource("arxiv", ARXIV), FakeSource("openalex", OPENALEX)],
                              topic="speculative decoding", year_from=2023, fetcher=fetcher)
        assert report.found == {"arxiv": 2, "openalex": 2}
        assert report.unique == 3 and report.added == 3
        assert report.fulltext == {"ok": 2, "unavailable": 1}
        assert library.topic_summary()[0]["papers"] == 3

        # повторный сбор: ничего нового, полные тексты не качаются заново
        again = ingest_topic(library, [FakeSource("arxiv", ARXIV)], topic="speculative decoding", fetcher=fetcher)
        assert again.added == 0 and again.already_in_library == 2
        assert sum(again.fulltext.values()) == 0
        assert len(fetcher.fetched) == 3


def test_query_defaults_to_topic_and_params_pass_through():
    source = FakeSource("openalex", OPENALEX)
    with Library(":memory:") as library:
        ingest_topic(library, [source], topic="ML в бизнесе", limit=5, year_to=2024, language="ru")
    assert source.calls == [("ML в бизнесе", 5, None, 2024, "ru")]


def test_search_sources_keeps_only_papers_of_the_author():
    from research_assistant.ingest.models import PaperRecord
    from research_assistant.ingest.pipeline import search_sources

    class Source:
        name = "fake"

        def __init__(self):
            self.kwargs = None

        def search(self, query, limit=20, year_from=None, year_to=None, language=None, author=None):
            self.kwargs = {"query": query, "author": author}
            return [PaperRecord(title="Paper about Geoffrey Hinton winning the Nobel prize", sources=["fake"],
                                authors=["Some Journalist"]),
                    PaperRecord(title="Layer normalization for recurrent networks", sources=["fake"],
                                authors=["Jimmy Lei Ba", "Geoffrey E. Hinton"])]

    source = Source()
    result = search_sources([source], "", author="Geoffrey Hinton")
    assert source.kwargs == {"query": "", "author": "Geoffrey Hinton"}
    assert [r.title for r in result.records] == ["Layer normalization for recurrent networks"]
    assert len(search_sources([Source()], "layer").records) == 2  # без автора не фильтруем
