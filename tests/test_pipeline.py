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


def test_fulltexts_are_fetched_from_different_hosts_in_turn():
    from research_assistant.ingest.pipeline import interleave_by_host

    def row(number, **fields):
        return {"id": number, "arxiv_id": None, "pdf_url": None, "landing_url": None, "doi": None, **fields}

    papers = [row(1, arxiv_id="2401.00001"), row(2, arxiv_id="2401.00002"), row(3, arxiv_id="2401.00003"),
              row(4, pdf_url="https://mdpi.com/a.pdf"), row(5, doi="10.1/x"), row(6, pdf_url="https://mdpi.com/b.pdf"),
              row(7)]
    order = [p["id"] for p in interleave_by_host(papers)]
    assert sorted(order) == [1, 2, 3, 4, 5, 6, 7]
    assert order[:4] == [1, 4, 5, 7]  # по одной статье с каждого сайта, начиная с самого большого
    assert order[4:] == [2, 6, 3]  # arXiv с его паузой в 15 с чередуется с остальными, пока они есть


def test_fulltexts_in_several_threads_keep_one_host_in_one_thread():
    import threading

    from research_assistant.ingest.pipeline import fetch_fulltexts, split_by_host

    class ThreadFetcher:
        def __init__(self):
            self.threads = {}

        def fetch(self, paper):
            host = "arxiv" if paper["arxiv_id"] else "journal"
            self.threads.setdefault(host, set()).add(threading.get_ident())
            if paper["title"].startswith("Broken"):
                raise RuntimeError("boom")
            return FulltextResult("ok", path=f"{paper['id']}.pdf", kind="pdf")

    records = [PaperRecord(title=f"Arxiv paper number {i} on decoding", sources=["arxiv"], arxiv_id=f"2401.0000{i}")
               for i in range(5)]
    records += [PaperRecord(title=f"Journal paper number {i} on decoding", sources=["openalex"],
                            openalex_id=f"W{i}", pdf_url=f"https://journal.org/{i}.pdf") for i in range(4)]
    records.append(PaperRecord(title="Broken paper that makes the fetcher fail", sources=["openalex"],
                               openalex_id="W99", pdf_url="https://journal.org/99.pdf"))
    with Library(":memory:") as library:
        topic_id = library.get_or_create_topic("t")
        for record in records:
            library.link_topic(library.upsert_paper(record)[0], topic_id)
        buckets = split_by_host(library.papers_for_fulltext(topic_id), workers=3)
        assert sorted(len(b) for b in buckets) == [5, 5]  # два сайта — два потока, третий не нужен

        fetcher, counter, lines = ThreadFetcher(), {"ok": 0, "failed": 0}, []
        fetch_fulltexts(library, topic_id, fetcher, ("pending",), counter, lines.append, workers=3)
        assert counter == {"ok": 9, "failed": 1} and len(lines) == 10
        assert all(len(threads) == 1 for threads in fetcher.threads.values())  # один сайт — один поток
        assert library.papers_for_fulltext(topic_id) == []  # статусы записаны в базу вызывающим потоком
        failed = library.papers_for_fulltext(topic_id, ("failed",))
        assert len(failed) == 1 and "RuntimeError: boom" in failed[0]["fulltext_error"]
