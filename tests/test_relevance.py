from research_assistant.ingest.download import FulltextResult
from research_assistant.ingest.models import PaperRecord
from research_assistant.ingest.pipeline import add_references, ingest_topic
from research_assistant.ingest.relevance import TermFilter
from research_assistant.store.db import Library

FOOD = "ресторан|общественное питание|кафе|restaurant|food service"
AI = "искусственный интеллект|нейросеть|machine learning|AI"


def record(title, abstract=None, **keys):
    return PaperRecord(title=title, sources=["openalex"], abstract=abstract, **keys)


def test_term_filter_matches_word_forms_and_phrases():
    food = TermFilter.parse([FOOD])
    assert food.matches(record("Применение ИИ в ресторанном бизнесе"))
    assert food.matches(record("Цифровизация", abstract="На предприятиях общественного питания внедряют роботов."))
    assert food.matches(record("Machine learning for restaurants revenue"))
    assert not food.matches(record("Оценка состояния насосного оборудования"))
    assert not food.matches(record("Питание и общественное здоровье"))  # слова фразы есть, но не подряд
    assert not food.matches(record("Кафедра экономики"))  # «кафе» — не начало слова «кафедра»


def test_every_group_must_match():
    both = TermFilter.parse([FOOD, AI])
    assert both.matches(record("Нейросети в сфере общественного питания"))
    assert both.matches(record("AI in restaurant operations"))
    assert not both.matches(record("Маркетинг ресторана"))
    assert not both.matches(record("Machine learning in medicine"))


def test_parse_ignores_empty_specs():
    assert TermFilter.parse(None) is None
    assert TermFilter.parse(["", " | , "]) is None
    assert len(TermFilter.parse(["ресторан, кафе; столовая"]).groups[0]) == 3


class FakeSource:
    name = "openalex"

    def __init__(self, by_query):
        self.by_query = by_query
        self.calls = []

    def search(self, query, limit=20, year_from=None, year_to=None, language=None):
        self.calls.append(query)
        return self.by_query.get(query, [])


RU = record("Искусственный интеллект в ресторанном бизнесе", openalex_id="W1")
EN = record("Artificial intelligence in restaurants: a review", openalex_id="W2")
PUMPS = record("Оценка состояния насосного оборудования", openalex_id="W3")


def test_ingest_runs_every_query_and_counts_a_paper_once():
    source = FakeSource({"ии ресторан": [RU, PUMPS], "AI restaurant": [EN, RU]})
    with Library(":memory:") as library:
        report = ingest_topic(library, [source], topic="ИИ в общепите", queries=["ии ресторан", "AI restaurant", " "],
                              relevance=TermFilter.parse([FOOD]))
        assert source.calls == ["ии ресторан", "AI restaurant"]
        assert report.unique == 2 and report.added == 2 and report.off_topic == 1
        assert report.found == {"openalex": 4}
        assert library.topic_summary()[0]["papers"] == 2
        titles = [p["title"] for p in library.list_papers("ИИ в общепите")]
        assert PUMPS.title not in titles


def test_query_goes_first_and_topic_is_the_default_query():
    source = FakeSource({})
    with Library(":memory:") as library:
        ingest_topic(library, [source], topic="тема", query="главный", queries=["второй", "главный"])
        ingest_topic(library, [source], topic="тема")
    assert source.calls == ["главный", "второй", "тема"]


class FakeFetcher:
    def fetch(self, paper):
        return FulltextResult("ok", path=f"{paper['id']}.pdf", kind="pdf")


def test_add_references_keeps_going_after_a_bad_reference():
    def resolve(ref):
        if ref == "bad":
            raise ValueError("не разобрать")
        return {"10.1/a": RU, "10.1/b": EN}[ref]

    with Library(":memory:") as library:
        report = add_references(library, ["10.1/a", "bad", "10.1/b", "10.1/a", " "], "ИИ в общепите", resolve,
                                fetcher=FakeFetcher())
        assert report.added == 2 and report.unique == 2
        assert list(report.errors) == ["bad"]
        assert report.fulltext == {"ok": 2}
        again = add_references(library, ["10.1/a"], "ИИ в общепите", resolve)
        assert again.added == 0 and again.already_in_library == 1
        # чужие ожидающие статьи темы add не качает: их может в это время качать ingest
        waiting, _ = library.upsert_paper(PUMPS)
        library.link_topic(waiting, library.get_or_create_topic("ИИ в общепите"))
        add_references(library, ["10.1/b"], "ИИ в общепите", resolve, fetcher=FakeFetcher())
        assert library.get_paper(waiting)["fulltext_status"] == "pending"
