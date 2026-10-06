import json

import pytest

from research_assistant.ingest.models import PaperRecord
from research_assistant.store.db import Library


@pytest.fixture
def library():
    with Library(":memory:") as lib:
        yield lib


def arxiv_record(**overrides):
    values = dict(title="Decoding Speculative Decoding", sources=["arxiv"], arxiv_id="2402.01528", year=2024,
                  abstract="short", authors=["A"])
    values.update(overrides)
    return PaperRecord(**values)


def test_insert_then_merge_by_other_key(library):
    paper_id, created = library.upsert_paper(arxiv_record())
    assert created

    from_openalex = PaperRecord(title="Something else", sources=["openalex"], openalex_id="W4391556558",
                                doi="https://doi.org/10.48550/arxiv.2402.01528", venue="arXiv",
                                abstract="a longer abstract text", authors=["A", "B"])
    same_id, created = library.upsert_paper(from_openalex)
    assert (same_id, created) == (paper_id, False)

    row = library.get_paper(paper_id)
    assert row["title"] == "Decoding Speculative Decoding"
    assert row["openalex_id"] == "W4391556558"
    assert row["abstract"] == "a longer abstract text"
    assert json.loads(row["authors"]) == ["A", "B"]
    assert row["sources"] == "arxiv,openalex"
    assert row["fulltext_status"] == "pending"


def test_merge_by_title_when_ids_differ(library):
    first, _ = library.upsert_paper(arxiv_record())
    published = PaperRecord(title="Decoding speculative decoding.", sources=["openalex"],
                            doi="10.3929/ethz-c-000806091", year=2025)
    second, created = library.upsert_paper(published)
    assert second == first and not created
    assert library.get_paper(first)["doi"] == "10.3929/ethz-c-000806091"
    assert library.get_paper(first)["year"] == 2024


def test_does_not_steal_unique_key_from_another_paper(library):
    library.upsert_paper(PaperRecord(title="Paper with the doi we will clash", sources=["openalex"], doi="10.1/x"))
    target, _ = library.upsert_paper(arxiv_record())
    clash = PaperRecord(title="Decoding Speculative Decoding", sources=["openalex"], doi="10.1/x")
    # находится по DOI первая статья, поэтому вторая не трогается
    hit, created = library.upsert_paper(clash)
    assert not created and hit != target
    assert library.get_paper(target)["doi"] is None


def test_topics_links_and_summary(library):
    topic = library.get_or_create_topic("speculative  decoding")
    assert library.get_or_create_topic("speculative decoding") == topic
    paper_id, _ = library.upsert_paper(arxiv_record())
    ru_id, _ = library.upsert_paper(PaperRecord(title="Машинное обучение в бизнесе", sources=["openalex"],
                                                openalex_id="W2894934818", language="ru"))
    assert library.link_topic(paper_id, topic)
    assert not library.link_topic(paper_id, topic)
    library.link_topic(ru_id, topic)
    library.set_fulltext(paper_id, "ok", path="data/fulltext/1.pdf", kind="pdf")

    assert library.topic_summary() == [{"name": "speculative decoding", "papers": 2, "fulltext": 1, "russian": 1}]
    assert [p["id"] for p in library.papers_for_fulltext(topic)] == [ru_id]
    assert [p["id"] for p in library.list_papers("speculative decoding")] == [paper_id, ru_id]


def test_empty_topic_in_summary(library):
    library.get_or_create_topic("empty")
    assert library.topic_summary() == [{"name": "empty", "papers": 0, "fulltext": 0, "russian": 0}]
