import pytest

from research_assistant.ingest.dedup import dedupe, merge_records
from research_assistant.ingest.models import PaperRecord, normalize_arxiv_id, normalize_doi, title_key


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("https://doi.org/10.1145/ABC.123", "10.1145/abc.123"),
        ("http://dx.doi.org/10.18411/trnio-12-2022-491", "10.18411/trnio-12-2022-491"),
        ("doi:10.1000/xyz", "10.1000/xyz"),
        ("not a doi", None),
        (None, None),
    ],
)
def test_normalize_doi(raw, expected):
    assert normalize_doi(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2203.16487v6", "2203.16487"),
        ("http://arxiv.org/abs/2203.16487v6", "2203.16487"),
        ("https://arxiv.org/pdf/2603.03251", "2603.03251"),
        ("arXiv:2402.01528", "2402.01528"),
        ("https://doi.org/10.48550/arxiv.2402.01528", "2402.01528"),
        ("10.48550/arXiv.2402.01528", "2402.01528"),
        ("cs.CL/0612001v2", "cs.cl/0612001"),
        ("https://doi.org/10.1145/123", None),
        ("", None),
    ],
)
def test_normalize_arxiv_id(raw, expected):
    assert normalize_arxiv_id(raw) == expected


def test_title_key_ignores_case_punctuation_and_yo():
    assert title_key("Decoding Speculative Decoding!") == title_key("decoding  speculative-decoding")
    assert title_key("Ёмкость нейронных сетей: обзор") == title_key("емкость нейронных сетей обзор")


def test_short_titles_are_not_keys():
    assert title_key("Introduction") is None


def test_record_derives_arxiv_id_from_doi():
    record = PaperRecord(title="Speculative Speculative Decoding", sources=["openalex"],
                         doi="https://doi.org/10.48550/arxiv.2603.03251")
    assert record.arxiv_id == "2603.03251"
    assert record.external_id == "arxiv:2603.03251"
    assert ("arxiv", "2603.03251") in record.keys()


def test_merge_prefers_first_fills_gaps_and_keeps_longest_abstract():
    arxiv = PaperRecord(title="Decoding Speculative Decoding", sources=["arxiv"], arxiv_id="2402.01528",
                        abstract="short", authors=["A"], year=2024)
    openalex = PaperRecord(title="Decoding speculative decoding", sources=["openalex"], openalex_id="W1",
                           abstract="a much longer abstract", authors=["A", "B"], venue="ICLR", year=2025)
    merged = merge_records(arxiv, openalex)
    assert merged.year == 2024
    assert merged.venue == "ICLR"
    assert merged.openalex_id == "W1"
    assert merged.abstract == "a much longer abstract"
    assert merged.authors == ["A", "B"]
    assert merged.sources == ["arxiv", "openalex"]


def test_dedupe_joins_by_any_key_transitively():
    records = [
        PaperRecord(title="Paper one about retrieval augmented generation", sources=["arxiv"], arxiv_id="2005.11401"),
        PaperRecord(title="Some other title entirely different", sources=["openalex"],
                    doi="10.48550/arXiv.2005.11401", openalex_id="W9"),
        PaperRecord(title="Third record", sources=["openalex"], openalex_id="W9", venue="NeurIPS"),
        PaperRecord(title="Unrelated paper on speculative decoding", sources=["arxiv"], arxiv_id="2203.16487"),
    ]
    merged = dedupe(records)
    assert len(merged) == 2
    assert merged[0].venue == "NeurIPS"
    assert merged[0].sources == ["arxiv", "openalex"]
