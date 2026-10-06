import pymupdf
import pytest

from research_assistant.ingest.models import PaperRecord
from research_assistant.parse.pipeline import PARSER_VERSION, parse_paper, parse_papers
from research_assistant.store.db import Library


@pytest.fixture
def library():
    with Library(":memory:") as lib:
        yield lib


def make_pdf(path):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "Abstract", fontsize=11, fontname="hebo")
    page.insert_text((72, 120), "Drafting with a small model speeds up decoding (see Fig. 1). It is exact.",
                     fontsize=11, fontname="helv")
    doc.save(path)
    doc.close()


def add_paper(library, title, **fulltext):
    paper_id, _ = library.upsert_paper(PaperRecord(title=title, sources=["openalex"],
                                                   abstract=fulltext.pop("abstract", None)))
    if fulltext:
        library.set_fulltext(paper_id, **fulltext)
    return library.get_paper(paper_id)


def test_parse_pdf_stores_exact_sentence_slices(library, tmp_path):
    path = tmp_path / "1.pdf"
    make_pdf(path)
    paper = add_paper(library, "Speculative decoding paper", status="ok", path=str(path), kind="pdf")

    outcome = parse_paper(library, paper)
    assert (outcome.status, outcome.parser, outcome.sentences) == ("ok", "pdf", 2)

    [section] = library.paper_structure(paper["id"])
    assert section["kind"] == "abstract"
    [passage] = section["passages"]
    assert passage["source"] == "fulltext" and passage["page"] == 1
    for sentence in passage["sentences"]:
        assert passage["text"][sentence["char_start"]:sentence["char_end"]] == sentence["text"]
    assert passage["sentences"][0]["text"].endswith("(see Fig. 1).")
    assert library.get_parse(paper["id"])["parser_version"] == PARSER_VERSION


def test_metadata_fallback_and_empty(library):
    with_abstract = add_paper(library, "Paper with abstract only", abstract="An abstract. Two sentences.")
    nothing = add_paper(library, "Paper with nothing at all here")
    assert parse_paper(library, with_abstract).status == "metadata"
    assert parse_paper(library, nothing).status == "empty"
    [section] = library.paper_structure(with_abstract["id"])
    assert section["passages"][0]["source"] == "metadata"
    assert section["passages"][0]["page"] is None


def test_broken_pdf_fails_but_keeps_metadata(library, tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.7 not really a pdf")
    paper = add_paper(library, "Broken file paper title", abstract="Still searchable.",
                      status="ok", path=str(path), kind="pdf")
    outcome = parse_paper(library, paper)
    assert outcome.status == "failed" and outcome.error
    assert outcome.passages == 1


def test_reparse_replaces_structure_and_respects_version(library):
    add_paper(library, "Paper with abstract only", abstract="An abstract.")
    assert sum(parse_papers(library).values()) == 1
    assert sum(parse_papers(library).values()) == 0  # уже разобрано текущей версией
    assert sum(parse_papers(library, force=True).values()) == 1
    count = library.conn.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
    assert count == 1  # старый разбор удалён, а не продублирован
