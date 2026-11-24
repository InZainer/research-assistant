import pymupdf

from research_assistant.agent.tools import ToolContext, search_library
from research_assistant.ingest.models import PaperRecord
from research_assistant.parse.passages import document_passages, paragraph_passages
from research_assistant.parse.pdf import pdf_units, table_rows
from research_assistant.parse.pipeline import parse_paper
from research_assistant.parse.structure import Paragraph, build_document
from research_assistant.parse.units import Unit
from research_assistant.retrieval.search import Searcher
from research_assistant.store.db import Library


def line(text, x, y):
    return (text, float(x), float(y), float(y) + 10)


def test_cells_on_one_height_make_a_row():
    lines = [line("Algorithm", 50, 100), line("Starbucks", 150, 100), line("McDon.", 220, 100),
             line("SVR", 50, 112), line("0.73", 150, 112), line("0.81", 220, 112),
             line("RankNet", 50, 124), line("0.72", 150, 124), line("0.79", 220, 124)]
    assert table_rows(lines) == ["Algorithm | Starbucks | McDon.", "SVR | 0.73 | 0.81", "RankNet | 0.72 | 0.79"]


def test_row_given_as_one_line_is_split_at_trailing_numbers():
    lines = [line("Feature Starbucks Dun.Don.", 50, 100), line("Density 0.60 0.79 0.73", 50, 112),
             line("Jensen Quality 0.54 0.81 0.72", 50, 124), line("Entropy 65% 1,204 0.69", 50, 136)]
    assert table_rows(lines)[1:] == ["Density | 0.60 | 0.79 | 0.73", "Jensen Quality | 0.54 | 0.81 | 0.72",
                                     "Entropy | 65% | 1,204 | 0.69"]


def test_prose_is_not_a_table():
    prose = [line("is the top feature for Starbucks and McDonald's scoring 0.70 and", 50, 100),
             line("0.78 respectively. Interestingly, as suggested by the results,", 50, 112),
             line("the NDCG@10 score of the best feature grows to 0.66 and 0.81", 50, 124),
             line("when the competitive features are added to the model.", 50, 136)]
    assert table_rows(prose) is None
    assert table_rows([line("Table 5 | 0.73 | 0.81", 50, 100)]) is None  # одна строка — не таблица


def test_table_units_become_one_citable_paragraph():
    units = [
        Unit("Results", 1, 1, bold=True, size=1.3),
        Unit("The ranking of the features is stable across the three chains that we study in this", 1, 2, size=1.0),
        Unit("Feature | Starbucks | McDon.", 1, 3, size=1.0, kind="table"),
        Unit("Density | 0.60 | 0.73", 1, 3, size=1.0, kind="table"),
        Unit("Entropy | 0.65 | 0.69", 1, 4, size=1.0, kind="table"),  # соседний блок той же таблицы
        Unit("paper, and the mobility features are the strongest ones.", 1, 5, size=1.0),
    ]
    document = build_document(units)
    paragraphs = document.sections[0].paragraphs
    assert [p.kind for p in paragraphs] == ["text", "table"]
    assert paragraphs[1].text == "Feature | Starbucks | McDon.\nDensity | 0.60 | 0.73\nEntropy | 0.65 | 0.69"
    assert paragraphs[0].text.endswith("strongest ones.")  # предложение продолжилось после таблицы

    passage = document_passages(document)[0].passages[1]
    assert passage.kind == "table"
    rows = [passage.text[s.start:s.end] for s in passage.sentences]
    assert rows == ["Feature | Starbucks | McDon.", "Density | 0.60 | 0.73", "Entropy | 0.65 | 0.69"]


def test_long_table_is_cut_between_rows():
    text = "\n".join(f"model {i} | 0.{i:02d} | 0.{i + 1:02d}" for i in range(60))
    passages = paragraph_passages(Paragraph(text, [(0, 4)], kind="table"), max_words=50)
    assert len(passages) > 1 and all(p.kind == "table" for p in passages)
    rows = [p.text[s.start:s.end] for p in passages for s in p.sentences]
    assert rows == text.split("\n")


def make_pdf(path):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 80), "Abstract", fontsize=11, fontname="hebo")
    page.insert_text((72, 100), "We compare ranking models for store placement on check-in data.", fontsize=11)
    page.insert_text((72, 114), "The table below reports the average NDCG at ten for every chain.", fontsize=11)
    for row, cells in enumerate([("Algorithm", "Starbucks", "McDon."), ("SVR", "0.73", "0.81"),
                                 ("RankNet", "0.72", "0.79"), ("Baseline", "0.48", "0.53")]):
        for column, cell in enumerate(cells):
            page.insert_text((72 + 110 * column, 160 + 12 * row), cell, fontsize=8)  # мельче основного текста
    doc.save(path)
    doc.close()


def test_pdf_table_is_parsed_stored_and_shown_to_the_agent(tmp_path):
    path = tmp_path / "paper.pdf"
    make_pdf(path)
    units, _ = pdf_units(path)
    assert [u.text for u in units if u.kind == "table"] == [
        "Algorithm | Starbucks | McDon.", "SVR | 0.73 | 0.81", "RankNet | 0.72 | 0.79", "Baseline | 0.48 | 0.53"]

    with Library(":memory:") as library:
        paper_id, _ = library.upsert_paper(PaperRecord(title="Store placement", sources=["openalex"]))
        library.set_fulltext(paper_id, "ok", str(path), "pdf")
        assert parse_paper(library, library.get_paper(paper_id)).status == "ok"
        rows = library.conn.execute(
            """SELECT s.text, substr(p.text, s.char_start + 1, s.char_end - s.char_start) AS cut
               FROM sentences s JOIN passages p ON p.id = s.passage_id WHERE p.kind = 'table'""").fetchall()
        assert [r["text"] for r in rows][1] == "SVR | 0.73 | 0.81"
        assert all(r["text"] == r["cut"] for r in rows)  # инвариант цитат выполняется и для строк таблицы
        out = search_library(ToolContext(library=library, searcher=Searcher(library)), "RankNet Starbucks")
        assert "[table] [s" in out and "RankNet | 0.72 | 0.79" in out
