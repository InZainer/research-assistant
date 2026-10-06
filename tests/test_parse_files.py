"""Разбор настоящих файлов: PDF генерируется PyMuPDF прямо в тесте, HTML — по образцу CyberLeninka."""

import pymupdf
import pytest

from research_assistant.parse.html import cyberleninka_units
from research_assistant.parse.passages import document_passages
from research_assistant.parse.pdf import pdf_units
from research_assistant.parse.structure import build_document

BODY = "Speculative decoding drafts several tokens with a small model and verifies them in parallel."


def write_line(page, y, text, size=11, bold=False):
    page.insert_text((72, y), text, fontsize=size, fontname="hebo" if bold else "helv")


@pytest.fixture
def sample_pdf(tmp_path):
    doc = pymupdf.open()
    for number in (1, 2):
        page = doc.new_page(width=595, height=842)
        write_line(page, 30, "Proceedings of a Workshop 2026", size=9)  # колонтитул
        write_line(page, 820, str(number), size=11)  # номер страницы
        page.insert_text((20, 600), "arXiv:2601.00001v1 [cs.CL]", fontsize=20, rotate=90)  # штамп на полях
    first, second = doc[0], doc[1]
    write_line(first, 100, "A Tiny Test Paper", size=16, bold=True)
    write_line(first, 140, "Abstract", bold=True)
    write_line(first, 160, BODY)
    write_line(first, 200, "1 Introduction", size=13, bold=True)
    write_line(first, 220, "The idea is simple and it is explained in a para-")
    write_line(first, 234, "graph that goes on to the next page")
    write_line(first, 700, "1Footnote: code is available online for everyone.", size=8)
    write_line(second, 100, "and ends there with a full stop.")
    write_line(second, 140, "References", bold=True)
    write_line(second, 160, "Smith J. Some cited paper. 2020.")
    path = tmp_path / "paper.pdf"
    doc.save(path)
    doc.close()
    return path


def test_pdf_units_drop_headers_page_numbers_and_rotated_text(sample_pdf):
    units, n_pages = pdf_units(sample_pdf)
    texts = [u.text for u in units]
    assert n_pages == 2
    assert "Proceedings of a Workshop 2026" not in texts
    assert not any(t.startswith("arXiv:") for t in texts)
    assert "1" not in texts and "2" not in texts
    assert not any("Footnote" in t for t in texts)  # мелкий шрифт
    assert any(u.bold for u in units if u.text == "1 Introduction")


def test_pdf_end_to_end_structure(sample_pdf):
    units, _ = pdf_units(sample_pdf)
    document = build_document(units, title="A Tiny Test Paper")
    assert [(s.title, s.kind) for s in document.sections] == [
        ("Abstract", "abstract"), ("1 Introduction", "introduction")]
    sections = document_passages(document)
    [intro] = sections[1].passages
    assert intro.text == ("The idea is simple and it is explained in a paragraph that goes on to the next page "
                          "and ends there with a full stop.")
    assert intro.page == 1
    assert "Smith" not in " ".join(p.text for s in sections for p in s.passages)


HTML = """<html><head><meta name="citation_title" content="Машинное обучение в медицине"></head><body>
<div class="ocr">
<p>МАШИННОЕ ОБУЧЕНИЕ В МЕДИЦИНЕ</p>
<p>Иванов Иван Иванович - студент</p>
<p>АННОТАЦИЯ</p>
<p>Работа описывает применение машинного обучения в медицине и его ограничения.</p>
<p>Ключевые слова: машинное обучение; медицина</p>
<p>European science № 4 (53) ■ 18</p>
<p>Машинное обучение применяют для диагностики заболеваний по снимкам, т.е. для</p>
<p>классификации изображений врачами и алгоритмами (см. рис. 2).</p>
<p>Модель\t0.05\t1.23\t41.72</p>
<p>19 ■ European science № 4 (53)</p>
<p>ЗАКЛЮЧЕНИЕ</p>
<p>Машинное обучение полезно, но требует проверки врачом.</p>
<p>Список литературы:</p>
<p>1.\tИванов И. Статья. 2020.</p>
</div></body></html>"""


def test_cyberleninka_html_end_to_end():
    units = cyberleninka_units(HTML)
    texts = [u.text for u in units]
    assert not any("European science" in t for t in texts)  # колонтитулы обеих чётностей
    assert not any("0.05" in t for t in texts)  # строка таблицы
    assert all(u.page is None for u in units)

    document = build_document(units, title="Машинное обучение в медицине")
    assert [(s.title, s.kind) for s in document.sections] == [
        ("АННОТАЦИЯ", "abstract"), ("", "body"), ("ЗАКЛЮЧЕНИЕ", "conclusion")]
    body = document.sections[1].paragraphs[0].text
    assert body == ("Машинное обучение применяют для диагностики заболеваний по снимкам, т.е. для "
                    "классификации изображений врачами и алгоритмами (см. рис. 2).")
    assert "Иванов" not in " ".join(p.text for s in document.sections for p in s.paragraphs)


def test_html_without_text_container():
    assert cyberleninka_units("<html><body><p>nothing</p></body></html>") == []
