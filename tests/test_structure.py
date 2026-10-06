from research_assistant.parse.passages import document_passages, page_at, paragraph_passages
from research_assistant.parse.structure import (
    Paragraph,
    build_document,
    continues,
    heading_candidate,
    join_lines,
    section_kind,
)
from research_assistant.parse.units import Unit, clean_text, header_key

BODY = "This is an ordinary body sentence that is long enough to keep."


class Doc:
    """Конструктор строк: каждый вызов — новый блок, если не передан block."""

    def __init__(self):
        self.units = []
        self.block = 0

    def line(self, text, page=1, bold=False, size=1.0, y=0.3, block=None):
        if block is None:
            self.block += 1
            block = self.block
        self.units.append(Unit(text=text, page=page, block=block, bold=bold, size=size, y=y))
        return self

    def heading(self, text, page=1):
        return self.line(text, page=page, bold=True, size=1.2)


def titles(document):
    return [(s.title, s.kind) for s in document.sections]


# --- отдельные правила ---

def test_heading_candidate_rules():
    assert heading_candidate(Unit("3 Speculative Decoding", 1, 1, bold=True))
    assert heading_candidate(Unit("UNFAIRNESS MITIGATION", 1, 1))  # капс без жирного (ICLR)
    assert heading_candidate(Unit("Введение", 1, 1))  # известное название без оформления
    assert heading_candidate(Unit("1", 1, 1))  # номер раздела отдельной строкой
    assert not heading_candidate(Unit("This sentence ends with a period.", 1, 1, bold=True))
    assert not heading_candidate(Unit("• Intermediate variables", 1, 1, bold=True))
    assert not heading_candidate(Unit("MAX_TOK=512; LR=1e-4", 1, 1))
    assert not heading_candidate(Unit("Figure 2: Overview", 1, 1, bold=True))
    assert not heading_candidate(Unit("lowercase start", 1, 1, bold=True))


def test_section_kinds():
    assert section_kind("1 Introduction") == "introduction"
    assert section_kind("References / Список литературы") == "references"
    assert section_kind("Список литературы:") == "references"
    assert section_kind("4.3 Analysis") == "discussion"
    assert section_kind("ЗАКЛЮЧЕНИЕ") == "conclusion"
    assert section_kind("Speculative Decoding") == "other"


def test_continuation_rules():
    assert continues("the draft model is", "small and fast.")
    assert continues("as follows:", "• first item")
    assert continues("speed-up on task T is", "ˆy≤j and simultaneously")
    assert not continues("A finished sentence.", "lowercase continuation")
    assert not continues("no period here", "Capitalized new paragraph")
    assert not continues("no period here", "• bullet")


def test_dehyphenation_uses_document_vocabulary():
    assert join_lines("speculative decod-", "ing works", set()) == "speculative decoding works"
    assert join_lines("the draft-", "then-verify idea", {"draft-then-verify"}) == "the draft-then-verify idea"
    assert join_lines("model Llama-", "2 is", set()) == "model Llama-2 is"
    assert join_lines("first line", "second line", set()) == "first line second line"


def test_clean_text_and_header_key():
    soft_hyphen, bom, zero_width = chr(0xAD), chr(0xFEFF), chr(0x200B)
    assert clean_text(f"{bom}de{soft_hyphen}coding  {zero_width} x") == "decoding x"
    assert header_key("European science № 4 (53) ■ 18") == header_key("19 ■ European science № 4 (53)")
    assert header_key("Table 1") is None


def test_page_tracking_in_passages():
    paragraph = Paragraph("First on page one. Second starts on page two.", [(0, 1), (19, 2)])
    [passage] = paragraph_passages(paragraph)
    assert [s.page for s in passage.sentences] == [1, 2]
    assert page_at([(0, 3)], 100) == 3


def test_long_paragraph_split_on_sentence_boundaries():
    text = " ".join(f"Sentence number {i} has exactly seven words." for i in range(80))
    passages = paragraph_passages(Paragraph(text, [(0, 1)]), max_words=100)
    assert len(passages) > 1
    assert all(len(p.text.split()) <= 100 for p in passages)
    assert " ".join(p.text for p in passages) == text


# --- сборка документа ---

def test_structure_of_a_typical_paper():
    d = Doc()
    d.line("A Great Paper Title About Things", bold=True, size=1.5)
    d.line("Author Name, University")
    d.heading("Abstract").line(f"{BODY} Abstract text.")
    d.heading("1").line("Introduction", bold=True, size=1.2, block=d.block)
    d.line("The method works well and we explain why in a para-")
    d.line("graph that continues in the next block", page=1)
    d.line("and finishes on the second page.", page=2)
    d.line("c = arg max", page=2)  # формула
    d.line("1Footnote text with details here.", page=2, y=0.9)
    d.line("Figure 1: A caption of the figure.", page=2)
    d.heading("2 Method", page=2).line(BODY, page=2)
    d.heading("2.1 Speedup Details", page=2).line(BODY, page=2)
    d.heading("References", page=2).line("Smith J. A paper. 2020.", page=2)
    d.heading("A Implementation Details", page=3).line(BODY, page=3)

    document = build_document(d.units, title="A Great Paper Title About Things")
    assert document.has_abstract
    assert titles(document) == [
        ("Abstract", "abstract"), ("1 Introduction", "introduction"), ("2 Method", "method"),
        ("2.1 Speedup Details", "method"), ("A Implementation Details", "appendix"),
    ]
    intro = document.sections[1]
    texts = [p.text for p in intro.paragraphs]
    assert "paragraph that continues in the next block and finishes on the second page." in texts[0]
    assert intro.paragraphs[0].pages == [(0, 1), (texts[0].index("and finishes"), 2)]
    assert [p.kind for p in intro.paragraphs] == ["text", "caption"]
    everything = " ".join(p.text for s in document.sections for p in s.paragraphs)
    assert "arg max" not in everything and "Footnote" not in everything and "Smith J." not in everything


def test_lettered_heading_before_references_is_not_appendix():
    d = Doc().heading("Abstract").line(BODY).heading("A Simple Baseline").line(BODY)
    assert titles(build_document(d.units))[1] == ("A Simple Baseline", "other")


def test_bold_paragraph_is_not_a_heading():
    d = Doc().heading("Abstract").line(BODY)
    d.line("Altogether our method is on average", bold=True)
    d.line("thirty percent faster than baselines.", bold=True, block=d.block)
    document = build_document(d.units)
    assert titles(document) == [("Abstract", "abstract")]
    assert document.sections[0].paragraphs[-1].text.startswith("Altogether")


def test_empty_parent_section_is_kept():
    d = Doc().heading("Abstract").line(BODY).heading("4 Experiments").heading("4.1 Settings").line(BODY)
    assert titles(build_document(d.units)) == [
        ("Abstract", "abstract"), ("4 Experiments", "experiments"), ("4.1 Settings", "experiments")]


def test_collection_article_found_by_title_and_cut_at_next_article():
    d = Doc()
    d.line("Tail of the previous article in the proceedings volume.", page=1)
    d.line("Применение машинного обучения в медицине", page=1, bold=True)
    d.line("Иванов И.И.", page=1)
    d.heading("Аннотация").line("Короткая аннотация статьи о машинном обучении в медицине.")
    d.line("Ключевые слова: машинное обучение, медицина")
    d.heading("Введение")
    for _ in range(30):
        d.line("Основной текст статьи содержит достаточно много слов для проверки.", page=2)
    d.line("Следующая статья сборника", page=3, bold=True)
    d.heading("Аннотация", page=3).line("Аннотация чужой статьи, которую брать нельзя.", page=3)

    document = build_document(d.units, title="Применение машинного обучения в медицине")
    assert titles(document) == [("Аннотация", "abstract"), ("Введение", "introduction")]
    everything = " ".join(p.text for s in document.sections for p in s.paragraphs)
    assert "previous article" not in everything and "чужой" not in everything and "Ключевые" not in everything


def test_runin_abstract_covers_only_its_paragraph():
    d = Doc()
    d.line("Аннотация: краткое изложение работы в одном абзаце текста.", page=None)
    d.line("Основной текст начинается здесь и идёт дальше без заголовков.", page=None)
    document = build_document(d.units)
    assert titles(document) == [("Abstract", "abstract"), ("", "body")]


def test_document_passages_keeps_section_order():
    d = Doc().heading("Abstract").line(BODY).heading("1 Introduction").line(BODY)
    sections = document_passages(build_document(d.units))
    assert [s.kind for s in sections] == ["abstract", "introduction"]
    assert all(s.passages[0].sentences for s in sections)


def test_whole_journal_issue_picks_title_followed_by_abstract():
    title = "Цифровые технологии в продажах сельскохозяйственной продукции"
    d = Doc()
    d.line("Содержание", page=1, bold=True)
    d.line(title, page=1, bold=True)  # строка оглавления
    for n in range(50):
        d.line(f"Другая статья выпуска номер {n} с длинным названием.", page=2)
    d.line("1 Университет дружбы народов", page=9, bold=True)
    d.line(title, page=9, bold=True, block=d.block)  # заголовок слипся с аффилиацией: 12 слов вместе
    d.line("Аннотация: в статье анализируется цифровизация аграрного сектора.", page=9)
    d.heading("Введение", page=9).line(BODY, page=9)

    document = build_document(d.units, title=title)
    assert titles(document) == [("Abstract", "abstract"), ("Введение", "introduction")]
