from pathlib import Path

import pytest

from research_assistant.agent.loop import AgentResult, Claim
from research_assistant.answer.citations import load_citations
from research_assistant.ingest.models import PaperRecord
from research_assistant.parse.passages import SectionOut, paragraph_passages
from research_assistant.parse.structure import Paragraph
from research_assistant.retrieval.papers import PaperHit
from research_assistant.retrieval.search import Searcher
from research_assistant.store.db import Library
from research_assistant.ui.chats import ChatStore
from research_assistant.ui.views import (
    answer_view,
    describe_event,
    friendly_error,
    highlight,
    hit_rows,
    link_label,
    md_escape,
    paper_card,
    sentence_context,
    source_link,
)

APP = Path(__file__).parents[1] / "src" / "research_assistant" / "ui" / "app.py"


def fill(lib: Library) -> None:
    topic = lib.get_or_create_topic("speculative decoding")
    papers = [
        (PaperRecord(title="Speculative decoding with draft models", sources=["arxiv"], year=2024, language="en",
                     arxiv_id="2401.00001", authors=["Yaniv Leviathan", "Matan Kalman"],
                     abstract="Draft models propose tokens that the target model verifies."),
         "Speedup drops when the draft model disagrees with the target model. Rare tokens are the main cause."),
        (PaperRecord(title="Tree attention for verification", sources=["openalex"], year=2025, language="en",
                     doi="10.1000/tree", authors=["Tri Dao"]),
         "Tree-based verification checks many candidate continuations in one forward pass. It needs more memory."),
    ]
    for record, text in papers:
        paper_id, _ = lib.upsert_paper(record)
        lib.link_topic(paper_id, topic)
        section = SectionOut("Method", "method", 1)
        section.passages = paragraph_passages(Paragraph(text, [(0, 3)]))
        lib.save_structure(paper_id, [section], status="ok", parser="pdf", version=1)


@pytest.fixture
def lib():
    with Library(":memory:") as library:
        fill(library)
        yield library


def sentence_id(lib, startswith):
    return lib.conn.execute("SELECT id FROM sentences WHERE text LIKE ?", (startswith + "%",)).fetchone()[0]


def answered(lib) -> AgentResult:
    sid = sentence_id(lib, "Speedup drops")
    citations = load_citations(lib, [sid])
    return AgentResult(question="Почему?", answer="Из-за расхождения моделей.",
                       claims=[Claim("Ускорение падает.", [citations[sid]])], steps=2, tool_calls=2)


def test_source_link_prefers_pdf_page():
    assert source_link({"pdf_url": "https://x/a.pdf"}, 3) == "https://x/a.pdf#page=3"
    assert source_link({"arxiv_id": "2401.00001"}, 2) == "https://arxiv.org/pdf/2401.00001#page=2"
    assert source_link({"landing_url": "https://cyberleninka.ru/article/n/x"}, 5) == "https://cyberleninka.ru/article/n/x"
    assert source_link({"doi": "10.1/x"}) == "https://doi.org/10.1/x"
    assert source_link({}) is None
    assert link_label("https://x/a.pdf#page=3", 3) == "Открыть PDF на с. 3"
    assert link_label("https://doi.org/10.1/x", None) == "Открыть источник"


def test_sentence_context_splits_passage_around_quote(lib):
    sid = sentence_id(lib, "Rare tokens")
    before, sentence, after = sentence_context(lib, sid)
    assert sentence == "Rare tokens are the main cause."
    assert before.startswith("Speedup drops") and after == ""
    assert sentence_context(lib, 10**6) is None


def test_answer_view_builds_citation_cards(lib):
    view = answer_view(lib, answered(lib), steps=["Ищу в библиотеке: «x»"])
    [(claim, [card])] = view.claims
    assert claim == "Ускорение падает."
    assert card.quote == "Speedup drops when the draft model disagrees with the target model."
    assert card.where == "Speculative decoding with draft models (2024), раздел «Method», с. 3"
    assert card.link == "https://arxiv.org/pdf/2401.00001#page=3" and card.page == 3
    assert card.context[1] == card.quote
    assert view.verified and view.steps == ["Ищу в библиотеке: «x»"] and "шагов 2" in view.stats


def test_answer_view_marks_unverified_answer(lib):
    view = answer_view(lib, AgentResult(question="?", answer="", stop_reason="max_steps"))
    assert not view.verified and view.answer == "(ассистент не дал ответа)"


def test_describe_event_speaks_human():
    assert describe_event("tool", {"name": "search_library", "arguments": '{"query": "draft speedup"}'}) == \
        "Ищу в библиотеке: «draft speedup»"
    # поиск по самому вопросу (prefetch) пишется строкой, а не JSON
    assert describe_event("tool", {"name": "search_library", "arguments": "Почему?"}) == "Ищу в библиотеке: «Почему?»"
    assert describe_event("tool", {"name": "find_papers", "arguments": '{"author": "Tri Dao"}'}) == \
        "Ищу статьи: автор Tri Dao"
    assert describe_event("tool", {"name": "search_external", "arguments": '{"author": "Geoffrey Hinton"}'}) == \
        "Ищу новые статьи в OpenAlex и arXiv: автор Geoffrey Hinton"
    assert describe_event("tool", {"name": "final_answer", "arguments": "{}", "error": None}) == \
        "Ответ готов, цитаты проверены"
    assert "не получилось" in describe_event("tool", {"name": "read_passage", "arguments": "{}", "error": "bad id"})
    assert describe_event("llm", {}) is None


def test_friendly_error_explains_missing_model_server():
    class APIConnectionError(Exception):
        pass

    text = friendly_error(APIConnectionError("refused"), "http://localhost:1234/v1", "qwen3.5-4b")
    assert "LM Studio" in text and "qwen3.5-4b" in text
    assert "ValueError" in friendly_error(ValueError("x"), None, None)


def test_chat_store_round_trip_and_history(lib, tmp_path):
    store = ChatStore(tmp_path / "chats")
    chat = store.new("  Почему   падает ускорение спекулятивного декодирования при слабой черновой модели и что делать?")
    assert chat.title.endswith("…") and "  " not in chat.title
    for _ in range(4):
        chat.messages.append(answer_view(lib, answered(lib), steps=["шаг"]))
    store.save(chat)

    [loaded] = store.list()
    assert loaded.id == chat.id and len(loaded.messages) == 4
    card = loaded.messages[0].claims[0][1][0]
    assert card.quote.startswith("Speedup drops") and isinstance(card.context, tuple)
    assert [m["role"] for m in loaded.history(memory=3)] == ["user", "assistant"] * 3

    store.delete(chat.id)
    assert store.list() == [] and store.load(chat.id) is None
    with pytest.raises(ValueError):
        store.load("../secret")


def test_chat_store_skips_broken_files(tmp_path):
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    assert ChatStore(tmp_path).list() == []


def test_hit_rows_and_paper_card(lib):
    hits = Searcher(lib).search("draft model speedup", k=2, mode="bm25")
    [first, *_] = hit_rows(lib, hits)
    assert first["title"] == "Speculative decoding with draft models"
    assert first["link"] == "https://arxiv.org/pdf/2401.00001#page=3" and first["section"] == "Method"
    card = paper_card(lib, PaperHit(dict(lib.get_paper(1)), 1.0, ["title", "semantic"]))
    assert card["authors"] == "Yaniv Leviathan, Matan Kalman" and card["fulltext"]
    assert card["topics"] == ["speculative decoding"]
    assert card["matched"] == ["совпадение в названии", "похожа по смыслу"]


def test_md_escape_and_highlight():
    assert md_escape("cost $O(n)$ * 2") == r"cost \$O\(n\)\$ \* 2"
    assert highlight("Draft models are slow.", "черновая draft model") == \
        ":orange-background[Draft] :orange-background[models] are slow\\."
    assert highlight("a*b", None) == "a\\*b"


def app_test(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    with Library(tmp_path / "library.sqlite") as library:
        fill(library)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("EMBED_MODEL", "")  # без сервера эмбеддингов
    return AppTest.from_file(str(APP), default_timeout=30)


def test_app_pages_render_without_model(tmp_path, monkeypatch):
    app = app_test(tmp_path, monkeypatch).run()
    assert not app.exception
    pages = app.sidebar.radio[0]
    assert pages.options == ["💬 Вопрос по статьям", "📄 Статьи", "🔍 Поиск цитат"]
    assert any("Задайте вопрос" in md.value for md in app.markdown)  # приветствие с примерами

    app = pages.set_value("📄 Статьи").run()
    assert not app.exception
    text = " ".join(md.value for md in app.markdown)
    assert "Speculative decoding with draft models" in text and "Tree attention" in text

    app.text_input[1].input("Dao")  # поле «Автор»
    app = next(b for b in app.button if b.label == "Найти").click().run()
    text = " ".join(md.value for md in app.markdown)
    assert "Tree attention" in text and "Speculative decoding with draft models" not in text


def test_app_opens_saved_chat(tmp_path, monkeypatch):
    with Library(":memory:") as library:
        fill(library)
        store = ChatStore(tmp_path / "chats")
        chat = store.new("Почему падает ускорение?")
        chat.messages.append(answer_view(library, answered(library), steps=["Ищу в библиотеке: «x»"]))
        store.save(chat)
    app = app_test(tmp_path, monkeypatch).run()
    history = [b for b in app.sidebar.button if b.label == "Почему падает ускорение?"]
    assert history, "сохранённый чат есть в боковой панели"
    app = history[0].click().run()
    assert not app.exception
    text = " ".join(md.value for md in app.markdown)
    assert "Из\\-за расхождения моделей\\." in text and "Speedup drops" in text


def test_external_card_marks_papers_already_in_library(lib):
    from research_assistant.ui.views import external_card

    known = PaperRecord(title="Speculative decoding with draft models", sources=["openalex"], arxiv_id="2401.00001")
    new = PaperRecord(title="Layer normalization", sources=["arxiv"], arxiv_id="1607.06450", year=2016,
                      authors=["Jimmy Lei Ba", "Jamie Kiros", "Geoffrey E. Hinton"])
    assert external_card(lib, known)["in_library"]
    card = external_card(lib, new)
    assert not card["in_library"] and card["has_pdf"] and card["id"] == "arxiv:1607.06450"
    assert card["link"] == "https://arxiv.org/pdf/1607.06450" and card["authors"].endswith("Geoffrey E. Hinton")


def test_found_papers_survive_saving_and_show_under_answer(tmp_path, monkeypatch):
    found = PaperRecord(title="Layer normalization", sources=["arxiv"], arxiv_id="1607.06450", year=2016,
                        authors=["Jimmy Lei Ba", "Geoffrey E. Hinton"])
    with Library(":memory:") as library:
        fill(library)
        view = answer_view(library, AgentResult(question="Найди статьи Hinton", answer="Нашёл одну.", not_found=True),
                           found=[found])
        store = ChatStore(tmp_path / "chats")
        chat = store.new("Найди статьи Hinton")
        chat.messages.append(view)
        store.save(chat)
    assert store.load(chat.id).messages[0].found_papers[0]["arxiv_id"] == "1607.06450"

    app = app_test(tmp_path, monkeypatch).run()
    app = next(b for b in app.sidebar.button if b.label == "Найди статьи Hinton").click().run()
    assert not app.exception
    text = " ".join(md.value for md in app.markdown)
    assert "Layer normalization" in text and "Найдено в интернете: 1" in text
    assert [c.label for c in app.checkbox] == ["Добавить в библиотеку"]
    assert not any("не нашлось мест" in i.value for i in app.info)  # «не найдено в библиотеке» не путает
