import numpy as np
import pytest

from research_assistant.agent.tools import ToolContext, ToolError, find_papers as find_papers_tool, search_library
from research_assistant.ingest.models import PaperRecord
from research_assistant.parse.passages import SectionOut, paragraph_passages
from research_assistant.parse.structure import Paragraph
from research_assistant.retrieval.papers import author_matches, find_papers
from research_assistant.retrieval.search import Searcher
from research_assistant.store.db import Library

PAPERS = [
    ("Speculative decoding with a weak draft model", ["Yaniv Leviathan", "Matan Kalman"], 2023, "en", "speculative",
     "We study how the acceptance rate of draft tokens limits the speedup."),
    ("Tree attention for verification", ["Tri Dao", "Avner May"], 2025, "en", "speculative",
     "Verification of many candidate continuations in one forward pass."),
    ("Искусственный интеллект в медицине", ["А. Барсуков", "М.Д. Панов"], 2024, "ru", "медицина",
     "Обзор применения нейронных сетей для диагностики заболеваний."),
]


@pytest.fixture
def lib():
    with Library(":memory:") as library:
        for title, authors, year, language, topic, abstract in PAPERS:
            paper_id, _ = library.upsert_paper(PaperRecord(title=title, sources=["openalex"], authors=authors,
                                                           year=year, language=language, abstract=abstract))
            library.link_topic(paper_id, library.get_or_create_topic(topic))
            section = SectionOut("Abstract", "abstract", 1)
            section.passages = paragraph_passages(Paragraph(abstract, [(0, 1)]))
            library.save_structure(paper_id, [section], status="ok", parser="pdf", version=1)
        yield library


def titles(hits):
    return [h.paper["title"] for h in hits]


def test_author_matches_surnames_initials_and_case():
    assert author_matches(["Tri Dao", "Avner May"], "dao")
    assert author_matches(["Tri Dao"], "Tri Dao")
    assert author_matches(["А. Барсуков"], "Барсуков А.")  # инициалы не мешают
    assert author_matches(["Пётр Ёлкин"], "елкин")
    assert not author_matches(["Tri Dao"], "Leviathan")
    assert not author_matches(["Tri Dao", "Avner May"], "Tri May")  # слова должны быть у одного автора


def test_find_by_author_and_filters(lib):
    assert titles(find_papers(lib, author="Dao")) == ["Tree attention for verification"]
    assert titles(find_papers(lib, author="барсуков")) == ["Искусственный интеллект в медицине"]
    # без запроса — все статьи под фильтрами, новые первыми
    assert titles(find_papers(lib, language="en")) == ["Tree attention for verification",
                                                         "Speculative decoding with a weak draft model"]
    assert titles(find_papers(lib, topic="медицина")) == ["Искусственный интеллект в медицине"]
    assert titles(find_papers(lib, year_from=2024, year_to=2024)) == ["Искусственный интеллект в медицине"]


def test_find_by_words_needs_all_words(lib):
    hits = find_papers(lib, "verification forward pass")
    assert titles(hits) == ["Tree attention for verification"] and hits[0].matched == ["title", "abstract"]
    assert find_papers(lib, "verification draft") == []  # слова из разных статей — ни одна не подходит целиком
    # словоформы: «диагностика» в запросе, «диагностики» в аннотации
    hits = find_papers(lib, "диагностика")
    assert titles(hits) == ["Искусственный интеллект в медицине"] and hits[0].matched == ["abstract"]
    assert find_papers(lib, "quantum chromodynamics") == []


class KeywordEmbedder:
    """Подменный эмбеддер: «смысл» — наличие ключевых слов на любом языке."""

    VOCAB = [("draft", "черновая", "черновой"), ("verification", "проверка"), ("медицин", "diagnos", "диагност")]

    def _vector(self, text):
        text = text.lower()
        vector = np.array([float(any(w in text for w in words)) for words in self.VOCAB]) + 1e-3
        return vector / np.linalg.norm(vector)

    def embed_documents(self, texts):
        return np.array([self._vector(t) for t in texts])

    def embed_query(self, text):
        return self._vector(text)


def semantic_searcher(lib, tmp_path):
    from research_assistant.retrieval.search import build_index
    from research_assistant.store.vectors import VectorIndex

    searcher = Searcher(lib, embedder=KeywordEmbedder(), index=VectorIndex(tmp_path / "v.npz"))
    build_index(lib, searcher.embedder, searcher.index)
    return searcher


def test_semantic_hits_come_after_word_matches(lib, tmp_path):
    searcher = semantic_searcher(lib, tmp_path)
    # русский запрос к английской аннотации: по словам не совпадает, по смыслу находится
    hits = find_papers(lib, "черновая модель", searcher=searcher)
    assert titles(hits)[0] == "Speculative decoding with a weak draft model"
    assert hits[0].matched == ["semantic"]
    hits = find_papers(lib, "verification", searcher=searcher)
    assert hits[0].matched == ["title", "abstract"]
    assert all(h.matched == ["semantic"] for h in hits[1:])


def test_find_papers_tool_and_search_by_paper_ids(lib):
    ctx = ToolContext(library=lib, searcher=Searcher(lib))
    text = find_papers_tool(ctx, author="Leviathan")
    assert 'paper 1: "Speculative decoding with a weak draft model"' in text and "Yaniv Leviathan" in text
    assert "search_library with paper_ids" in text
    assert "No papers found" in find_papers_tool(ctx, author="Nobody")
    with pytest.raises(ToolError):
        find_papers_tool(ctx)

    only_second = search_library(ctx, "verification draft tokens", paper_ids=[2])
    assert "paper 2:" in only_second and "paper 1:" not in only_second
