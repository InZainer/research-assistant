import numpy as np
import pytest

from research_assistant.ingest.models import PaperRecord
from research_assistant.nlp.embed import OpenAIEmbedder, prefixes_for
from research_assistant.nlp.text import analyze, detect_language
from research_assistant.parse.passages import SectionOut, paragraph_passages
from research_assistant.parse.structure import Paragraph
from research_assistant.retrieval.search import Searcher, build_index, embedding_text, rrf
from research_assistant.store.db import Library, PassageFilter
from research_assistant.store.vectors import VectorIndex, index_path


# --- анализатор ---

def test_analyze_stems_both_languages_and_drops_stopwords():
    assert analyze("Обучения нейронных сетей") == analyze("обучение нейронной сети")
    assert analyze("The decoding of models") == analyze("decoded model")
    assert analyze("и в на the of") == []
    assert analyze("Ёлка") == analyze("елка")
    assert analyze("GPT-4 has 175 B") == ["gpt", "4", "175", "b"]


def test_detect_language():
    assert detect_language("Машинное обучение в медицине") == "ru"
    assert detect_language("Speculative decoding with LLM") == "en"


# --- библиотека с несколькими статьями ---

PAPERS = {
    "Speculative decoding with draft models": (2024, "en", {
        "method": "The draft model proposes several tokens and the target model verifies them in parallel.",
        "limitations": "Speedup drops when the draft model disagrees with the target model on rare tokens.",
    }),
    "Tree attention for verification": (2025, "en", {
        "method": "Tree-based verification checks many candidate continuations in one forward pass.",
    }),
    "Машинное обучение в диагностике": (2022, "ru", {
        "results": "Нейронные сети обучения на снимках достигли точности диагностики девяносто процентов.",
    }),
}


@pytest.fixture
def library():
    with Library(":memory:") as lib:
        topic = lib.get_or_create_topic("speculative decoding")
        for title, (year, language, sections) in PAPERS.items():
            paper_id, _ = lib.upsert_paper(PaperRecord(title=title, sources=["openalex"], year=year, language=language))
            if language == "en":
                lib.link_topic(paper_id, topic)
            outs = []
            for kind, text in sections.items():
                section = SectionOut(kind.title(), kind, 1)
                section.passages = paragraph_passages(Paragraph(text, [(0, 1)]))
                outs.append(section)
            lib.save_structure(paper_id, outs, status="ok", parser="pdf", version=1)
        yield lib


def passage_texts(library, ranked):
    found = library.passages_by_ids([i for i, _ in ranked])
    return [found[i]["text"] for i, _ in ranked]


def test_bm25_finds_by_stemmed_words(library):
    ranked = library.search_fts("verification of candidates", 5)  # candidates → candidate
    assert passage_texts(library, ranked)[0].startswith("Tree-based verification")
    ru = library.search_fts("обучение нейронной сети", 5)
    assert passage_texts(library, ru)[0].startswith("Нейронные сети")


def test_bm25_filters(library):
    assert len(library.search_fts("model", 10, PassageFilter(section_kinds=["limitations"]))) == 1
    assert library.search_fts("сети", 10, PassageFilter(topic="speculative decoding")) == []
    assert len(library.search_fts("model", 10, PassageFilter(year_from=2025))) == 0
    assert len(library.search_fts("verification", 10, PassageFilter(year_from=2025))) == 1
    assert library.search_fts("и в на", 5) == []  # только стоп-слова


def test_fts_follows_reparse(library):
    paper_id = library.conn.execute("SELECT id FROM papers WHERE title LIKE 'Tree%'").fetchone()[0]
    library.save_structure(paper_id, [], status="empty", parser=None, version=2)
    assert library.search_fts("tree verification", 5) == [] or all(
        library.passages_by_ids([i])[i]["paper_id"] != paper_id for i, _ in library.search_fts("tree verification", 5)
    )
    assert library.conn.execute("SELECT COUNT(*) FROM passages_fts").fetchone()[0] == \
        library.conn.execute("SELECT COUNT(*) FROM passages").fetchone()[0]


def test_existing_database_gets_fts_backfilled(tmp_path):
    path = tmp_path / "lib.sqlite"
    with Library(path) as lib:
        paper_id, _ = lib.upsert_paper(PaperRecord(title="Some paper on decoding speed", sources=["arxiv"]))
        section = SectionOut("Abstract", "abstract", 1)
        section.passages = paragraph_passages(Paragraph("Decoding speed matters a lot.", [(0, 1)]))
        lib.save_structure(paper_id, [section], status="ok", parser="pdf", version=1)
        lib.conn.execute("DELETE FROM passages_fts")
        lib.conn.commit()
    with Library(path) as lib:
        assert lib.search_fts("decoding", 5)


# --- RRF и плотный поиск ---

def test_rrf_prefers_items_high_in_both_lists():
    fused = rrf({"bm25": [1, 2, 3], "dense": [3, 1, 4]})
    assert [i for i, _, _ in fused][:2] == [1, 3]
    assert fused[0][2] == {"bm25": 1, "dense": 2}


class FakeEmbedder:
    """Детерминированные «эмбеддинги»: мешок основ слов по фиксированному словарю."""

    vocabulary = ["draft", "verif", "tree", "speedup", "сет", "диагност", "candid", "token"]

    def _vector(self, text):
        stems = analyze(text)
        v = np.array([sum(s.startswith(w) for s in stems) for w in self.vocabulary], dtype=np.float32) + 1e-3
        return v / np.linalg.norm(v)

    def embed_documents(self, texts):
        return np.stack([self._vector(t) for t in texts])

    def embed_query(self, text):
        return self._vector(text)


def test_build_index_and_hybrid_search(library, tmp_path):
    index = VectorIndex(tmp_path / "vectors.npz")
    stats = build_index(library, FakeEmbedder(), index)
    assert stats == {"total": 4, "added": 4, "removed": 0}
    assert build_index(library, FakeEmbedder(), VectorIndex.load(tmp_path / "vectors.npz"))["added"] == 0

    searcher = Searcher(library, FakeEmbedder(), index)
    dense = searcher.search("tree candidates", k=2, mode="dense")
    assert dense[0].passage["title"] == "Tree attention for verification"
    hybrid = searcher.search("draft model speedup", k=3, mode="hybrid")
    assert set(hybrid[0].ranks) == {"bm25", "dense"}
    assert hybrid[0].passage["section_kind"] in {"method", "limitations"}
    filtered = searcher.search("диагностика", k=3, mode="dense", where=PassageFilter(language="en"))
    assert all(h.passage["language"] == "en" for h in filtered)


def test_hybrid_without_embeddings_falls_back_to_bm25(library):
    hits = Searcher(library).search("verification", k=2, mode="hybrid")
    assert hits and set(hits[0].ranks) == {"bm25"}
    with pytest.raises(RuntimeError):
        Searcher(library).search("verification", mode="dense")


def test_vector_index_keeps_only_existing_ids(tmp_path):
    index = VectorIndex(tmp_path / "v.npz")
    index.add([1, 2, 3], np.eye(3, dtype=np.float32))
    assert index.keep_only({1, 3}) == 1
    assert index.ids.tolist() == [1, 3]
    assert index.missing([1, 2, 3]) == [2]
    with pytest.raises(ValueError):
        index.add([4], np.ones((1, 5), dtype=np.float32))
    assert index_path(tmp_path, "text-embedding/nomic v1.5").name == "text-embedding_nomic_v1.5.npz"


def test_embedding_text_and_prefixes():
    assert embedding_text({"title": "Paper", "section": "Method", "text": "Body."}) == "Paper — Method\n\nBody."
    assert prefixes_for("text-embedding-nomic-embed-text-v1.5") == ("search_query: ", "search_document: ")
    assert prefixes_for("intfloat/multilingual-e5-large") == ("query: ", "passage: ")
    assert prefixes_for("bge-m3") == ("", "")


def test_openai_embedder_batches_and_normalizes():
    class FakeClient:
        def __init__(self):
            self.calls = []
            self.embeddings = self

        def create(self, model, input):
            self.calls.append(list(input))
            data = [type("D", (), {"index": i, "embedding": [3.0, 4.0]}) for i in range(len(input))]
            return type("R", (), {"data": list(reversed(data))})

    client = FakeClient()
    embedder = OpenAIEmbedder("multilingual-e5-small", client=client)
    vectors = embedder.embed_documents([f"t{i}" for i in range(40)])
    assert vectors.shape == (40, 2) and np.allclose(np.linalg.norm(vectors, axis=1), 1)
    assert [len(c) for c in client.calls] == [32, 8]
    assert client.calls[0][0] == "passage: t0"
    embedder.embed_query("q")
    assert client.calls[-1] == ["query: q"]


# --- переранжирование и перевод запроса ---

class JsonReplyClient:
    """Подменная чат-модель: отвечает заданными текстами по очереди и запоминает запросы."""

    def __init__(self, replies):
        from types import SimpleNamespace

        self._ns = SimpleNamespace
        self.replies = list(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        message = self._ns(content=reply, tool_calls=None)
        return self._ns(choices=[self._ns(message=message)], usage=self._ns(prompt_tokens=10, completion_tokens=2))


def test_llm_reranker_reorders_and_keeps_unranked():
    from research_assistant.llm import ChatModel
    from research_assistant.retrieval.rerank import LLMReranker

    client = JsonReplyClient(['{"ranking": [3, 1, 99, 3]}', RuntimeError("model down")])
    reranker = LLMReranker(ChatModel("fake", client=client))
    candidates = [(10, "alpha"), (20, "beta"), (30, "gamma")]
    assert reranker.rerank("q", candidates) == [30, 10, 20]  # 99 и повтор отброшены, 20 дописан в конец
    assert client.requests[0]["response_format"]["type"] == "json_schema"
    assert reranker.rerank("q", candidates) == [10, 20, 30]  # сбой модели — исходный порядок
    assert reranker.tokens == 12


def test_query_translator_only_for_non_english_and_cached():
    from research_assistant.llm import ChatModel
    from research_assistant.retrieval.rerank import QueryTranslator

    client = JsonReplyClient(['"draft model verification"\nextra line'])
    translator = QueryTranslator(ChatModel("fake", client=client))
    assert translator.to_english("speculative decoding") is None
    assert translator.to_english("проверка черновой модели") == "draft model verification"
    assert translator.to_english("проверка черновой модели") == "draft model verification"
    assert len(client.requests) == 1


def test_search_with_translation_and_rerank(library):
    class Translator:
        def to_english(self, query):
            return "tree verification candidates"

    class ReverseReranker:
        def rerank(self, query, candidates):
            return [i for i, _ in reversed(candidates)]

    searcher = Searcher(library, translator=Translator(), reranker=ReverseReranker())
    plain = searcher.search("проверка деревом", k=3, mode="bm25")
    translated = searcher.search("проверка деревом", k=3, mode="bm25", translate=True)
    assert plain == [] and translated[0].passage["title"] == "Tree attention for verification"
    assert set(translated[0].ranks) == {"bm25_en"}
    reranked = searcher.search("model", k=10, mode="bm25", rerank=True)
    assert "rerank" in reranked[0].ranks
    baseline = searcher.search("model", k=10, mode="bm25")
    assert [h.passage_id for h in reranked] == [h.passage_id for h in reversed(baseline)]
