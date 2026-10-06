import json
from types import SimpleNamespace

import pytest

from research_assistant.agent.tools import ToolContext, ToolError, list_entities, search_library
from research_assistant.evaluation.entities import GoldEntities, evaluate_entities, prf, score_type
from research_assistant.ingest.models import PaperRecord
from research_assistant.llm import ChatModel
from research_assistant.nlp.annotate import annotate_papers, annotation_text
from research_assistant.nlp.entities import clean, contains_stems, entity_key, occurs_in, parse_json, split_abbreviation
from research_assistant.nlp.keywords import extract_keywords
from research_assistant.parse.passages import SectionOut, paragraph_passages
from research_assistant.parse.structure import Paragraph
from research_assistant.retrieval.search import Searcher
from research_assistant.store.db import Library, PassageFilter


# --- нормализация и проверка по тексту ---

def test_occurs_in_ignores_word_forms():
    assert occurs_in("диагностика заболеваний", "ИИ помогает в диагностике заболеваний и лечении")
    assert occurs_in("acceptance rate", "the token acceptance rates grow")
    assert not occurs_in("chat reasoning", "chat and math reasoning")
    assert not occurs_in("", "anything")
    assert contains_stems(["a", "b", "c"], ["b", "c"]) and not contains_stems(["a", "b"], ["a", "c"])


def test_clean_splits_abbreviations_drops_numbers_generic_and_ungrounded():
    text = "We introduce Approximate Speculative Decoding (ASD) on GSM8K, 30% faster; the model is good."
    raw = {"method": ["Approximate Speculative Decoding (ASD)", "model", "EAGLE-2"],
           "dataset": ["GSM8K", "gsm8k"], "metric": ["30%", "5x"], "task": []}
    result = clean(raw, text)
    assert result.entities["method"] == ["Approximate Speculative Decoding", "ASD"]
    assert result.entities["dataset"] == ["GSM8K"]  # дубль по ключу схлопнут
    assert result.entities["metric"] == []
    dropped = {name for _, name in result.dropped}
    assert {"model", "EAGLE-2", "30%", "5x"} <= dropped
    assert clean(raw, text, grounded=False).entities["method"] == ["Approximate Speculative Decoding", "ASD", "EAGLE-2"]
    assert split_abbreviation("Plain name") == ["Plain name"]
    assert entity_key("Диагностики") == entity_key("диагностика")


def test_parse_json_tolerates_wrapping_text():
    assert parse_json('Sure! {"method": ["x"]} done') == {"method": ["x"]}
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_keywords_skip_generic_single_words():
    text = ("Speculative decoding uses a draft model. The draft model proposes tokens. HSDDW improves the draft model. "
            "Models and tokens are everywhere. The window size of HSDDW adapts. Speculative decoding is fast.") * 3
    keywords = [k for k, _ in extract_keywords(text, top=6, language="en")]
    assert "HSDDW" in keywords
    assert "tokens" not in keywords and "model" not in keywords
    assert extract_keywords("") == []


# --- метрики оценки ---

def test_score_type_with_aliases_and_lenient_match():
    gold = ["Approximate Speculative Decoding|ASD", "GSM8K"]
    assert score_type(["ASD", "Approximate Speculative Decoding", "MATH"], gold) == (1, 1, 1)
    assert score_type(["speedup ratio"], ["speedup"]) == (0, 1, 1)
    assert score_type(["speedup ratio"], ["speedup"], lenient=True) == (1, 0, 0)
    assert prf(1, 1, 1)["f1"] == pytest.approx(0.5)


class JsonClient:
    """Подменная модель для извлечения: отвечает заданным JSON."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.requests = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        message = SimpleNamespace(content=json.dumps(self.payloads.pop(0)), tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                               usage=SimpleNamespace(prompt_tokens=50, completion_tokens=5))


def test_evaluate_entities_compares_raw_and_grounded():
    gold = [GoldEntities(paper="P", text="We evaluate EAGLE-2 on MT-Bench.",
                         entities={"method": ["EAGLE-2"], "dataset": ["MT-Bench"], "metric": [], "task": []})]
    client = JsonClient([{"method": ["EAGLE-2", "Medusa"], "dataset": ["MT-Bench"], "metric": [], "task": []}])
    report = evaluate_entities(ChatModel("fake", client=client), gold)
    raw, grounded = report["strict"]["raw"]["all"]["micro"], report["strict"]["grounded"]["all"]["micro"]
    assert (raw["tp"], raw["fp"]) == (2, 1)  # Medusa выдумана
    assert (grounded["tp"], grounded["fp"]) == (2, 0)  # проверка по тексту её убрала
    assert client.requests[0]["response_format"]["type"] == "json_schema"


# --- хранилище и разметка библиотеки ---

def add_paper(library, title, sections, topic=None):
    paper_id, _ = library.upsert_paper(PaperRecord(title=title, sources=["arxiv"], year=2025, language="en"))
    if topic:
        library.link_topic(paper_id, library.get_or_create_topic(topic))
    outs = []
    for kind, texts in sections.items():
        section = SectionOut(kind.title(), kind, 1)
        for text in texts:
            section.passages += paragraph_passages(Paragraph(text, [(0, 1)]))
        outs.append(section)
    library.save_structure(paper_id, outs, status="ok", parser="pdf", version=1)
    return paper_id


@pytest.fixture
def library():
    with Library(":memory:") as lib:
        add_paper(lib, "Dynamic depth for EAGLE", {
            "abstract": ["We extend EAGLE-2 with a dynamic depth and evaluate on MT-Bench."],
            "experiments": ["On MT-Bench the speedup reaches 3.16x.", "Unrelated passage about memory."],
        }, topic="sd")
        add_paper(lib, "Another drafting paper", {
            "abstract": ["Drafting on MT-Bench and GSM8K with a small model."],
        }, topic="sd")
        yield lib


def test_annotate_papers_links_mentions_and_survives_reparse(library):
    payloads = [
        {"method": ["EAGLE-2", "Medusa"], "dataset": ["MT-Bench"], "metric": ["speedup"], "task": []},
        {"method": [], "dataset": ["MT-Bench", "GSM8K"], "metric": [], "task": []},
    ]
    statuses = annotate_papers(library, ChatModel("fake", client=JsonClient(payloads)))
    assert statuses == {"ok": 2}
    assert library.paper_entities(1) == {"method": ["EAGLE-2"], "dataset": ["MT-Bench"], "metric": ["speedup"]}
    assert library.conn.execute("SELECT dropped FROM annotations WHERE paper_id = 1").fetchone()[0] == 1
    assert annotate_papers(library, ChatModel("fake", client=JsonClient([]))) == {}  # уже размечено

    stats = {r["name"]: r["papers"] for r in library.entity_stats("dataset", topic="sd")}
    assert stats == {"MT-Bench": 2, "GSM8K": 1}

    mt_bench = library.passage_ids(PassageFilter(entity="mt-bench"))
    assert len(mt_bench) == 3  # аннотация и эксперимент первой статьи + аннотация второй
    assert library.passage_ids(PassageFilter(entity="EAGLE")) == library.passage_ids(PassageFilter(entity="EAGLE-2"))

    # переразбор статьи удаляет фрагменты — упоминания должны восстановиться по новым
    section = SectionOut("Abstract", "abstract", 1)
    section.passages = paragraph_passages(Paragraph("A new text about EAGLE-2 on MT-Bench.", [(0, 1)]))
    library.save_structure(1, [section], status="ok", parser="pdf", version=2)
    assert len(library.passage_ids(PassageFilter(entity="MT-Bench", paper_ids=[1]))) == 1


def test_annotation_text_uses_summary_sections(library):
    text = annotation_text(library, library.get_paper(1))
    assert text.startswith("Dynamic depth for EAGLE") and "3.16x" in text


def test_agent_entity_tools(library):
    annotate_papers(library, ChatModel("fake", client=JsonClient([
        {"method": ["EAGLE-2"], "dataset": ["MT-Bench"], "metric": [], "task": []},
        {"method": [], "dataset": ["MT-Bench", "GSM8K"], "metric": [], "task": []},
    ])))
    ctx = ToolContext(library=library, searcher=Searcher(library))
    listing = list_entities(ctx, "dataset", topic="sd")
    assert "MT-Bench: 2 papers" in listing and "GSM8K: 1 papers" in listing
    with pytest.raises(ToolError):
        list_entities(ctx, "person")
    hits = search_library(ctx, "speedup", entity="MT-Bench")
    assert "3.16x" in hits and "Unrelated passage" not in hits


def test_failed_annotation_is_retried(library):
    class Broken:
        model, reasoning_effort = "broken", None
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: (_ for _ in ()).throw(RuntimeError("fetch failed")))))

    assert annotate_papers(library, Broken(), paper_ids=[1]) == {"failed": 1}
    payload = [{"method": ["EAGLE-2"], "dataset": [], "metric": [], "task": []}]
    assert annotate_papers(library, ChatModel("fake", client=JsonClient(payload)), paper_ids=[1]) == {"ok": 1}
