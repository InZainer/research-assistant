import json

import pytest

from research_assistant.evaluation.retrieval import (
    GoldItem,
    GoldQuestion,
    evaluate,
    format_report,
    load_questions,
    mrr_at_k,
    ndcg_at_k,
    question_group,
    recall_at_k,
    resolve,
)
from research_assistant.ingest.models import PaperRecord
from research_assistant.parse.passages import SectionOut, paragraph_passages
from research_assistant.parse.structure import Paragraph
from research_assistant.retrieval.search import Searcher
from research_assistant.store.db import Library


def test_metrics():
    ranked = [5, 3, 9, 1]
    assert recall_at_k(ranked, {3, 1}, 2) == 0.5
    assert recall_at_k(ranked, {3, 1}, 4) == 1.0
    assert mrr_at_k(ranked, {9}, 10) == pytest.approx(1 / 3)
    assert mrr_at_k(ranked, {7}, 10) == 0.0
    assert ndcg_at_k([3, 5], {3}, 10) == 1.0
    assert 0 < ndcg_at_k([5, 3], {3}, 10) < 1
    assert recall_at_k(ranked, set(), 5) == 0.0


def test_question_group():
    item = GoldItem("Paper", "an english quote")
    assert question_group(GoldQuestion("q", "Как?", "ru", [item])) == "ru→en"
    assert question_group(GoldQuestion("q", "How?", "en", [item])) == "en→en"


@pytest.fixture
def library():
    with Library(":memory:") as lib:
        texts = {
            "Paper about drafting": "Small draft models propose tokens.   The target verifies them in parallel.",
            "Paper about caches": "Caching keys and values speeds up decoding considerably.",
        }
        for title, text in texts.items():
            paper_id, _ = lib.upsert_paper(PaperRecord(title=title, sources=["arxiv"]))
            section = SectionOut("Abstract", "abstract", 1)
            section.passages = paragraph_passages(Paragraph(text, [(0, 1)]))
            lib.save_structure(paper_id, [section], status="ok", parser="pdf", version=1)
        yield lib


def test_resolve_by_title_and_whitespace_insensitive_quote(library):
    question = GoldQuestion("q1", "?", "en", [GoldItem("paper  about DRAFTING", "tokens. The target verifies")])
    assert len(resolve(library, question)) == 1
    missing = GoldQuestion("q2", "?", "en", [GoldItem("No such paper", "tokens")])
    assert resolve(library, missing) == set()


def test_evaluate_end_to_end(library, tmp_path):
    gold = tmp_path / "gold.jsonl"
    rows = [
        {"id": "a", "question": "how do draft models verify tokens", "language": "en",
         "relevant": [{"paper": "Paper about drafting", "quote": "Small draft models propose tokens."}]},
        {"id": "b", "question": "unrelated", "language": "en",
         "relevant": [{"paper": "Missing paper", "quote": "x"}]},
    ]
    gold.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    questions = load_questions(gold)
    assert questions[0].status == "draft"

    report = evaluate(Searcher(library), questions, ["bm25"], k=10)
    assert report["unresolved"] == ["b"]
    summary = report["modes"]["bm25"]["summary"]
    assert summary["all"]["n"] == 1 and summary["all"]["recall@10"] == 1.0
    assert "en→en" in summary
    assert "bm25" in format_report(report)
