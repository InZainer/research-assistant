import json
from types import SimpleNamespace

import pytest

from research_assistant.agent.loop import run_agent
from research_assistant.agent.tools import (
    ToolContext,
    ToolError,
    add_papers,
    available_schemas,
    library_overview,
    read_passage,
    search_external,
    search_library,
)
from research_assistant.answer.citations import format_citation, load_citations, parse_sentence_id
from research_assistant.ingest.models import PaperRecord
from research_assistant.llm import ChatModel
from research_assistant.parse.passages import SectionOut, paragraph_passages
from research_assistant.parse.structure import Paragraph
from research_assistant.retrieval.search import Searcher
from research_assistant.store.db import Library


# --- подменная модель: заранее заданные ходы ---

def call(name, args, call_id=None):
    arguments = args if isinstance(args, str) else json.dumps(args)
    return SimpleNamespace(id=call_id or f"call_{name}", function=SimpleNamespace(name=name, arguments=arguments))


class ScriptedClient:
    """Отвечает по сценарию. Элемент сценария — список вызовов или текст; функция получает сообщения."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})  # снимок на момент запроса
        turn = self.script.pop(0)
        if callable(turn):
            turn = turn(kwargs["messages"])
        content, calls = (turn, None) if isinstance(turn, str) else (None, turn)
        message = SimpleNamespace(content=content, tool_calls=calls)
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=10)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=usage)


def model(script):
    return ChatModel("fake-model", client=ScriptedClient(script), reasoning_effort="none")


@pytest.fixture
def ctx():
    with Library(":memory:") as lib:
        topic = lib.get_or_create_topic("speculative decoding")
        texts = [
            ("Speculative decoding with draft models", 2024, "limitations",
             "Speedup drops when the draft model disagrees with the target model. Rare tokens are the main cause."),
            ("Tree attention for verification", 2025, "method",
             "Tree-based verification checks many candidate continuations in one forward pass. It needs more memory."),
        ]
        for title, year, kind, text in texts:
            paper_id, _ = lib.upsert_paper(PaperRecord(title=title, sources=["openalex"], year=year, language="en"))
            lib.link_topic(paper_id, topic)
            section = SectionOut(kind.title(), kind, 1)
            section.passages = paragraph_passages(Paragraph(text, [(0, 3)]))
            lib.save_structure(paper_id, [section], status="ok", parser="pdf", version=1)
        yield ToolContext(library=lib, searcher=Searcher(lib))


def sentence_id(ctx, startswith):
    return ctx.library.conn.execute("SELECT id FROM sentences WHERE text LIKE ?", (startswith + "%",)).fetchone()[0]


def last_tool_observation(messages):
    return [m for m in messages if m["role"] == "tool"][-1]["content"]


# --- цикл ---

def test_happy_path_with_citations_and_log(ctx, tmp_path):
    sid = sentence_id(ctx, "Speedup drops")
    script = [
        [call("search_library", {"query": "why does speedup drop"})],
        [call("final_answer", {"answer": "Из-за расхождения моделей.",
                               "claims": [{"text": "Ускорение падает при расхождении моделей.", "sentence_ids": [f"s{sid}"]}]})],
    ]
    log = tmp_path / "run.jsonl"
    result = run_agent("Почему падает ускорение?", ctx, model(script), log_path=log)

    assert result.stop_reason == "final_answer"
    assert result.steps == 2 and result.tool_calls == 2 and result.tool_errors == 0
    assert result.usage.prompt_tokens == 200 and result.usage.completion_tokens == 20
    [claim] = result.claims
    assert claim.citations[0].text == "Speedup drops when the draft model disagrees with the target model."
    assert claim.citations[0].page == 3
    events = [json.loads(line)["event"] for line in log.read_text(encoding="utf-8").splitlines()]
    assert events == ["start", "llm", "tool", "llm", "tool", "finish"]


def test_unseen_sentence_id_is_rejected_then_fixed(ctx):
    sid = sentence_id(ctx, "Tree-based")
    script = [
        [call("final_answer", {"answer": "x", "claims": [{"text": "c", "sentence_ids": [f"s{sid}"]}]})],
        lambda messages: (_ for _ in ()).throw(AssertionError("not reached")) if "not in your tool results"
        not in last_tool_observation(messages) else [call("search_library", {"query": "tree verification"})],
        [call("final_answer", {"answer": "x", "claims": [{"text": "c", "sentence_ids": [sid]}]})],
    ]
    result = run_agent("q", ctx, model(script))
    assert result.stop_reason == "final_answer" and result.tool_errors == 1
    assert result.claims[0].citations[0].sentence_id == sid


def test_errors_become_observations(ctx):
    seen = []
    script = [
        [call("search_library", "{not json")],
        lambda m: seen.append(last_tool_observation(m)) or [call("teleport", {})],
        lambda m: seen.append(last_tool_observation(m)) or [call("read_passage", {"passage_id": 999})],
        lambda m: seen.append(last_tool_observation(m)) or [call("search_library", {"query": "x", "section_kinds": ["bogus"]})],
        lambda m: seen.append(last_tool_observation(m)) or [call("search_library", {"qeury": "typo"})],
        lambda m: seen.append(last_tool_observation(m)) or [call("final_answer", {"answer": "нет данных", "claims": [], "not_found": True})],
    ]
    result = run_agent("q", ctx, model(script))
    assert result.stop_reason == "final_answer" and result.not_found
    assert result.tool_errors == 5
    assert "not valid JSON" in seen[0]
    assert "unknown tool teleport" in seen[1]
    assert "no passage or sentence with id 999" in seen[2]
    assert "unknown section_kinds" in seen[3]
    assert "wrong arguments" in seen[4]


def test_text_reply_is_nudged_then_given_up(ctx):
    client = ScriptedClient(["просто текст", "снова текст", "и опять"])
    result = run_agent("q", ctx, ChatModel("fake", client=client))
    assert result.stop_reason == "no_final_answer"
    assert result.answer == "и опять"
    assert sum(1 for m in client.requests[-1]["messages"] if m["role"] == "user") == 3  # вопрос + 2 подсказки


def test_max_steps(ctx):
    script = [[call("library_overview", {})] for _ in range(3)]
    result = run_agent("q", ctx, model(script), max_steps=3)
    assert result.stop_reason == "max_steps" and result.steps == 3


def test_history_and_reasoning_effort_reach_the_model(ctx):
    client = ScriptedClient([[call("final_answer", {"answer": "ok", "claims": [], "not_found": True})]])
    history = [{"role": "user", "content": "раньше"}, {"role": "assistant", "content": "ответ"}]
    run_agent("q", ctx, ChatModel("fake", client=client, reasoning_effort="none"), history=history)
    request = client.requests[0]
    assert [m["role"] for m in request["messages"]] == ["system", "user", "assistant", "user"]
    assert request["extra_body"] == {"reasoning_effort": "none"}
    assert {t["function"]["name"] for t in request["tools"]} == {
        "search_library", "read_passage", "library_overview", "list_entities", "compare_papers", "final_answer"}


# --- инструменты ---

def test_search_library_marks_seen_sentences(ctx):
    text = search_library(ctx, "draft model disagrees", k=1)
    sid = sentence_id(ctx, "Speedup drops")
    assert f"[s{sid}]" in text and "page 3" in text and sid in ctx.seen_sentences
    assert "No passages found" in search_library(ctx, "квантовая хромодинамика")
    with pytest.raises(ToolError):
        search_library(ctx, "  ")


def test_read_passage_and_overview(ctx):
    passage_id = ctx.library.conn.execute("SELECT id FROM passages ORDER BY id").fetchone()[0]
    assert "Rare tokens are the main cause." in read_passage(ctx, passage_id, neighbors=0)
    assert "speculative decoding: 2 papers" in library_overview(ctx)
    assert "Tree attention for verification" in library_overview(ctx, topic="speculative decoding")
    with pytest.raises(ToolError):
        library_overview(ctx, topic="no such topic")


def test_external_search_and_add(ctx):
    record = PaperRecord(title="New paper on drafting", sources=["arxiv"], arxiv_id="2501.00001", year=2025)
    ctx.sources = [SimpleNamespace(name="arxiv", search=lambda *a, **k: [record])]
    added = []
    ctx.add_papers_fn = lambda records, topic: added.append((records, topic)) or "Added 1"
    assert "arxiv:2501.00001" in search_external(ctx, "drafting")
    assert add_papers(ctx, ["arxiv:2501.00001"], "speculative decoding") == "Added 1"
    assert added[0][0][0].title == "New paper on drafting"
    with pytest.raises(ToolError, match="unknown ids"):
        add_papers(ctx, ["arxiv:9999.99999"], "t")
    assert {s["function"]["name"] for s in available_schemas(ctx)} >= {"search_external", "add_papers"}


def test_citation_helpers(ctx):
    assert [parse_sentence_id(x) for x in (12, "12", "s12", "[s12]", "S12", "abc")] == [12, 12, 12, 12, 12, None]
    sid = sentence_id(ctx, "It needs more memory")
    citation = load_citations(ctx.library, [sid])[sid]
    assert format_citation(citation) == (
        f"«It needs more memory.» — Tree attention for verification (2025), раздел «Method», с. 3 [s{sid}]"
    )


def test_read_passage_accepts_sentence_id_by_mistake(ctx):
    sid = sentence_id(ctx, "It needs more memory")
    text = read_passage(ctx, f"s{sid}", neighbors=0)
    assert f"s{sid} is a sentence id" in text and "Tree-based verification" in text
    assert "It needs more memory" in read_passage(ctx, str(sid), neighbors=0)  # голое число тоже разрешается
    with pytest.raises(ToolError, match="bad id"):
        read_passage(ctx, "passage one")


def test_research_budget_leaves_only_final_answer(ctx):
    client = ScriptedClient(
        [[call("library_overview", {}, call_id=f"c{i}")] for i in range(6)]
        + [[call("final_answer", {"answer": "нет", "claims": [], "not_found": True})]]
    )
    result = run_agent("q", ctx, ChatModel("fake", client=client), max_steps=12)
    assert result.stop_reason == "final_answer"
    tool_names = [{t["function"]["name"] for t in r["tools"]} for r in client.requests]
    assert tool_names[5] != {"final_answer"} and tool_names[6] == {"final_answer"}
    observations = [m["content"] for m in client.requests[6]["messages"] if m["role"] == "tool"]
    assert "call final_answer now" in observations[-1]


def test_last_step_offers_only_final_answer(ctx):
    client = ScriptedClient([[call("library_overview", {})], [call("final_answer", {"answer": "ok", "claims": [], "not_found": True})]])
    run_agent("q", ctx, ChatModel("fake", client=client), max_steps=2)
    assert {t["function"]["name"] for t in client.requests[1]["tools"]} == {"final_answer"}


def test_section_filter_keeps_unstructured_papers(ctx):
    paper_id, _ = ctx.library.upsert_paper(PaperRecord(title="Paper without any headings at all", sources=["openalex"]))
    section = SectionOut("", "body", 1)
    section.passages = paragraph_passages(Paragraph("We compared Naive Bayes and logistic regression classifiers.", [(0, None)]))
    ctx.library.save_structure(paper_id, [section], status="ok", parser="html", version=1)
    assert "Naive Bayes" in search_library(ctx, "naive bayes classifiers", section_kinds=["results"])


# --- этап 7: сравнение статей, однократный RAG, цитаты текстом, оценка агента ---

def test_compare_papers(ctx):
    from research_assistant.agent.tools import compare_papers

    ctx.library.save_annotation(1, {"dataset": ["MT-Bench"], "method": ["draft model"]}, [], status="ok",
                                annotator="t", version=1)
    text = compare_papers(ctx, [1, 2], "speedup and memory", k=1)
    assert "=== paper 1" in text and "=== paper 2" in text and "datasets: MT-Bench" in text
    assert "[s" in text  # предложения с ID, на них можно ссылаться
    with pytest.raises(ToolError):
        compare_papers(ctx, [1], "x")
    with pytest.raises(ToolError, match="does not exist"):
        compare_papers(ctx, [1, 999], "x")


def test_single_shot_searches_once_and_offers_only_final_answer(ctx):
    sid = sentence_id(ctx, "Speedup drops")
    client = ScriptedClient([[call("final_answer", {"answer": "a", "claims": [{"text": "c", "sentence_ids": [sid]}]})]])
    result = run_agent("draft model disagrees speedup", ctx, ChatModel("fake", client=client), single_shot=True)
    assert result.stop_reason == "final_answer" and result.steps == 1
    request = client.requests[0]
    assert {t["function"]["name"] for t in request["tools"]} == {"final_answer"}
    assert any(m["role"] == "tool" and f"[s{sid}]" in m["content"] for m in request["messages"])


def test_quotes_mode_accepts_free_text_quotes(ctx):
    client = ScriptedClient([[call("final_answer", {"answer": "a", "claims": [
        {"text": "c", "quote": "Speedup drops when the draft model disagrees with the target model."}]})]])
    result = run_agent("q", ctx, ChatModel("fake", client=client), answer_mode="quotes")
    assert result.stop_reason == "final_answer" and result.claims == []
    assert result.quotes == [("c", "Speedup drops when the draft model disagrees with the target model.")]
    schema = next(t for t in client.requests[0]["tools"] if t["function"]["name"] == "final_answer")
    assert "quote" in schema["function"]["parameters"]["properties"]["claims"]["items"]["properties"]


def test_agent_evaluation_metrics(ctx):
    from research_assistant.evaluation.agent import evaluate_agent, passages_with_quote, wilson
    from research_assistant.evaluation.retrieval import GoldItem, GoldQuestion

    low, high = wilson(8, 10)
    assert 0.4 < low < 0.8 < high < 1.0 and wilson(0, 0) == (0.0, 0.0)
    assert len(passages_with_quote(ctx.library, "Rare tokens   are the main cause.")) == 1
    assert passages_with_quote(ctx.library, "Rare tokens are the MAIN cause.") == set()

    questions = [
        GoldQuestion("en1", "why does speedup drop", "en",
                     [GoldItem("Speculative decoding with draft models", "Rare tokens are the main cause.")]),
        GoldQuestion("en2", "which model needs tree verification", "en",  # поиск поднимает обе статьи
                     [GoldItem("Tree attention for verification", "checks many candidate continuations")]),
    ]
    good = sentence_id(ctx, "Rare tokens")  # фрагмент статьи 1: золото для вопроса 1, мимо для вопроса 2

    def turn(sid):
        return lambda messages: [call("final_answer", {"answer": "a", "claims": [{"text": "c", "sentence_ids": [sid]}]})]

    # вопрос 1 — цитата из золотого фрагмента (успех), вопрос 2 — цитата из другой статьи (неуспех)
    client = ScriptedClient([turn(good)] * 4)
    report = evaluate_agent(lambda: ToolContext(library=ctx.library, searcher=ctx.searcher),
                            ChatModel("fake", client=client), questions, variant="single", runs=2)
    summary = report["summary"]["all"]
    assert summary["n"] == 4 and summary["pass@1"] == 0.5 and summary["pass^2"] == 0.5
    assert summary["accepted"] == 1.0 and summary["exact_quote_rate"] == 1.0
    assert "en→en" in report["summary"]


def test_quote_with_copied_id_tag_counts_as_exact(ctx):
    from research_assistant.evaluation.agent import passages_with_quote

    sid = sentence_id(ctx, "Rare tokens")
    assert len(passages_with_quote(ctx.library, f"[s{sid}] Rare tokens are the main cause.")) == 1


def test_prefetch_keeps_tools(ctx):
    client = ScriptedClient([[call("final_answer", {"answer": "нет", "claims": [], "not_found": True})]])
    run_agent("draft model speedup", ctx, ChatModel("fake", client=client), prefetch=True)
    request = client.requests[0]
    assert "search_library" in {t["function"]["name"] for t in request["tools"]}
    assert request["messages"][-1]["role"] == "tool"
