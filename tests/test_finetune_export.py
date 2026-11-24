import copy
import json
from types import SimpleNamespace

import pytest

from research_assistant.agent.prompts import NUDGE_FINAL_ANSWER
from research_assistant.agent.tools import FUNCTIONS, ToolContext, available_schemas
from research_assistant.evaluation.retrieval import GoldItem, GoldQuestion
from research_assistant.finetune.export import (
    ASSISTANT_PREFIX,
    est_tokens,
    export,
    strip_think,
    to_sft,
    validate_example,
)
from research_assistant.finetune.trajectories import (
    accept,
    choose_best,
    collect,
    error_kind,
    is_recovery,
    load_records,
    prompt_hash,
)
from research_assistant.ingest.models import PaperRecord
from research_assistant.llm import ChatModel
from research_assistant.parse.passages import SectionOut, paragraph_passages
from research_assistant.parse.pipeline import PARSER_VERSION
from research_assistant.parse.structure import Paragraph
from research_assistant.retrieval.search import Searcher
from research_assistant.store.db import Library

SPEEDUP = "Speculative decoding with draft models"
TREE = "Tree attention for verification"


# --- подменная модель (как в test_agent.py): заранее заданные ходы ---

def call(name, args, call_id=None):
    arguments = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
    return SimpleNamespace(id=call_id or f"call_{name}", function=SimpleNamespace(name=name, arguments=arguments))


class ScriptedClient:
    def __init__(self, script, base_url=None):
        self.script = list(script)
        if base_url is not None:
            self.base_url = base_url
        self.temperatures = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.temperatures.append(kwargs["temperature"])
        turn = self.script.pop(0)
        if isinstance(turn, Exception):
            raise turn
        if callable(turn):
            turn = turn(kwargs["messages"])
        content, calls = (turn, None) if isinstance(turn, str) else (None, turn)
        message = SimpleNamespace(content=content, tool_calls=calls)
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=10)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=usage)


def teacher(script, base_url=None):
    return ChatModel("mercury-test", client=ScriptedClient(script, base_url))


@pytest.fixture
def library():
    with Library(":memory:") as lib:
        topic = lib.get_or_create_topic("speculative decoding")
        texts = [
            (SPEEDUP, "limitations",
             "Speedup drops when the draft model disagrees with the target model. Rare tokens are the main cause."),
            (TREE, "method",
             "Tree-based verification checks many candidate continuations in one forward pass. It needs more memory."),
        ]
        for title, kind, text in texts:
            paper_id, _ = lib.upsert_paper(PaperRecord(title=title, sources=["openalex"], year=2024, language="en"))
            lib.link_topic(paper_id, topic)
            section = SectionOut(kind.title(), kind, 1)
            section.passages = paragraph_passages(Paragraph(text, [(0, 3)]))
            lib.save_structure(paper_id, [section], status="ok", parser="pdf", version=1)
        yield lib


def make_context(library):
    return lambda: ToolContext(library=library, searcher=Searcher(library))


def sid(library, startswith):
    return library.conn.execute("SELECT id FROM sentences WHERE text LIKE ?", (startswith + "%",)).fetchone()[0]


def paper_id(library, title):
    return library.conn.execute("SELECT id FROM papers WHERE title = ?", (title,)).fetchone()[0]


def question(qid="tr-fact-000001", text="Why does speedup drop with a draft model?", language="en", kind="fact",
             relevant=((SPEEDUP, "Speedup drops when the draft model disagrees"),), **meta):
    return GoldQuestion(qid, text, language, [GoldItem(p, q) for p, q in relevant], kind=kind,
                        meta={"split": "train", **meta})


def good_script(library, answer="Speedup drops when the draft model disagrees with the target.", sentence=None,
                query="draft model disagreement speedup"):
    sentence = sentence or sid(library, "Speedup drops")
    return [
        [call("search_library", {"query": query, "k": 4})],
        [call("final_answer", {"answer": answer,
                               "claims": [{"text": "The draft model disagrees with the target model.",
                                           "sentence_ids": [f"s{sentence}"]}]})],
    ]


def collect_one(library, tmp_path, q=None, script=None, name="t.jsonl", **kwargs):
    q = q or question()
    out = tmp_path / name
    collect([q], make_context(library), teacher(script or good_script(library)), out, samples=1, log=lambda s: None,
            **kwargs)
    return [r for r in load_records(out) if r["qid"] == q.id][-1]


# --- сбор ---

def test_collect_writes_spec_record(library, tmp_path):
    q = question()
    model = teacher(good_script(library) + good_script(library))
    out = tmp_path / "train.jsonl"
    counts = collect([q], make_context(library), model, out, samples=2, log=lambda s: None)
    assert counts == {"written": 2, "skipped": 0, "errors": 0}
    first, second = load_records(out)
    assert (first["qid"], first["sample"], second["sample"]) == (q.id, 0, 1)
    assert (first["temperature"], second["temperature"]) == (0.2, 0.7)
    assert model.client.temperatures == [0.2, 0.2, 0.7, 0.7] and model.temperature == 0.2  # восстановлена
    assert first["kind"] == "fact" and first["model"] == "mercury-test" and first["error"] is None
    assert first["prompt_hash"] == prompt_hash() and first["parser_version"] == PARSER_VERSION
    roles = [m["role"] for m in first["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant", "tool"]
    assert first["messages"][2]["tool_calls"][0]["id"] == "single_shot_search"
    ctx = make_context(library)()
    assert [t["function"]["name"] for t in first["tools"]] == [t["function"]["name"] for t in available_schemas(ctx)]
    result = first["result"]
    assert result["stop_reason"] == "final_answer" and result["steps"] == 2 and result["tool_calls"] == 2
    assert result["citations"] == [sid(library, "Speedup drops")] and result["nudges"] == 0
    assert result["usage"] == {"prompt": 200, "completion": 20}
    assert result["gold_hit"] and result["cited_papers"] == result["gold_papers"] == [paper_id(library, SPEEDUP)]


def test_collect_resumes_and_retries_errors(library, tmp_path):
    q = question()
    out = tmp_path / "train.jsonl"
    counts = collect([q], make_context(library), teacher([RuntimeError("429 rate limit")]), out, samples=1,
                     log=lambda s: None)
    assert counts["errors"] == 1 and "429" in load_records(out)[0]["error"]
    collect([q], make_context(library), teacher(good_script(library)), out, samples=1, log=lambda s: None)
    # готовые пары пропускаются: пустой сценарий упал бы при вызове модели
    counts = collect([q], make_context(library), teacher([]), out, samples=1, log=lambda s: None)
    assert counts == {"written": 0, "skipped": 1, "errors": 0}
    assert [bool(r["error"]) for r in load_records(out)] == [True, False]


def test_collect_refuses_local_student(library, tmp_path):
    model = teacher([], base_url="http://localhost:1234/v1")
    with pytest.raises(ValueError, match="локальная"):
        collect([question()], make_context(library), model, tmp_path / "t.jsonl", log=lambda s: None)
    assert not (tmp_path / "t.jsonl").exists()


# --- отсев ---

def with_extra_call(record, name, args, observation):
    """Вставляет вызов модели перед финальным ходом."""
    record = copy.deepcopy(record)
    turn = {"role": "assistant", "content": "",
            "tool_calls": [{"id": f"x_{name}", "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}]}
    record["messages"][-2:-2] = [turn, {"role": "tool", "tool_call_id": f"x_{name}", "content": observation}]
    record["result"]["tool_calls"] += 1
    record["result"]["steps"] += 1
    return record


def test_accept_good_trajectory(library, tmp_path):
    record = collect_one(library, tmp_path)
    assert accept(record, question(), set()) == []


def test_accept_rejection_reasons(library, tmp_path):
    record = collect_one(library, tmp_path)
    q = question()
    leaked = sid(library, "Speedup drops")
    assert "test_leak" in accept(record, q, {leaked})
    assert "test_leak" in accept(record, q, set(), {paper_id(library, SPEEDUP)})
    assert accept(with_extra_call(record, "add_papers", {"ids": ["1"], "topic": "x"}, "Added"), q) == ["add_papers"]
    assert accept(with_extra_call(record, "search_external", {"query": "x"}, "Found"), q) == ["search_external"]

    nudged = copy.deepcopy(record)
    nudged["messages"][4:4] = [{"role": "assistant", "content": "text"},
                               {"role": "user", "content": NUDGE_FINAL_ANSWER}]
    assert "nudge" in accept(nudged, q)

    long = copy.deepcopy(record)
    long["result"].update(steps=5, tool_calls=9)
    assert {"too_many_steps", "too_many_tool_calls"} <= set(accept(long, q))

    degraded = copy.deepcopy(record)
    degraded["messages"][3]["content"] = "[Semantic search is unavailable, keyword search only]\n" + "Found"
    assert accept(degraded, q) == ["semantic_search_unavailable"]

    stopped = copy.deepcopy(record)
    stopped["result"]["stop_reason"] = "max_steps"
    assert "stop_reason:max_steps" in accept(stopped, q)
    assert accept({**record, "error": "APIError"}, q) == ["error"]

    other = question(relevant=((TREE, "Tree-based verification"),))
    other_record = collect_one(library, tmp_path, other, name="o.jsonl")  # процитирована другая статья
    assert accept(other_record, other) == ["wrong_paper"]


def test_accept_language_of_answer_and_claims(library, tmp_path):
    q = question()
    russian = collect_one(library, tmp_path, script=good_script(library, answer="Ускорение падает при расхождении."))
    assert accept(russian, q) == ["language"]
    ru_question = question(text="Почему падает ускорение?", language="ru")
    claims_en = collect_one(library, tmp_path, ru_question, good_script(library, answer="Из-за расхождения моделей."),
                            name="ru.jsonl")
    assert accept(claims_en, ru_question) == ["language"]  # ответ по-русски, а утверждение — по-английски


def test_accept_recovery_is_flag_not_reason(library, tmp_path):
    script = [[call("final_answer", {"answer": "x", "claims": [{"text": "c", "sentence_ids": ["s999999"]}]})]]
    script += good_script(library)
    record = collect_one(library, tmp_path, script=script)
    assert record["result"]["tool_errors"] == 1
    assert accept(record, question()) == [] and is_recovery(record)

    broken = [[call("search_library", "{not json")]] + good_script(library)
    record = collect_one(library, tmp_path, script=broken, name="b.jsonl")
    assert "tool_errors" in accept(record, question()) and not is_recovery(record)


def test_recovery_follows_error_kind(library, tmp_path, monkeypatch):
    def boom(ctx, **kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setitem(FUNCTIONS, "library_overview", boom)
    first_calls = {
        "bad_args": call("search_library", {"qury": "draft"}),  # TypeError в цикле
        "bad_args_not_object": call("search_library", "[1]"),
        "tool_error": call("search_library", {"query": " "}),
        "json": call("search_library", "{not json"),
        "unknown_tool": call("no_such_tool", {}),
        "env": call("library_overview", {}),
    }
    expected = {"bad_args": True, "bad_args_not_object": True, "tool_error": True, "json": False,
                "unknown_tool": False, "env": False}
    for label, first in first_calls.items():
        record = collect_one(library, tmp_path, script=[[first]] + good_script(library), name=f"{label}.jsonl")
        kind = label.removesuffix("_not_object")
        assert record["result"]["error_kinds"] == [kind], label  # вид пишется при сборе
        assert is_recovery(record) is expected[label], label
        assert ("tool_errors" in accept(record, question())) is not expected[label], label
        old = copy.deepcopy(record)  # запись без error_kinds: вид восстанавливается по сообщениям
        del old["result"]["error_kinds"]
        assert is_recovery(old) is expected[label], label
    assert error_kind("final_answer", "Fix final_answer: claim 1 cites unknown s9") == "answer_rejected"
    assert error_kind("final_answer", "Answer accepted.") is None
    assert error_kind("search_library", "Error in search_library: OSError: x") == "env"


def test_accept_unanswerable_and_external(library, tmp_path):
    q = question("tr-un-1", "What GPU did the authors of the moon landing use?", kind="unanswerable", relevant=())
    abstain = [[call("final_answer", {"answer": "The library has no such information.", "claims": [],
                                      "not_found": True})]]
    assert accept(collect_one(library, tmp_path, q, abstain), q) == []
    answered = collect_one(library, tmp_path, q, good_script(library), name="a.jsonl")
    assert "answered_unanswerable" in accept(answered, q)
    # not_found с утверждениями годится: при экспорте утверждения убираются (их игнорирует и сам агент)
    sid = library.conn.execute("SELECT id FROM sentences LIMIT 1").fetchone()[0]
    noisy = [[call("search_library", {"query": "moon gpu"})],
             [call("final_answer", {"answer": "Not in the library.", "not_found": True,
                                    "claims": [{"text": "Related fact.", "sentence_ids": [f"s{sid}"]}]})]]
    noisy_record = collect_one(library, tmp_path, q, noisy, name="n.jsonl")
    assert accept(noisy_record, q) == []
    example = to_sft(noisy_record, q, "train")
    final = example["messages"][-1]["tool_calls"][0]["function"]["arguments"]
    assert final["claims"] == [] and final["not_found"] is True
    many = [[call("search_library", {"query": f"moon gpu {i}"})] for i in range(4)] + abstain
    assert "too_many_searches" in accept(collect_one(library, tmp_path, q, many, name="m.jsonl"), q)

    ext = question("tr-ext-1", "Find new papers about speculative decoding", kind="external", relevant=())
    record = collect_one(library, tmp_path, ext, abstain, name="e.jsonl")
    assert accept(record, ext) == ["no_search_external"]
    with_search = with_extra_call(record, "search_external", {"query": "speculative decoding"}, "Found 2 papers")
    assert accept(with_search, ext) == []
    found = collect_one(library, tmp_path, ext, good_script(library), name="e2.jsonl")
    assert "expected_not_found" in accept(found, ext)


def test_accept_compare_needs_both_papers(library, tmp_path):
    q = question("tr-cmp-1", "Compare draft-model speedup and tree verification memory", kind="compare",
                 relevant=((SPEEDUP, "Speedup drops"), (TREE, "It needs more memory.")))
    assert accept(collect_one(library, tmp_path, q), q) == ["missing_paper"]
    both = [[call("search_library", {"query": "tree verification memory"})],
            [call("final_answer", {"answer": "Draft speedup drops on disagreement; tree verification needs memory.",
                                   "claims": [{"text": "Speedup drops on disagreement.",
                                               "sentence_ids": [sid(library, "Speedup drops")]},
                                              {"text": "Tree verification needs more memory.",
                                               "sentence_ids": [sid(library, "It needs more memory")]}]})]]
    assert accept(collect_one(library, tmp_path, q, both, name="c.jsonl"), q) == []


def test_choose_best_prefers_gold_then_fewer_steps(library, tmp_path):
    record = collect_one(library, tmp_path)
    short = copy.deepcopy(record)
    short.update(sample=1)
    short["result"]["steps"] = 1
    no_gold = copy.deepcopy(short)
    no_gold.update(sample=2)
    no_gold["result"]["gold_hit"] = False
    assert choose_best([record, short, no_gold])["sample"] == 1
    assert choose_best([]) is None


# --- пример SFT ---

def test_to_sft_renders_template_and_masks(library, tmp_path):
    record = collect_one(library, tmp_path)
    example = to_sft(record, question(), "train")
    assert example["id"] == "tr-fact-000001-s0" and example["split"] == "train" and example["kind"] == "fact"
    messages = example["messages"]
    assert messages[-1]["role"] == "assistant" and all(m.get("content") != "Answer accepted." for m in messages)
    calls = [c for m in messages for c in m.get("tool_calls") or []]
    assert all(isinstance(c["function"]["arguments"], dict) for c in calls)
    assert calls[0]["function"]["arguments"] == {"query": question().question, "k": 6}
    assert [m["weight"] for m in messages if m["role"] == "assistant"] == [0, 1, 1]
    text = example["text"]
    assert text.startswith("<|im_start|>system\n# Tools") and text.count("<|im_start|>system") == 1
    assert "<tool_call>\n<function=search_library>\n<parameter=query>" in text
    user_end = text.index("<|im_end|>", text.index("<|im_start|>user\n"))
    assert text.count("<|im_start|>assistant\n") == text[user_end:].count(ASSISTANT_PREFIX) == 3
    spans = example["train_spans"]
    assert len(spans) == 2 and all(text[s:e].endswith("<|im_end|>") for s, e in spans)
    prefetch_at = text.index("<function=search_library>\n<parameter=query>\n" + question().question)
    final_at = text.index("<function=final_answer>")
    assert not any(s <= prefetch_at < e for s, e in spans)
    assert any(s <= final_at < e for s, e in spans)
    assert text.endswith("</tool_call><|im_end|>\n")
    meta = example["meta"]
    assert meta["qid"] == record["qid"] and meta["prompt_hash"] == prompt_hash() and meta["teacher"] == "mercury-test"
    assert meta["est_tokens"] == est_tokens(text) and meta["steps"] == 2
    assert validate_example(example) == []


def test_to_sft_drops_non_dict_arguments_and_strips_think(library, tmp_path):
    record = collect_one(library, tmp_path)
    broken = copy.deepcopy(record)
    broken["messages"][4]["tool_calls"][0]["function"]["arguments"] = "[1, 2]"
    assert to_sft(broken, question(), "train") is None
    thinking = copy.deepcopy(record)
    thinking["messages"][4]["content"] = "<think>hmm, search more</think>\n\nLet me search."
    example = to_sft(thinking, question(), "train")
    assert example["messages"][4]["content"] == "Let me search."
    assert "hmm" not in example["text"] and validate_example(example) == []
    assert strip_think("<think>unclosed") == "" and strip_think("a</think>b") == "b"


def test_recovery_turn_has_no_weight(library, tmp_path):
    script = [[call("final_answer", {"answer": "x", "claims": [{"text": "c", "sentence_ids": ["s999999"]}]})]]
    record = collect_one(library, tmp_path, script=script + good_script(library))
    example = to_sft(record, question(), "train")
    assert [m["weight"] for m in example["messages"] if m["role"] == "assistant"] == [0, 0, 1, 1]
    assert len(example["train_spans"]) == 2 and validate_example(example) == []


def test_followup_history_is_outside_spans(library, tmp_path):
    history = [{"role": "user", "content": "What limits speculative decoding?"},
               {"role": "assistant", "content": "Mostly the HISTORY-ANSWER about draft disagreement."}]
    q = question("tr-fu-1", "And why does its speedup drop?", kind="followup", history=history)
    record = collect_one(library, tmp_path, q)
    assert record["messages"][1:3] == history
    example = to_sft(record, q, "train")
    assert [m["weight"] for m in example["messages"] if m["role"] == "assistant"] == [0, 0, 1, 1]
    text = example["text"]
    at = text.index("HISTORY-ANSWER")
    assert not any(s <= at < e for s, e in example["train_spans"])
    # ход истории рендерится без <think>, как в инференсе: он до последнего вопроса пользователя
    assert "<|im_start|>assistant\nMostly the HISTORY-ANSWER" in text
    assert validate_example(example) == []


def test_validate_example_catches_broken_examples(library, tmp_path):
    example = to_sft(collect_one(library, tmp_path), question(), "train")

    def broken(change):
        bad = copy.deepcopy(example)
        change(bad)
        return validate_example(bad)

    assert "system" in broken(lambda e: e["messages"].insert(2, {"role": "system", "content": "x"}))
    assert any(p.startswith("order") for p in broken(lambda e: e["messages"].pop(3)))  # нет ответа инструмента
    assert "last_not_final_answer" in broken(lambda e: e["messages"].append({"role": "tool", "tool_call_id": "y",
                                                                             "content": "Answer accepted."}))
    assert "history_order" in broken(lambda e: e["messages"].insert(1, {"role": "assistant", "content": "x"}))
    assert "arguments_not_dict" in broken(
        lambda e: e["messages"][-1]["tool_calls"][0]["function"].update(arguments="{}"))
    assert "think_leak" in broken(lambda e: e["messages"][-1].update(content="<think>x</think>"))
    assert "span_end" in broken(lambda e: e["train_spans"][0].__setitem__(1, e["train_spans"][0][1] - 1))
    assert "span_start" in broken(lambda e: e["train_spans"][0].__setitem__(0, e["train_spans"][0][0] - 1))
    assert "span_count" in broken(lambda e: e["train_spans"].pop())


def test_est_tokens():
    assert est_tokens("") == 0 and est_tokens("abc") == 1 and est_tokens("a" * 34) == 10 and est_tokens("a" * 35) == 11


# --- экспорт ---

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def question_row(q):
    return {"id": q.id, "question": q.question, "language": q.language, "topic": None, "status": "draft",
            "relevant": [{"paper": i.paper, "quote": i.quote} for i in q.relevant], "kind": q.kind, "meta": q.meta}


def test_export_writes_files_and_caps(library, tmp_path):
    root = tmp_path / "finetune"
    good = question("tr-fact-000001")
    recov = question("tr-fact-000002")
    leak = question("tr-fact-000003", relevant=((TREE, "Tree-based verification"),))
    bad = question("tr-fact-000004")
    trajectories = root / "trajectories" / "train.jsonl"
    rejected_answer = [[call("final_answer", {"answer": "x", "claims": [{"text": "c", "sentence_ids": ["s999999"]}]})]]
    tree_script = good_script(library, sentence=sid(library, "Tree-based"),
                              answer="Tree verification checks many continuations at once.", query="tree verification")
    scripts = [(good, good_script(library) + good_script(library)), (recov, rejected_answer + good_script(library)),
               (leak, tree_script), (bad, ["just text", "more text", "still text"])]
    for q, script in scripts:
        collect([q], make_context(library), teacher(script), trajectories, samples=2 if q is good else 1,
                log=lambda s: None)
    # Повтор сбора, упавший на API после удачи, не вытесняет удачную запись.
    with trajectories.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"qid": good.id, "sample": 0, "kind": "fact", "error": "APIError: 500"}) + "\n")
    write_jsonl(root / "questions" / "train.jsonl", [question_row(q) for q, _ in scripts])
    (root / "splits.json").write_text(json.dumps({"test": [paper_id(library, TREE)], "val": [], "train": []}))

    stats = export(root / "trajectories", root / "questions", root, ["train"], log=lambda s: None)
    examples = load_records(root / "sft" / "train.jsonl")
    assert [e["id"] for e in examples] == ["tr-fact-000001-s0"]  # recovery: 0.1 · 1 / 0.9 → 0 мест
    rejected = load_records(root / "rejected.jsonl")
    reasons = {(r["qid"], r["sample"]): r["reasons"] for r in rejected}
    assert reasons[("tr-fact-000001", 1)] == ["not_chosen"]  # принят, но выбран s0
    assert reasons[("tr-fact-000002", 0)] == ["recovery_cap"]
    assert "test_leak" in reasons[("tr-fact-000003", 0)]
    assert "nudge" in reasons[("tr-fact-000004", 0)]
    assert len(rejected) == 4 and not any(r["error"] for r in rejected)
    saved = json.loads((root / "sft" / "stats.json").read_text(encoding="utf-8"))
    assert saved == stats
    split = stats["splits"]["train"]
    assert split["examples"] == 1 and split["by_kind"] == {"fact": 1} and split["by_language"] == {"en": 1}
    assert split["hard"] == 2 and stats["hard"]["train"] == ["tr-fact-000003", "tr-fact-000004"]
    assert split["recovery"] == 0 and split["recovery_capped_questions"] == 1
    assert stats["rejection_reasons"]["recovery_cap"] == 1 and sum(split["tokens"]["histogram"].values()) == 1

    # Доля 0.5 при одном обычном примере даёт одно место: recovery остаётся, ход с ошибкой — без веса.
    stats = export(root / "trajectories", root / "questions", root, ["train"], recovery_share=0.5,
                   log=lambda s: None)
    examples = {e["id"]: e for e in load_records(root / "sft" / "train.jsonl")}
    assert sorted(examples) == ["tr-fact-000001-s0", "tr-fact-000002-s0"]
    recovered = examples["tr-fact-000002-s0"]
    assert recovered["meta"]["recovery"] and not examples["tr-fact-000001-s0"]["meta"]["recovery"]
    assert [m["weight"] for m in recovered["messages"] if m["role"] == "assistant"] == [0, 0, 1, 1]
    assert stats["splits"]["train"]["recovery"] == 1 and "recovery_cap" not in stats["rejection_reasons"]
    assert stats["splits"]["train"]["recovery_capped_questions"] == 0

    stats = export(root / "trajectories", root / "questions", root, ["train"], recovery_share=0.5, max_tokens=100,
                   log=lambda s: None)
    assert stats["splits"]["train"]["examples"] == 0 and stats["rejection_reasons"]["too_long"] == 3


def test_recovery_cap_falls_back_to_plain_sample(library, tmp_path):
    root = tmp_path / "finetune"
    q = question()
    rejected_answer = [[call("final_answer", {"answer": "x", "claims": [{"text": "c", "sentence_ids": ["s999999"]}]})]]
    recovery = collect_one(library, tmp_path, script=rejected_answer + good_script(library), name="r.jsonl")
    plain = collect_one(library, tmp_path, name="p.jsonl")
    plain["sample"] = 1
    plain["result"]["gold_hit"] = False  # принят (статья верная), но золотой фрагмент не процитирован
    assert is_recovery(recovery) and choose_best([recovery, plain]) is recovery
    write_jsonl(root / "trajectories" / "train.jsonl", [recovery, plain])
    write_jsonl(root / "questions" / "train.jsonl", [question_row(q)])
    (root / "splits.json").write_text(json.dumps({"test": [], "val": [], "train": []}))

    stats = export(root / "trajectories", root / "questions", root, ["train"], log=lambda s: None)
    assert [e["id"] for e in load_records(root / "sft" / "train.jsonl")] == [f"{q.id}-s1"]
    assert [(r["sample"], r["reasons"]) for r in load_records(root / "rejected.jsonl")] == [(0, ["recovery_cap"])]
    split = stats["splits"]["train"]
    assert split["examples"] == 1 and split["hard"] == 0 and split["recovery_capped_questions"] == 0


def test_final_answer_drops_not_found_false(library, tmp_path):
    root = tmp_path / "finetune"
    script = good_script(library)
    explicit = json.loads(script[1][0].function.arguments) | {"not_found": False}
    script[1] = [call("final_answer", explicit)]
    record = collect_one(library, tmp_path, script=script)
    example = to_sft(record, question(), "train")
    final = example["messages"][-1]["tool_calls"][0]["function"]["arguments"]
    assert "not_found" not in final and "<parameter=not_found>" not in example["text"]
    assert "not_found" in record["messages"][-2]["tool_calls"][0]["function"]["arguments"]  # в записи как было
    assert example["meta"]["dropped_not_found_false"] == 1 and validate_example(example) == []

    q = question("tr-un-1", "What GPU did the authors of the moon landing use?", kind="unanswerable", relevant=())
    abstain = [[call("final_answer", {"answer": "The library has no such information.", "claims": [],
                                      "not_found": True})]]
    kept = to_sft(collect_one(library, tmp_path, q, abstain, name="u.jsonl"), q, "train")
    assert kept["messages"][-1]["tool_calls"][0]["function"]["arguments"]["not_found"] is True
    assert kept["meta"]["dropped_not_found_false"] == 0

    write_jsonl(root / "trajectories" / "train.jsonl", [record])
    write_jsonl(root / "questions" / "train.jsonl", [question_row(question())])
    stats = export(root / "trajectories", root / "questions", root, {"test": [], "val": [], "train": []},
                   log=lambda s: None)
    assert stats["splits"]["train"]["dropped_not_found_false"] == 1


def test_export_val_split_and_missing_splits(library, tmp_path):
    root = tmp_path / "finetune"
    val = question("tr-fact-000101", split="val")
    collect([val], make_context(library), teacher(good_script(library)), root / "trajectories" / "val.jsonl",
            samples=1, log=lambda s: None)
    write_jsonl(root / "questions" / "val.jsonl", [question_row(val)])

    with pytest.raises(ValueError, match="splits.json"):
        export(root / "trajectories", root / "questions", root, ["train", "val"], log=lambda s: None)
    assert not (root / "sft").exists()

    stats = export(root / "trajectories", root / "questions", root, {"test": [], "val": [], "train": []},
                   log=lambda s: None)
    assert list(stats["splits"]) == ["train", "val"]
    assert stats["splits"]["train"]["examples"] == 0 and stats["splits"]["val"]["examples"] == 1
    (example,) = load_records(root / "sft" / "val.jsonl")
    assert example["split"] == "val" and example["id"] == "tr-fact-000101-s0"

    # Утечка проверяется и для val: статья ответа — тестовая.
    stats = export(root / "trajectories", root / "questions", root,
                   {"test": [paper_id(library, SPEEDUP)], "val": [], "train": []}, log=lambda s: None)
    assert stats["splits"]["val"]["examples"] == 0 and stats["rejection_reasons"] == {"test_leak": 1}


def test_teacher_hint_is_used_in_collection_but_not_in_training_text():
    from research_assistant.agent.prompts import SYSTEM_PROMPT
    from research_assistant.finetune.export import build_example
    from research_assistant.finetune.trajectories import TEACHER_HINTS, teacher_hint

    question = GoldQuestion(id="q1", question="Why?", language="en", relevant=[])
    assert teacher_hint(question) == TEACHER_HINTS["en"]
    record = {"qid": "q1", "sample": 0, "model": "mercury-3", "prompt_hash": "h", "result": {"steps": 1},
              "messages": [
        {"role": "system", "content": SYSTEM_PROMPT + "\n" + TEACHER_HINTS["en"]},
        {"role": "user", "content": "Why?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "final_answer", "arguments": '{"answer": "Not found.", "claims": [], "not_found": true}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "Answer accepted."},
    ], "tools": []}
    example, reason = build_example(record, question, "train")
    assert reason is None and example["messages"][0]["content"] == SYSTEM_PROMPT
    assert TEACHER_HINTS["en"] not in example["text"]

    record["messages"][0] = {"role": "system", "content": "Some other prompt"}
    assert build_example(record, question, "train") == (None, "foreign_system_prompt")
