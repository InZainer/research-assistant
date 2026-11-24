"""Метрики поведения агента по журналу (docs/finetune.md, «Метрики поведения»), без сети и модели."""

import json
from types import SimpleNamespace

import pytest

from research_assistant.agent import prompts
from research_assistant.agent.loop import AgentResult, run_agent
from research_assistant.agent.tools import ToolContext, prompt_hash
from research_assistant.evaluation.agent import (
    error_kind,
    evaluate_agent,
    format_agent_report,
    journal_metrics,
    percentile,
    rescore_log,
    results_from_log,
    score_run,
)
from research_assistant.evaluation.retrieval import GoldItem, GoldQuestion, load_questions, question_group
from research_assistant.ingest.models import PaperRecord
from research_assistant.llm import ChatModel
from research_assistant.parse.passages import SectionOut, paragraph_passages
from research_assistant.parse.structure import Paragraph
from research_assistant.retrieval.search import Searcher
from research_assistant.store.db import Library


# --- подменная модель и библиотека (как в test_agent.py) ---

def call(name, args, call_id=None):
    arguments = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
    return SimpleNamespace(id=call_id or f"call_{name}", function=SimpleNamespace(name=name, arguments=arguments))


class ScriptedClient:
    """Отвечает по сценарию: элемент — список вызовов или текст."""

    def __init__(self, script):
        self.script = list(script)
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        turn = self.script.pop(0)
        content, calls = (turn, None) if isinstance(turn, str) else (None, turn)
        message = SimpleNamespace(content=content, tool_calls=calls)
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=10)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=usage)


def model(script):
    return ChatModel("fake-model", client=ScriptedClient(script))


@pytest.fixture
def ctx():
    with Library(":memory:") as lib:
        topic = lib.get_or_create_topic("speculative decoding")
        texts = [
            ("Speculative decoding with draft models", "limitations",
             "Speedup drops when the draft model disagrees with the target model. Rare tokens are the main cause."),
            ("Tree attention for verification", "method",
             "Tree-based verification checks many candidate continuations in one forward pass. It needs more memory."),
        ]
        for title, kind, text in texts:
            paper_id, _ = lib.upsert_paper(PaperRecord(title=title, sources=["openalex"], year=2025, language="en"))
            lib.link_topic(paper_id, topic)
            section = SectionOut(kind.title(), kind, 1)
            section.passages = paragraph_passages(Paragraph(text, [(0, 3)]))
            lib.save_structure(paper_id, [section], status="ok", parser="pdf", version=1)
        yield ToolContext(library=lib, searcher=Searcher(lib))


def sentence_id(ctx, startswith):
    return ctx.library.conn.execute("SELECT id FROM sentences WHERE text LIKE ?", (startswith + "%",)).fetchone()[0]


def read_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def answer(text, sid=None):
    """Ход модели: final_answer с цитатой или not_found."""
    if sid is None:
        return [call("final_answer", {"answer": text, "claims": [], "not_found": True})]
    return [call("final_answer", {"answer": text, "claims": [{"text": f"claim: {text}", "sentence_ids": [f"s{sid}"]}]})]


SPEEDUP = GoldQuestion("en1", "why does speedup drop?", "en",
                       [GoldItem("Speculative decoding with draft models", "Rare tokens are the main cause.")])


# --- prompt_hash ---

def test_prompt_hash_is_stable_and_tracks_the_prompt(monkeypatch):
    first = prompt_hash()
    assert first == prompt_hash() and len(first) == 40
    monkeypatch.setattr(prompts, "SYSTEM_PROMPT", prompts.SYSTEM_PROMPT + "\n9. Be brief.")
    assert prompt_hash() != first


# --- живой прогон через run_agent ---

def test_metrics_of_a_real_run(ctx, tmp_path):
    good = sentence_id(ctx, "Speedup drops")
    script = [
        [call("search_library", "{not json")],                                   # ошибка формата
        [call("read_passage", {"passage_id": 999}, "c1"),                       # ошибка инструмента
         call("search_library", {"query": "tree verification"}, "c2")],          # параллельный вызов
        [call("search_library", {"query": "Why does  SPEEDUP drop?", "k": 3})],  # повтор поиска шага 0
        [call("add_papers", {"ids": ["arxiv:1"], "topic": "t"})],                # добавить без просьбы
        [call("final_answer", {"answer": "x", "claims": [{"text": "c", "sentence_ids": ["s999999"]}]})],
        answer("Из-за расхождения моделей.", good),                              # ответ не на языке вопроса
    ]
    log = tmp_path / "run.jsonl"
    seen = []
    result = run_agent(SPEEDUP.question, ctx, model(script), log_path=log, prefetch=True,
                       on_event=lambda event, data: seen.append({"event": event, **data}))
    assert result.stop_reason == "final_answer" and result.nudges == 0

    events = read_events(log)
    assert events[0]["prompt_hash"] == prompt_hash()
    tools = [e for e in events if e["event"] == "tool"]
    assert [e["error_kind"] for e in tools] == [None, "json", "tool_error", None, None, "tool_error",
                                                "answer_rejected", None]
    assert all(e["final_only"] is False for e in events if e["event"] == "llm")
    assert events[-1]["nudges"] == 0

    metrics = journal_metrics(events, SPEEDUP)
    assert metrics["calls"] == 7 and metrics["llm_steps"] == 6 and metrics["steps"] == 6
    assert metrics["call_validity"] == pytest.approx(6 / 7)
    assert metrics["tool_error_rate"] == pytest.approx(2 / 7)
    assert metrics["redundant_rate"] == pytest.approx(1 / 7)
    assert metrics["first_answer_accept"] is False
    assert metrics["lang_match"] is False
    assert metrics["unauthorized"] == 1
    assert metrics["read_passage"] == 1 and metrics["parallel_steps"] == 1
    assert metrics["research_calls"] == 4 and not metrics["budget_exhausted"]
    assert not metrics["nudge"] and not metrics["max_steps"]
    assert metrics["tokens"] == 6 * 110
    assert metrics["errors"] == {"json": 1, "tool_error": 2, "answer_rejected": 1}

    # события из listener и из файла дают одно и то же; старый журнал без error_kind — тоже
    assert journal_metrics(seen, SPEEDUP) == metrics
    old = [{k: v for k, v in e.items() if k not in ("error_kind", "prompt_hash", "final_only", "nudges")}
           for e in events]
    assert journal_metrics(old, SPEEDUP) == metrics

    # просьба добавить — не нарушение
    asked = [{**events[0], "question": "Добавь статьи про speculative decoding"}, *events[1:]]
    assert journal_metrics(asked)["unauthorized"] == 0


def test_last_step_is_logged_as_final_only(ctx, tmp_path):
    log = tmp_path / "run.jsonl"
    run_agent("q", ctx, model([[call("search_library", {"query": "a"})], answer("нет")]), max_steps=2, log_path=log)
    assert [e["final_only"] for e in read_events(log) if e["event"] == "llm"] == [False, True]


# --- синтетические журналы ---

def test_error_kind_falls_back_to_text_for_old_journals():
    def kind(error, name="search_library"):
        return error_kind({"event": "tool", "name": name, "error": error})

    assert kind(None) is None
    assert kind("Error: arguments are not valid JSON (Expecting value). Call x again.") == "json"
    assert kind("Error: unknown tool teleport; available: search_library") == "unknown_tool"
    assert kind("Error: wrong arguments for search_library: unexpected keyword 'qeury'") == "bad_args"
    assert kind("Error: arguments must be a JSON object") == "bad_args"
    assert kind("Error: no passage or sentence with id 999") == "tool_error"
    assert kind("Error in search_library: OperationalError: database is locked") == "env"
    assert kind("Fix final_answer: ids ['s1'] were not in your tool results", "final_answer") == "answer_rejected"
    assert kind("final_answer.answer must be a non-empty string", "final_answer") == "answer_rejected"
    assert error_kind({"error": "Error: anything", "error_kind": "env"}) == "env"  # новый журнал — как записано


def synthetic_run(question, tool_calls, stop_reason="final_answer", answer_text="ok", nudge=False):
    """Журнал одного прогона: tool_calls — [(шаг, имя, аргументы, ошибка, наблюдение)]."""
    events = [{"event": "start", "question": question}]
    for step in sorted({c[0] for c in tool_calls if c[0] > 0}):
        names = [c for c in tool_calls if c[0] == step]
        events.append({"event": "llm", "step": step,
                       "tool_calls": [{"name": n, "arguments": a} for _, n, a, _, _ in names]})
    if nudge:
        events.append({"event": "nudge", "step": 1})
    for step, name, arguments, error, observation in tool_calls:
        events.append({"event": "tool", "step": step, "name": name, "arguments": arguments, "error": error,
                       "observation": observation})
    steps = max(c[0] for c in tool_calls)
    events.append({"event": "finish", "stop_reason": stop_reason, "steps": steps, "tool_calls": len(tool_calls),
                   "tool_errors": sum(bool(c[3]) for c in tool_calls), "not_found": False, "answer": answer_text,
                   "usage": {"prompt": 1000, "completion": 50, "calls": steps}, "seconds": 3.5})
    return events


def test_budget_nudge_and_max_steps_from_a_synthetic_journal():
    q = "how does tree verification work?"
    searches = [(i, "search_library", json.dumps({"query": f"tree {i}"}), None, "Found") for i in range(1, 7)]
    events = synthetic_run(q, [(0, "search_library", q, None, "Found"), *searches,
                               (7, "search_library", json.dumps({"query": "TREE 3", "k": 8}), None, "You already")],
                           stop_reason="max_steps", answer_text="", nudge=True)
    metrics = journal_metrics(events, GoldQuestion("t", q, "en", []))
    assert metrics["budget_exhausted"] and metrics["max_steps"] and metrics["nudge"] and metrics["nudges"] == 1
    assert metrics["redundant_rate"] == pytest.approx(1 / 7)
    assert metrics["first_answer_accept"] is None and metrics["lang_match"] is None  # ответа нет
    assert metrics["call_validity"] == 1.0 and metrics["tokens"] == 1050 and metrics["seconds"] == 3.5


def test_percentile():
    assert percentile([], 0.5) is None
    assert percentile([3, 1, 2], 0.5) == 2
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.95) == pytest.approx(9.55)


# --- вопросы без ответа, группы, пересчёт ---

def test_question_group_without_relevant_and_new_fields(tmp_path):
    path = tmp_path / "q.jsonl"
    rows = [
        {"id": "u1", "question": "Сколько стоит GPU?", "language": "ru", "relevant": [], "kind": "unanswerable",
         "meta": {"split": "test", "history": []}},
        {"id": "u2", "question": "GPU price?", "language": "en", "relevant": []},
    ]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    ru, en = load_questions(path)
    assert ru.kind == "unanswerable" and ru.meta["split"] == "test"
    assert en.kind == "fact" and en.meta == {}
    assert question_group(ru) == "ru→none" and question_group(en) == "en→none"


def test_unanswerable_success_is_not_found(ctx):
    nf = AgentResult(question="q", answer="нет", not_found=True)
    assert score_run(ctx.library, set(), nf, kind="unanswerable")["success"] is True
    assert score_run(ctx.library, set(), nf, kind="unanswerable")["success_paper"] is True
    assert score_run(ctx.library, set(), nf)["success"] is False
    answered = AgentResult(question="q", answer="да")
    assert score_run(ctx.library, set(), answered, kind="unanswerable")["success"] is False
    stuck = AgentResult(question="q", answer="", not_found=True, stop_reason="max_steps")
    assert score_run(ctx.library, set(), stuck, kind="unanswerable")["success"] is False


def test_rescore_counts_runs_and_abstention(ctx, tmp_path):
    good = sentence_id(ctx, "Rare tokens")
    unanswerable = GoldQuestion("u1", "Сколько стоят rare tokens на видеокарте?", "ru", [], kind="unanswerable")
    tree = GoldQuestion("en2", "what does tree verification check?", "en",
                        [GoldItem("Tree attention for verification", "checks many candidate continuations")])
    log = tmp_path / "eval.jsonl"
    # прогон 1: верно, честный not_found, ложный not_found;
    # прогон 2: верно, ответ без опоры, ложный not_found не на языке вопроса
    scripts = [
        (SPEEDUP, answer("Rare tokens.", good)), (unanswerable, answer("В библиотеке нет цен.")),
        (tree, answer("Not found.")),
        (SPEEDUP, answer("Rare tokens.", good)), (unanswerable, answer("Дорого.", good)), (tree, answer("Не найдено.")),
    ]
    for q, turn in scripts:
        run_agent(q.question, ctx, model([turn]), log_path=log, prefetch=True)

    report = rescore_log(ctx.library, [SPEEDUP, unanswerable, tree], log, "prefetch")
    assert report["runs"] == 2 and report["prompt_hash"] == prompt_hash()
    assert report["journal_prompt_hashes"] == [prompt_hash()]
    assert [(r["id"], r["run"]) for r in report["rows"]] == [
        ("en1", 1), ("u1", 1), ("en2", 1), ("en1", 2), ("u1", 2), ("en2", 2)]
    s = report["summary"]["all"]
    assert s["pass@1"] == pytest.approx(3 / 6) and s["pass^2"] == pytest.approx(1 / 3)
    assert s["abstention_precision"] == pytest.approx(1 / 3)  # 1 честный not_found из 3
    assert s["abstention_recall"] == pytest.approx(1 / 2)
    assert s["false_not_found"] == pytest.approx(2 / 4)
    assert s["call_validity"] == 1.0 and s["redundant_rate"] == 0.0 and s["first_answer_accept"] == 1.0
    assert s["lang_match"] == pytest.approx(5 / 6)  # мимо только «Не найдено.» на английский вопрос
    assert set(report["summary"]) == {"all", "en→en", "ru→none"}
    assert report["summary"]["ru→none"]["pass^2"] == 0.0

    # явное число прогонов тоже принимается
    assert "pass^3" in rescore_log(ctx.library, [SPEEDUP], log, "prefetch", runs=3)["summary"]["all"]

    text = format_agent_report(report)
    assert "поведение" in text and "ru→none" in text and "Внимание" not in text
    report["journal_prompt_hashes"] = ["0" * 40]
    assert "Внимание" in format_agent_report(report)

    # тексты утверждений сохраняются
    first = results_from_log(ctx.library, log)[0]
    assert first.claims[0].text == "claim: Rare tokens." and first.claims[0].citations[0].sentence_id == good


def test_evaluate_agent_and_rescore_agree(ctx, tmp_path):
    good = sentence_id(ctx, "Rare tokens")
    client = ScriptedClient([
        [call("search_library", {"query": "why does speedup drop?"})],  # повтор поиска шага 0
        answer("Rare tokens.", good),
    ])
    report = evaluate_agent(lambda: ToolContext(library=ctx.library, searcher=ctx.searcher),
                            ChatModel("fake", client=client), [SPEEDUP], variant="prefetch", log_dir=tmp_path)
    live = report["summary"]["all"]
    assert live["redundant_rate"] == 0.5 and live["read_passage_per_q"] == 0.0 and live["pass@1"] == 1.0
    assert report["rows"][0]["kind"] == "fact"
    [log] = tmp_path.glob("eval-agent-prefetch-*.jsonl")
    again = rescore_log(ctx.library, [SPEEDUP], log, "prefetch")["summary"]["all"]
    keys = ("call_validity", "redundant_rate", "first_answer_accept", "lang_match", "unauthorized",
            "read_passage_per_q", "pass@1", "pass@1_paper", "steps", "tokens")
    assert {k: again[k] for k in keys} == {k: live[k] for k in keys}
    assert "поведение" in format_agent_report(report)


# --- виды ошибок, которых нет в живом прогоне выше ---

def test_error_kind_of_format_and_env_errors(ctx, tmp_path, monkeypatch):
    from research_assistant.agent import loop

    def broken(ctx, **kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setitem(loop.FUNCTIONS, "list_papers", broken)
    script = [
        [call("teleport", {"to": "mars"})],                       # неизвестный инструмент
        [call("search_library", {"qeury": "speedup"})],           # не те аргументы (TypeError)
        [call("search_library", "[1, 2]")],                       # JSON, но не объект
        [call("list_papers", {})],                                # сбой среды
        answer("нет"),
    ]
    log = tmp_path / "run.jsonl"
    run_agent("q", ctx, model(script), log_path=log)
    tools = [e for e in read_events(log) if e["event"] == "tool"]
    expected = ["unknown_tool", "bad_args", "bad_args", "env", None]
    assert [e["error_kind"] for e in tools] == expected
    # старый журнал без error_kind даёт те же виды по тексту ошибки
    assert [error_kind({k: v for k, v in e.items() if k != "error_kind"}) for e in tools] == expected
    metrics = journal_metrics(read_events(log))
    assert metrics["errors"] == {"unknown_tool": 1, "bad_args": 2, "env": 1}
    assert metrics["call_validity"] == pytest.approx(2 / 5) and metrics["tool_error_rate"] == 0.0


# --- followup: вопрос задаётся с историей; пересчёт различает вопросы с одинаковым текстом ---

class RecordingClient(ScriptedClient):
    """ScriptedClient, который запоминает сообщения каждого запроса."""

    def __init__(self, script):
        super().__init__(script)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs["messages"])
        return super().create(**kwargs)


HISTORY_A = [{"role": "user", "content": "Что такое speculative decoding?"},
             {"role": "assistant", "content": "Ускорение генерации черновой моделью."}]
HISTORY_B = [{"role": "user", "content": "Что такое tree attention?"},
             {"role": "assistant", "content": "Проверка многих продолжений за один проход."}]


def followups():
    """Два followup с одинаковым вторым вопросом и отклонённый кандидат с тем же текстом и историей."""
    first = GoldQuestion("f1", "а какие у неё ограничения?", "ru",
                         [GoldItem("Speculative decoding with draft models", "Rare tokens are the main cause.")],
                         kind="followup", meta={"history": HISTORY_A})
    second = GoldQuestion("f2", "а какие у неё ограничения?", "ru",
                          [GoldItem("Tree attention for verification", "It needs more memory.")],
                          kind="followup", meta={"history": HISTORY_B})
    rejected = GoldQuestion("f2x", "а какие у неё ограничения?", "ru", [], kind="followup",
                            meta={"history": HISTORY_B, "accepted": False})
    return first, second, rejected


def test_evaluate_agent_passes_followup_history(ctx, tmp_path):
    first, second, rejected = followups()
    rare, memory = sentence_id(ctx, "Rare tokens"), sentence_id(ctx, "It needs more memory")
    speedup = [call("search_library", {"query": "speedup rare tokens"})]
    tree = [call("search_library", {"query": "tree verification memory"})]
    client = RecordingClient([speedup, answer("Редкие токены.", rare), tree, answer("Нужно больше памяти.", memory),
                              tree, answer("Нужно больше памяти.", memory)])
    report = evaluate_agent(lambda: ToolContext(library=ctx.library, searcher=ctx.searcher),
                            ChatModel("fake", client=client), [first, second, rejected], log_dir=tmp_path)
    # модель видит предыдущий обмен между системным промптом и вопросом
    for request, history in zip(client.requests[::2], [HISTORY_A, HISTORY_B, HISTORY_B]):
        assert request[1:3] == history and request[3] == {"role": "user", "content": "а какие у неё ограничения?"}
    assert [r["success"] for r in report["rows"]] == [True, True, False]

    [log] = tmp_path.glob("eval-agent-agent-*.jsonl")
    starts = [e for e in read_events(log) if e["event"] == "start"]
    assert [(e["qid"], e["history"]) for e in starts] == [("f1", HISTORY_A), ("f2", HISTORY_B), ("f2x", HISTORY_B)]

    # новый журнал: по qid, каждый прогон — своему вопросу, без предупреждений
    again = rescore_log(ctx.library, [first, second, rejected], log, "agent")
    assert [(r["id"], r["run"], r["success"]) for r in again["rows"]] == [
        ("f1", 1, True), ("f2", 1, True), ("f2x", 1, False)]
    assert again["runs"] == 1 and again["warnings"] == []
    # набор сузили до одного вопроса — чужие прогоны не приписываются ему по тексту
    assert [r["id"] for r in rescore_log(ctx.library, [second], log, "agent")["rows"]] == ["f2"]

    # старый журнал без qid, но с историей: f1 отделяется по истории, f2 и f2x — по порядку, с предупреждением
    old_log = tmp_path / "old.jsonl"
    old_log.write_text("".join(json.dumps({k: v for k, v in e.items() if k != "qid"}, ensure_ascii=False) + "\n"
                               for e in read_events(log)), encoding="utf-8")
    old = rescore_log(ctx.library, [first, second, rejected], old_log, "agent")
    assert [(r["id"], r["run"]) for r in old["rows"]] == [("f1", 1), ("f2", 1), ("f2x", 1)]
    assert len(old["warnings"]) == 1 and "Внимание: в журнале нет ID" in format_agent_report(old)

    # совсем старый журнал (ни qid, ни истории): все три — по порядку
    oldest_log = tmp_path / "oldest.jsonl"
    oldest_log.write_text("".join(
        json.dumps({k: v for k, v in e.items() if k not in ("qid", "history")}, ensure_ascii=False) + "\n"
        for e in read_events(log)), encoding="utf-8")
    oldest = rescore_log(ctx.library, [first, second, rejected], oldest_log, "agent")
    assert [r["id"] for r in oldest["rows"]] == ["f1", "f2", "f2x"] and len(oldest["warnings"]) == 1
