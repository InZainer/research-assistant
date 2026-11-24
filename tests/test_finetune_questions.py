import json
import re
from types import SimpleNamespace

import pytest

from research_assistant.evaluation.retrieval import GoldItem, GoldQuestion, load_questions, resolve
from research_assistant.finetune import questions as q
from research_assistant.finetune.questions import (
    CHECK_PROMPT,
    JUDGE_PROMPT,
    SECTION_WEIGHTS,
    Candidate,
    HeldOut,
    TeacherConfigError,
    build_relevant,
    check_candidate,
    format_plan,
    gold_rank,
    paper_quota,
    plan_sampling,
    sample_passages,
    shares_ngram,
    subject_key,
    summarize_plan,
    synthesize,
)
from research_assistant.finetune.splits import (
    jaccard,
    load_splits,
    near_duplicate,
    save_splits,
    split_of_paper,
    split_papers,
)
from research_assistant.ingest.models import PaperRecord
from research_assistant.llm import ChatModel
from research_assistant.nlp.text import detect_language
from research_assistant.parse.passages import SectionOut, paragraph_passages
from research_assistant.parse.structure import Paragraph
from research_assistant.retrieval.search import Searcher
from research_assistant.store.db import Library

SD, MED = "speculative decoding", "машинное обучение в медицине"


# --- библиотека из шести статей: en и ru, аннотация из метаданных, подписи, благодарности, сущности ---

def en_text(name: str, i: int) -> str:
    return (f"During trial {i} the {name} system achieved a throughput gain of {i + 2}.5 times over vanilla "
            f"autoregressive sampling. Memory overhead stayed below {10 + i} percent because the {name} cache was "
            f"pruned after every accepted block. Latency spikes appeared only when the batch exceeded "
            f"{32 * (i + 1)} sequences on a single accelerator.")


def ru_text(name: str, i: int) -> str:
    return (f"В испытании {i} система {name} повысила чувствительность до {80 + i} процентов на снимках из "
            f"областной больницы. Ложноположительные срабатывания составили {i + 3} процента после калибровки "
            f"порога на отложенной выборке. Обучение заняло {i + 5} часов на одном графическом ускорителе.")


def add_paper(lib, title, language, topic, authors, sections, source="fulltext"):
    paper_id, _ = lib.upsert_paper(PaperRecord(title=title, sources=["openalex"], year=2024, language=language,
                                               authors=authors))
    lib.link_topic(paper_id, lib.get_or_create_topic(topic))
    out = []
    for kind, paragraphs in sections:
        section = SectionOut(kind.title(), kind, 1)
        for paragraph in paragraphs:
            text, paragraph_kind = paragraph if isinstance(paragraph, tuple) else (paragraph, "text")
            section.passages += paragraph_passages(Paragraph(text, [(0, 1)], paragraph_kind))
        out.append(section)
    lib.save_structure(paper_id, out, status="ok", parser="pdf", version=1, source=source)
    return paper_id


@pytest.fixture
def lib():
    with Library(":memory:") as library:
        mt_bench = ("The MT-Bench suite was used to measure answer quality of the drafted continuations in every "
                    "configuration. Scores were averaged over three seeds to reduce variance of the judge, and ties were "
                    "broken by a second annotator.")
        add_paper(library, "Speculative decoding with draft models", "en", SD, ["Alice Smith", "Bob Jones"], [
            ("abstract", [en_text("Lookahead", 0)]),
            ("results", [en_text("Lookahead", i) for i in range(1, 9)] + [mt_bench]),
            ("acknowledgements", [en_text("Lookahead", 50)]),
            ("other", [(en_text("Lookahead", 60), "caption"), "Too short passage."]),
            ("discussion", [en_text("Lookahead", 70) + " " + en_text("Lookahead", 71)
                            + " Results vary across hardware."]),
        ])
        add_paper(library, "Tree attention for verification", "en", SD, ["Carol White"], [
            ("abstract", [en_text("Arbor", 0)]),
            ("method", [en_text("Arbor", i) for i in range(1, 5)]),
            ("experiments", [mt_bench.replace("drafted", "tree-verified")]),
        ])
        add_paper(library, "Medusa heads for parallel decoding", "en", SD, ["Dan Brown"], [
            ("abstract", [en_text("Medusa", 0)]),
            ("results", [en_text("Medusa", i) for i in range(1, 4)]),
        ])
        add_paper(library, "Машинное обучение в диагностике пневмонии", "ru", MED, ["Иван Петров"], [
            ("abstract", [ru_text("ПневмоНет", 0)]),
            ("results", [ru_text("ПневмоНет", i) for i in range(1, 6)]),
        ])
        add_paper(library, "Нейросети для анализа ЭКГ", "ru", MED, ["Мария Иванова"], [
            ("abstract", [ru_text("КардиоНет", 0)]),
            ("results", [ru_text("КардиоНет", i) for i in range(1, 4)]),
        ])
        add_paper(library, "Draft verification survey", "en", SD, ["Eve Black"],
                  [("abstract", [en_text("Survey", 0)])], source="metadata")
        for paper_id in (1, 2):
            library.save_annotation(paper_id, {"dataset": ["MT-Bench"], "method": ["Lookahead"] if paper_id == 1 else []},
                                    [], status="ok", annotator="test", version=1)
        yield library


@pytest.fixture
def gold():
    return [GoldQuestion("g1", "What throughput gain does Medusa reach?", "en",
                         [GoldItem("medusa heads for parallel decoding.", "Memory overhead stayed below 11 percent")])]


def passages_of(lib, title):
    return [r[0] for r in lib.conn.execute(
        "SELECT p.id FROM passages p JOIN papers pa ON pa.id = p.paper_id WHERE pa.title = ? ORDER BY p.order_idx",
        (title,))]


def sentences_of(lib, passage_id):
    return [(r[0], r[1]) for r in lib.conn.execute(
        "SELECT id, text FROM sentences WHERE passage_id = ? ORDER BY order_idx", (passage_id,))]


# --- подменный учитель: отвечает по типу запроса ---

TAG_RE = re.compile(r"\[s(\d+)\] ([^\n]*)")
EN_WORDS = ["robustness", "fairness", "scalability", "energy", "stability", "portability", "interpretability",
            "efficiency", "reliability", "transferability", "privacy", "cost", "usability"]
RU_WORDS = ["устойчивость", "масштабируемость", "энергопотребление", "стабильность", "переносимость",
            "интерпретируемость", "эффективность", "надёжность", "воспроизводимость", "стоимость", "удобство"]


def ask(sid: int, ru: bool, template: str) -> str:
    words = RU_WORDS if ru else EN_WORDS
    return template.format(a=words[sid % len(words)], b=words[(sid * 7 + 3) % len(words)], n=sid)


def teacher_reply(system: str, user: str) -> str:
    tags = [(int(i), text) for i, text in TAG_RE.findall(user)]
    ru = "in Russian" in system
    if system == CHECK_PROMPT:
        return json.dumps({"sentence_ids": [i for i, _ in tags]})
    if system == JUDGE_PROMPT:
        return json.dumps({"answering_ids": []})
    sid = tags[0][0] if tags else 0
    if "two-turn conversation" in system:
        return json.dumps({
            "first_question": ask(sid, ru, "Какую {a} обсуждают в варианте {n}?" if ru else "Which {a} is discussed for variant {n}?"),
            "first_answer": "Краткий ответ." if ru else "A short answer.",
            "question": ask(sid, ru, "А что с {b}?" if ru else "And what about its {b}?"),
            "sentence_ids": [tags[1][0]],
        })
    if "does NOT answer" in system or "these topics do not cover" in system:
        return json.dumps({"question": ask(sid, ru, "Какой кластер дал {a} для узла {n}?" if ru
                                           else "Which GPU cluster gave {a} for node {n}?")})
    if "compares the two papers" in system:
        first, second = user.split("Passage 2.")
        ids = [TAG_RE.findall(first)[0][0], TAG_RE.findall(second)[0][0]]
        return json.dumps({"question": ask(sid, ru, "Чем подходы различаются по {a} в паре {n}?" if ru
                                           else "How do the approaches differ in {a} for pair {n}?"),
                           "sentence_ids": [int(i) for i in ids]})
    entity = re.search(r'mentions the \w+ "([^"]+)"', system)
    if entity:
        sid = next(i for i, text in tags if entity.group(1) in text)
        return json.dumps({"question": ask(sid, ru, "Какой набор данных применяли для {a} в опыте {n}?" if ru
                                           else "Which benchmark served {a} checks in study {n}?"),
                           "sentence_ids": [sid]})
    return json.dumps({"question": ask(sid, ru, "Какую {a} и {b} показал вариант {n}?" if ru
                                       else "Which {a} and {b} did variant {n} show?"),
                       "sentence_ids": [sid]})


class Teacher:
    """Как ScriptedClient в test_agent.py, но отвечает функцией по системному промпту; fail — исключения подряд."""

    def __init__(self, reply=teacher_reply, fail=()):
        self.reply, self.fail, self.calls = reply, list(fail), []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        system, user = kwargs["messages"][0]["content"], kwargs["messages"][1]["content"]
        self.calls.append(system)
        if self.fail:
            raise self.fail.pop(0)
        message = SimpleNamespace(content=self.reply(system, user), tool_calls=None)
        usage = SimpleNamespace(prompt_tokens=50, completion_tokens=5)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=usage)


def teacher(**kwargs):
    client = Teacher(**kwargs)
    return ChatModel("fake-teacher", client=client), client


def run(lib, tmp_path, kinds, model=None, **kwargs):
    model = model or teacher()[0]
    out = tmp_path / "questions" / "train.jsonl"
    stats = synthesize(lib, Searcher(lib), model, kwargs.pop("papers", [1, 2, 4, 5, 6]), kinds, 0, out,
                       log=lambda m: None, sleep=lambda s: None, **kwargs)
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()] if out.exists() else []
    return stats, records, out


# --- сплит ---

def test_split_is_deterministic_and_disjoint(lib, gold, tmp_path):
    splits = split_papers(lib, gold, val_papers=2, seed=0)
    assert splits == split_papers(lib, gold, val_papers=2, seed=0)
    assert splits["test"] == [3]  # название в золоте нормализуется
    train, val, test = (set(splits[name]) for name in ("train", "val", "test"))
    assert not (train & val or train & test or val & test)
    assert train | val | test == {1, 2, 3, 4, 5, 6}
    languages = {lib.get_paper(pid)["language"] for pid in val}
    assert len(val) == 2 and languages == {"en", "ru"}  # стратификация по теме и языку
    path = tmp_path / "splits.json"
    save_splits(splits, path)
    assert load_splits(path) == splits
    assert split_of_paper(splits, 3) == "test" and split_of_paper(splits, 99) is None


def test_split_reports_missing_gold_papers(lib):
    missing = [GoldQuestion("g", "q", "en", [GoldItem("No such paper", "x")])]
    assert split_papers(lib, missing)["gold_missing"] == ["No such paper"]


def test_near_duplicate():
    assert near_duplicate("How fast is speculative decoding?", "how FAST is the speculative decoding", 0.6)
    assert not near_duplicate("How fast is speculative decoding?", "Какие данные использовали?", 0.6)
    assert jaccard("", "x") == 0.0


# --- выборка фрагментов ---

def test_sample_passages_filters_and_limits(lib):
    sampled = sample_passages(lib, [1, 6], per_paper_max=30, seed=0)
    paper1 = [p for p in sampled if p["paper_id"] == 1]
    kinds = {p["section_kind"] for p in sampled}
    assert "acknowledgements" not in kinds
    assert all(len(p["text"]) >= 200 for p in sampled)
    assert not any("Lookahead system" in p["text"] and "trial 60" in p["text"] for p in sampled)  # подпись
    assert [p["source"] for p in sampled if p["paper_id"] == 6] == ["metadata"]  # аннотация из метаданных
    orders = [p["order_idx"] for p in paper1]
    assert all(b - a > 1 for a, b in zip(orders, orders[1:]))  # без соседних фрагментов
    eligible = 11  # аннотация, 9 результатов, обсуждение
    assert len(paper1) <= paper_quota(eligible) == 10
    assert all(p["weight"] == SECTION_WEIGHTS[p["section_kind"]] for p in sampled)
    assert sample_passages(lib, [1, 6], seed=0) == sampled
    assert len(sample_passages(lib, [1], per_paper_max=2)) == 2
    assert paper_quota(950) == 30 and paper_quota(4) == 6


def test_sampling_prefers_heavier_sections():
    import random

    passages = [{"id": 1, "order_idx": 0, "weight": 1.0}, {"id": 2, "order_idx": 5, "weight": 0.3}]
    picks = [q._sample_paper(passages, 1, random.Random(seed))[0]["id"] for seed in range(300)]
    assert picks.count(1) / len(picks) > 0.65  # ожидание 1 / (1 + 0.3) ≈ 0.77


# --- план без модели (--dry-run) ---

def test_plan_without_model(lib):
    kinds = {"fact_en": 3, "fact_xl": 2, "fact_ru": 2, "entity": 1, "paper_lookup": 2, "unanswerable": 2,
             "followup": 1, "external": 2, "compare": 1}
    plan = plan_sampling(lib, [1, 2, 4, 5, 6], kinds, split="train", seed=0)
    assert [c.id for c in plan] == [c.id for c in plan_sampling(lib, [1, 2, 4, 5, 6], kinds, split="train", seed=0)]
    assert len({c.id for c in plan}) == len(plan)
    by_id = {c.id: c for c in plan}
    for c in plan:
        if c.kind in ("fact_en", "fact_xl"):
            assert c.source_language == "en"
        if c.kind == "fact_ru":
            assert c.source_language == "ru" and c.language == "ru"
    [compare] = [c for c in plan if c.kind == "compare"]
    assert compare.paper_ids == [1, 2] and compare.meta["entity"] == "MT-Bench"
    assert all(c.id.startswith("tr-") for c in by_id.values())
    summary = summarize_plan(plan, kinds)
    assert summary["by_kind"]["fact_en"] == {"planned": 3, "requested": 3}
    assert summary["total"] == len(plan) == sum(kinds.values())
    assert "Кандидатов в плане" in format_plan(summary)
    # --limit берёт начало плана, и в нём уже несколько типов
    assert len({c.kind for c in plan[:6]}) >= 3
    with pytest.raises(ValueError):
        plan_sampling(lib, [1], {"bogus": 1})


def test_plan_shortfall_when_passages_run_out(lib):
    plan = plan_sampling(lib, [5], {"fact_ru": 50, "fact_en": 5})
    assert summarize_plan(plan)["by_kind"] == {"fact_ru": {"planned": 2, "requested": None}}


# --- синтез ---

def test_synthesize_all_kinds_accepts_verbatim_gold(lib, gold, tmp_path):
    kinds = {"fact_en": 2, "fact_xl": 1, "fact_ru": 1, "entity": 1, "paper_lookup": 1, "unanswerable": 1,
             "followup": 1, "external": 1, "compare": 1}
    stats, records, out = run(lib, tmp_path, kinds, gold_questions=gold)
    rejected = {r["id"]: r["meta"]["reasons"] for r in records if not r["meta"]["accepted"]}
    assert not rejected
    assert stats["written"] == stats["accepted"] == len(records) == 10
    assert stats["usage"]["calls"] > 0
    for record in records:
        assert set(record) == {"id", "question", "language", "topic", "status", "relevant", "kind", "meta"}
        assert record["status"] == "draft" and detect_language(record["question"]) == record["language"]
        assert "gold_rank" in record["meta"]
    kinds_seen = {r["kind"] for r in records}
    assert kinds_seen == {"fact", "entity", "paper_lookup", "unanswerable", "followup", "external", "compare"}
    for record in records:
        if record["kind"] in ("unanswerable", "external"):
            assert record["relevant"] == []
            continue
        sentences = dict(sentence for pid in record["meta"].get("passage_ids", [record["meta"]["passage_id"]])
                         for sentence in sentences_of(lib, pid))
        quotes = [sentences[sid] for sid in record["meta"]["sentence_ids"]]
        assert [item["quote"] for item in record["relevant"]] == quotes  # дословно предложения по id
        titles = {row["title"] for row in lib.conn.execute("SELECT title FROM papers")}
        assert {item["paper"] for item in record["relevant"]} <= titles
    [compare] = [r for r in records if r["kind"] == "compare"]
    assert len(compare["relevant"]) == 2 and compare["meta"]["paper_ids"] == [1, 2]
    [followup] = [r for r in records if r["kind"] == "followup"]
    assert [m["role"] for m in followup["meta"]["history"]] == ["user", "assistant"]
    [xl] = [r for r in records if r["meta"]["subkind"] == "fact_xl"]
    assert xl["language"] == "ru" and xl["meta"]["source_language"] == "en"
    [unanswerable] = [r for r in records if r["kind"] == "unanswerable"]
    assert unanswerable["meta"]["judge_hits"] is not None
    # записи читаются как золотой набор
    loaded = {g.id: g for g in load_questions(out)}
    assert loaded[compare["id"]].kind == "compare" and len(resolve(lib, loaded[compare["id"]])) == 2


def test_resume_skips_written_candidates(lib, tmp_path):
    stats, records, out = run(lib, tmp_path, {"fact_en": 2})
    assert stats["written"] == 2
    broken, client = teacher(reply=lambda s, u: (_ for _ in ()).throw(AssertionError("teacher must not be called")))
    stats, again, _ = run(lib, tmp_path, {"fact_en": 2}, model=broken)
    assert stats["skipped"] == 2 and stats["written"] == 0 and not client.calls
    assert again == records
    stats, _, _ = run(lib, tmp_path, {"fact_en": 3}, limit=1)
    assert stats["written"] == 1 and stats["skipped"] == 2


def test_malformed_json_is_a_reason(lib, tmp_path):
    model, _ = teacher(reply=lambda s, u: "Sure! Here is a question: what is the speedup?")
    stats, records, _ = run(lib, tmp_path, {"fact_en": 1})
    stats, records, _ = run(lib, tmp_path, {"fact_en": 1}, model=model, resume=False)
    assert stats["rejected"] == 1 and records[0]["meta"]["reasons"][0].startswith("bad_json")
    model, _ = teacher(reply=lambda s, u: json.dumps({"question": "Which speedup?", "sentence_ids": "s1"}))
    _, records, _ = run(lib, tmp_path, {"fact_en": 1}, model=model, resume=False)
    assert records[0]["meta"]["reasons"] == ["bad_json: sentence_ids — не список id"]


class RateLimitError(Exception):
    pass


class BadRequestError(Exception):
    status_code = 400


def test_transient_errors_are_retried(lib, tmp_path):
    model, client = teacher(fail=[RateLimitError("429"), RateLimitError("429")])
    stats, records, _ = run(lib, tmp_path, {"fact_en": 1}, model=model)
    assert stats["accepted"] == 1 and client.calls.count(client.calls[0]) == 3


def test_exhausted_retries_leave_candidate_for_next_run(lib, tmp_path):
    model, _ = teacher(fail=[RateLimitError("429")] * 4)
    stats, records, _ = run(lib, tmp_path, {"fact_en": 1}, model=model, retries=3)
    assert stats["api_errors"] == 1 and records == []
    stats, records, _ = run(lib, tmp_path, {"fact_en": 1})
    assert stats["accepted"] == 1


def test_permanent_api_error_is_a_reason(lib, tmp_path):
    model, _ = teacher(fail=[BadRequestError("context too long")])
    stats, records, _ = run(lib, tmp_path, {"fact_en": 1}, model=model)
    assert records[0]["meta"]["reasons"][0].startswith("api_error: BadRequestError")


def test_test_split_skips_gold_neighbourhood(lib, gold):
    exclude = q.gold_neighbourhood(lib, gold)
    medusa = passages_of(lib, "Medusa heads for parallel decoding")
    assert exclude == set(medusa[0:3])  # золотой фрагмент (trial 1) и его соседи
    plan = plan_sampling(lib, [3], {"fact_en": 10}, split="test", exclude_passages=exclude)
    assert [c.passage_ids for c in plan] == [[medusa[3]]]


class AuthenticationError(Exception):
    status_code = 401


class NotFoundError(Exception):
    status_code = 404


@pytest.mark.parametrize("error", [AuthenticationError("bad key"), NotFoundError("no such model"),
                                   type("PaymentRequired", (Exception,), {"status_code": 402})("pay")])
def test_config_error_stops_synthesis_without_records(lib, tmp_path, error):
    model, client = teacher(fail=[error])
    with pytest.raises(TeacherConfigError):
        run(lib, tmp_path, {"fact_en": 3}, model=model)
    assert len(client.calls) == 1  # дальше по плану не пошли
    _, records, out = run(lib, tmp_path, {"fact_en": 3})
    assert not out.with_name("train.jsonl.tmp").exists()
    assert len(records) == 3 and all(r["meta"]["accepted"] for r in records)


def test_failed_records_are_retried_on_resume(lib, tmp_path):
    model, _ = teacher(fail=[BadRequestError("context too long")])
    _, records, _ = run(lib, tmp_path, {"fact_en": 2}, model=model)
    assert [r["meta"]["reasons"][0].split(":")[0] for r in records if r["meta"]["reasons"]] == ["api_error"]
    stats, again, out = run(lib, tmp_path, {"fact_en": 2})
    assert stats["retried"] == 1 and stats["written"] == 1 and stats["skipped"] == 1
    assert len(again) == 2 and len({r["id"] for r in again}) == 2 and all(r["meta"]["accepted"] for r in again)


def test_truncated_tail_is_repaired_on_resume(lib, tmp_path):
    _, records, out = run(lib, tmp_path, {"fact_en": 2})
    text = out.read_text(encoding="utf-8")
    out.write_text('[1]\n{"question": "без id"}\n' + text[:-20], encoding="utf-8")  # оборванная запись
    stats, again, _ = run(lib, tmp_path, {"fact_en": 2})
    assert stats["skipped"] == 1 and stats["written"] == 1
    assert [r["id"] for r in again] == [r["id"] for r in records]
    assert len(load_questions(out)) == 2  # файл снова читается как набор вопросов


def test_test_split_requires_gold(lib, gold, tmp_path):
    out = tmp_path / "questions" / "test.jsonl"
    with pytest.raises(ValueError):
        synthesize(lib, Searcher(lib), teacher()[0], [3], {"fact_en": 5}, 0, out, log=lambda m: None)
    synthesize(lib, Searcher(lib), teacher()[0], [3], {"fact_en": 5}, 0, out, log=lambda m: None, gold_questions=gold)
    medusa = passages_of(lib, "Medusa heads for parallel decoding")
    assert [r["meta"]["passage_id"] for r in load_records(out)] == [medusa[3]]  # без золота и соседей


def load_records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_train_drops_near_duplicates_of_test(lib, tmp_path):
    kinds = {"fact_en": 2, "external": 2}
    _, records, out = run(lib, tmp_path, kinds)
    fact = next(r for r in records if r["kind"] == "fact")
    external = next(r for r in records if r["kind"] == "external")
    test_records = [{**fact, "id": "te-1", "question": fact["question"].upper() + " now"},
                    {**external, "id": "te-2", "question": "x", "meta": {**external["meta"], "accepted": True,
                                                                         "subject": external["meta"]["subject"].upper()}}]
    (out.parent / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in test_records), encoding="utf-8")
    stats, again, _ = run(lib, tmp_path, kinds)
    assert stats["marked_near_test"] == 2
    marked = {r["id"]: r["meta"] for r in again if "near_duplicate_test" in r["meta"]["reasons"]}
    assert set(marked) == {fact["id"], external["id"]} and not any(m["accepted"] for m in marked.values())
    # новые внешние запросы не берут тему из test
    held = subject_key(external["meta"]["subject"])
    new = [r for r in again if r["kind"] == "external" and r["id"] not in marked]
    assert new and all(subject_key(r["meta"]["subject"]) != held for r in new)
    stats, _, _ = run(lib, tmp_path, kinds)
    assert stats["marked_near_test"] == 0 and stats["written"] == 0  # повторный запуск ничего не меняет


def test_external_subjects_are_deduplicated(lib):
    lib.save_annotation(2, {"method": ["Speculative Decoding", "SD"]}, [], status="ok", annotator="test", version=2)
    plan = plan_sampling(lib, [1, 2], {"external": 50})
    keys = [(subject_key(c.meta["subject"]), c.language) for c in plan]
    assert len(keys) == len(set(keys)) == 4  # speculative decoding и Lookahead, по разу на язык
    assert "SD" not in {c.meta["subject"] for c in plan}
    plan = plan_sampling(lib, [1, 2], {"external": 50}, exclude_subjects={subject_key("speculative decoding")})
    assert {c.meta["subject"] for c in plan} == {"Lookahead"}


# --- проверки кандидата ---

def candidate(lib, passage_id, question, sentence_ids, kind="fact_en", language="en", **extra):
    source = detect_language(lib.conn.execute("SELECT text FROM passages WHERE id = ?", (passage_id,)).fetchone()[0])
    paper_id = lib.conn.execute("SELECT paper_id FROM passages WHERE id = ?", (passage_id,)).fetchone()[0]
    return Candidate(id=f"c-{passage_id}", kind=kind, split="train", paper_ids=[paper_id], passage_ids=[passage_id],
                     language=language, source_language=source, question=question, sentence_ids=sentence_ids,
                     **extra)


@pytest.fixture
def results(lib):
    """Фрагмент результатов статьи 1 (trial 1) и id его предложений."""
    passage_id = passages_of(lib, "Speculative decoding with draft models")[1]
    return passage_id, [sid for sid, _ in sentences_of(lib, passage_id)]


GOOD_EN = "Which memory budget does the Lookahead pruning scheme keep?"


def test_good_candidate_passes(lib, results):
    passage_id, sids = results
    c = candidate(lib, passage_id, GOOD_EN, [sids[1]])
    relevant = build_relevant(lib, c)
    assert [item.quote for item in relevant] == [sentences_of(lib, passage_id)[1][1]]
    assert check_candidate(lib, c) == []
    assert check_candidate(lib, c, model=teacher()[0]) == []


def test_non_adjacent_sentences_become_separate_quotes(lib, results):
    passage_id, sids = results
    c = candidate(lib, passage_id, GOOD_EN, [sids[0], sids[2]])
    assert len(build_relevant(lib, c)) == 2 and check_candidate(lib, c) == []
    c = candidate(lib, passage_id, GOOD_EN, [sids[0], sids[1]])
    [item] = build_relevant(lib, c)
    assert item.quote.startswith("During trial 1") and item.quote.endswith("accepted block.")


@pytest.mark.parametrize("question, reason", [
    ("Memory overhead stayed below 11 percent because the Lookahead cache was pruned?", "copies_quote"),
    ("What does this paper say about the Lookahead memory budget?", "not_self_contained"),
    ("Сколько памяти занимает кэш Lookahead?", "wrong_language: ожидался en"),
    ("", "empty_question"),
])
def test_question_reasons(lib, results, question, reason):
    passage_id, sids = results
    assert reason in check_candidate(lib, candidate(lib, passage_id, question, [sids[1]]))


def test_gold_reasons(lib, results):
    passage_id, sids = results
    other_passage = passages_of(lib, "Speculative decoding with draft models")[3]
    foreign = sentences_of(lib, other_passage)[0][0]
    assert "foreign_sentence_ids" in check_candidate(lib, candidate(lib, passage_id, GOOD_EN, [sids[1], foreign]))
    assert check_candidate(lib, candidate(lib, passage_id, GOOD_EN, [])) == ["no_sentence_ids"]
    long_passage = passages_of(lib, "Speculative decoding with draft models")[-1]
    long_ids = [sid for sid, _ in sentences_of(lib, long_passage)]
    short = next(sid for sid, text in sentences_of(lib, long_passage) if text == "Results vary across hardware.")
    assert "quote_too_short" in check_candidate(lib, candidate(lib, long_passage, GOOD_EN, [short]))
    assert "quote_too_long" in check_candidate(lib, candidate(lib, long_passage, GOOD_EN, long_ids))
    c = candidate(lib, passage_id, GOOD_EN, [sids[1]])
    fake = [GoldItem("Speculative decoding with draft models", "a quote that is nowhere in the library at all")]
    assert "unresolved_quote" in check_candidate(lib, c, fake)


def test_copy_check_is_skipped_for_cross_lingual(lib, results):
    passage_id, sids = results
    c = candidate(lib, passage_id, "Сколько памяти занимает кэш Lookahead после обрезки?", [sids[1]],
                  kind="fact_xl", language="ru")
    assert check_candidate(lib, c) == []
    assert shares_ngram("one two three four five six seven", "zero two three four five six seven eight")
    assert not shares_ngram("one two three four five", "one two three four five")


def test_near_duplicates(lib, results, gold):
    passage_id, sids = results
    c = candidate(lib, passage_id, "What throughput gain does the Medusa system reach?", [sids[1]])
    assert "near_duplicate_gold" in check_candidate(lib, c, gold_questions=gold)
    c = candidate(lib, passage_id, GOOD_EN, [sids[1]])
    assert check_candidate(lib, c, others=[GOOD_EN.upper()]) == ["near_duplicate_candidate"]
    assert check_candidate(lib, c, held_out=HeldOut(texts=[GOOD_EN + " today"])) == ["near_duplicate_test"]
    # внешние запросы из одного шаблона похожи текстом, но дублями считаются только по теме
    external = Candidate(id="e", kind="external", split="train", paper_ids=[], passage_ids=[], language="en",
                         question="Find new papers about Medusa on the internet", meta={"subject": "Medusa"})
    others = ["Find new papers about Lookahead on the internet"]
    assert check_candidate(lib, external, others=others, held_out=HeldOut(texts=others)) == []
    assert check_candidate(lib, external, held_out=HeldOut(subjects={subject_key("medusa")})) == ["near_duplicate_test"]


def test_answerability_check_with_teacher(lib, results):
    passage_id, sids = results
    c = candidate(lib, passage_id, GOOD_EN, [sids[1]])
    nobody, _ = teacher(reply=lambda s, u: json.dumps({"sentence_ids": []}))
    assert check_candidate(lib, c, model=nobody) == ["not_answerable"]
    other, _ = teacher(reply=lambda s, u: json.dumps({"sentence_ids": [f"s{sids[0]}"]}))
    assert check_candidate(lib, c, model=other) == ["not_answerable"]
    garbage, _ = teacher(reply=lambda s, u: "no idea")
    assert check_candidate(lib, c, model=garbage) == ["answerability_bad_json"]
    # плохой вопрос отсеивается без вызова учителя
    calls, client = teacher()
    check_candidate(lib, candidate(lib, passage_id, "What does this paper do?", [sids[1]]), model=calls)
    assert client.calls == []


def test_unanswerable_judge(lib, results):
    passage_id, _ = results
    c = candidate(lib, passage_id, "Which GPU cluster trained Lookahead drafts?", [], kind="unanswerable")
    searcher = Searcher(lib)
    hits = [h.passage_id for h in searcher.search(c.question, k=10)]
    assert hits
    answers, client = teacher(reply=lambda s, u: json.dumps({"answering_ids": [f"p{hits[0]}"]}))
    assert check_candidate(lib, c, model=answers, searcher=searcher) == ["answered_by_library"]
    assert client.calls == [JUDGE_PROMPT]
    nobody, _ = teacher()
    assert check_candidate(lib, c, model=nobody, searcher=searcher) == []
    garbage, _ = teacher(reply=lambda s, u: "[]")
    assert check_candidate(lib, c, model=garbage, searcher=searcher) == ["judge_bad_json"]


def test_entity_and_compare_reasons(lib):
    passage_id = passages_of(lib, "Speculative decoding with draft models")[9]  # абзац с MT-Bench
    sids = [sid for sid, _ in sentences_of(lib, passage_id)]
    meta = {"entity": "MT-Bench", "entity_type": "dataset"}
    c = candidate(lib, passage_id, "Which benchmark judged Lookahead answer quality?", [sids[1]], kind="entity",
                  meta=meta)
    assert check_candidate(lib, c) == ["entity_not_in_quote"]
    c = candidate(lib, passage_id, "Was MT-Bench used to judge Lookahead answers?", [sids[0]], kind="entity",
                  meta=meta)
    assert check_candidate(lib, c) == ["entity_in_question"]
    tree = passages_of(lib, "Tree attention for verification")[-1]
    pair = Candidate(id="c", kind="compare", split="train", paper_ids=[1, 2], passage_ids=[passage_id, tree],
                     language="en", source_language="en", question="How do Lookahead and Arbor judges differ?",
                     sentence_ids=[sids[0]], meta=meta)
    assert "compare_needs_both_papers" in check_candidate(lib, pair)
    pair.sentence_ids = [sids[0], sentences_of(lib, tree)[0][0]]
    assert check_candidate(lib, pair) == []


def test_external_asks_to_add(lib):
    c = Candidate(id="e", kind="external", split="train", paper_ids=[], passage_ids=[], language="ru",
                  question="Найди и добавь в библиотеку статьи про speculative decoding")
    assert check_candidate(lib, c) == ["asks_to_add"]


def test_gold_rank(lib, results):
    passage_id, _ = results
    assert gold_rank(Searcher(lib), "trial 1 Lookahead throughput gain", [passage_id]) == 1
    assert gold_rank(Searcher(lib), "квантовая хромодинамика", [passage_id]) is None


def test_reference_list_passages_are_not_sampled():
    from research_assistant.finetune.questions import looks_like_references

    refs = ('1 Obermeyer, Z. (2016). "Predicting the future." New England Journal of Medicine, 375(13), 1216-1219. '
            'doi:10.1056/NEJMp1606181. 2 Reddy, S. (2019). Journal of the American Board, 32(3), pp. 4-7.')
    assert looks_like_references(refs)
    assert not looks_like_references("Speculative decoding (Leviathan et al., 2023) drafts tokens with a small "
                                     "model and verifies them with the target model in one pass.")


def test_empty_teacher_reply_is_retried():
    from types import SimpleNamespace

    from research_assistant.finetune.questions import ask_teacher

    replies = iter(["", '{"question": "q?", "sentence_ids": [1]}'])

    class Model:
        def complete(self, messages, tools=None):
            message = SimpleNamespace(content=next(replies))
            return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    pauses = []
    assert ask_teacher(Model(), "system", "user", sleep=pauses.append).startswith('{"question"')
    assert pauses == [2]
