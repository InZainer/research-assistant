import json
from types import SimpleNamespace

import pytest

from research_assistant.agent.coverage import check_coverage, format_coverage
from research_assistant.agent.reading import format_reading, paper_windows, read_paper
from research_assistant.ingest.models import PaperRecord
from research_assistant.llm import ChatModel
from research_assistant.parse.passages import SectionOut, paragraph_passages
from research_assistant.parse.structure import Paragraph
from research_assistant.retrieval.search import Searcher
from research_assistant.store.db import Library

ROOMS = {
    "abstract": "We generate furniture layouts for residential rooms. The method supports L-shaped floor plans.",
    "method": "A language model places clusters of furniture inside the room polygon. Collisions are resolved by rules.",
    "results": "The plausibility score is 2.74 on rectangular rooms. In-bounds rate stays at 1.000 for every plan.",
}
TABLE = "Setting | Plaus.\nRectangular | 2.73\nNon-rectangular | 2.48"
REVENUE = {"abstract": "We predict restaurant revenue across cities with gradient boosting on demographic data."}


def call(name, args):
    return SimpleNamespace(id=f"call_{name}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))


class ScriptedClient:
    """Отвечает по сценарию: элемент — список вызовов инструментов, текст или функция от сообщений."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        turn = self.script.pop(0)
        if callable(turn):
            turn = turn(kwargs["messages"])
        content, calls = (turn, None) if isinstance(turn, str) else (None, turn)
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=10)
        message = SimpleNamespace(content=content, tool_calls=calls)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=usage)


def model(script):
    client = ScriptedClient(script)
    return ChatModel("fake-model", client=client), client


def add(library, title, sections, table=None, status="ok", topic="layouts"):
    paper_id, _ = library.upsert_paper(PaperRecord(title=title, sources=["openalex"], year=2026, language="en"))
    library.link_topic(paper_id, library.get_or_create_topic(topic))
    outs = []
    for kind, text in sections.items():
        section = SectionOut(kind.title(), kind, 1)
        section.passages = paragraph_passages(Paragraph(text, [(0, 2)]))
        if table and kind == "results":
            section.passages += paragraph_passages(Paragraph(table, [(0, 3)], kind="table"))
        outs.append(section)
    library.save_structure(paper_id, outs, status=status, parser="pdf" if status == "ok" else "metadata", version=2,
                           source="fulltext" if status == "ok" else "metadata")
    return paper_id


@pytest.fixture
def library():
    with Library(":memory:") as lib:
        yield lib


def sid(library, startswith):
    return library.conn.execute("SELECT id FROM sentences WHERE text LIKE ?", (startswith + "%",)).fetchone()[0]


# --- чтение статьи целиком ---

def test_windows_keep_reading_order_and_mark_tables(library):
    paper_id = add(library, "PolyLayout", ROOMS, table=TABLE)
    whole = paper_windows(library, paper_id, window_chars=10_000)
    assert len(whole) == 1 and len(whole[0].sentence_ids) == 9
    text = whole[0].text
    assert text.index("## Abstract [abstract]") < text.index("## Method [method]") < text.index("## Results [results]")
    assert f"[table] [s{sid(library, 'Rectangular | 2.73')}] Rectangular | 2.73" in text
    parts = paper_windows(library, paper_id, window_chars=220)
    assert len(parts) > 1
    assert set().union(*(w.sentence_ids for w in parts)) == whole[0].sentence_ids  # ничего не потеряно
    assert all(w.text.startswith("## ") for w in parts)  # в каждом окне видно, какой это раздел


def test_one_window_report_with_citations_absences_and_inconsistency(library, tmp_path):
    paper_id = add(library, "PolyLayout", ROOMS, table=TABLE)
    shape, score, row = sid(library, "The method supports"), sid(library, "The plausibility"), sid(library, "Rectangular |")
    report = {
        "summary": "Метод расставляет мебель и в непрямоугольных комнатах.",
        "findings": [
            {"aspect": "method", "text": "Поддерживает L-образные планы.", "sentence_ids": [f"s{shape}"]},
            {"aspect": "result", "text": "Выдуманный результат.", "sentence_ids": ["s999999"]},
        ],
        "not_covered": ["Рестораны и кафе не рассматриваются."],
        "inconsistencies": [{"text": "В тексте 2.74, в таблице 2.73.", "sentence_ids": [f"s{score}", f"[s{row}]"]}],
    }
    chat, client = model([[call("paper_report", report)]])
    log = tmp_path / "read.jsonl"
    reading = read_paper(library, chat, paper_id, aspects=["рестораны и кафе"], window_chars=10_000, log_path=log)

    assert reading.stop_reason == "report" and reading.windows == 1 and reading.source == "fulltext"
    assert [f.text for f in reading.findings] == ["Поддерживает L-образные планы."]
    assert reading.findings[0].citations[0].text == "The method supports L-shaped floor plans."  # текст подставил код
    assert reading.dropped == 1  # утверждение с выдуманным ID в отчёт не попало
    assert [c.text for c in reading.inconsistencies[0].citations] == [
        "The plausibility score is 2.74 on rectangular rooms.", "Rectangular | 2.73"]
    assert reading.not_covered == ["Рестораны и кафе не рассматриваются."]
    user = client.requests[0]["messages"][1]["content"]
    assert "рестораны и кафе" in user and "Write the report in Russian." in user
    assert [t["function"]["name"] for t in client.requests[0]["tools"]] == ["paper_report"]
    events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [e["event"] for e in events] == ["start", "llm", "finish"] and events[1]["usage"]["prompt"] == 100
    text = format_reading(reading)
    assert "прочитано целиком: 9 предложений" in text and "Чего в статье нет" in text and "Расхождения" in text


def test_long_paper_is_read_in_parts_and_reported_from_notes(library):
    paper_id = add(library, "PolyLayout", ROOMS, table=TABLE)
    shape, score = sid(library, "The method supports"), sid(library, "The plausibility")
    windows = paper_windows(library, paper_id, window_chars=220)

    def notes(messages):
        user = messages[1]["content"]
        found = [{"aspect": "result", "text": "Plausibility 2.74.", "sentence_ids": [f"s{score}"]}] \
            if f"[s{score}]" in user else []
        if f"[s{shape}]" in user:
            found.append({"aspect": "method", "text": "L-shaped plans.", "sentence_ids": [f"s{shape}"]})
        return [call("take_notes", {"notes": found})]

    def report(messages):
        user = messages[1]["content"]
        assert "L-shaped plans." in user and "Plausibility 2.74." in user  # отчёт пишется по заметкам
        assert "We generate furniture layouts" not in user  # а не по тексту статьи
        return [call("paper_report", {
            "summary": "ok", "not_covered": [], "inconsistencies": [],
            "findings": [{"aspect": "result", "text": "2.74", "sentence_ids": [f"s{score}"]},
                         {"aspect": "data", "text": "Не из заметок.", "sentence_ids": [f"s{score + 1}"]}],
        })]

    chat, client = model([notes] * len(windows) + [report])
    reading = read_paper(library, chat, paper_id, question="What does it report?", window_chars=220)
    assert reading.windows == len(windows) and reading.usage.calls == len(windows) + 1
    assert [f.text for f in reading.findings] == ["2.74"] and reading.dropped == 1
    second = client.requests[1]["messages"][1]["content"]
    assert "Your notes on the earlier parts" in second or len(windows) == 1
    assert "Write the report in English." in second


def test_abstract_only_paper_cannot_prove_absence(library):
    paper_id = add(library, "Revenue", REVENUE, status="metadata")
    first = sid(library, "We predict restaurant revenue")
    chat, client = model([[call("paper_report", {
        "summary": "Только аннотация.", "inconsistencies": [], "not_covered": ["Нет расположения внутри города."],
        "findings": [{"aspect": "task", "text": "Прогноз выручки.", "sentence_ids": [f"s{first}"]}]})]])
    reading = read_paper(library, chat, paper_id)
    assert reading.source == "metadata" and reading.not_covered == []
    assert "ONLY" in client.requests[0]["messages"][1]["content"].upper()
    assert "только аннотация" in format_reading(reading)


def test_model_without_report_and_paper_without_text(library):
    paper_id = add(library, "PolyLayout", ROOMS)
    chat, client = model(["Статья про мебель.", "Всё ещё текст."])
    reading = read_paper(library, chat, paper_id, window_chars=10_000)
    assert reading.stop_reason == "no_report" and len(client.requests) == 2  # одна повторная просьба
    assert "не оформила отчёт" in format_reading(reading)

    empty, _ = library.upsert_paper(PaperRecord(title="No text at all here", sources=["openalex"]))
    chat, client = model([])
    assert read_paper(library, chat, empty).stop_reason == "empty" and client.requests == []
    with pytest.raises(ValueError):
        read_paper(library, chat, 999)


# --- проверка «чего нет» по всем статьям темы ---

def test_coverage_asks_every_paper_and_reports_the_limits(library, tmp_path):
    rooms = add(library, "PolyLayout", ROOMS)
    revenue = add(library, "Revenue prediction", REVENUE, status="metadata")
    other = add(library, "Decoding", {"abstract": "Speculative decoding speeds up language models."}, topic="llm")
    shape = sid(library, "The method supports")

    def judge(messages):
        user = messages[1]["content"]
        if "PolyLayout" in user:
            assert f"[s{shape}]" in user
            return [call("verdict", {"verdict": "yes", "note": "Поддерживает L-образные комнаты.",
                                     "sentence_ids": [f"s{shape}"]})]
        assert "ONLY THE ABSTRACT" in user
        return [call("verdict", {"verdict": "partial", "note": "Похоже.", "sentence_ids": ["s424242"]})]

    expand = [call("search_phrasings", {"phrasings": ["furniture layout L-shaped floor plans", "restaurant revenue"]})]
    chat, client = model([expand, judge, judge])
    log = tmp_path / "coverage.jsonl"
    report = check_coverage(library, Searcher(library), chat, "расстановка мебели в непрямоугольной комнате",
                            topic="layouts", log_path=log)

    assert report.scope == 2 and other not in {v.paper_id for v in report.verdicts}  # чужая тема не проверяется
    assert report.phrasings[0] == "расстановка мебели в непрямоугольной комнате" and len(report.phrasings) == 3
    by_id = {v.paper_id: v for v in report.verdicts}
    assert by_id[rooms].verdict == "yes"
    assert by_id[rooms].citations[0].text == "The method supports L-shaped floor plans."
    assert by_id[revenue].verdict == "unverified" and not by_id[revenue].fulltext  # «похоже» без цитаты — не факт
    text = format_coverage(report)
    assert "Проверено моделью: 2 из 2" in text and "только аннотация" in text
    assert "утверждать его новизну нельзя" in text and f"(id: {rooms}" in text
    events = [json.loads(line)["event"] for line in log.read_text(encoding="utf-8").splitlines()]
    assert events[0] == "start" and events[-1] == "finish" and events.count("verdict") == 2


def test_coverage_with_no_direct_match_says_how_strong_the_absence_is(library):
    add(library, "PolyLayout", ROOMS)
    add(library, "Revenue prediction", REVENUE)
    add(library, "Unrelated", {"abstract": "Protein folding with diffusion models on molecular data."})
    no = [call("verdict", {"verdict": "no", "note": "", "sentence_ids": []})]
    chat, client = model([no])
    report = check_coverage(library, Searcher(library), chat, "restaurant revenue prediction", topic="layouts",
                            limit=1, expand=False)
    assert report.phrasings == ["restaurant revenue prediction"] and len(client.requests) == 1
    assert [v.title for v in report.verdicts] == ["Revenue prediction"]  # самая близкая к формулировке
    assert len(report.unchecked) == 2 and report.no_hits == 2
    text = format_coverage(report)
    assert "Прямого рассмотрения аспекта не найдено" in text and "2 статей не проверялись" in text
    assert "--limit 3" in text


def test_coverage_survives_model_failures(library):
    add(library, "PolyLayout", ROOMS)

    def boom(messages):
        raise RuntimeError("503")

    chat, _ = model([boom, boom])
    report = check_coverage(library, Searcher(library), chat, "furniture layouts", topic="layouts")
    assert report.phrasings == ["furniture layouts"]  # без переформулировок, но проверка идёт
    assert report.verdicts[0].verdict == "error" and "503" in report.verdicts[0].note
    assert check_coverage(library, Searcher(library), chat, "x", topic="нет такой темы").scope == 0
