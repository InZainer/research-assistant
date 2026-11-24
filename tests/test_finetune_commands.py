import json

import pytest

from research_assistant.cli import main
from research_assistant.config import load_settings
from research_assistant.finetune.commands import DEFAULT_COUNTS, _teacher, parse_counts
from research_assistant.finetune.questions import KINDS
from research_assistant.ingest.models import PaperRecord
from research_assistant.parse.passages import SectionOut, paragraph_passages
from research_assistant.parse.structure import Paragraph
from research_assistant.store.db import Library


def test_parse_counts_and_defaults_cover_known_kinds():
    assert parse_counts("fact_en=3, unanswerable=2", "train") == {"fact_en": 3, "unanswerable": 2}
    assert parse_counts(None, "val") == DEFAULT_COUNTS["val"]
    assert all(set(counts) <= set(KINDS) for counts in DEFAULT_COUNTS.values())
    with pytest.raises(ValueError):
        parse_counts("fact_en", "train")


def test_teacher_must_not_be_the_local_student(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)  # .env мог выставить его в другом тесте через main()
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "lm-studio")
    monkeypatch.setenv("MODEL", "qwen3.5-4b")
    with pytest.raises(ValueError, match="облачной"):
        _teacher(load_settings(env_file=None))


def test_ft_split_writes_splits_under_data_dir(tmp_path, monkeypatch, capsys):
    with Library(tmp_path / "library.sqlite") as library:
        for i, title in enumerate(["Gold paper about drafting", "Train paper about trees", "Another paper on caches"]):
            paper_id, _ = library.upsert_paper(PaperRecord(title=title, sources=["openalex"], language="en",
                                                           year=2024))
            library.link_topic(paper_id, library.get_or_create_topic("speculative decoding"))
            section = SectionOut("Abstract", "abstract", 1)
            section.passages = paragraph_passages(Paragraph(f"Sentence number {i} explains the method well.", [(0, 1)]))
            library.save_structure(paper_id, [section], status="ok", parser="pdf", version=1)
    gold = tmp_path / "gold.jsonl"
    gold.write_text(json.dumps({"id": "en01", "question": "q?", "language": "en", "relevant": [
        {"paper": "Gold paper about drafting", "quote": "Sentence number 0 explains the method well."}]}) + "\n",
        encoding="utf-8")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert main(["ft-split", "--gold", str(gold), "--val-papers", "1"]) == 0
    splits = json.loads((tmp_path / "finetune" / "splits.json").read_text(encoding="utf-8"))
    assert splits["test"] == [1] and len(splits["val"]) == 1 and len(splits["train"]) == 1
    assert main(["ft-export"]) in (0, 2)  # без траекторий — пустой экспорт или понятная ошибка, но не падение
    capsys.readouterr()


def test_ft_collect_without_split_gives_clear_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert main(["ft-collect", "--split", "train"]) == 2
    assert "ft-split" in capsys.readouterr().out
