import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_sft.py"
NOTEBOOK = ROOT / "notebooks" / "finetune_qwen35_colab.ipynb"
DOC = ROOT / "docs" / "finetune.md"


def load_script():
    """Импорт скрипта по пути: torch и unsloth при этом не нужны."""
    spec = importlib.util.spec_from_file_location("train_sft", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_sft"] = module  # dataclass ищет модуль в sys.modules
    spec.loader.exec_module(module)
    return module


ts = load_script()
IGNORE = ts.IGNORE_INDEX


class CharTokenizer:
    """Подменный токенайзер: один символ — один токен, offset_mapping как у fast-токенайзера."""

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=True):
        assert return_offsets_mapping and not add_special_tokens
        return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def test_import_without_torch(monkeypatch):
    # None в sys.modules делает импорт невозможным: импорт сверху модуля упал бы с ImportError
    for name in ("torch", "unsloth", "transformers", "peft", "trl", "gguf"):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "train_sft", sys.modules["train_sft"])
    module = load_script()
    assert module.DEFAULT_TEMPLATE.exists()


def test_labels_inside_outside_and_edges():
    # отрезок [4, 11): токен 0 вне, токен 1 заходит слева, 2 внутри, 3 выходит справа, 4 вне
    offsets = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)]
    ids = [10, 11, 12, 13, 14]
    assert ts.labels_from_spans(offsets, [[4, 11]], ids) == [IGNORE, IGNORE, 12, 13, IGNORE]
    assert ts.edge_tokens(offsets, [[4, 11]]) == (1, 1)
    # точные границы: ничего не режется
    assert ts.labels_from_spans(offsets, [[3, 9]], ids) == [IGNORE, 11, 12, IGNORE, IGNORE]
    assert ts.edge_tokens(offsets, [[3, 9]]) == (0, 0)
    # несколько отрезков и токен нулевой ширины
    offsets = [(0, 2), (2, 2), (2, 4), (4, 6), (6, 8)]
    assert ts.labels_from_spans(offsets, [[6, 8], [2, 4]], [1, 2, 3, 4, 5]) == [IGNORE, 2, 3, IGNORE, 5]
    with pytest.raises(ValueError):
        ts.labels_from_spans([(0, 1)], [[0, 1]], [1, 2])


def test_trainable_segments():
    assert ts.trainable_segments([IGNORE, 5, 6, IGNORE, 7]) == [(1, 3), (4, 5)]
    assert ts.trainable_segments([IGNORE, IGNORE]) == []


def test_encode_row_decodes_spans_exactly():
    text = "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\nANSWER<|im_end|>\n"
    start = text.index("ANSWER")
    end = text.index("<|im_end|>", start) + len("<|im_end|>")
    row = {"id": "x", "text": text, "train_spans": [[start, end]], "meta": {"est_tokens": 20}}
    example = ts.encode_row(CharTokenizer(), row)
    tok = CharTokenizer()
    segments = ts.trainable_segments(example["labels"])
    assert [tok.decode(example["input_ids"][i:j]) for i, j in segments] == ["ANSWER<|im_end|>"]
    assert example["attention_mask"] == [1] * len(text)
    # процессор VL-модели разворачивается до токенайзера
    wrapper = type("Processor", (), {"tokenizer": tok})()
    assert ts.text_tokenizer(wrapper) is tok and ts.text_tokenizer(tok) is tok


def test_encode_rows_skips_broken(capsys):
    rows = [{"id": "ok", "text": "abcdef", "train_spans": [[2, 4]]},
            {"id": "out", "text": "abc", "train_spans": [[1, 9]]},
            {"id": "empty", "text": "abc", "train_spans": []}]
    examples = ts.encode_rows(CharTokenizer(), rows)
    assert [e["id"] for e in examples] == ["ok"]
    assert "битых строк: 2" in capsys.readouterr().out


def test_check_row():
    assert ts.check_row({"text": "abcdef", "train_spans": [[0, 2], [3, 6]]}) == []
    assert ts.check_row({"text": "", "train_spans": [[0, 1]]}) == ["нет text"]
    assert "пересекается" in ts.check_row({"text": "abcdef", "train_spans": [[0, 4], [3, 6]]})[0]
    # битые отрезки — сообщение, а не исключение
    for spans in ([[0]], [[0, 1, 2]], [["0", 1]], [[0.5, 2]], [5], [[True, 2]], "0-1"):
        assert len(ts.check_row({"text": "abcdef", "train_spans": spans})) == 1, spans


def example(labels):
    return {"input_ids": list(range(len(labels))), "labels": labels, "attention_mask": [1] * len(labels)}


def test_drop_or_truncate():
    short = example([IGNORE, 1, 2])
    assert ts.drop_or_truncate(short, 3) is short
    # обучаемый ход за пределом — пример отбрасывается, а не режется
    assert ts.drop_or_truncate(example([IGNORE, 1, 2, 3]), 3) is None
    # за пределом только токены без потерь — хвост отрезается
    cut = ts.drop_or_truncate(example([IGNORE, 1, 2, IGNORE, IGNORE]), 3)
    assert cut["input_ids"] == [0, 1, 2] and cut["labels"] == [IGNORE, 1, 2] and cut["attention_mask"] == [1, 1, 1]
    # после обрезки не осталось обучаемых токенов — отбросить
    assert ts.drop_or_truncate(example([IGNORE, IGNORE, IGNORE, IGNORE]), 3) is None


def test_filter_by_length_counts():
    examples = [example([IGNORE, 1]), example([IGNORE, 1, 2, 3]), example([1, 2, 3, IGNORE])]
    kept, dropped, truncated = ts.filter_by_length(examples, 3)
    assert len(kept) == 2 and dropped == 1 and truncated == 1


def test_length_stats():
    stats = ts.length_stats([100, 200, 300, 5000], thresholds=(150, 4096))
    assert stats["count"] == 4 and stats["min"] == 100 and stats["max"] == 5000
    assert stats["p50"] == 200
    assert stats["over"] == {"150": 3, "4096": 1}
    assert ts.length_stats([])["count"] == 0


def test_collator_pads_lists():
    collate = ts.PadCollator(pad_token_id=0)
    batch = collate([{"input_ids": [5, 6, 7], "labels": [IGNORE, 6, 7]},
                     {"input_ids": [8], "labels": [8], "attention_mask": [1]}])
    assert batch["input_ids"] == [[5, 6, 7], [8, 0, 0]]
    assert batch["labels"] == [[IGNORE, 6, 7], [8, IGNORE, IGNORE]]
    assert batch["attention_mask"] == [[1, 1, 1], [1, 0, 0]]


def test_collator_multiple_and_tensor_hook():
    collate = ts.PadCollator(pad_token_id=9, pad_to_multiple_of=4, to_tensor=tuple)
    batch = collate([{"input_ids": [1, 2, 3, 4, 5], "labels": [1, 2, 3, 4, 5]}])
    assert batch["input_ids"] == ([1, 2, 3, 4, 5, 9, 9, 9],)
    assert batch["attention_mask"] == ([1, 1, 1, 1, 1, 0, 0, 0],)


def test_lora_targets_and_coverage():
    import re

    regex = re.compile(ts.lora_target_regex())
    assert regex.fullmatch("model.language_model.layers.0.linear_attn.in_proj_qkv")
    assert regex.fullmatch("model.language_model.layers.3.self_attn.q_proj")
    assert not regex.fullmatch("model.visual.blocks.0.attn.out_proj")
    assert not regex.fullmatch("model.language_model.layers.0.linear_attn.conv1d")
    names = [f"m.layers.0.linear_attn.{n}" for n in ts.DELTANET_MODULES]
    coverage = ts.lora_coverage(names + ["m.visual.blocks.0.mlp.up_proj"])
    assert coverage["counts"]["in_proj_z"] == 1
    assert coverage["visual"] == ["m.visual.blocks.0.mlp.up_proj"]
    assert "q_proj" in coverage["missing"] and "in_proj_qkv" not in coverage["missing"]


def test_pick_precision():
    assert ts.pick_precision("auto", True, (12, 0)) == "bf16"   # RTX 5070
    assert ts.pick_precision("auto", False, (7, 5)) == "fp32"   # T4
    assert ts.pick_precision("auto", False, None) == "fp32"
    assert ts.pick_precision("fp32", True, (12, 0)) == "fp32"
    with pytest.raises(SystemExit):
        ts.pick_precision("bf16", False, (7, 5))


def test_native_bf16_ignores_emulation():
    # на T4 torch.cuda.is_bf16_supported() отвечает True из-за эмуляции; решаем по capability
    assert ts.native_bf16((12, 0)) and ts.native_bf16((8, 0))
    assert not ts.native_bf16((7, 5)) and not ts.native_bf16(None)
    with pytest.raises(SystemExit):
        ts.pick_precision("bf16", ts.native_bf16((7, 5)), (7, 5))  # явный --bf16 на T4


def test_steps():
    assert ts.total_steps(160, 1, 16, 2, -1) == 20
    assert ts.total_steps(170, 1, 16, 2, -1) == 22
    assert ts.total_steps(1000, 1, 16, 2, 30) == 30
    assert ts.warmup_steps(20, 0.04) == 1 and ts.warmup_steps(500, 0.04) == 20


def test_install_chat_template(tmp_path):
    template = tmp_path / "t.jinja"
    template.write_text("{{ messages }}\n", encoding="utf-8")
    model_dir = tmp_path / "merged"
    model_dir.mkdir()
    (model_dir / "tokenizer_config.json").write_text(json.dumps({"chat_template": "old", "eos_token": "x"}))
    (model_dir / "chat_template.json").write_text("{}")
    ts.install_chat_template(model_dir, template)
    assert (model_dir / "chat_template.jinja").read_text(encoding="utf-8") == "{{ messages }}\n"
    config = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert config == {"chat_template": "{{ messages }}\n", "eos_token": "x"}
    assert not (model_dir / "chat_template.json").exists()
    assert ts.same_template("{{ messages }}", "{{ messages }}\n") and not ts.same_template(None, "x")


def test_read_rows_and_hashes(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = [{"text": "a", "train_spans": [[0, 1]], "meta": {"prompt_hash": h}} for h in ("h1", "h1", "h2")]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    assert len(ts.read_rows(path)) == 3 and len(ts.read_rows(path, limit=2)) == 2
    assert ts.prompt_hashes(ts.read_rows(path)) == ["h1", "h2"]


def test_cli_help_and_errors(capsys):
    with pytest.raises(SystemExit) as exit_info:
        ts.main(["--help"])
    assert exit_info.value.code == 0
    assert "--max-len" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exit_info:
        ts.main([])  # нет --train
    assert exit_info.value.code == 2
    with pytest.raises(SystemExit) as exit_info:
        ts.main(["--merge-from", "ckpt"])  # нет --merge-out
    assert exit_info.value.code == 2
    with pytest.raises(SystemExit) as exit_info:
        ts.main(["--train", "t.jsonl", "--val", "v.jsonl", "--eval-steps", "50", "--save-steps", "70"])
    assert exit_info.value.code == 2
    for flags in (["--eval-steps", "0"], ["--save-steps", "0"], ["--eval-steps", "-5"]):
        with pytest.raises(SystemExit) as exit_info:
            ts.main(["--train", "t.jsonl", "--val", "v.jsonl", *flags])
        assert exit_info.value.code == 2, flags


def test_drop_warning():
    assert ts.drop_warning(0, 10, 4096) is None and ts.drop_warning(5, 10, 4096) is None
    assert "4096" in ts.drop_warning(6, 10, 4096)
    assert ts.drop_warning(0, 0, 4096) is None


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return str(path)


def dialog(answer):
    text = f"<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>\n"
    start = text.index(answer)
    return {"text": text, "train_spans": [[start, start + len(answer) + len("<|im_end|>")]]}


def fake_transformers(monkeypatch, loaded):
    """transformers с AutoTokenizer, который отдаёт посимвольный токенайзер."""
    module = types.ModuleType("transformers")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(name):
            loaded.append(name)
            return CharTokenizer()

    module.AutoTokenizer = AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", module)


def test_dry_run_with_fake_tokenizer(tmp_path, monkeypatch, capsys):
    loaded = []
    fake_transformers(monkeypatch, loaded)
    rows = [{"id": "short", **dialog("OK"), "meta": {"est_tokens": 20, "prompt_hash": "h"}},
            {"id": "long", **dialog("A" * 200), "meta": {"prompt_hash": "h"}},
            {"id": "bad", "text": "abc", "train_spans": [[0]]}]
    path = write_jsonl(tmp_path / "train.jsonl", rows)
    assert ts.main(["--train", path, "--val", path, "--dry-run", "--max-len", "100", "--tokenizer", "tok"]) == 0
    out = capsys.readouterr().out
    assert loaded == ["tok"]
    assert "Строк: train 3, val 3; prompt_hash: ?, h" in out
    assert "пропуск bad: битый отрезок" in out
    assert "train: 2 примеров" in out and ">100: 1" in out
    assert "при --max-len 100: останется 1, отброшено 1, обрезан хвост 0" in out
    assert "Обучаемые участки примера short:" in out and "'OK<|im_end|>'" in out
    assert "токенов на est_tokens" in out and "ВНИМАНИЕ: при --max-len" not in out
    # всё длиннее предела — предупреждение о доле отброшенных
    ts.main(["--train", path, "--dry-run", "--max-len", "20"])
    assert "ВНИМАНИЕ: при --max-len 20 отброшено 2 из 2" in capsys.readouterr().out


def fake_gguf(monkeypatch, template):
    """gguf.GGUFReader: строковое поле хранит байты в parts[data[0]]."""
    module = types.ModuleType("gguf")

    class Field:
        def __init__(self, value):
            self.parts = [[len(value)], list(value.encode("utf-8"))]
            self.data = [1]

    class GGUFReader:
        def __init__(self, path):
            self.fields = {} if template is None else {"tokenizer.chat_template": Field(template)}

    module.GGUFReader = GGUFReader
    monkeypatch.setitem(sys.modules, "gguf", module)


def test_check_gguf(tmp_path, monkeypatch, capsys):
    template = tmp_path / "t.jinja"
    template.write_text("{{ шаблон }}\n", encoding="utf-8")
    fake_gguf(monkeypatch, "{{ шаблон }}")
    assert ts.main(["--check-gguf", "m.gguf", "--chat-template", str(template)]) == 0
    assert "совпадает" in capsys.readouterr().out
    fake_gguf(monkeypatch, "{{ другой }}")
    assert ts.check_gguf("m.gguf", template) == 1
    assert "ОТЛИЧАЕТСЯ" in capsys.readouterr().out
    fake_gguf(monkeypatch, None)
    assert ts.check_gguf("m.gguf", template) == 1
    assert "нет tokenizer.chat_template" in capsys.readouterr().out


def test_notebook_is_valid():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    for cell in notebook["cells"]:
        assert cell["cell_type"] in ("markdown", "code")
        assert isinstance(cell["source"], list) and all(isinstance(line, str) for line in cell["source"])
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and "execution_count" in cell
    code = "".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "train_sft.py" in code and "--max-steps 30" in code
    assert "API_KEY" not in code  # ключи в ноутбуках не пишутся


def minimal_prompt_tokens():
    """Оценка токенов кратчайшего примера: системный промпт, схемы и один вопрос без ответа."""
    from research_assistant.agent.prompts import SYSTEM_PROMPT
    from research_assistant.agent.tools import SCHEMAS
    from research_assistant.finetune.export import est_tokens, render

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "Что такое X?"}]
    return est_tokens(render(messages, SCHEMAS, add_generation_prompt=True))


def test_smoke_max_len_fits_minimal_example():
    # иначе фильтр длины отбрасывает все примеры и пробный запуск падает до первого шага;
    # запас 1.2 — est_tokens (len / 3.4) может занижать реальное число токенов
    floor = 1.2 * minimal_prompt_tokens()
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    smoke = ["".join(c["source"]) for c in notebook["cells"]
             if c["cell_type"] == "code" and "--max-steps 30" in "".join(c["source"])]
    assert len(smoke) == 1
    lengths = [int(n) for n in re.findall(r"--max-len (\d+)", smoke[0])]
    assert lengths and all(n >= floor for n in lengths), (lengths, floor)
    documented = [int(n) for n in re.findall(r"30 шагов с `--max-len (\d+)", DOC.read_text(encoding="utf-8"))]
    assert documented and all(n >= floor for n in documented), (documented, floor)
    assert ts.build_parser().get_default("max_len") >= floor
