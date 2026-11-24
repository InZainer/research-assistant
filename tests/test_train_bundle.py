import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "scripts" / "train_bundle"
NAME = "rtx5070_train_bundle"
REQUIRED = {
    "README_RTX5070.md", "MANIFEST.json", "data/train.jsonl", "data/val.jsonl", "data/stats.json",
    "train/train_sft.py", "train/qwen3_5_chat_template.jinja", "train/requirements-train.txt",
    "train/setup_wsl.sh", "train/run_docker.sh", "train/smoke.sh", "train/train_full.sh",
    "train/convert_gguf.sh", "train/verify_bundle.py", "train/bundle_tools.py", "train/common.sh",
    "notebook/finetune_qwen35_colab.ipynb",
}
FAKE_KEY = "sk-" + "Zq7" * 10  # собирается на лету: в тестах нет настоящих ключей


def load(path, name, monkeypatch=None, register=False):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if monkeypatch is not None:
        monkeypatch.setitem(sys.modules, name, module)  # dataclass ищет модуль в sys.modules
    elif register:
        sys.modules[name] = module  # уникальное имя: не пересекается с train_sft из test_train_script.py
    spec.loader.exec_module(module)
    return module


mb = load(ROOT / "scripts" / "make_train_bundle.py", "make_train_bundle")
vb = load(TEMPLATES / "verify_bundle.py", "verify_bundle")
bt = load(TEMPLATES / "bundle_tools.py", "bundle_tools")
ts = load(ROOT / "scripts" / "train_sft.py", "train_sft_for_bundle", register=True)


def sft_row(i, prompt_hash="h1"):
    text = f"<|im_start|>system\nS<|im_end|>\n<|im_start|>assistant\nответ {i}<|im_end|>\n"
    return {"id": f"ex-{i}", "split": "train", "kind": "fact", "text": text, "train_spans": [[40, len(text) - 1]],
            "meta": {"prompt_hash": prompt_hash, "est_tokens": 20}}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def repo(tmp_path):
    """Похожий на репозиторий каталог: исходники комплекта, данные и файлы, которым в комплекте не место."""
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "train_sft.py", root / "scripts" / "train_sft.py")
    template = root / "src" / "research_assistant" / "finetune" / "qwen3_5_chat_template.jinja"
    template.parent.mkdir(parents=True)
    template.write_bytes(b"{{ messages }}\r\n{%- if tools %}tools{% endif %}\r\n")
    (root / "notebooks").mkdir()
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": [
        {"cell_type": "code", "metadata": {}, "execution_count": 3, "source": ["print(1)\n"],
         "outputs": [{"output_type": "stream", "name": "stdout", "text": ["/content/drive/секрет\n"]}]}]}
    (root / "notebooks" / "finetune_qwen35_colab.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
    sft = root / "data" / "finetune" / "sft"
    sft.mkdir(parents=True)
    write_jsonl(sft / "train.jsonl", [sft_row(1), sft_row(2)])
    (sft / "val.jsonl").write_text("", encoding="utf-8")
    stats = {"max_tokens": 10240, "splits": {
        "train": {"examples": 2, "tokens": {"mean": 20, "p50": 20, "p95": 20, "max": 20}, "by_kind": {"fact": 2}},
        "val": {"examples": 0, "tokens": {"mean": 0, "p50": 0, "p95": 0, "max": 0}}}}
    (sft / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
    # то, что не должно попасть в комплект
    (root / ".env").write_text(f"OPENAI_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    (root / "data" / "library.sqlite").write_bytes(b"SQLite format 3\x00")
    (root / "data" / "fulltext").mkdir()
    (root / "data" / "fulltext" / "1.txt").write_text("полный текст", encoding="utf-8")
    (root / "runs").mkdir()
    (root / "runs" / "chat-1.jsonl").write_text('{"role": "user"}\n', encoding="utf-8")
    return root


def build(repo, tmp_path, **kwargs):
    return mb.build(repo, tmp_path / "dist", version=kwargs.pop("version", "test"), templates=TEMPLATES, **kwargs)


def zip_files(zip_path):
    with zipfile.ZipFile(zip_path) as archive:
        return {info.filename[len(NAME) + 1:]: archive.read(info) for info in archive.infolist()}, archive.infolist()


def test_bundle_contents_and_manifest(repo, tmp_path):
    result = build(repo, tmp_path)
    files, infos = zip_files(result["zip"])
    assert REQUIRED <= set(files)
    assert all(info.filename.startswith(NAME + "/") for info in infos)

    manifest = json.loads(files["MANIFEST.json"])
    listed = {entry["path"]: entry for entry in manifest["files"]}
    assert set(listed) == set(files) - {"MANIFEST.json"}
    for path, entry in listed.items():
        assert entry["size"] == len(files[path])
        assert entry["sha256"] == mb.sha256_bytes(files[path])
    assert manifest["version"] == "test"
    assert manifest["mode"] == "smoke" and "smoke" in manifest["mode_note"]
    assert manifest["data"]["train"]["examples"] == 2 and manifest["data"]["val"]["examples"] == 0
    assert manifest["data"]["prompt_hashes"] == ["h1"]
    assert manifest["data"]["train"]["tokens"]["p50"] == 20
    assert manifest["stack"]["unsloth"] == "2026.11.3" and manifest["stack"]["torch"] == "2.8.0"
    assert manifest["stack"]["transformers"] == "5.2.0" and manifest["stack"]["unsloth-zoo"] == "2026.11.3"
    assert manifest["source"]["git_commit"] is None  # во временной папке нет git — сборка не падает
    assert any("val пуст" in w for w in manifest["warnings"])

    # скрипты — LF и исполняемые; шаблон без CR; ноутбук без выводов
    for path, data in files.items():
        if path.endswith(".sh"):
            assert b"\r" not in data, path
            assert data.startswith(b"#") and b"common.sh" in data or path == "train/common.sh"
    modes = {info.filename[len(NAME) + 1:]: (info.external_attr >> 16) & 0o777 for info in infos}
    assert modes["train/smoke.sh"] == 0o755 and modes["data/train.jsonl"] == 0o644
    assert b"\r" not in files["train/qwen3_5_chat_template.jinja"]
    cell = json.loads(files["notebook/finetune_qwen35_colab.ipynb"])["cells"][0]
    assert cell["outputs"] == [] and cell["execution_count"] is None

    readme = files["README_RTX5070.md"].decode("utf-8")
    assert "{{" not in readme and "qwen3.5-4b-ra-smoke" in readme and "val: 0" in readme
    # папка совпадает с архивом
    assert (result["stage"] / "train" / "smoke.sh").read_bytes() == files["train/smoke.sh"]


def test_no_forbidden_files_or_secrets(repo, tmp_path):
    result = build(repo, tmp_path)
    files, _ = zip_files(result["zip"])
    assert mb.forbidden_paths(files) == []
    for path, data in files.items():
        text = data.decode("utf-8", errors="replace")
        assert FAKE_KEY not in text and "SQLite format" not in text and "полный текст" not in text, path
        assert mb.find_secrets(text) == [], path
    assert not any(name.endswith((".env", ".sqlite")) or "runs/" in name or "fulltext" in name for name in files)


def test_find_secrets_and_forbidden_paths():
    assert mb.find_secrets(f"key: {FAKE_KEY}")
    assert mb.find_secrets("OPENAI_API_KEY=abcdef123456")
    assert mb.find_secrets("MERCURY_API_KEY = 'abcdef123456'")
    assert mb.find_secrets('$env:OPENAI_API_KEY = "lm-studio"') == []
    assert mb.find_secrets("OPENAI_API_KEY=$KEY и OPENAI_API_KEY=<ключ>") == []
    assert mb.find_secrets("task-oriented risk-free desk-top") == []
    assert mb.forbidden_paths([".env", "data/library.sqlite", "data/fulltext/1.txt", "runs/a.jsonl",
                               "data/train.jsonl", "train/smoke.sh"]) == [
        ".env", "data/library.sqlite", "data/fulltext/1.txt", "runs/a.jsonl"]


def test_refuses_empty_or_missing_train(repo, tmp_path, capsys):
    train = repo / "data" / "finetune" / "sft" / "train.jsonl"
    train.write_text("\n  \n", encoding="utf-8")
    assert mb.main(["--root", str(repo), "--out", str(tmp_path / "dist")]) == 2
    assert "train.jsonl отсутствует или пуст" in capsys.readouterr().err
    train.unlink()
    assert mb.main(["--root", str(repo), "--out", str(tmp_path / "dist")]) == 2
    assert not (tmp_path / "dist" / f"{NAME}.zip").exists()


def test_refuses_secret_in_data(repo, tmp_path, capsys):
    rows = [sft_row(1)]
    rows[0]["text"] += f"OPENAI_API_KEY={FAKE_KEY}"
    write_jsonl(repo / "data" / "finetune" / "sft" / "train.jsonl", rows)
    assert mb.main(["--root", str(repo), "--out", str(tmp_path / "dist")]) == 2
    assert "секрет" in capsys.readouterr().err


def test_cli_warns_on_empty_val(repo, tmp_path, capsys):
    assert mb.main(["--root", str(repo), "--out", str(tmp_path / "dist"), "--version", "v1"]) == 0
    out = capsys.readouterr().out
    assert "ВНИМАНИЕ: val пуст" in out and "режим smoke" in out
    assert (tmp_path / "dist" / f"{NAME}.zip").is_file()


def test_full_mode_with_val(repo, tmp_path):
    sft = repo / "data" / "finetune" / "sft"
    write_jsonl(sft / "val.jsonl", [sft_row(3), sft_row(4, prompt_hash="h2")])
    result = build(repo, tmp_path)
    manifest = result["manifest"]
    assert manifest["mode"] == "full" and manifest["model"]["lmstudio_new_id"] == "qwen3.5-4b-ra"
    assert manifest["data"]["val"]["examples"] == 2
    assert manifest["data"]["prompt_hashes"] == ["h1", "h2"]
    assert not any("val пуст" in w for w in manifest["warnings"])
    assert any("несколько prompt_hash" in w for w in manifest["warnings"])
    assert any("stats.json: val" in w for w in manifest["warnings"])  # в stats.json val 0, в файле 2


def test_build_is_reproducible(repo, tmp_path):
    first = mb.build(repo, tmp_path / "a", templates=TEMPLATES)
    second = mb.build(repo, tmp_path / "b", templates=TEMPLATES)
    assert first["zip"].read_bytes() == second["zip"].read_bytes()
    assert first["manifest"]["version"].startswith(first["manifest"]["date"])
    third = mb.build(repo, tmp_path / "a", templates=TEMPLATES)  # пересборка поверх старой папки
    assert third["zip"].read_bytes() == first["zip"].read_bytes()


def test_refuses_to_overwrite_foreign_folder(repo, tmp_path):
    stage = tmp_path / "dist" / NAME
    stage.mkdir(parents=True)
    (stage / "мой файл.txt").write_text("не трогать", encoding="utf-8")
    with pytest.raises(mb.BundleError):
        build(repo, tmp_path)
    assert (stage / "мой файл.txt").exists()


def test_verify_bundle_detects_tampering(repo, tmp_path, capsys):
    stage = build(repo, tmp_path)["stage"]
    assert vb.main([str(stage)]) == 0
    assert "Комплект цел" in capsys.readouterr().out
    script = stage / "train" / "smoke.sh"
    script.write_bytes(script.read_bytes().replace(b"\n", b"\r\n"))  # как после редактора с CRLF
    assert vb.main([str(stage)]) == 1
    assert "train/smoke.sh" in capsys.readouterr().out
    data = stage / "data" / "train.jsonl"
    content = bytearray(data.read_bytes())
    content[10] ^= 1  # тот же размер, другой хеш
    data.write_bytes(bytes(content))
    (stage / "data" / "stats.json").unlink()
    problems = dict(vb.verify(stage))
    assert problems["data/train.jsonl"] == "sha256 не совпадает" and problems["data/stats.json"] == "файла нет"
    (stage / "MANIFEST.json").unlink()
    assert vb.main([str(stage)]) == 2


def test_bundled_train_sft_uses_sibling_template(repo, tmp_path, monkeypatch):
    stage = build(repo, tmp_path)["stage"]
    module = load(stage / "train" / "train_sft.py", "bundled_train_sft", monkeypatch)
    assert module.DEFAULT_TEMPLATE == (stage / "train" / "qwen3_5_chat_template.jinja").resolve()
    assert module.DEFAULT_TEMPLATE.exists()


def test_choose_max_len():
    assert bt.choose_max_len([3000, 3500, 4000, 4500, 5000], [4096, 6144, 8192], 3) == 4096
    assert bt.choose_max_len([3000, 4500, 5000, 5100, 9000], [8192, 4096, 6144], 4) == 6144
    assert bt.choose_max_len([9000, 9500], [4096, 8192], 4) == 8192  # не помещается — наибольший
    assert bt.choose_max_len([3000], [4096, 8192], 4) == 4096  # min_kept не больше числа примеров


def test_loss_summary_and_latest_state(tmp_path):
    history = [{"loss": 2.0, "grad_norm": 1.0}, {"loss": 1.8}, {"loss": 1.2}, {"loss": 1.0}, {"eval_loss": 1.1}]
    summary = bt.loss_summary(history)
    assert summary["decreasing"] and not summary["nan"] and summary["eval_loss"] == 1.1
    assert bt.loss_summary([{"loss": 1.0}, {"loss": float("nan")}])["nan"]
    assert bt.loss_summary([])["records"] == 0
    for step in (5, 30, 10):
        state = tmp_path / f"checkpoint-{step}" / "trainer_state.json"
        state.parent.mkdir()
        state.write_text(json.dumps({"log_history": history}), encoding="utf-8")
    assert bt.latest_state(tmp_path).parent.name == "checkpoint-30"
    assert bt.main(["loss-report", "--out", str(tmp_path)]) == 0
    assert bt.latest_state(tmp_path / "нет") is None


def test_templates_are_consistent():
    readme = (TEMPLATES / "README_RTX5070.md").read_text(encoding="utf-8")
    for needle in ("verify_bundle.py", "setup_wsl.sh", "run_docker.sh", "smoke.sh", "convert_gguf.sh",
                   "lms import", "--context-length 16384", "eval-agent --variant prefetch",
                   "--provider lmstudio", "LLM_LMSTUDIO_MODEL", "0.80", "0.97", "make_train_bundle.py", "CRLF"):
        assert needle in readme, needle
    for name in ("setup_wsl.sh", "run_docker.sh", "smoke.sh", "train_full.sh", "convert_gguf.sh"):
        script = (TEMPLATES / name).read_text(encoding="utf-8")
        assert script.startswith("#!/usr/bin/env bash") and "set -euo pipefail" in script, name
    requirements = (TEMPLATES / "requirements-train.txt").read_text(encoding="utf-8")
    for pin in ("torch==2.8.0", "transformers==5.2.0", "trl==0.22.2", "unsloth[base]==2026.11.3",
                "unsloth_zoo[base]==2026.11.3", "torchao==0.16.0"):
        assert pin in requirements
    active = [line.strip() for line in requirements.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    # индекс PyTorch в файле ломает разбор зависимостей uv (first-index: certifi/setuptools из cu128)
    assert not [line for line in active if line.startswith("-")], active
    # Unsloth закреплён релизом, а не плавающей веткой git
    assert not [line for line in active if "git+" in line], active
    setup = (TEMPLATES / "setup_wsl.sh").read_text(encoding="utf-8")
    assert "--index-url https://download.pytorch.org/whl/cu128" in setup  # torch cu128 — отдельной командой
    assert "env-freeze.txt" in setup and "FREEZE" in setup
    docker = (TEMPLATES / "run_docker.sh").read_text(encoding="utf-8")
    for needle in ("--ipc=host", "--ulimit memlock=-1", "--ulimit stack=67108864", "HF_HOME=/workspace/ft/hf-cache",
                   "--user"):
        assert needle in docker, needle
    assert "не проверен" in readme.lower()
    assert "\ntriton" not in requirements  # triton отдельно не ставится
    assert "--chat-template" in (TEMPLATES / "smoke.sh").read_text(encoding="utf-8")
    assert "dist/" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()


# --- контракт скриптов комплекта с train_sft.py ---

TRAIN_SFT_OPTIONS = {opt for action in ts.build_parser()._actions for opt in action.option_strings}


def logical_lines(text):
    """Строки скрипта с учётом переносов `\\` (комментарии отброшены)."""
    text = re.sub(r"\\\n\s*", " ", text)
    return [line for line in text.splitlines() if not line.lstrip().startswith("#")]


@pytest.mark.parametrize("name", ["smoke.sh", "train_full.sh"])
def test_scripts_pass_only_known_train_sft_flags(name):
    lines = [line for line in logical_lines((TEMPLATES / name).read_text(encoding="utf-8"))
             if "train_sft.py" in line or re.match(r"\s*(\[.*\]\s*&&\s*)?\w*ARGS\+?=\(", line)]
    flags = {flag for line in lines for flag in re.findall(r"(?<![\w-])--[a-z][a-z0-9-]*", line)}
    assert {"--train", "--max-len", "--chat-template"} <= flags, name
    assert flags - TRAIN_SFT_OPTIONS == set(), f"{name}: неизвестные флаги train_sft.py"


def find_bash():
    bash = shutil.which("bash")
    if bash is None:
        return None
    lowered = bash.lower()
    if os.name == "nt" and ("system32" in lowered or "windowsapps" in lowered):
        return None  # это запуск WSL, а не Git Bash: пути Windows там не работают
    return bash


STUB = """#!/usr/bin/env bash
# Подмена python: записывает argv и создаёт слитую модель, если передан --merge-out.
(IFS=$'\\t'; printf '%s\\n' "$*") >> "$STUB_LOG"
prev=""
for arg in "$@"; do
    if [ "$prev" = "--merge-out" ]; then mkdir -p "$arg"; : > "$arg/config.json"; : > "$arg/chat_template.jinja"; fi
    prev="$arg"
done
exit 0
"""


def run_with_stub(stage, script, workdir):
    """Запускает скрипт комплекта с PY=заглушка; возвращает argv всех вызовов train_sft.py."""
    bash = find_bash()
    if bash is None:
        pytest.skip("нет bash (Git Bash / Linux)")
    workdir.mkdir(parents=True)
    stub = workdir / "stub-python"
    stub.write_bytes(STUB.encode("utf-8"))
    stub.chmod(0o755)
    log = workdir / "argv.log"
    env = {**os.environ, "PY": stub.as_posix(), "STUB_LOG": log.as_posix(), "FT_HOME": (workdir / "ft").as_posix(),
           "FT_NO_VENV": "1", "MAX_LEN": "4096"}
    env.pop("VIRTUAL_ENV", None)
    result = subprocess.run([bash, (stage / "train" / script).as_posix()], env=env, capture_output=True,
                            timeout=120, encoding="utf-8", errors="replace")
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [line.split("\t") for line in log.read_text(encoding="utf-8").splitlines()]
    return [call[1:] for call in calls if call and call[0].endswith("train_sft.py")]


@pytest.mark.parametrize("script", ["smoke.sh", "train_full.sh"])
def test_scripts_omit_val_when_empty_and_pass_it_otherwise(repo, tmp_path, script):
    stage = build(repo, tmp_path)["stage"]
    parser = ts.build_parser()

    calls = run_with_stub(stage, script, tmp_path / "empty")
    assert calls, "train_sft.py не вызывался"
    for argv in calls:
        assert "--val" not in argv and "--limit-val" not in argv, argv
        ts.validate_args(parser, parser.parse_args(argv))  # неизвестный флаг → SystemExit
    training = [argv for argv in calls if "--dry-run" not in argv]
    assert training and all("--merge-out" in argv and "--chat-template" in argv for argv in training)

    write_jsonl(stage / "data" / "val.jsonl", [sft_row(3)])
    calls = run_with_stub(stage, script, tmp_path / "with-val")
    assert calls and all("--val" in argv for argv in calls)
    for argv in calls:
        assert Path(argv[argv.index("--val") + 1]).name == "val.jsonl"
        ts.validate_args(parser, parser.parse_args(argv))
