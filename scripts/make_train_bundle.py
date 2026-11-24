"""Сборка комплекта для дообучения на ПК с RTX 5070: dist/rtx5070_train_bundle/ и .zip.

    uv run python scripts/make_train_bundle.py                       # из корня репозитория
    uv run python scripts/make_train_bundle.py --version 2026-11-09 --no-notebook
    uv run python scripts/make_train_bundle.py --root <репозиторий> --out <папка>

В комплект попадают только скрипт обучения, чат-шаблон, данные SFT (data/finetune/sft) и файлы
из scripts/train_bundle/. Никаких .env, ключей, базы SQLite, полных текстов, журналов runs/ и истории
чатов: список источников фиксирован, а собранные файлы дополнительно проверяются на секреты.
Сеть и модель не нужны. Код выхода 2 — сборка отказалась (пустой train, секрет, нет нужного файла).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = Path(__file__).resolve().parent / "train_bundle"
BUNDLE_NAME = "rtx5070_train_bundle"
BASE_MODEL = "unsloth/Qwen3.5-4B"
HF_BASE = "Qwen/Qwen3.5-4B"
LMSTUDIO_BASE_ID = "qwen3.5-4b"
PILOT_THRESHOLD = 200  # меньше примеров в train — пилотные данные

# путь в комплекте → путь в репозитории (обязательные источники)
SOURCES = {
    "train/train_sft.py": "scripts/train_sft.py",
    "train/qwen3_5_chat_template.jinja": "src/research_assistant/finetune/qwen3_5_chat_template.jinja",
}
DATA_DIR = "data/finetune/sft"
NOTEBOOK = ("notebook/finetune_qwen35_colab.ipynb", "notebooks/finetune_qwen35_colab.ipynb")

# путь в комплекте → файл в scripts/train_bundle/
TEMPLATE_FILES = {
    "README_RTX5070.md": "README_RTX5070.md",
    "train/requirements-train.txt": "requirements-train.txt",
    "train/common.sh": "common.sh",
    "train/setup_wsl.sh": "setup_wsl.sh",
    "train/run_docker.sh": "run_docker.sh",
    "train/smoke.sh": "smoke.sh",
    "train/train_full.sh": "train_full.sh",
    "train/convert_gguf.sh": "convert_gguf.sh",
    "train/bundle_tools.py": "bundle_tools.py",
    "train/verify_bundle.py": "verify_bundle.py",
}
EXECUTABLE_SUFFIXES = (".sh", ".py")

# Секреты: ключи OpenAI-совместимых API и HF, присваивание ключа в .env, приватные ключи.
# "lm-studio" — заглушка ключа LM Studio из инструкции, это не секрет.
ALLOWED_KEY_VALUES = {"lm-studio"}
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)
KEY_ASSIGNMENT = re.compile(r"\b[A-Z_]*API_KEY\b\s*=\s*[\"']?([^\s\"'`;]+)")
FORBIDDEN_NAME = re.compile(
    r"(^|/)\.env($|\.)|\.(sqlite|sqlite3|db)$|(^|/)fulltext(/|$)|(^|/)runs(/|$)|(^|/)chats?(/|$)|\.lnk$",
    re.IGNORECASE,
)


class BundleError(Exception):
    """Сборка отказалась; сообщение — для пользователя."""


# --- чистые функции ---

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def to_lf(data):
    """Переводы строк CRLF/CR → LF (скрипты запускаются в Linux)."""
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def count_rows(path):
    """Непустые строки JSONL; отсутствующий файл — 0."""
    path = Path(path)
    if not path.is_file():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def prompt_hashes(paths):
    """Все meta.prompt_hash в файлах JSONL (отсортированы)."""
    found = set()
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    found.add((json.loads(line).get("meta") or {}).get("prompt_hash") or "?")
    return sorted(found)


def find_secrets(text):
    """Подозрительные фрагменты: ключи API и присваивания *API_KEY=значение (кроме заглушек)."""
    hits = [m.group(0)[:12] + "…" for pattern in SECRET_PATTERNS for m in pattern.finditer(text)]
    for match in KEY_ASSIGNMENT.finditer(text):
        value = match.group(1)
        if value in ALLOWED_KEY_VALUES or value.startswith(("$", "<", "{")):
            continue
        hits.append(match.group(0)[:24] + "…")
    return hits


def forbidden_paths(names):
    """Пути, которым не место в комплекте (.env, база, полные тексты, журналы, чаты)."""
    return [name for name in names if FORBIDDEN_NAME.search(name.replace("\\", "/"))]


def clean_notebook(data):
    """Ноутбук без выводов ячеек и номеров запуска: в выводах могли остаться пути и данные."""
    notebook = json.loads(data.decode("utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    return (json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode("utf-8")


def data_summary(train_path, val_path, stats):
    """Сводка данных для MANIFEST: примеры по сплитам (по строкам файлов) и токены из stats.json."""
    splits = (stats or {}).get("splits", {})
    summary = {}
    for name, path in (("train", train_path), ("val", val_path)):
        split = splits.get(name, {})
        summary[name] = {"examples": count_rows(path), "tokens": split.get("tokens"),
                         "by_kind": split.get("by_kind"), "by_language": split.get("by_language")}
    summary["prompt_hashes"] = prompt_hashes([train_path, val_path])
    summary["max_tokens"] = (stats or {}).get("max_tokens")
    return summary


def tokens_line(tokens):
    if not tokens or not tokens.get("max"):
        return "нет данных"
    return f"mean {tokens['mean']} / p50 {tokens['p50']} / p95 {tokens['p95']} / max {tokens['max']} токенов"


def render_readme(template, values):
    """Подставляет {{KEY}}; оставшиеся плейсхолдеры — ошибка шаблона."""
    text = template
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", str(value))
    left = sorted(set(re.findall(r"\{\{[A-Z_]+\}\}", text)))
    if left:
        raise BundleError(f"в README_RTX5070.md не заполнены {', '.join(left)}")
    return text


def mode_values(mode, summary):
    train_n = summary["train"]["examples"]
    if mode == "smoke":
        note = (f"> **Сейчас это пробный (smoke) прогон.** val пуст, в train {train_n} примеров. "
                "Прироста качества не будет: прогон проверяет, что весь конвейер работает "
                "(обучение → слияние → GGUF → LM Studio → `eval-agent`). Полный прогон — после генерации "
                "полного датасета; тогда комплект пересобирается на ноутбуке (см. «Пересборка комплекта»).")
        return {"MODE": "smoke", "MODE_NOTE": note, "MODEL_ID": "qwen3.5-4b-ra-smoke",
                "MODE_TEXT": "val пуст: пробный прогон без eval loss, проверка конвейера"}
    note = (f"> **Полные данные:** train {train_n}, val {summary['val']['examples']}. Сначала короткий "
            "`smoke.sh` (проверка окружения и памяти), затем `train_full.sh` — раздел «Полный прогон».")
    return {"MODE": "full", "MODE_NOTE": note, "MODEL_ID": "qwen3.5-4b-ra",
            "MODE_TEXT": "есть val: полное обучение, лучший чекпоинт по eval loss"}


def pinned_stack(requirements):
    """Закреплённые версии из requirements-train.txt: {пакет: версия} для MANIFEST."""
    stack = {}
    for line in requirements.splitlines():
        line = line.split("#", 1)[0].strip()
        match = re.fullmatch(r"([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?\s*==\s*([^\s;,]+)", line)
        if match:
            stack[match.group(1).lower().replace("_", "-")] = match.group(2)
    return stack


def git_info(root):
    """Коммит и признак незакоммиченных изменений; без git — None."""
    def run(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=20, check=True).stdout

    try:
        commit = run("rev-parse", "HEAD").strip() or None
        dirty = bool(run("status", "--porcelain").strip())
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": dirty}


def zip_info(name, date):
    """Запись zip с фиксированной датой и правами Unix: содержимое архива воспроизводимо."""
    info = zipfile.ZipInfo(name, date_time=(date.year, date.month, date.day, 0, 0, 0))
    info.create_system = 3  # Unix: unzip восстановит права
    mode = 0o755 if name.endswith(EXECUTABLE_SUFFIXES) and "/train/" in f"/{name}" else 0o644
    info.external_attr = (0o100000 | mode) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


# --- сборка ---

def collect(root, include_notebook=True):
    """Содержимое комплекта: {путь в комплекте: байты}, сводка данных, предупреждения, источники."""
    root = Path(root)
    files, warnings, inputs = {}, [], []

    for target, source in SOURCES.items():
        path = root / source
        if not path.is_file():
            raise BundleError(f"нет {source} — без него комплект бесполезен")
        files[target] = to_lf(path.read_bytes())
        inputs.append(path)

    sft = root / DATA_DIR
    train, val, stats_path = sft / "train.jsonl", sft / "val.jsonl", sft / "stats.json"
    if count_rows(train) == 0:
        raise BundleError(f"{DATA_DIR}/train.jsonl отсутствует или пуст — сначала "
                          "`uv run research-assistant ft-export`")
    files["data/train.jsonl"] = train.read_bytes()
    inputs.append(train)
    files["data/val.jsonl"] = val.read_bytes() if val.is_file() else b""
    if val.is_file():
        inputs.append(val)
    if count_rows(val) == 0:
        warnings.append("val пуст — режим smoke: пробный прогон без eval loss")
    stats = None
    if stats_path.is_file():
        files["data/stats.json"] = stats_path.read_bytes()
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        inputs.append(stats_path)
    else:
        warnings.append("нет stats.json — сводка токенов в MANIFEST пуста")

    if include_notebook:
        target, source = NOTEBOOK
        path = root / source
        if path.is_file():
            files[target] = clean_notebook(path.read_bytes())
            inputs.append(path)
        else:
            warnings.append(f"нет {source} — ноутбук не включён")

    summary = data_summary(train, val, stats)
    for name in ("train", "val"):
        stated = ((stats or {}).get("splits", {}).get(name) or {}).get("examples")
        if stats is not None and stated is not None and stated != summary[name]["examples"]:
            warnings.append(f"stats.json: {name} {stated} примеров, а в файле {summary[name]['examples']}")
    if summary["train"]["examples"] < PILOT_THRESHOLD:
        warnings.append(f"в train {summary['train']['examples']} примеров (< {PILOT_THRESHOLD}) — пилотные данные, "
                        "дообучение только проверяет конвейер")
    if len(summary["prompt_hashes"]) > 1:
        warnings.append(f"в данных несколько prompt_hash: {', '.join(summary['prompt_hashes'])}")
    return files, summary, warnings, inputs


def build(root=ROOT, out=None, version=None, include_notebook=True, templates=TEMPLATES, name=BUNDLE_NAME):
    """Собирает папку и zip; возвращает сводку. BundleError — отказ (код 2)."""
    root, templates = Path(root), Path(templates)
    out = Path(out) if out else root / "dist"
    files, summary, warnings, inputs = collect(root, include_notebook)

    template_bytes = {}
    for target, source in TEMPLATE_FILES.items():
        path = templates / source
        if not path.is_file():
            raise BundleError(f"нет шаблона {path}")
        template_bytes[target] = to_lf(path.read_bytes())
        inputs.append(path)

    mtime = max(path.stat().st_mtime for path in inputs)
    date = datetime.fromtimestamp(mtime, timezone.utc).date()
    content_hash = sha256_bytes("".join(
        f"{target}:{sha256_bytes(data)}\n" for target, data in sorted({**files, **template_bytes}.items())
    ).encode("utf-8"))
    version = version or f"{date.isoformat()}-{content_hash[:8]}"
    mode = "smoke" if summary["val"]["examples"] == 0 else "full"
    source = git_info(root)

    values = {"VERSION": version, "TRAIN_EXAMPLES": summary["train"]["examples"],
              "VAL_EXAMPLES": summary["val"]["examples"], "TOKENS_LINE": tokens_line(summary["train"]["tokens"]),
              "PROMPT_HASHES": ", ".join(summary["prompt_hashes"]) or "нет",
              "GIT_COMMIT": (source["git_commit"] or "нет git")[:12]
              + (" (с незакоммиченными изменениями)" if source["git_dirty"] else ""),
              **mode_values(mode, summary)}
    readme = template_bytes.pop("README_RTX5070.md").decode("utf-8")
    files["README_RTX5070.md"] = render_readme(readme, values).encode("utf-8")
    files.update(template_bytes)

    bad = forbidden_paths(files)
    if bad:
        raise BundleError(f"в комплект попали запрещённые файлы: {', '.join(bad)}")
    for target, data in files.items():
        hits = find_secrets(data.decode("utf-8", errors="replace"))
        if hits:
            raise BundleError(f"похоже на секрет в {target}: {', '.join(hits[:3])} — сборка остановлена")
    crlf = [target for target, data in files.items() if target.endswith(".sh") and b"\r" in data]
    if crlf:
        raise BundleError(f"CR в скриптах: {', '.join(crlf)}")

    manifest = {
        "bundle": name,
        "version": version,
        "date": date.isoformat(),
        "mode": mode,
        "mode_note": ("smoke: val пуст — пробный прогон конвейера (обучение без eval loss → слияние → GGUF → LM Studio)"
                      if mode == "smoke" else "full: есть val — полное обучение с выбором чекпоинта по eval loss"),
        "source": {**source, "content_sha256": content_hash},
        "model": {"base": BASE_MODEL, "hf_base": HF_BASE, "lmstudio_base_id": LMSTUDIO_BASE_ID,
                  "lmstudio_new_id": values["MODEL_ID"]},
        "stack": pinned_stack(files["train/requirements-train.txt"].decode("utf-8")),
        "data": summary,
        "warnings": warnings,
        "files": [{"path": target, "size": len(data), "sha256": sha256_bytes(data)}
                  for target, data in sorted(files.items())],
    }
    files["MANIFEST.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    stage = out / name
    if stage.exists():
        if any(stage.iterdir()) and not (stage / "MANIFEST.json").is_file():
            raise BundleError(f"{stage} существует и не похож на комплект — удалите его вручную")
        shutil.rmtree(stage)
    for target, data in files.items():
        path = stage / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    zip_path = out / f"{name}.zip"
    tmp = zip_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for target in sorted(files):
            archive.writestr(zip_info(f"{name}/{target}", date), files[target])
    tmp.replace(zip_path)
    return {"stage": stage, "zip": zip_path, "manifest": manifest, "warnings": warnings}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="make_train_bundle.py",
                                     description="комплект дообучения для ПК с RTX 5070 (папка и zip)")
    parser.add_argument("--root", default=str(ROOT), help="корень репозитория (по умолчанию — этот)")
    parser.add_argument("--out", help="куда положить папку и zip (по умолчанию <root>/dist)")
    parser.add_argument("--version", help="версия комплекта (по умолчанию <дата>-<хеш содержимого>)")
    parser.add_argument("--no-notebook", action="store_true", help="не включать ноутбук Colab")
    args = parser.parse_args(argv)
    try:
        result = build(args.root, args.out, args.version, include_notebook=not args.no_notebook)
    except BundleError as error:
        print(f"Сборка отменена: {error}", file=sys.stderr)
        return 2
    manifest = result["manifest"]
    for warning in result["warnings"]:
        print(f"ВНИМАНИЕ: {warning}")
    size = result["zip"].stat().st_size
    print(f"Комплект {manifest['version']} (режим {manifest['mode']}): {len(manifest['files'])} файлов")
    print(f"  папка: {result['stage']}")
    print(f"  архив: {result['zip']} ({size / 1024:.0f} КБ)")
    print(f"  train {manifest['data']['train']['examples']}, val {manifest['data']['val']['examples']}; "
          f"prompt_hash {', '.join(manifest['data']['prompt_hashes'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
