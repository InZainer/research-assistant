"""Архив проекта для запуска на другой машине: код, AGENT.md, .env.example и готовая библиотека статей.

    uv run python scripts/make_project_bundle.py            # → dist/research_assistant_bundle.zip
    uv run python scripts/make_project_bundle.py --no-library

В архив не попадают .env, ключи, полные тексты статей (PDF), журналы прогонов, история чатов и данные
дообучения. Сборка отказывается (код 2), если в файлах найден ключ: по шаблонам и по реальным значениям из .env.
"""

import argparse
import json
import re
import sqlite3
import subprocess
import sys
import tempfile
import zipfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_train_bundle import SECRET_PATTERNS, git_info, sha256_bytes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_NAME = "research_assistant_bundle"
TOP = "research-assistant"  # папка внутри архива
EMBED_INDEX = "data/vectors/text-embedding-bge-m3.npz"  # индекс модели EMBED_MODEL по умолчанию
LIBRARY = "data/library.sqlite"
ALLOWED_KEY_VALUES = {"lm-studio", "ollama"}
# Построчно: «KEY=» с пустым значением не должно захватывать имя на следующей строке.
KEY_ASSIGNMENT = re.compile(r"^[ \t]*[A-Z_]*API_KEY[ \t]*=[ \t]*[\"']?([^\s\"'`;]+)", re.M)
FORBIDDEN = re.compile(
    r"(^|/)\.env$|(^|/)\.env\.(?!example$)|(^|/)data/(?!library\.sqlite$|vectors/)|(^|/)runs/|(^|/)dist/"
    r"|(^|/)\.venv/|__pycache__|\.lnk$|\.pyc$",
    re.IGNORECASE,
)
MIN_SECRET_LEN = 8
# Секретом считаем только значения переменных с такими именами: EMBED_MODEL=text-embedding-bge-m3 и пути
# встречаются в документации и секретами не являются.
SECRET_NAME = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD)$", re.IGNORECASE)


class BundleError(Exception):
    """Сборка отказалась; сообщение — для пользователя."""


def project_files(root: Path) -> list[str]:
    """Файлы проекта с учётом .gitignore: отслеживаемые git и новые неигнорируемые."""
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root,
                         capture_output=True, check=True, timeout=60).stdout
    names = [n for n in out.decode("utf-8").split("\0") if n]
    return sorted(n for n in names if (root / n).is_file())


def forbidden(names) -> list[str]:
    return [n for n in names if FORBIDDEN.search(n.replace("\\", "/"))]


def env_secrets(env_path: Path) -> list[str]:
    """Значения из .env, похожие на ключи: ни одно не должно встретиться в архиве."""
    if not env_path.is_file():
        return []
    values = []
    for line in env_path.read_text(encoding="utf-8").splitlines():
        name, sep, value = line.partition("=")
        value = value.strip().strip("'\"")
        if (sep and not line.lstrip().startswith("#") and SECRET_NAME.search(name.strip())
                and len(value) >= MIN_SECRET_LEN and value not in ALLOWED_KEY_VALUES):
            values.append(value)
    return values


def find_secrets(text: str) -> list[str]:
    hits = [m.group(0)[:12] + "…" for pattern in SECRET_PATTERNS for m in pattern.finditer(text)]
    for match in KEY_ASSIGNMENT.finditer(text):
        value = match.group(1)
        if value not in ALLOWED_KEY_VALUES and not value.startswith(("$", "<", "{", "…")):
            hits.append(match.group(0).strip()[:24] + "…")
    return hits


def snapshot_sqlite(path: Path) -> bytes:
    """Согласованная копия базы через backup API: база может быть открыта веб-интерфейсом."""
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "library.sqlite"
        source = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        dest = sqlite3.connect(target)
        try:
            source.backup(dest)
        finally:
            dest.close()
            source.close()
        return target.read_bytes()


def library_stats(data: bytes) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "library.sqlite"
        path.write_bytes(data)
        conn = sqlite3.connect(path)
        try:
            count = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
            stats = {"papers": count("SELECT COUNT(*) FROM papers"), "passages": count("SELECT COUNT(*) FROM passages"),
                     "sentences": count("SELECT COUNT(*) FROM sentences"),
                     "topics": [r[0] for r in conn.execute("SELECT name FROM topics ORDER BY id")]}
        finally:
            conn.close()
    return stats


def collect(root: Path, files: list[str] | None = None, include_library: bool = True) -> tuple[dict, dict]:
    """{путь в архиве: байты} и сводка для MANIFEST.json."""
    names = project_files(root) if files is None else sorted(files)
    names = [n for n in names if n not in forbidden(names)]
    content = {name: (root / name).read_bytes() for name in names}
    summary = {"library": None}
    if include_library:
        db, index = root / LIBRARY, root / EMBED_INDEX
        if not db.is_file():
            raise BundleError(f"нет {LIBRARY}: соберите библиотеку или запустите с --no-library")
        content[LIBRARY] = snapshot_sqlite(db)
        summary["library"] = library_stats(content[LIBRARY])
        if index.is_file():
            content[EMBED_INDEX] = index.read_bytes()
            summary["embed_index"] = EMBED_INDEX
        else:
            summary["embed_index"] = None  # без индекса поиск по словам (BM25)
    return content, summary


def check_secrets(content: dict, secrets: list[str]) -> list[str]:
    problems = []
    for name, data in content.items():
        for value in secrets:
            if value.encode("utf-8") in data:
                problems.append(f"{name}: значение из .env")
        if name.endswith((".sqlite", ".npz")):
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        problems += [f"{name}: {hit}" for hit in find_secrets(text)]
    return problems


def build(root: Path = ROOT, out: Path | None = None, files: list[str] | None = None, include_library: bool = True,
          version: str | None = None, env_path: Path | None = None) -> Path:
    root = Path(root)
    out = Path(out) if out else root / "dist"
    content, summary = collect(root, files, include_library)
    if "AGENT.md" not in content:
        raise BundleError("нет AGENT.md в корне проекта — инструкция для ИИ-агента обязательна")
    problems = check_secrets(content, env_secrets(env_path or root / ".env"))
    if problems:
        raise BundleError("в архив попали бы секреты, сборка остановлена:\n  " + "\n  ".join(problems[:20]))
    manifest = {
        "bundle": BUNDLE_NAME, "version": version or date.today().isoformat(), **git_info(root), **summary,
        "start_here": "AGENT.md",
        "files": {name: {"sha256": sha256_bytes(data), "size": len(data)} for name, data in sorted(content.items())},
    }
    content["MANIFEST.json"] = json.dumps(manifest, ensure_ascii=False, indent=1).encode("utf-8")
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / f"{BUNDLE_NAME}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(content.items()):
            archive.writestr(f"{TOP}/{name}", data)
    return zip_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Архив проекта для запуска на другой машине")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-library", action="store_true", help="без базы статей и индекса эмбеддингов")
    parser.add_argument("--version")
    args = parser.parse_args(argv)
    try:
        path = build(args.root, args.out, include_library=not args.no_library, version=args.version)
    except BundleError as error:
        print(f"Ошибка: {error}")
        return 2
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read(f"{TOP}/MANIFEST.json"))
    library = manifest.get("library")
    print(f"Архив: {path} ({path.stat().st_size / 1e6:.1f} МБ), файлов {len(manifest['files']) + 1}")
    if library:
        print(f"Библиотека: статей {library['papers']}, фрагментов {library['passages']}; индекс: "
              f"{manifest.get('embed_index') or 'нет (поиск по словам)'}")
    print("Начинать с research-assistant/AGENT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
