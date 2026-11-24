"""Проверка целостности комплекта: пересчитывает sha256 файлов и сверяет с MANIFEST.json.

    python3 train/verify_bundle.py            # из корня распакованного комплекта
    python3 train/verify_bundle.py <папка>    # или с явным путём

Только стандартная библиотека Python 3.8+. Код выхода: 0 — всё совпало, 1 — файл изменён или пропал,
2 — нет или не читается MANIFEST.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root):
    """Список проблем: [(путь, описание)]; пустой список — комплект цел."""
    root = Path(root)
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    problems = []
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file():
            problems.append((entry["path"], "файла нет"))
            continue
        size = path.stat().st_size
        if size != entry["size"]:
            problems.append((entry["path"], f"размер {size} вместо {entry['size']}"))
        elif sha256_file(path) != entry["sha256"]:
            problems.append((entry["path"], "sha256 не совпадает"))
    return problems


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    root = Path(argv[0]) if argv else Path(__file__).resolve().parent.parent
    try:
        manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
        problems = verify(root)
    except (OSError, ValueError, KeyError) as error:
        print(f"Не удалось прочитать {root / 'MANIFEST.json'}: {error}")
        return 2
    total = len(manifest["files"])
    if problems:
        for path, problem in problems:
            print(f"  {path}: {problem}")
        print(f"Комплект ПОВРЕЖДЁН: {len(problems)} из {total} файлов. Скопируйте архив заново "
              "(и не открывайте .sh в редакторах, которые меняют переводы строк на CRLF).")
        return 1
    print(f"Комплект цел: {total} файлов совпали с MANIFEST.json "
          f"(версия {manifest.get('version')}, режим {manifest.get('mode')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
