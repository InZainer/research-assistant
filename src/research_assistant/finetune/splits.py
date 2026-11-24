"""Сплит библиотеки по статьям: test — все статьи золотого набора, val — несколько других, train — остальное.

Делим по статьям, а не по вопросам: иначе вопросы к одному фрагменту попали бы и в обучение, и в тест.
Val стратифицирован по теме и языку, чтобы в нём были и английские, и русские статьи разных тем.
"""

import json
import random
from pathlib import Path

from research_assistant.ingest.models import normalize_title
from research_assistant.nlp.text import analyze
from research_assistant.store.db import Library

SPLITS = ("train", "val", "test")


def gold_paper_ids(library: Library, gold_questions) -> tuple[set[int], list[str]]:
    """(id статей, на которые ссылается золото; названия из золота, которых нет в библиотеке).

    Сопоставление по нормализованному названию — так же, как в evaluation.retrieval.resolve."""
    papers = {normalize_title(r["title"]): r["id"] for r in library.conn.execute("SELECT id, title FROM papers")}
    found, missing = set(), []
    for question in gold_questions:
        for item in question.relevant:
            paper_id = papers.get(normalize_title(item.paper))
            if paper_id is None:
                missing.append(item.paper)
            else:
                found.add(paper_id)
    return found, sorted(set(missing))


def _stratum(library: Library, paper) -> tuple[str, str]:
    topics = library.paper_topics(paper["id"])
    return (topics[0] if topics else "-"), (paper["language"] or "-")


def _quotas(sizes: dict, total: int) -> dict:
    """Доли страт пропорционально размеру (метод наибольших остатков)."""
    whole = sum(sizes.values())
    if not whole or total <= 0:
        return {key: 0 for key in sizes}
    exact = {key: total * size / whole for key, size in sizes.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    rest = sorted(sizes, key=lambda key: (-(exact[key] - quotas[key]), -sizes[key], key))
    for key in rest[: total - sum(quotas.values())]:
        quotas[key] += 1
    return quotas


def split_papers(library: Library, gold_questions, val_papers: int = 6, seed: int = 0,
                 gold_file: str | None = None) -> dict:
    """{"test": [...], "val": [...], "train": [...], "seed": …}. Детерминирован при том же seed и библиотеке.

    В val берутся только статьи с текстовыми фрагментами: статья без текста не даст ни одного вопроса."""
    test, missing = gold_paper_ids(library, gold_questions)
    with_text = {r[0] for r in library.conn.execute("SELECT DISTINCT paper_id FROM passages WHERE kind = 'text'")}
    rest = [p for p in library.conn.execute("SELECT id, language FROM papers ORDER BY id") if p["id"] not in test]
    strata: dict[tuple[str, str], list[int]] = {}
    for paper in rest:
        if paper["id"] in with_text:
            strata.setdefault(_stratum(library, paper), []).append(paper["id"])
    rng = random.Random(seed)
    val = []
    for key, quota in sorted(_quotas({k: len(v) for k, v in strata.items()}, val_papers).items()):
        ids = sorted(strata[key])
        rng.shuffle(ids)
        val += ids[:quota]
    val_set = set(val)
    return {
        "test": sorted(test),
        "val": sorted(val_set),
        "train": sorted(p["id"] for p in rest if p["id"] not in val_set),
        "seed": seed,
        "val_papers": val_papers,
        "gold_file": gold_file,
        "gold_missing": missing,
    }


def save_splits(splits: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")


def load_splits(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def split_of_paper(splits: dict, paper_id: int) -> str | None:
    return next((name for name in SPLITS if paper_id in splits.get(name, ())), None)


def jaccard(a: str, b: str) -> float:
    """Жаккар по множествам основ слов analyze(): перефразы с теми же словами дают высокое значение."""
    left, right = set(analyze(a)), set(analyze(b))
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def near_duplicate(a: str, b: str, threshold: float = 0.6) -> bool:
    return jaccard(a, b) >= threshold
