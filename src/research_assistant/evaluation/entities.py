"""Оценка извлечения сущностей на золотом наборе: precision, recall, F1 по типам.

Сравнение — по основам слов (entity_key), поэтому «диагностики» и «диагностика» совпадают.
Строгое совпадение — ключи равны; мягкое — один ключ целиком входит в другой подряд
(«EAGLE-2» и «EAGLE-2 model», «speedup» и «speedup ratio»).
Синонимы одной золотой сущности записываются через «|».
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

from research_assistant.nlp.entities import ENTITY_TYPES, Extraction, clean, contains_stems, extract_entities, parse_json
from research_assistant.nlp.text import analyze, detect_language


@dataclass
class GoldEntities:
    paper: str
    text: str
    entities: dict[str, list[str]]
    status: str = "draft"


def load_gold(path: str | Path) -> list[GoldEntities]:
    return [GoldEntities(**json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _matches(pred: list[str], gold_aliases: list[list[str]], lenient: bool) -> bool:
    for alias in gold_aliases:
        if pred == alias:
            return True
        if lenient and (contains_stems(pred, alias) or contains_stems(alias, pred)):
            return True
    return False


def score_type(predicted: list[str], gold: list[str], lenient: bool = False) -> tuple[int, int, int]:
    """(TP, FP, FN). Несколько предсказаний одной золотой сущности (полное имя и аббревиатура) — не ложные."""
    gold_aliases = [[analyze(alias) for alias in item.split("|")] for item in gold]
    matched = set()
    false_positive = 0
    for name in predicted:
        key = analyze(name)
        hit = next((i for i, aliases in enumerate(gold_aliases) if _matches(key, aliases, lenient)), None)
        if hit is None:
            false_positive += 1
        else:
            matched.add(hit)
    return len(matched), false_positive, len(gold) - len(matched)


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def summarize(rows: list[tuple[str, dict[str, list[str]], dict[str, list[str]]]], lenient: bool) -> dict:
    """rows: (язык, предсказания, золото). Метрики по типам, по языкам и общий micro-average."""
    groups = {"all": rows, **{lang: [r for r in rows if r[0] == lang] for lang in ("en", "ru")}}
    out = {}
    for name, group in groups.items():
        if not group:
            continue
        totals = {t: [0, 0, 0] for t in ENTITY_TYPES}
        for _, predicted, gold in group:
            for t in ENTITY_TYPES:
                for i, value in enumerate(score_type(predicted.get(t, []), gold.get(t, []), lenient)):
                    totals[t][i] += value
        micro = [sum(totals[t][i] for t in ENTITY_TYPES) for i in range(3)]
        out[name] = {"micro": prf(*micro), **{t: prf(*totals[t]) for t in ENTITY_TYPES}}
    return out


def evaluate_entities(model, gold: list[GoldEntities], log=lambda message: None) -> dict:
    """Модель вызывается один раз на текст; из её ответа считаются оба варианта — с проверкой по тексту и без."""
    variants = {"raw": [], "grounded": []}
    per_item = []
    tokens, started = 0, time.perf_counter()
    for number, item in enumerate(gold, 1):
        language = detect_language(item.text)
        try:
            extraction = extract_entities(model, item.text, grounded=False)
            raw = parse_json(extraction.raw)
            tokens += extraction.tokens
        except Exception as exc:  # сбой модели на одном тексте не останавливает оценку
            log(f"[{number}/{len(gold)}] ошибка: {exc}")
            raw, extraction = {}, Extraction()
        grounded = clean(raw, item.text, grounded=True)
        variants["raw"].append((language, extraction.entities, item.entities))
        variants["grounded"].append((language, grounded.entities, item.entities))
        per_item.append({"paper": item.paper, "language": language, "raw": extraction.entities,
                         "grounded": grounded.entities, "dropped": grounded.dropped, "gold": item.entities})
        log(f"[{number}/{len(gold)}] {item.paper[:60]}: отброшено проверкой {len(grounded.dropped)}")
    return {
        "model": model.model,
        "items": len(gold),
        "tokens": tokens,
        "seconds": time.perf_counter() - started,
        "strict": {name: summarize(rows, lenient=False) for name, rows in variants.items()},
        "lenient": {name: summarize(rows, lenient=True) for name, rows in variants.items()},
        "per_item": per_item,
    }


def format_entity_report(report: dict) -> str:
    lines = [f"Модель {report['model']}: {report['items']} текстов, {report['tokens']} токенов, "
             f"{report['seconds']:.0f} с"]
    header = f"{'сравнение':<9} {'вариант':<9} {'группа':<5} {'тип':<8} {'P':>5} {'R':>5} {'F1':>5}  TP/FP/FN"
    lines += [header, "-" * len(header)]
    for match in ("strict", "lenient"):
        for variant, groups in report[match].items():
            for group, metrics in groups.items():
                types = ["micro"] if group != "all" else ["micro", *ENTITY_TYPES]
                for t in types:
                    m = metrics[t]
                    lines.append(f"{match:<9} {variant:<9} {group:<5} {t:<8} {m['precision']:>5.2f} {m['recall']:>5.2f} "
                                 f"{m['f1']:>5.2f}  {m['tp']}/{m['fp']}/{m['fn']}")
    return "\n".join(lines)
