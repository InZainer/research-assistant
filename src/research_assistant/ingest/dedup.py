"""Слияние записей об одной и той же статье из разных источников."""

from dataclasses import fields, replace

from research_assistant.ingest.models import PaperRecord


def merge_records(first: PaperRecord, second: PaperRecord) -> PaperRecord:
    """Поля берутся из первой записи, пустые дополняются из второй.

    Исключения: аннотация — более длинная, авторы — более полный список, источники — объединение.
    """
    values = {}
    for f in fields(PaperRecord):
        a, b = getattr(first, f.name), getattr(second, f.name)
        values[f.name] = a if a not in (None, "", []) else b
    if second.abstract and len(second.abstract) > len(first.abstract or ""):
        values["abstract"] = second.abstract
    if len(second.authors) > len(first.authors):
        values["authors"] = list(second.authors)
    values["sources"] = list(dict.fromkeys(first.sources + second.sources))
    return replace(first, **values)


def dedupe(records: list[PaperRecord]) -> list[PaperRecord]:
    """Склеивает записи, у которых совпадает хотя бы один ключ (DOI, arXiv, OpenAlex, заголовок)."""
    merged: list[PaperRecord] = []
    index: dict[tuple[str, str], int] = {}
    for record in records:
        hit = next((index[key] for key in record.keys() if key in index), None)
        if hit is None:
            merged.append(record)
            hit = len(merged) - 1
        else:
            merged[hit] = merge_records(merged[hit], record)
        for key in merged[hit].keys():
            index.setdefault(key, hit)
    return merged
