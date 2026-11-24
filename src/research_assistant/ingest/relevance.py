"""Отсев посторонних статей при сборе: в названии или аннотации должны быть слова темы.

Поиск OpenAlex ищет и по полному тексту, а русские слова не приводит к основе, поэтому на запрос
«искусственный интеллект ресторан» приходят учебник по интеллектуальным системам и статья про насосы.
Фильтр проверяет запись сам: слова сравниваются по основам (nlp/text.analyze), фраза — как слова подряд.
"""

import re

from research_assistant.ingest.models import PaperRecord
from research_assistant.nlp.entities import contains_stems
from research_assistant.nlp.text import analyze


class TermFilter:
    """groups — список групп синонимов. Запись подходит, если из КАЖДОЙ группы нашёлся хотя бы один термин:
    [["ресторан", "общественное питание"], ["искусственный интеллект", "нейросеть"]] — «общепит И ИИ»."""

    def __init__(self, groups: list[list[str]]):
        self.groups = [[stems for stems in (analyze(term) for term in group) if stems] for group in groups]
        self.groups = [group for group in self.groups if group]

    @classmethod
    def parse(cls, specs: list[str] | None) -> "TermFilter | None":
        """Каждая строка — одна группа: термины через «|» или запятую («ресторан|кафе|restaurant»)."""
        groups = [[t.strip() for t in re.split(r"[|,;]", spec) if t.strip()] for spec in specs or []]
        groups = [group for group in groups if group]
        return cls(groups) if groups else None

    def matches(self, record: PaperRecord) -> bool:
        text = analyze(f"{record.title}. {record.abstract or ''}")
        return all(any(contains_stems(text, term) for term in group) for group in self.groups)
