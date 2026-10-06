"""Внешние источники статей. У каждого есть name и search(query, limit, year_from, year_to, language)."""

from typing import Protocol

from research_assistant.ingest.models import PaperRecord


class Source(Protocol):
    name: str

    def search(
        self,
        query: str,
        limit: int = 20,
        year_from: int | None = None,
        year_to: int | None = None,
        language: str | None = None,
    ) -> list[PaperRecord]: ...
