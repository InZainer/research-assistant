"""Библиотека статей в SQLite: темы, статьи, связи, состояние полных текстов."""

import json
import sqlite3
from pathlib import Path

from research_assistant.ingest.models import PaperRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    title_key TEXT,
    authors TEXT NOT NULL DEFAULT '[]',
    year INTEGER,
    abstract TEXT,
    doi TEXT UNIQUE,
    arxiv_id TEXT UNIQUE,
    openalex_id TEXT UNIQUE,
    venue TEXT,
    language TEXT,
    landing_url TEXT,
    pdf_url TEXT,
    sources TEXT NOT NULL DEFAULT '',
    fulltext_status TEXT NOT NULL DEFAULT 'pending',  -- pending | ok | failed | unavailable
    fulltext_kind TEXT,                               -- pdf | html
    fulltext_path TEXT,
    fulltext_error TEXT,
    added_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS papers_title_key ON papers(title_key);

CREATE TABLE IF NOT EXISTS paper_topics (
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    topic_id INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    query TEXT,
    added_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (paper_id, topic_id)
);
"""

KEY_COLUMNS = {"doi": "doi", "arxiv": "arxiv_id", "openalex": "openalex_id", "title": "title_key"}
UNIQUE_COLUMNS = ("doi", "arxiv_id", "openalex_id")
FILLABLE_COLUMNS = ("year", "doi", "arxiv_id", "openalex_id", "venue", "language", "landing_url", "pdf_url")


class Library:
    def __init__(self, path: str | Path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def get_or_create_topic(self, name: str) -> int:
        name = " ".join(name.split())
        row = self.conn.execute("SELECT id FROM topics WHERE name = ?", (name,)).fetchone()
        if row:
            return row["id"]
        with self.conn:
            return self.conn.execute("INSERT INTO topics(name) VALUES (?)", (name,)).lastrowid

    def find_paper(self, record: PaperRecord) -> int | None:
        for kind, value in record.keys():
            column = KEY_COLUMNS[kind]
            row = self.conn.execute(f"SELECT id FROM papers WHERE {column} = ?", (value,)).fetchone()
            if row:
                return row["id"]
        return None

    def upsert_paper(self, record: PaperRecord) -> tuple[int, bool]:
        """Добавляет статью или дополняет найденную по любому ключу. Возвращает (id, создана ли)."""
        paper_id = self.find_paper(record)
        with self.conn:
            if paper_id is None:
                cursor = self.conn.execute(
                    """INSERT INTO papers(title, title_key, authors, year, abstract, doi, arxiv_id, openalex_id,
                                          venue, language, landing_url, pdf_url, sources)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        record.title, record.title_key, json.dumps(record.authors, ensure_ascii=False), record.year,
                        record.abstract, record.doi, record.arxiv_id, record.openalex_id, record.venue,
                        record.language, record.landing_url, record.pdf_url, ",".join(record.sources),
                    ),
                )
                return cursor.lastrowid, True
            self._fill_missing(paper_id, record)
            return paper_id, False

    def _fill_missing(self, paper_id: int, record: PaperRecord) -> None:
        row = self.conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        updates: dict[str, object] = {}
        for column in FILLABLE_COLUMNS:
            value = getattr(record, column)
            if row[column] is not None or value is None:
                continue
            if column in UNIQUE_COLUMNS and self._owned_by_other(column, value, paper_id):
                continue  # редкий случай: ключ уже у другой статьи, не склеиваем три записи в одну
            updates[column] = value
        if record.abstract and len(record.abstract) > len(row["abstract"] or ""):
            updates["abstract"] = record.abstract
        if len(record.authors) > len(json.loads(row["authors"])):
            updates["authors"] = json.dumps(record.authors, ensure_ascii=False)
        sources = list(dict.fromkeys([s for s in row["sources"].split(",") if s] + record.sources))
        if ",".join(sources) != row["sources"]:
            updates["sources"] = ",".join(sources)
        if updates:
            assignments = ", ".join(f"{column} = ?" for column in updates)
            self.conn.execute(f"UPDATE papers SET {assignments} WHERE id = ?", (*updates.values(), paper_id))

    def _owned_by_other(self, column: str, value: object, paper_id: int) -> bool:
        row = self.conn.execute(f"SELECT id FROM papers WHERE {column} = ? AND id != ?", (value, paper_id)).fetchone()
        return row is not None

    def link_topic(self, paper_id: int, topic_id: int, query: str | None = None) -> bool:
        """Привязывает статью к теме. Возвращает False, если связь уже была."""
        with self.conn:
            cursor = self.conn.execute(
                "INSERT OR IGNORE INTO paper_topics(paper_id, topic_id, query) VALUES (?, ?, ?)",
                (paper_id, topic_id, query),
            )
        return cursor.rowcount == 1

    def get_paper(self, paper_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()

    def papers_for_fulltext(self, topic_id: int | None = None, statuses: tuple[str, ...] = ("pending",)) -> list[sqlite3.Row]:
        marks = ",".join("?" * len(statuses))
        if topic_id is None:
            sql = f"SELECT * FROM papers WHERE fulltext_status IN ({marks}) ORDER BY id"
            return self.conn.execute(sql, statuses).fetchall()
        sql = f"""SELECT p.* FROM papers p JOIN paper_topics pt ON pt.paper_id = p.id
                  WHERE pt.topic_id = ? AND p.fulltext_status IN ({marks}) ORDER BY p.id"""
        return self.conn.execute(sql, (topic_id, *statuses)).fetchall()

    def set_fulltext(
        self, paper_id: int, status: str, path: str | None = None, kind: str | None = None, error: str | None = None
    ) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE papers SET fulltext_status = ?, fulltext_path = ?, fulltext_kind = ?, fulltext_error = ? WHERE id = ?",
                (status, path, kind, error, paper_id),
            )

    def topic_summary(self) -> list[dict]:
        rows = self.conn.execute(
            """SELECT t.name, COUNT(pt.paper_id) AS papers,
                      SUM(p.fulltext_status = 'ok') AS fulltext,
                      SUM(p.language = 'ru') AS russian
               FROM topics t
               LEFT JOIN paper_topics pt ON pt.topic_id = t.id
               LEFT JOIN papers p ON p.id = pt.paper_id
               GROUP BY t.id ORDER BY t.created_at, t.id"""
        ).fetchall()
        return [{k: (row[k] or 0) if k != "name" else row[k] for k in row.keys()} for row in rows]

    def list_papers(self, topic: str | None = None, limit: int = 50) -> list[sqlite3.Row]:
        if topic is None:
            return self.conn.execute("SELECT * FROM papers ORDER BY id LIMIT ?", (limit,)).fetchall()
        return self.conn.execute(
            """SELECT p.* FROM papers p JOIN paper_topics pt ON pt.paper_id = p.id
               JOIN topics t ON t.id = pt.topic_id WHERE t.name = ? ORDER BY p.id LIMIT ?""",
            (" ".join(topic.split()), limit),
        ).fetchall()
