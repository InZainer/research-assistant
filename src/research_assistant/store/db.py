"""Библиотека статей в SQLite: темы, статьи, связи, состояние полных текстов."""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from research_assistant.ingest.models import PaperRecord
from research_assistant.nlp.text import analyze

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

CREATE TABLE IF NOT EXISTS parses (
    paper_id INTEGER PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,
    status TEXT NOT NULL,             -- ok | metadata | empty | failed
    parser TEXT,                      -- pdf | html | metadata
    parser_version INTEGER NOT NULL,
    has_abstract INTEGER NOT NULL DEFAULT 0,
    n_pages INTEGER,
    error TEXT,
    parsed_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sections (
    id INTEGER PRIMARY KEY,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    order_idx INTEGER NOT NULL,
    title TEXT NOT NULL,
    kind TEXT NOT NULL,
    level INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS passages (
    id INTEGER PRIMARY KEY,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    section_id INTEGER NOT NULL REFERENCES sections(id) ON DELETE CASCADE,
    order_idx INTEGER NOT NULL,
    page INTEGER,
    kind TEXT NOT NULL,               -- text | caption
    source TEXT NOT NULL,             -- fulltext | metadata
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS passages_paper ON passages(paper_id, order_idx);

-- Цитата = sentences.text; это ровно passages.text[char_start:char_end].
CREATE TABLE IF NOT EXISTS sentences (
    id INTEGER PRIMARY KEY,
    passage_id INTEGER NOT NULL REFERENCES passages(id) ON DELETE CASCADE,
    order_idx INTEGER NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    page INTEGER,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS sentences_passage ON sentences(passage_id, order_idx);

-- Полнотекстовый индекс по основам слов (nlp/text.py), rowid = passages.id.
-- Колонки: текст фрагмента, заголовок раздела, заголовок статьи — с разными весами в BM25.
CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5(body, section, title, tokenize = 'unicode61');

-- Разметка: сущности статьи (метод, датасет, метрика, задача), их упоминания во фрагментах, ключевые слова.
-- key — основы слов (nlp/text.py): «диагностики» и «диагностика» дают один ключ.
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    key TEXT NOT NULL,
    UNIQUE (paper_id, type, key)
);
CREATE INDEX IF NOT EXISTS entities_key ON entities(type, key);

CREATE TABLE IF NOT EXISTS entity_mentions (
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    passage_id INTEGER NOT NULL REFERENCES passages(id) ON DELETE CASCADE,
    PRIMARY KEY (entity_id, passage_id)
);
CREATE INDEX IF NOT EXISTS entity_mentions_passage ON entity_mentions(passage_id);

CREATE TABLE IF NOT EXISTS keywords (
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    keyword TEXT NOT NULL,
    score REAL NOT NULL,
    PRIMARY KEY (paper_id, keyword)
);

CREATE TABLE IF NOT EXISTS annotations (
    paper_id INTEGER PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,
    status TEXT NOT NULL,             -- ok | failed
    annotator TEXT,                   -- модель
    version INTEGER NOT NULL,
    dropped INTEGER NOT NULL DEFAULT 0,  -- сколько сущностей отброшено: нет в тексте
    error TEXT,
    annotated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

# Веса колонок passages_fts в BM25: текст важнее заголовков.
BM25_WEIGHTS = (1.0, 0.5, 0.3)


@dataclass
class PassageFilter:
    """Ограничения поиска. Пустые поля не ограничивают."""

    topic: str | None = None
    year_from: int | None = None
    year_to: int | None = None
    section_kinds: list[str] | None = None
    language: str | None = None
    paper_ids: list[int] | None = None
    include_captions: bool = True
    entity: str | None = None  # фрагменты, где упоминается сущность (по основам слов, допускается часть: «eagle» → «EAGLE-2»)

    def sql(self) -> tuple[str, list]:
        """Условие WHERE над алиасами p (passages), s (sections), pa (papers)."""
        conditions, params = [], []
        if self.topic:
            conditions.append(
                "p.paper_id IN (SELECT pt.paper_id FROM paper_topics pt JOIN topics t ON t.id = pt.topic_id"
                " WHERE t.name = ?)"
            )
            params.append(" ".join(self.topic.split()))
        if self.year_from:
            conditions.append("pa.year >= ?")
            params.append(self.year_from)
        if self.year_to:
            conditions.append("pa.year <= ?")
            params.append(self.year_to)
        if self.section_kinds:
            conditions.append(f"s.kind IN ({','.join('?' * len(self.section_kinds))})")
            params.extend(self.section_kinds)
        if self.language:
            conditions.append("pa.language = ?")
            params.append(self.language)
        if self.paper_ids:
            conditions.append(f"p.paper_id IN ({','.join('?' * len(self.paper_ids))})")
            params.extend(self.paper_ids)
        if not self.include_captions:
            conditions.append("p.kind = 'text'")
        if self.entity:
            key = " ".join(analyze(self.entity))
            conditions.append(
                "p.id IN (SELECT m.passage_id FROM entity_mentions m JOIN entities e ON e.id = m.entity_id"
                " WHERE e.key = ? OR (' ' || e.key || ' ') LIKE ?)"
            )
            params.extend([key, f"% {key} %"])
        return (" AND ".join(conditions) or "1"), params


PASSAGE_JOIN = "passages p JOIN sections s ON s.id = p.section_id JOIN papers pa ON pa.id = p.paper_id"

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
        self._ensure_fts()

    def _ensure_fts(self) -> None:
        """Достраивает полнотекстовый индекс для фрагментов, разобранных до его появления."""
        passages = self.conn.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
        indexed = self.conn.execute("SELECT COUNT(*) FROM passages_fts").fetchone()[0]
        if passages != indexed:
            self.rebuild_fts()

    def rebuild_fts(self) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM passages_fts")
            rows = self.conn.execute(
                f"SELECT p.id, p.text, s.title AS section, pa.title FROM {PASSAGE_JOIN}"
            ).fetchall()
            self.conn.executemany(
                "INSERT INTO passages_fts(rowid, body, section, title) VALUES (?, ?, ?, ?)",
                [(r["id"], *_fts_columns(r["text"], r["section"], r["title"])) for r in rows],
            )

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

    def papers_to_parse(
        self, version: int, topic_id: int | None = None, paper_ids: list[int] | None = None, force: bool = False
    ) -> list[sqlite3.Row]:
        """Статьи без разбора или разобранные старой версией парсера (или все при force)."""
        conditions, params = [], []
        if not force:
            conditions.append("(r.paper_id IS NULL OR r.parser_version < ?)")
            params.append(version)
        if topic_id is not None:
            conditions.append("p.id IN (SELECT paper_id FROM paper_topics WHERE topic_id = ?)")
            params.append(topic_id)
        if paper_ids:
            conditions.append(f"p.id IN ({','.join('?' * len(paper_ids))})")
            params.extend(paper_ids)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT p.* FROM papers p LEFT JOIN parses r ON r.paper_id = p.id {where} ORDER BY p.id"
        return self.conn.execute(sql, params).fetchall()

    def save_structure(
        self,
        paper_id: int,
        sections,
        *,
        status: str,
        parser: str | None,
        version: int,
        source: str = "fulltext",
        has_abstract: bool = False,
        n_pages: int | None = None,
        error: str | None = None,
    ) -> dict[str, int]:
        """Заменяет разбор статьи целиком. sections — список SectionOut из parse.passages."""
        counts = {"sections": 0, "passages": 0, "sentences": 0}
        paper_title = self.conn.execute("SELECT title FROM papers WHERE id = ?", (paper_id,)).fetchone()["title"]
        with self.conn:
            self.conn.execute(
                "DELETE FROM passages_fts WHERE rowid IN (SELECT id FROM passages WHERE paper_id = ?)", (paper_id,)
            )
            self.conn.execute("DELETE FROM sections WHERE paper_id = ?", (paper_id,))
            passage_idx = 0
            for section_idx, section in enumerate(sections):
                section_id = self.conn.execute(
                    "INSERT INTO sections(paper_id, order_idx, title, kind, level) VALUES (?, ?, ?, ?, ?)",
                    (paper_id, section_idx, section.title, section.kind, section.level),
                ).lastrowid
                counts["sections"] += 1
                for passage in section.passages:
                    passage_id = self.conn.execute(
                        """INSERT INTO passages(paper_id, section_id, order_idx, page, kind, source, text)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (paper_id, section_id, passage_idx, passage.page, passage.kind, source, passage.text),
                    ).lastrowid
                    passage_idx += 1
                    self.conn.execute(
                        "INSERT INTO passages_fts(rowid, body, section, title) VALUES (?, ?, ?, ?)",
                        (passage_id, *_fts_columns(passage.text, section.title, paper_title)),
                    )
                    self.conn.executemany(
                        """INSERT INTO sentences(passage_id, order_idx, char_start, char_end, page, text)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        [
                            (passage_id, i, s.start, s.end, s.page, passage.text[s.start:s.end])
                            for i, s in enumerate(passage.sentences)
                        ],
                    )
                    counts["passages"] += 1
                    counts["sentences"] += len(passage.sentences)
            self.conn.execute(
                """INSERT OR REPLACE INTO parses(paper_id, status, parser, parser_version, has_abstract, n_pages, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (paper_id, status, parser, version, int(has_abstract), n_pages, error),
            )
            # Переразбор удалил фрагменты, а с ними упоминания сущностей: восстанавливаем по новым фрагментам.
            self._link_mentions(paper_id)
        return counts

    def paper_structure(self, paper_id: int) -> list[dict]:
        """Разделы статьи с фрагментами и предложениями — для просмотра и проверки разбора."""
        sections = []
        for section in self.conn.execute(
            "SELECT * FROM sections WHERE paper_id = ? ORDER BY order_idx", (paper_id,)
        ).fetchall():
            passages = []
            for passage in self.conn.execute(
                "SELECT * FROM passages WHERE section_id = ? ORDER BY order_idx", (section["id"],)
            ).fetchall():
                sentences = self.conn.execute(
                    "SELECT * FROM sentences WHERE passage_id = ? ORDER BY order_idx", (passage["id"],)
                ).fetchall()
                passages.append({**dict(passage), "sentences": [dict(s) for s in sentences]})
            sections.append({**dict(section), "passages": passages})
        return sections

    def search_fts(self, query: str, k: int, where: PassageFilter | None = None) -> list[tuple[int, float]]:
        """BM25 по основам слов. Слова запроса объединяются через OR: BM25 сам поднимет фрагменты,
        где совпало больше редких слов. Возвращает (id фрагмента, оценка), лучшие первыми."""
        terms = list(dict.fromkeys(analyze(query)))
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)
        condition, params = (where or PassageFilter()).sql()
        weights = ", ".join(str(w) for w in BM25_WEIGHTS)
        rows = self.conn.execute(
            f"""SELECT p.id, bm25(passages_fts, {weights}) AS rank
                FROM passages_fts JOIN passages p ON p.id = passages_fts.rowid
                JOIN sections s ON s.id = p.section_id JOIN papers pa ON pa.id = p.paper_id
                WHERE passages_fts MATCH ? AND {condition}
                ORDER BY rank LIMIT ?""",
            [match, *params, k],
        ).fetchall()
        return [(row["id"], -row["rank"]) for row in rows]

    def passage_ids(self, where: PassageFilter | None = None) -> set[int]:
        condition, params = (where or PassageFilter()).sql()
        return {r[0] for r in self.conn.execute(f"SELECT p.id FROM {PASSAGE_JOIN} WHERE {condition}", params)}

    def passages_for_embedding(self, ids: list[int] | None = None) -> list[sqlite3.Row]:
        """Текст для эмбеддинга: заголовок статьи и раздела добавляют фрагменту контекст."""
        sql = f"SELECT p.id, p.text, s.title AS section, pa.title FROM {PASSAGE_JOIN}"
        if ids is None:
            return self.conn.execute(sql + " ORDER BY p.id").fetchall()
        rows = []
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            rows += self.conn.execute(sql + f" WHERE p.id IN ({','.join('?' * len(chunk))})", chunk).fetchall()
        return sorted(rows, key=lambda r: r["id"])

    def passages_by_ids(self, ids: list[int]) -> dict[int, dict]:
        """Фрагменты с данными статьи и раздела — для выдачи результатов поиска."""
        if not ids:
            return {}
        rows = self.conn.execute(
            f"""SELECT p.id, p.text, p.page, p.kind, p.source, p.paper_id, s.title AS section, s.kind AS section_kind,
                       pa.title, pa.year, pa.language, pa.authors
                FROM {PASSAGE_JOIN} WHERE p.id IN ({','.join('?' * len(ids))})""",
            ids,
        ).fetchall()
        return {r["id"]: dict(r) for r in rows}

    def save_annotation(
        self,
        paper_id: int,
        entities: dict[str, list[str]],
        keywords: list[tuple[str, float]],
        *,
        status: str,
        annotator: str | None,
        version: int,
        dropped: int = 0,
        error: str | None = None,
    ) -> int:
        """Заменяет разметку статьи. Возвращает число упоминаний сущностей во фрагментах."""
        with self.conn:
            self.conn.execute("DELETE FROM entities WHERE paper_id = ?", (paper_id,))
            self.conn.execute("DELETE FROM keywords WHERE paper_id = ?", (paper_id,))
            for entity_type, names in entities.items():
                for name in names:
                    key = " ".join(analyze(name))
                    if key:
                        self.conn.execute(
                            "INSERT OR IGNORE INTO entities(paper_id, type, name, key) VALUES (?, ?, ?, ?)",
                            (paper_id, entity_type, name, key),
                        )
            self.conn.executemany(
                "INSERT OR IGNORE INTO keywords(paper_id, keyword, score) VALUES (?, ?, ?)",
                [(paper_id, keyword, score) for keyword, score in keywords],
            )
            mentions = self._link_mentions(paper_id)
            self.conn.execute(
                """INSERT OR REPLACE INTO annotations(paper_id, status, annotator, version, dropped, error)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (paper_id, status, annotator, version, dropped, error),
            )
        return mentions

    def _link_mentions(self, paper_id: int) -> int:
        """Привязывает сущности статьи ко всем её фрагментам, где они встречаются (основы слов подряд)."""
        from research_assistant.nlp.entities import contains_stems

        entities = self.conn.execute("SELECT id, key FROM entities WHERE paper_id = ?", (paper_id,)).fetchall()
        if not entities:
            return 0
        self.conn.execute(
            "DELETE FROM entity_mentions WHERE entity_id IN (SELECT id FROM entities WHERE paper_id = ?)", (paper_id,)
        )
        passages = self.conn.execute("SELECT id, text FROM passages WHERE paper_id = ?", (paper_id,)).fetchall()
        stems = {row["id"]: analyze(row["text"]) for row in passages}
        pairs = [
            (entity["id"], passage_id)
            for entity in entities
            for passage_id, passage_stems in stems.items()
            if contains_stems(passage_stems, entity["key"].split())
        ]
        self.conn.executemany("INSERT OR IGNORE INTO entity_mentions(entity_id, passage_id) VALUES (?, ?)", pairs)
        return len(pairs)

    def papers_to_annotate(
        self, version: int, topic_id: int | None = None, paper_ids: list[int] | None = None, force: bool = False
    ) -> list[sqlite3.Row]:
        """Разобранные статьи без разметки или с разметкой старой версии (или все при force)."""
        conditions, params = ["r.status IN ('ok', 'metadata')"], []
        if not force:
            conditions.append("(a.paper_id IS NULL OR a.version < ? OR a.status = 'failed')")  # сбой — повторить
            params.append(version)
        if topic_id is not None:
            conditions.append("p.id IN (SELECT paper_id FROM paper_topics WHERE topic_id = ?)")
            params.append(topic_id)
        if paper_ids:
            conditions.append(f"p.id IN ({','.join('?' * len(paper_ids))})")
            params.extend(paper_ids)
        sql = f"""SELECT p.* FROM papers p JOIN parses r ON r.paper_id = p.id
                  LEFT JOIN annotations a ON a.paper_id = p.id WHERE {' AND '.join(conditions)} ORDER BY p.id"""
        return self.conn.execute(sql, params).fetchall()

    def paper_entities(self, paper_id: int) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for row in self.conn.execute("SELECT type, name FROM entities WHERE paper_id = ? ORDER BY id", (paper_id,)):
            result.setdefault(row["type"], []).append(row["name"])
        return result

    def paper_keywords(self, paper_id: int, limit: int = 10) -> list[str]:
        rows = self.conn.execute(
            "SELECT keyword FROM keywords WHERE paper_id = ? ORDER BY score LIMIT ?", (paper_id, limit)
        )
        return [r["keyword"] for r in rows]

    def entity_stats(self, entity_type: str | None = None, topic: str | None = None, limit: int = 20) -> list[dict]:
        """Сущности по числу статей: какие датасеты/методы/метрики встречаются в библиотеке или теме."""
        conditions, params = [], []
        if entity_type:
            conditions.append("e.type = ?")
            params.append(entity_type)
        if topic:
            conditions.append(
                "e.paper_id IN (SELECT pt.paper_id FROM paper_topics pt JOIN topics t ON t.id = pt.topic_id WHERE t.name = ?)"
            )
            params.append(" ".join(topic.split()))
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self.conn.execute(
            f"""SELECT e.type, e.key, MIN(e.name) AS name, COUNT(DISTINCT e.paper_id) AS papers,
                       GROUP_CONCAT(DISTINCT e.paper_id) AS paper_ids,
                       (SELECT COUNT(*) FROM entity_mentions m JOIN entities e2 ON e2.id = m.entity_id
                        WHERE e2.type = e.type AND e2.key = e.key) AS mentions
                FROM entities e {where}
                GROUP BY e.type, e.key ORDER BY papers DESC, mentions DESC LIMIT ?""",
            [*params, limit],
        ).fetchall()
        return [dict(r) for r in rows]

    def get_parse(self, paper_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM parses WHERE paper_id = ?", (paper_id,)).fetchone()

    def list_papers(self, topic: str | None = None, limit: int = 50) -> list[sqlite3.Row]:
        if topic is None:
            return self.conn.execute("SELECT * FROM papers ORDER BY id LIMIT ?", (limit,)).fetchall()
        return self.conn.execute(
            """SELECT p.* FROM papers p JOIN paper_topics pt ON pt.paper_id = p.id
               JOIN topics t ON t.id = pt.topic_id WHERE t.name = ? ORDER BY p.id LIMIT ?""",
            (" ".join(topic.split()), limit),
        ).fetchall()


def _fts_columns(text: str, section: str | None, title: str | None) -> tuple[str, str, str]:
    return " ".join(analyze(text)), " ".join(analyze(section or "")), " ".join(analyze(title or ""))
