"""Поиск по библиотеке: BM25 (FTS5), плотный поиск (эмбеддинги) и их слияние Reciprocal Rank Fusion.

RRF складывает не оценки (у BM25 и косинуса разные шкалы), а места в списках: 1 / (60 + место).
Фрагмент, который высоко в обоих списках, поднимается выше фрагмента, высокого только в одном.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from research_assistant.store.db import Library, PassageFilter
from research_assistant.store.vectors import VectorIndex

RRF_K = 60
CANDIDATES = 50  # сколько кандидатов берём из каждого списка перед слиянием
MODES = ("bm25", "dense", "hybrid")
EMBED_BATCH = 64


@dataclass
class Hit:
    passage_id: int
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # место в каждом списке, с 1
    passage: dict = field(default_factory=dict)


def rrf(rankings: dict[str, list[int]], k: int = RRF_K) -> list[tuple[int, float, dict[str, int]]]:
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    for name, ids in rankings.items():
        for place, passage_id in enumerate(ids, 1):
            scores[passage_id] = scores.get(passage_id, 0.0) + 1.0 / (k + place)
            ranks.setdefault(passage_id, {})[name] = place
    ordered = sorted(scores, key=lambda i: (-scores[i], min(ranks[i].values())))
    return [(i, scores[i], ranks[i]) for i in ordered]


def embedding_text(row) -> str:
    """Что эмбеддим: заголовок статьи и раздела дают фрагменту контекст («о чём эта статья»)."""
    head = " — ".join(part for part in (row["title"], row["section"]) if part)
    return f"{head}\n\n{row['text']}" if head else row["text"]


class Searcher:
    def __init__(self, library: Library, embedder=None, index: VectorIndex | None = None):
        self.library = library
        self.embedder = embedder
        self.index = index

    @property
    def dense_ready(self) -> bool:
        return self.embedder is not None and self.index is not None and len(self.index) > 0

    def bm25(self, query: str, k: int, where: PassageFilter | None = None) -> list[tuple[int, float]]:
        return self.library.search_fts(query, k, where)

    def dense(self, query: str, k: int, where: PassageFilter | None = None) -> list[tuple[int, float]]:
        if not self.dense_ready:
            raise RuntimeError("плотный поиск не настроен: задайте EMBED_MODEL и выполните index")
        allowed = self.library.passage_ids(where) if where is not None and where.sql()[0] != "1" else None
        return self.index.search(self.embedder.embed_query(query), k, allowed)

    def search(
        self, query: str, k: int = 10, mode: str = "hybrid", where: PassageFilter | None = None
    ) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"неизвестный режим {mode}; доступны {', '.join(MODES)}")
        if mode == "hybrid" and not self.dense_ready:
            mode = "bm25"  # без эмбеддингов гибрид вырождается в BM25
        if mode == "hybrid":
            rankings = {
                "bm25": [i for i, _ in self.bm25(query, CANDIDATES, where)],
                "dense": [i for i, _ in self.dense(query, CANDIDATES, where)],
            }
            fused = rrf(rankings)[:k]
            hits = [Hit(i, score, ranks) for i, score, ranks in fused]
        else:
            ranked = self.bm25(query, k, where) if mode == "bm25" else self.dense(query, k, where)
            hits = [Hit(i, score, {mode: place}) for place, (i, score) in enumerate(ranked, 1)]
        passages = self.library.passages_by_ids([h.passage_id for h in hits])
        for hit in hits:
            hit.passage = passages.get(hit.passage_id, {})
        return hits


def build_index(
    library: Library, embedder, index: VectorIndex, log: Callable[[str], None] = lambda message: None
) -> dict[str, int]:
    """Досчитывает эмбеддинги для новых фрагментов и выкидывает векторы удалённых. Сохраняет после
    каждой пачки, поэтому прерванное построение продолжается с места остановки."""
    all_ids = [row["id"] for row in library.conn.execute("SELECT id FROM passages ORDER BY id")]
    removed = index.keep_only(set(all_ids))
    missing = index.missing(all_ids)
    log(f"Фрагментов {len(all_ids)}, уже в индексе {len(all_ids) - len(missing)}, считать {len(missing)}")
    for start in range(0, len(missing), EMBED_BATCH):
        rows = library.passages_for_embedding(missing[start:start + EMBED_BATCH])
        vectors = embedder.embed_documents([embedding_text(r) for r in rows])
        index.add([r["id"] for r in rows], vectors)
        index.save()
        log(f"  {min(start + EMBED_BATCH, len(missing))}/{len(missing)}")
    if removed and not missing:
        index.save()
    return {"total": len(all_ids), "added": len(missing), "removed": removed}
