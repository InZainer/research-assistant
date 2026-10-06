"""Векторный индекс фрагментов: матрица в .npz и поиск перебором.

Для учебной библиотеки (тысячи–десятки тысяч фрагментов) перебор по нормированной матрице занимает
миллисекунды, отдельная векторная БД не нужна. Файл свой для каждой модели эмбеддингов.
"""

import re
from pathlib import Path

import numpy as np


def index_path(directory: Path, model: str) -> Path:
    return directory / f"{re.sub(r'[^A-Za-z0-9._-]+', '_', model)}.npz"


class VectorIndex:
    def __init__(self, path: Path, ids: np.ndarray | None = None, matrix: np.ndarray | None = None):
        self.path = path
        self.ids = ids if ids is not None else np.zeros(0, dtype=np.int64)
        self.matrix = matrix if matrix is not None else np.zeros((0, 0), dtype=np.float32)

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        if not path.exists():
            return cls(path)
        with np.load(path) as data:
            return cls(path, data["ids"], data["matrix"])

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.path, ids=self.ids, matrix=self.matrix)

    def __len__(self) -> int:
        return len(self.ids)

    def missing(self, ids: list[int]) -> list[int]:
        present = set(self.ids.tolist())
        return [i for i in ids if i not in present]

    def keep_only(self, ids: set[int]) -> int:
        """Убирает векторы фрагментов, которых больше нет (статью переразобрали). Возвращает число удалённых."""
        mask = np.isin(self.ids, np.fromiter(ids, dtype=np.int64, count=len(ids)))
        removed = int((~mask).sum())
        self.ids, self.matrix = self.ids[mask], self.matrix[mask] if len(self.matrix) else self.matrix
        return removed

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        if not ids:
            return
        if len(self.ids) == 0:
            self.ids, self.matrix = np.asarray(ids, dtype=np.int64), vectors.astype(np.float32)
            return
        if vectors.shape[1] != self.matrix.shape[1]:
            raise ValueError(f"размерность {vectors.shape[1]} не совпадает с индексом {self.matrix.shape[1]}")
        self.ids = np.concatenate([self.ids, np.asarray(ids, dtype=np.int64)])
        self.matrix = np.vstack([self.matrix, vectors.astype(np.float32)])

    def search(self, query: np.ndarray, k: int, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        if len(self.ids) == 0:
            return []
        scores = self.matrix @ query.astype(np.float32)
        if allowed is not None:
            mask = np.isin(self.ids, np.fromiter(allowed, dtype=np.int64, count=len(allowed)))
            scores = np.where(mask, scores, -np.inf)
        top = np.argsort(-scores)[:k]
        return [(int(self.ids[i]), float(scores[i])) for i in top if np.isfinite(scores[i])]
