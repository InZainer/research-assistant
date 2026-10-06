"""Эмбеддинги через OpenAI-совместимый /v1/embeddings: LM Studio локально или облачный сервис.

Часть моделей обучена с префиксами запроса и документа и без них работает заметно хуже.
"""

import numpy as np

# (подстрока в имени модели, префикс запроса, префикс документа)
PREFIXES = [
    ("nomic-embed", "search_query: ", "search_document: "),
    ("e5-", "query: ", "passage: "),
]
BATCH_SIZE = 32


def prefixes_for(model: str) -> tuple[str, str]:
    name = model.casefold()
    for marker, query, document in PREFIXES:
        if marker in name:
            return query, document
    return "", ""


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class OpenAIEmbedder:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, client=None):
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
        self.client = client
        self.model = model
        self.query_prefix, self.document_prefix = prefixes_for(model)

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for start in range(0, len(texts), BATCH_SIZE):
            response = self.client.embeddings.create(model=self.model, input=texts[start:start + BATCH_SIZE])
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return normalize(np.asarray(vectors, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed([self.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([self.query_prefix + text])[0]
