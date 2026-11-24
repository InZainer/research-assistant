"""Эмбеддинги через OpenAI-совместимый /v1/embeddings: LM Studio локально или облачный сервис.

Часть моделей обучена с префиксами запроса и документа и без них работает заметно хуже.
"""

import numpy as np

# (подстрока в имени модели, префикс запроса, префикс документа)
PREFIXES = [
    ("nomic-embed", "search_query: ", "search_document: "),
    # e5-instruct — другая схема, чем у обычных e5: инструкция у запроса, документы без префикса. Правило стоит
    # выше общего «e5-», потому что берётся первое совпадение.
    ("e5-large-instruct", "Instruct: Given a question, retrieve passages from scientific papers that answer it\nQuery: ", ""),
    ("e5-", "query: ", "passage: "),
    # snowflake-arctic-embed v2: префикс только у запроса (карточка модели).
    ("arctic-embed", "query: ", ""),
    # FRIDA (ai-forever): префиксы задач поиска из карточки модели.
    ("frida", "search_query: ", "search_document: "),
    # Giga-Embeddings-instruct (Сбер): инструкция у запроса, документы без префикса.
    ("giga-embeddings", "Instruct: Given a question, retrieve passages from scientific papers that answer it\nQuery: ", ""),
    # Qwen3-Embedding: инструкция только у запроса, документы без префикса (так в карточке модели).
    ("qwen3-embedding", "Instruct: Given a question, retrieve passages from scientific papers that answer it\nQuery: ", ""),
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

            # Локальный сервер (LM Studio) либо запущен, либо нет: повторы с паузами только задерживают ответ
            # агента на десятки секунд, когда он выключен.
            local = any(host in (base_url or "") for host in ("localhost", "127.0.0.1"))
            client = OpenAI(base_url=base_url, api_key=api_key or "not-needed", max_retries=0 if local else 2)
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
