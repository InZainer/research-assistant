> generated_by: nexus-mapper v2
> verified_at: 2026-10-22
> provenance: рёбра систем посчитаны по imports из raw/ast_nodes.json (AST-backed); последовательность вызовов в ask — inferred from manual inspection of cli.py, agent/session.py, agent/loop.py

# Зависимости

## Системы (по импортам AST, число — количество рёбер imports)

```mermaid
graph TD
    CLI["cli.py"] -->|14| STORE["Хранилище store/"]
    CLI -->|6| INGEST["Сбор ingest/"]
    CLI -->|5| EVAL["Оценка evaluation/"]
    CLI -->|6| AGENT["Агент agent/ + answer/"]
    CLI --> PARSE["Разбор parse/"]
    CLI --> ANNOT["Разметка nlp/annotate,entities,keywords"]
    CLI --> RETR["Поиск retrieval/ + nlp/embed"]

    AGENT --> RETR
    AGENT --> STORE
    AGENT -->|session.py| INGEST
    AGENT -->|session.py| PARSE
    AGENT -->|session.py| ANNOT

    EVAL --> AGENT
    EVAL --> STORE
    EVAL --> ANNOT

    INGEST --> STORE
    PARSE --> STORE
    ANNOT --> STORE
    RETR --> STORE
    RETR -.->|parse_json| ANNOT
    STORE -.->|contains_stems, ленивый импорт| ANNOT

    STORE --> TEXT["nlp/text.py анализатор"]
    RETR --> TEXT
    ANNOT --> TEXT
    STORE --> MODELS["ingest/models.py PaperRecord"]
    INGEST --> MODELS
    PARSE --> MODELS
```

Пунктир — зависимости «вверх»: нижние слои (хранилище, поиск) берут утилиты из модуля разметки `nlp/entities.py`. Цикла на уровне файлов нет (`nlp/entities.py` не импортирует `store`), но на уровне каталогов `store ↔ nlp` замкнуто через `nlp/annotate.py → store/db.py`. Кандидат на рефакторинг: вынести `contains_stems` и `parse_json` в `nlp/text.py` или отдельный `nlp/util.py`.

## Поток команды `ask` (inferred from manual inspection)

```mermaid
sequenceDiagram
    participant U as Пользователь
    participant C as cli.py (cmd_ask)
    participant S as agent/session.py
    participant L as agent/loop.py
    participant T as agent/tools.py
    participant R as retrieval/search.py
    participant D as store/db.py
    participant M as LLM (LM Studio)
    U->>C: research-assistant ask "вопрос"
    C->>S: make_agent(settings, library)
    S-->>C: ToolContext, ChatModel
    C->>L: run_agent(prefetch=True)
    L->>T: search_library(вопрос, k=6)
    T->>R: Searcher.search(hybrid)
    R->>D: search_fts + VectorIndex
    L->>M: сообщения + инструменты
    loop до final_answer или бюджета (6 вызовов, 12 шагов)
        M-->>L: tool_calls
        L->>T: инструмент → наблюдение (ошибка тоже наблюдение)
    end
    M-->>L: final_answer(claims, sentence_ids)
    L->>D: load_citations (проверка: ID видел агент)
    L-->>C: AgentResult + журнал runs/*.jsonl
```

## Внешние зависимости (pyproject.toml)
openai, httpx, protego, pymupdf, beautifulsoup4, razdel, snowballstemmer, numpy, yake, python-dotenv. Внешние сервисы: OpenAlex API, arXiv API и arxiv.org/pdf, страницы cyberleninka.ru, OpenAI-совместимый сервер модели и эмбеддингов (по `.env`).
