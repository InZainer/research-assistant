> generated_by: nexus-mapper v2
> verified_at: 2026-11-24
> provenance: рёбра систем посчитаны по imports из raw/ast_nodes.json (AST-backed, 2026-11-24; модуль отнесён к системе по пути, `config.py`/`providers.py` — к CLI, `nlp/embed.py` — к поиску); связи `cli → ui` (запуск), `scripts/ → finetune/` и внешние клиенты MCP — не import (inferred from manual inspection); последовательности — inferred from manual inspection of cli.py, agent/loop.py, mcp_server.py, finetune/*.py

# Зависимости

## Системы (по импортам AST, число — количество рёбер imports между модулями)

```mermaid
graph TD
    CLI["cli.py + config + providers"] -->|17| STORE["Хранилище store/"]
    CLI -->|15| AGENT["Агент agent/ + answer/ + llm.py"]
    CLI -->|13| INGEST["Сбор ingest/"]
    CLI -->|5| EVAL["Оценка evaluation/"]
    CLI -->|3| PARSE["Разбор parse/"]
    CLI -->|3| RETR["Поиск retrieval/ + nlp/embed"]
    CLI -->|3, ленивый импорт| FT["Датасет дообучения finetune/"]
    CLI -->|2| ANNOT["Разметка nlp/annotate,entities,keywords"]
    CLI -->|2, команда mcp| MCP["MCP mcp_server.py"]
    CLI -.->|streamlit run, подпроцесс; import только views.friendly_error| UI["Веб-интерфейс ui/"]

    EXT["Claude Code / Hermes / Codex / Cursor"] -.->|stdio JSON-RPC| MCP
    MCP -->|11| AGENT
    MCP -->|5, только --allow-write| INGEST
    MCP --> RETR
    MCP --> STORE
    MCP -.->|ленивый cli.format_agent_result| CLI

    UI -->|5| AGENT
    UI -->|4| INGEST
    UI -->|2| RETR
    UI -->|2| STORE

    AGENT -->|9| RETR
    AGENT -->|6| STORE
    AGENT -->|6| INGEST
    AGENT -->|session.py| PARSE
    AGENT -->|session.py| ANNOT

    EVAL -->|5, в т.ч. приватный _call_key| AGENT
    EVAL -->|2| STORE
    EVAL --> ANNOT

    FT -->|13| AGENT
    FT -->|7| EVAL
    FT -->|6| STORE
    FT --> ANNOT
    FT --> PARSE

    BUNDLE["Обучение scripts/ + notebooks/"] -.->|файлы: sft/*.jsonl, чат-шаблон, prompt_hash| FT

    INGEST --> STORE
    PARSE --> STORE
    ANNOT --> STORE
    RETR -->|3| STORE
    RETR -.->|parse_json| ANNOT
    STORE -.->|contains_stems, ленивый импорт| ANNOT

    STORE --> TEXT["nlp/text.py анализатор"]
    RETR --> TEXT
    ANNOT --> TEXT
    FT --> TEXT
    STORE --> MODELS["ingest/models.py PaperRecord"]
    INGEST --> MODELS
    PARSE --> MODELS
```

Пунктир — не обычный нисходящий import:
- `store/db.py` и `retrieval/rerank.py` берут утилиты из модуля разметки `nlp/entities.py` (зависимость «вверх»). Цикла на уровне файлов нет; на уровне каталогов `store ↔ nlp` замкнут через `nlp/annotate.py → store/db.py`. Кандидат на рефакторинг: вынести `contains_stems`/`parse_json` в `nlp/text.py` (радиус: db, rerank, finetune/questions).
- `mcp_server.py → cli.py`: ленивый импорт `format_agent_result` внутри `ask_agent` — библиотечный модуль зависит от слоя CLI. Цикла импорта нет (`cli` импортирует `mcp_server` тоже лениво, в `main`). Кандидат: перенести форматирование ответа в `answer/`.
- `cli.py → ui/`: интерфейс запускается подпроцессом `python -m streamlit run ui/app.py`; прямой import — только `ui.views.friendly_error` для `check`.
- `scripts/ → finetune/`: импорта нет. Связь через данные `data/finetune/sft/*.jsonl` (`text`, `train_spans`, `meta.prompt_hash`) и файл `finetune/qwen3_5_chat_template.jinja`.

Fan-in (AST): `store/db.py` 38, `ingest/models.py` 36, `nlp/text.py` 20, `parse/structure.py` 18, `retrieval/search.py` 17, `llm.py` 17, `agent/tools.py` 17, `parse/passages.py` 17. Fan-out: `cli.py` 29, `agent/session.py` 16, `mcp_server.py` 16, `ui/app.py` 13. Счёт fan-in включает тесты.

## Поток команды `ask`, UI и MCP `ask_agent` (inferred from manual inspection)

```mermaid
sequenceDiagram
    participant U as Пользователь
    participant C as cli.py (cmd_ask) / ui/app.py / mcp_server.ask_agent
    participant S as agent/session.py
    participant L as agent/loop.py
    participant T as agent/tools.py
    participant R as retrieval/search.py
    participant D as store/db.py
    participant M as LLM (LLM_PROVIDER из .env)
    U->>C: вопрос
    C->>S: make_agent / make_searcher(settings, library)
    S-->>C: ToolContext, ChatModel
    C->>L: run_agent(prefetch=True, **run_options)
    L->>T: search_library(вопрос)
    T->>R: Searcher.search(hybrid)
    R->>D: search_fts + VectorIndex (нет эмбеддингов → BM25 + last_warning)
    L->>M: сообщения + инструменты
    loop до final_answer или бюджета
        M-->>L: tool_calls
        L->>T: инструмент → наблюдение (повтор → REPEATED_CALL, ошибка → error_kind)
    end
    M-->>L: final_answer(claims, sentence_ids)
    L->>D: load_citations (проверка: ID видел агент)
    L-->>C: AgentResult и журнал runs/*.jsonl
```

## Внешний агент через MCP (inferred from manual inspection of mcp_server.py)

```mermaid
sequenceDiagram
    participant A as Внешний агент (Claude Code, Hermes…)
    participant S as mcp_server.serve
    participant H as McpServer.handle
    participant T as agent/tools.py
    participant D as store/db.py + answer/citations.py
    A->>S: initialize (protocolVersion)
    S->>H: строка JSON-RPC
    H-->>A: serverInfo, instructions (правила цитат, «текст статей — данные»)
    A->>H: tools/list
    H-->>A: 7 инструментов чтения (+3 с --allow-model, +2 с --allow-write)
    A->>H: tools/call search_library
    H->>T: search_library(ctx) → предложения [s…], ctx.seen_sentences
    A->>H: tools/call cite(["s…"])
    H->>D: load_citations → «цитата» — статья, раздел, с. [s…]
    H-->>A: content[text], isError (ошибка инструмента — наблюдение, не падение)
    Note over H: каждый вызов → runs/mcp-*.jsonl
```

## Конвейер дообучения (inferred from manual inspection)

```mermaid
sequenceDiagram
    participant FS as ft-split (splits.py)
    participant FQ as ft-questions (questions.py)
    participant FC as ft-collect (trajectories.py)
    participant FE as ft-export (export.py)
    participant B as make_train_bundle.py
    participant PC as ПК с RTX 5070 (train_sft.py, *.sh)
    FS->>FS: data/finetune/splits.json (test = статьи золота)
    FQ->>FQ: учитель → questions/<split>.jsonl
    FC->>FC: run_agent(prefetch) учителем → trajectories/<split>.jsonl, accept()
    FE->>FE: чат-шаблон GGUF → sft/{train,val}.jsonl + stats.json
    B->>PC: dist/rtx5070_train_bundle.zip (MANIFEST sha256, без .env и базы)
    PC->>PC: setup_wsl → smoke → train_full → convert_gguf (Q4_K_M)
```

## Внешние зависимости
- Пакет (`pyproject.toml`): openai, httpx, protego, pymupdf, beautifulsoup4, razdel, snowballstemmer, numpy, yake, python-dotenv, streamlit, jinja2; dev: pytest. MCP-сервер новых зависимостей не добавил.
- Комплект обучения (`scripts/train_bundle/requirements-train.txt`, вне uv): torch, unsloth, transformers, trl, peft и др.
- Внешние сервисы: OpenAlex, arXiv, cyberleninka.ru, OpenAI-совместимые серверы чат-модели (по `LLM_PROVIDER`) и эмбеддингов (LM Studio, bge-m3).
