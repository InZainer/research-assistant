> generated_by: nexus-mapper v2
> verified_at: 2026-10-22
> provenance: AST-backed (Python, полное структурное покрытие); границы систем проверены вручную по импортам (`grep`, `query_graph.py --impact`); домены — группировка из PLAN.md (inferred)

# Системы

Все пути — от корня репозитория, код пакета в `src/research_assistant/`.

## Сбор статей — `ingest/`
- `sources/openalex.py`, `sources/arxiv.py` — поиск; OpenAlex основной (в том числе `language:ru`), arXiv API часто отвечает 429, поэтому короткие повторы (`BACKOFF = (20,)`).
- `models.py` — `PaperRecord`, нормализация DOI/arXiv/заголовка. Фактически общая доменная модель: 18 импортёров, включая `store/db.py`.
- `dedup.py` — слияние записей по любому совпавшему ключу.
- `http.py` — `HttpClient`: пауза на хост, повторы на 429/5xx, дисковый кэш, таймаут на запрос.
- `robots.py` — `RobotsPolicy` на `protego` (шаблоны `*` и Crawl-delay).
- `download.py` — PDF или HTML CyberLeninka; проверка robots.txt и после редиректа.
- `pipeline.py` — `search_sources`, `ingest_topic`.

## Разбор полных текстов — `parse/`
- `pdf.py` (PyMuPDF), `html.py` (CyberLeninka) → строки `Unit` (`units.py`).
- `structure.py` (407 строк, самый сложный модуль) — заголовки, шапка, литература и приложения, склейка абзацев, снятие переносов, статьи внутри сборников.
- `sentences.py` — razdel + правила для научных сокращений и ссылок.
- `passages.py` — фрагменты ≤220 слов со страницами предложений.
- `pipeline.py` — `parse_paper`/`parse_papers`, `PARSER_VERSION`, запасной вариант «аннотация из метаданных».

## Разметка — `nlp/annotate.py`, `nlp/entities.py`, `nlp/keywords.py`
- `entities.py` — промпт и JSON-схема, проверка по тексту через основы слов (`occurs_in`, `contains_stems`). Здесь же общие утилиты `contains_stems` и `parse_json`, которые используют хранилище и поиск (см. dependencies.md).
- `keywords.py` — YAKE с отсевом общих слов.
- `annotate.py` — `annotate_papers`, `ANNOTATOR_VERSION`; вход модели — аннотация, заключение, начало экспериментов (≤900 слов).

## Хранилище — `store/`
- `db.py` (596 строк) — `Library`: схема (topics, papers, sections, passages, sentences, parses, entities, entity_mentions, keywords, annotations, `passages_fts`), upsert со слиянием, `save_structure` (разбор + FTS5 + упоминания в одной транзакции), `PassageFilter`, `search_fts`, `entity_stats`.
- `vectors.py` — `VectorIndex`: матрица в `.npz` на модель эмбеддингов, поиск перебором.

## Поиск — `retrieval/`, `nlp/embed.py`
- `search.py` — `Searcher.search(mode, translate, rerank)`, `rrf`, `build_index`.
- `rerank.py` — `LLMReranker` (listwise, выключен: на 4B вредит), `QueryTranslator`.
- `nlp/embed.py` — `OpenAIEmbedder`, префиксы для nomic/e5/qwen3-embedding.

## Агент — `agent/`, `answer/`, `llm.py`
- `loop.py` — `run_agent`: лимит шагов, ошибки как наблюдения, бюджет исследования (6 вызовов), режимы `prefetch` (по умолчанию в CLI), `single_shot`, `answer_mode="quotes"`; журнал `RunLog` в JSONL.
- `tools.py` — схемы и реализации 7 инструментов + `final_answer`.
- `prompts.py` — системный промпт (английский, язык ответа = язык вопроса).
- `session.py` — точка сборки: `make_searcher`, `make_agent`, `make_add_papers`; её вызывает и CLI (`search`, `ask`, `chat`, `eval-agent`).
- `answer/citations.py` — загрузка и формат цитат по ID предложений.
- `llm.py` — `ChatModel`, `Usage`; `reasoning_effort` для Qwen3.5 в LM Studio.

## Оценка — `evaluation/`, `eval/`
- `retrieval.py` — золото задано цитатой, Recall@k/MRR/nDCG по группам en→en / ru→en / ru→ru.
- `entities.py` — P/R/F1 строго и мягко, сравнение «с проверкой по тексту / без».
- `agent.py` — успех строгий (золотой фрагмент) и мягкий (золотая статья), pass@1 + Вильсон, pass^k, `rescore_log`.
- `eval/retrieval_gold.jsonl`, `eval/entities_gold.jsonl` — черновые (status `draft`); `eval/translate_queries.py` — скрипт эксперимента.

## Командная строка — `cli.py`, `config.py`
- `cli.py` (639 строк) — 14 команд: check, ingest, library, parse, show, index, search, eval-retrieval, annotate, entities, eval-entities, eval-agent, ask, chat. Импортирует 20 внутренних модулей.
- `config.py` — `Settings` из `.env` (значения ключей не анализировались).

## Веб-интерфейс — `ui/` (этап 8, дописано вручную после картирования)
- `views.py` — данные для страниц без Streamlit: `source_link` (PDF со страницей `#page=N`), `sentence_context`, `answer_view`/`CitationCard`, `describe_event` (шаги агента словами), `friendly_error`, `highlight`, `hit_rows`, `paper_card`. Тесты — `tests/test_ui.py`.
- `chats.py` — `ChatStore`: чат = JSON в `data/chats/`, память агента из сохранённых ответов.
- `app.py` — страница Streamlit, три раздела; агент через `agent/session.make_agent` с `prefetch=True` и `on_event`, поиск статей через `retrieval/papers.find_papers`. База открывается на каждый перезапуск скрипта (SQLite не делится между потоками).

## Поиск статей — `retrieval/papers.py` (дописано вручную)
- `find_papers`: автор (начала слов в имени одного автора), все слова запроса в названии/аннотации по основам, затем до 5 похожих по смыслу аннотаций (плотный поиск по разделам `abstract`). Использует веб-интерфейс и инструмент агента `find_papers`. Тесты — `tests/test_papers.py`.
- Сбор по автору (дописано вручную): `search_sources(author=...)` → `OpenAlexSource.author_ids` (профили со сверкой имени `ingest/models.author_matches`) и arXiv `au:`; результаты фильтруются по списку авторов.
