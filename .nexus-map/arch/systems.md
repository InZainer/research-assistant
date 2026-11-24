> generated_by: nexus-mapper v2
> verified_at: 2026-11-24
> provenance: AST-backed (Python, полное структурное покрытие, 100 файлов, без усечения); 7 bash-скриптов — только уровень модуля (unsupported language downgrade: описание их шагов — manual inspection); `.ipynb`, `.jinja`, `.md` — manual inspection; границы систем проверены по импортам (raw/ast_nodes.json, grep). Числа строк — по AST на 2026-11-24.

# Системы

Все пути — от корня репозитория, код пакета в `src/research_assistant/` (63 модуля, ~10.9 тыс. строк).

## Сбор статей — `ingest/`, `queries/`
- `sources/openalex.py` (177 строк), `sources/arxiv.py` — поиск по теме и по автору (`OpenAlexSource.author_ids` со сверкой имени `models.author_matches`, arXiv `au:`); OpenAlex — курсор (`limit` до 2000), `open_access_only`, `country`, `by_doi`. arXiv часто отвечает 429 (`BACKOFF = (20,)`).
- `models.py` — `PaperRecord`, нормализация DOI/arXiv/заголовка, `external_id`. Общая доменная модель: 36 импортёров.
- `dedup.py` — слияние записей по любому совпавшему ключу. `http.py` — `HttpClient`: пауза на хост, повторы на 429/5xx, дисковый кэш. `robots.py` — `RobotsPolicy` на `protego`.
- `download.py` — PDF или HTML CyberLeninka; PDF со страницы статьи (`discover`). `landing.py` — метатеги Highwire: ссылка на PDF и `record_from_page`.
- `relevance.py` — `TermFilter`: группы слов темы по основам (флаг `--require`). `resolve.py` — `resolve_reference`: DOI, arXiv или адрес страницы → `PaperRecord` (команда `add`).
- `pipeline.py` (239 строк) — `search_sources(author=...)`, `ingest_topic(queries=[...], relevance=TermFilter)`, `fetch_fulltexts`, `add_references`.
- `queries/` — наборы запросов для широких тем и `collect.sh` (bash, только уровень модуля).

## Разбор полных текстов — `parse/`
- `pdf.py` (PyMuPDF; `table_rows` — таблица с числами по геометрии строк), `html.py` → строки `Unit` (`units.py`, `kind = "table"`).
- `structure.py` (433 строки) — заголовки, шапка, литература и приложения, склейка абзацев, сборники.
- `sentences.py` — razdel + правила сокращений. `passages.py` — фрагменты ≤220 слов; фрагмент `table`, строка таблицы — предложение с ячейками через ` | `.
- `pipeline.py` — `parse_paper`/`parse_papers`, `PARSER_VERSION = 2`; статус `kept`, если файла полного текста нет, а прежний разбор есть.

## Разметка — `nlp/annotate.py`, `nlp/entities.py`, `nlp/keywords.py`
- `entities.py` — промпт, JSON-схема, проверка по тексту через основы. Здесь же общие `contains_stems` и `parse_json` (их берут `store/db.py`, `retrieval/rerank.py`, `finetune/questions.py`).
- `keywords.py` — YAKE. `annotate.py` — `annotate_papers`, `ANNOTATOR_VERSION`.

## Хранилище — `store/`
- `db.py` (627 строк, 38 импортёров — главный хаб) — `Library`: схема, upsert со слиянием, `save_structure` (разбор + FTS5 + упоминания в одной транзакции), `PassageFilter`, `search_fts`, `entity_stats`.
- `vectors.py` — `VectorIndex`: матрица `.npz` на модель эмбеддингов.

## Поиск — `retrieval/`, `nlp/embed.py`
- `search.py` — `Searcher.search(mode, translate, rerank)`, `rrf`, `build_index`. Если сервер эмбеддингов не ответил, гибрид падает на BM25, причина — в `last_warning`/`dense_failed`; `mode="dense"` бросает исключение. Переводчик запроса срабатывает только без эмбеддингов или при неполном индексе.
- `papers.py` — `find_papers`: автор, все слова запроса в названии/аннотации по основам, затем до 5 похожих аннотаций плотным поиском; `authors_of`.
- `rerank.py` — `LLMReranker` (выключен: на 4B вредит), `QueryTranslator`. `nlp/embed.py` — `OpenAIEmbedder`; для localhost `max_retries=0`.

## Агент — `agent/`, `answer/`, `llm.py`
- `loop.py` (333 строки) — `run_agent`: лимит шагов (`MAX_STEPS`, по умолчанию 12), бюджет вызовов (`research_calls`), `MAX_NUDGES=2`, режимы `prefetch`/`single_shot`/`answer_mode="quotes"`; защита от точных повторов (`_call_key` → `REPEATED_CALL`), `error_kind` в журнале, `prompt_hash` в событии start, `observation_chars`, `hints`.
- `tools.py` (531 строка) — 8 инструментов (`search_library`, `read_passage`, `library_overview`, `find_papers`, `list_entities`, `compare_papers`, `search_external`, `add_papers`) + `final_answer`; `SCHEMAS`; `prompt_hash()`; `ToolContext.seen_sentences`.
- `prompts.py` — `SYSTEM_PROMPT` (8 правил), `NUDGE_FINAL_ANSWER`, `answer_hints` (язык, полнота — вне `prompt_hash`).
- `session.py` (98 строк) — `make_searcher`, `make_agent`, `make_add_papers`, `run_options(settings)`: сборка для CLI, UI, MCP и `ft-collect`.
- `reading.py` (389 строк) — `read_paper`: окна текста статьи (`paper_windows`), заметки по окнам, отчёт `paper_report` (находки, `not_covered`, `inconsistencies`), проверка ID. `coverage.py` (344 строки) — `check_coverage`: переформулировки аспекта, `rank_papers`, вердикт `yes/partial/no` по каждой статье. Свои схемы, в `SCHEMAS` и `prompt_hash` не входят.
- `answer/citations.py` — цитаты по ID (`load_citations`, `format_citation`). `llm.py` — `ChatModel`, `Usage`, `reasoning_effort`.

## MCP для внешних агентов — `mcp_server.py`, `skills/`, `.mcp.json`
- `mcp_server.py` (422 строки) — JSON-RPC 2.0 по stdio без SDK: `McpServer.handle` (initialize, ping, tools/list, tools/call, пакеты), `serve` (UTF-8, пропуск BOM), `build_server` (уровни: чтение / `--allow-model` / `--allow-write`), `cite` (ID → цитата из базы, предупреждение о невиденных ID), `project_root`.
- Инструменты — функции `agent/tools.py` с копией параметров `SCHEMAS` и своими описаниями: `prompt_hash` не меняется (тест). Модельные — `ask_agent` (через `run_agent`), `read_paper`, `check_coverage`; запись — `search_external`, `add_papers` (через `session.make_add_papers`).
- `skills/research-library/SKILL.md` — навык для Claude Code и Hermes; `.mcp.json` — проектный конфиг Claude Code; подключение клиентов — `docs/mcp.md`. Тесты — `tests/test_mcp_server.py`.

## Оценка — `evaluation/`, `eval/`
- `retrieval.py` — Recall@k/MRR/nDCG по группам; `GoldQuestion` с `kind` (fact, entity, compare, paper_lookup, unanswerable, followup, external) и `meta`.
- `entities.py` — P/R/F1 строго и мягко.
- `agent.py` (504 строки) — строгий/мягкий успех, pass@1 + Вильсон, pass^k; `journal_metrics` — метрики поведения по журналу; `rescore_log` сопоставляет прогоны по `qid`.
- `eval/retrieval_gold.jsonl` (30 вопросов, статьи из него = test-сплит), `eval/entities_gold.jsonl`, `eval/translate_queries.py`.

## Командная строка — `cli.py`, `config.py`, `providers.py`
- `cli.py` (916 строк) — 20 команд: check, providers, ingest, add, parse, show, library, index, search, eval-retrieval, ask, chat, read, coverage, ui, mcp, eval-entities, annotate, entities, eval-agent. Команды `ft-*` регистрируются ленивым импортом `finetune.commands`. `format_agent_result` — общий для `ask`, `chat` и MCP `ask_agent`.
- `config.py` — `Settings` из `.env` (значения не читались; `.env` есть, `.env.example` — шаблон). `providers.py` (122 строки) — профили `LLM_PROVIDER` (lmstudio, ollama, inception, openai, anthropic, openrouter, deepseek, gemini, custom), переменные `LLM_<ИМЯ>_*`.

## Веб-интерфейс — `ui/`
- `views.py` — данные страниц без Streamlit (`answer_view`/`CitationCard`, `describe_event`, `friendly_error`, `paper_card`). Тесты — `tests/test_ui.py`.
- `chats.py` — `ChatStore`: чат = JSON в `data/chats/`.
- `app.py` (486 строк) — три раздела; агент через `make_agent` с `prefetch=True` и `on_event`. Запускается `research-assistant ui` подпроцессом `streamlit run`.

## Датасет дообучения — `finetune/`
Конвейер `ft-split → ft-questions → ft-collect → ft-export`, всё пишет в `data/finetune/`. Подробно — `docs/finetune.md`. Запуск отложен (AGENT.md).
- `splits.py` — test = статьи золотого набора, val ≈ 6 статей, train — остальное; Жаккар почти-дублей ≥ 0.6.
- `questions.py` (1060 строк, самый крупный модуль) — учитель видит фрагмент с `[s<id>]`, называет ID, код собирает дословное золото; отклонённые с `reasons`.
- `trajectories.py` — `collect` тем же путём, что деплой (`run_agent(prefetch)`); `accept()` — rejection sampling.
- `export.py` — рендер `qwen3_5_chat_template.jinja` (jinja2 sandbox), `train_spans`, валидация. `commands.py` — парсеры и запуск `ft-*`.

## Обучение и комплекты — `scripts/`, `notebooks/`
Вне пакета: не импортирует `research_assistant` (grep), связан с `finetune/` только файлами.
- `scripts/train_sft.py` — LoRA SFT `unsloth/Qwen3.5-4B`, labels из `train_spans`, `--check-gguf`. Чистые функции — `tests/test_train_script.py`.
- `scripts/make_train_bundle.py` — комплект `dist/rtx5070_train_bundle.zip` с `MANIFEST.json` (sha256), поиск секретов. `scripts/make_project_bundle.py` — архив проекта с библиотекой для другой машины (`git ls-files` + проверка секретов). Тесты — `test_train_bundle.py`, `test_project_bundle.py`.
- `scripts/train_bundle/` — инструкция RTX 5070, bash-скрипты (`setup_wsl.sh`, `smoke.sh`, `train_full.sh`, `convert_gguf.sh`, `run_docker.sh` — не проверен, `common.sh`), `verify_bundle.py`, `bundle_tools.py`.
- `notebooks/finetune_qwen35_colab.ipynb` — пробный запуск на Colab T4.
