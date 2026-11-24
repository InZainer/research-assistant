> generated_by: nexus-mapper v2
> verified_at: 2026-11-24
> provenance: статический анализ импортов тестов из raw/ast_nodes.json; тесты `test_train_*` грузят скрипты через `importlib` по пути (manual inspection). Сам маппер тесты не запускал; число проходящих взято из прогона `uv run pytest` в этой же сессии 2026-11-24.

# Тестовое покрытие (статическое)

31 тестовый файл в `tests/`, ~6.3 тыс. строк, 352 функции `test_*` (по `grep`; с параметризацией pytest собирает 376). Прогон 2026-11-24: 374 прошли, 2 пропущены в PowerShell (`test_train_bundle.py` — нужен bash; в Git Bash проходят). Фикстуры `tests/fixtures/` — настоящие ответы arXiv и OpenAlex. Правило проекта (AGENTS.md): тесты не ходят в сеть и не вызывают модель — модель, HTTP и источники подменяются (`ScriptedClient`, `JsonClient`, `RecordingClient`, `Teacher`, `httpx.MockTransport`).

## Кто что тестирует

| Модуль | Тесты |
|---|---|
| `ingest/sources/*`, `http.py`, `download.py`, `pipeline.py`, `models.py` | test_sources, test_http, test_download, test_pipeline, test_models |
| `ingest/landing.py`, `resolve.py` | test_landing |
| `ingest/relevance.py` | test_relevance |
| `parse/*` | test_structure, test_sentences, test_parse_files, test_parse_pipeline, test_tables |
| `store/db.py` | test_db + косвенно большинство файлов (38 импортёров) |
| `store/vectors.py`, `retrieval/search.py`, `rerank.py`, `nlp/embed.py` | test_search (в т.ч. откат на BM25) |
| `retrieval/papers.py` | test_papers, test_ui |
| `nlp/annotate.py`, `entities.py`, `keywords.py`, `evaluation/entities.py` | test_annotation |
| `agent/loop.py`, `agent/tools.py`, `answer/citations.py` | test_agent (в т.ч. защита от повторов), test_ui, test_eval_behaviour, test_mcp_server |
| `agent/reading.py`, `agent/coverage.py` | test_reading |
| `agent/prompts.py` | косвенно: test_finetune_export, test_train_script, test_mcp_server (через `prompt_hash`) |
| `mcp_server.py`, `skills/research-library/SKILL.md` | test_mcp_server: протокол, уровни доступа, `cite` и инвариант цитат, ошибки как наблюдения, stdio, BOM, `prompt_hash` не меняется, заголовок навыка |
| `evaluation/agent.py` | test_eval_behaviour, test_agent |
| `evaluation/retrieval.py` | test_eval_retrieval, test_agent, test_finetune_* |
| `ui/views.py`, `ui/chats.py` | test_ui |
| `finetune/*` | test_finetune_questions, test_finetune_export, test_finetune_commands |
| `cli.py` | test_cli (только `evaluate_check`), test_mcp_server (парсер `mcp --help`), test_finetune_commands |
| `config.py`, `providers.py` | test_config, test_providers |
| `scripts/train_sft.py` | test_train_script (чистые функции без torch) |
| `scripts/make_train_bundle.py`, `scripts/train_bundle/*.py` | test_train_bundle |
| `scripts/make_project_bundle.py` | test_project_bundle |

## Пробелы (evidence gap)
- **Нет прямых тестов:** `agent/session.py` (сборка для CLI, UI, MCP и `ft-collect`), `ui/app.py` (486 строк, виджеты Streamlit) — ни один тест их не импортирует.
- **MCP:** модельные инструменты (`ask_agent`, `read_paper`, `check_coverage`) в тестах только перечисляются через `build_server`; их вызов на живой модели не проверялся. Совместимость с конкретными клиентами (Claude Code, Hermes, Codex) — вручную, не тестами.
- **Bash-скрипты** (`scripts/train_bundle/*.sh`, `queries/collect.sh`) статически не тестируются; `run_docker.sh` — «НЕ ПРОВЕРЕНО на RTX 5070».
- **Команды CLI** (`cmd_ingest`, `cmd_search`, `cmd_read`, `cmd_coverage`, `cmd_eval_agent` и др.) не покрыты: тестируется логика ниже.
- **Качество на живой модели** — не тесты, а эксперименты E1–E6 (`docs/experiments.md`); золотые наборы `eval/*.jsonl` — по 30 вопросов.
