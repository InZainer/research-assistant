> generated_by: nexus-mapper v2
> verified_at: 2026-10-22
> provenance: статический анализ импортов тестов из raw/ast_nodes.json; тесты при картировании не запускались

# Тестовое покрытие (статическое)

16 тестовых файлов в `tests/`, ~2 тыс. строк, фикстуры `tests/fixtures/` (настоящие ответы arXiv и OpenAlex). Правило проекта (AGENTS.md): тесты не ходят в сеть и не вызывают модель — модель, HTTP и источники подменяются (`ScriptedClient`, `JsonClient`, `httpx.MockTransport`, PDF генерируется PyMuPDF прямо в тесте).

## Кто что тестирует

| Модуль | Тесты |
|---|---|
| `ingest/*` | test_sources, test_http, test_download, test_pipeline, test_models |
| `parse/*` | test_structure, test_sentences, test_parse_files, test_parse_pipeline |
| `store/db.py` | test_db + косвенно 6 файлов (агент, разметка, поиск, оценка) |
| `store/vectors.py`, `retrieval/*`, `nlp/embed.py`, `nlp/text.py` | test_search |
| `nlp/annotate.py`, `entities.py`, `keywords.py`, `evaluation/entities.py` | test_annotation |
| `agent/loop.py`, `agent/tools.py`, `answer/citations.py`, `evaluation/agent.py` | test_agent |
| `evaluation/retrieval.py` | test_eval_retrieval, test_agent |
| `cli.py` | test_cli (только `evaluate_check`; команды CLI не тестируются) |
| `config.py` | test_config |

## Пробелы (evidence gap)
- **Нет прямых тестов:** `agent/session.py` (точка сборки, 85 строк) и `agent/prompts.py` (константы). Сборка агента проверяется только живыми прогонами `ask`/`eval-agent`.
- **Команды CLI** (`cmd_ingest`, `cmd_search`, `cmd_eval_agent` и др.) не покрыты: тестируется логика ниже, а не разбор аргументов и вывод.
- **Качество на живой модели** проверяется не тестами, а экспериментами (`docs/experiments.md`); золотые наборы `eval/*.jsonl` имеют статус `draft`.
- **Не запускалось при картировании:** число проходящих тестов здесь не измерено; по журналу работы проекта последний известный прогон — 148 тестов (коммит 68586f7), это не проверено в этом анализе.
