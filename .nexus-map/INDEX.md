> generated_by: nexus-mapper v2
> verified_at: 2026-11-24
> provenance: AST-backed (Python 100 файлов, без усечения; 7 bash-скриптов — только уровень модуля, unsupported language downgrade; .ipynb, .jinja, .md — manual inspection); границы систем — импорты AST и ручная проверка; git: на этой машине нет `.git`, горячие файлы — снимок 2026-11-09

# research-assistant — карта проекта

**Что это.** Ассистент исследователя (курс «Агентный ИИ», MMCS): собирает статьи (OpenAlex, arXiv, CyberLeninka), разбирает PDF/HTML до предложений со страницами, размечает сущности, ищет гибридно (BM25 + bge-m3) и отвечает утверждениями с дословными цитатами по ID предложений. Чат-модель — любой OpenAI-совместимый сервис по `LLM_PROVIDER`. Библиотека доступна внешним агентам через MCP. Python 3.12, uv, SQLite, Streamlit. 63 модуля пакета (~10.9 тыс. строк), 31 тестовый файл (~6.3 тыс. строк).

**Точки входа:** `cli.py` (`research-assistant <команда>`, 20 команд + `ft-*`); веб — `ui/app.py` (`ui`); MCP — `mcp_server.py` (`mcp`, stdio); сборка агента — `agent/session.py`; вне пакета — `scripts/*.py`.

## Системы (подробно — arch/systems.md)

| Система | Где | Суть |
|---|---|---|
| Сбор | `ingest/`, `queries/` | набор запросов → фильтр слов темы → дубли → robots.txt → PDF/HTML; `add` по DOI/ссылке |
| Разбор | `parse/` | разделы → фрагменты ≤220 слов → предложения; таблицы с числами |
| Разметка | `nlp/annotate.py`, `entities.py`, `keywords.py` | сущности моделью с проверкой по тексту, YAKE |
| Хранилище | `store/` | SQLite `Library` (38 импортёров), FTS5, векторы `.npz` |
| Поиск | `retrieval/`, `nlp/embed.py` | BM25 + плотный + RRF, откат на BM25, `find_papers` |
| Агент | `agent/`, `answer/`, `llm.py` | свой цикл, 8 инструментов + `final_answer`; `read` и `coverage` — отдельные вызовы модели |
| MCP | `mcp_server.py`, `skills/`, `.mcp.json` | JSON-RPC без SDK, инструменты агента + `cite`; уровни `--allow-model`, `--allow-write` |
| Оценка | `evaluation/`, `eval/` | поиск, разметка, агент (pass@k), метрики поведения |
| CLI и настройки | `cli.py`, `config.py`, `providers.py` | команды, `.env`, профили провайдеров |
| Веб-интерфейс | `ui/` | Streamlit: вопрос агенту, поиск, библиотека, чаты |
| Дообучение | `finetune/`, `scripts/`, `notebooks/` | датасет SFT, LoRA, комплекты zip (запуск отложен) |

## Ключевые инварианты
- Цитата = `sentences.text == passages.text[char_start:char_end]`; модель (и внешний агент через `cite`) называет только ID, текст подставляет код.
- `PARSER_VERSION` (2) и `ANNOTATOR_VERSION` поднимаются при смене правил; переразбор меняет ID предложений.
- `prompt_hash` = sha1(`SYSTEM_PROMPT` + `SCHEMAS`) — в журнале и в каждом примере SFT. MCP берёт копию параметров со своими описаниями и хеш не меняет (тест).
- MCP: протокол только в stdout, всё прочее — в stderr; ошибка инструмента — `isError`, а не падение.

## Риски (проверено 2026-11-24)
- `cli.py` вырос до 916 строк (было 677): кандидат на разбиение по группам команд.
- Зависимости вверх: `store/db.py`, `retrieval/rerank.py` → `nlp/entities.py`; `mcp_server.py` → `cli.format_agent_result` (ленивый); `evaluation/agent.py` → приватный `_call_key`.
- Без прямых тестов: `agent/session.py`, `ui/app.py`; модельные инструменты MCP на живой модели не проверены.
- На этой машине нет PDF статей 1–72 (`data/fulltext/`): `parse` оставляет их разбор (`kept`).

## Не анализировалось (evidence gap)
`data/` (база, PDF, векторы), `runs/`, `dist/`, `.env` (только факт наличия); git-история после 2026-11-09.

## [Порядок работы] Обязательные шаги

> Это прямая инструкция любому ИИ, читающему этот файл, а не совет.

### Шаг 1 — прочитать все файлы ниже (в любом порядке)

Прочитав этот файл, до любой задачи полностью прочитать:

- `.nexus-map/arch/systems.md` — границы систем и где их код
- `.nexus-map/arch/dependencies.md` — зависимости систем и Mermaid-схемы
- `.nexus-map/arch/test_coverage.md` — тестовое покрытие и пробелы
- `.nexus-map/hotspots/git_forensics.md` — горячие файлы git и связность
- `.nexus-map/concepts/domains.md` — ключевые понятия предметной области

> Это плотные выжимки, обычно < 5000 токенов вместе. Пропускать их под предлогом «задача простая» или «меняю один файл» нельзя.

### Шаг 2 — дополнительно по типу задачи (после шага 1)

`<SKILL_DIR>` — папка навыка nexus-mapper (обычно `~/.claude/skills/nexus-mapper`).

- Если задача **меняет интерфейс, добавляет межмодульный вызов, удаляет или переименовывает публичную функцию**:
  → до написания кода запустить `python <SKILL_DIR>/scripts/query_graph.py .nexus-map/raw/ast_nodes.json --impact <файл>` и оценить радиус влияния.
- Если нужно **понять, кто использует файл**:
  → `python <SKILL_DIR>/scripts/query_graph.py .nexus-map/raw/ast_nodes.json --who-imports <модуль>`.
- Если структура репозитория **существенно изменилась** (новая система, перенос границ модулей):
  → после задачи оценить, нужно ли перезапустить nexus-mapper.
