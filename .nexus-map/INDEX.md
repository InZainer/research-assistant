> generated_by: nexus-mapper v2
> verified_at: 2026-10-22
> provenance: AST-backed (Python, 62 файла, без усечения); системные границы и домены — ручная проверка импортов и PLAN.md, помечены в concept_model.json

# research-assistant — карта проекта

**Что это.** CLI-агент «ассистент исследователя» (курс «Агентный ИИ», MMCS): собирает статьи по теме (OpenAlex, arXiv, страницы CyberLeninka), разбирает PDF/HTML до предложений со страницами, размечает сущности, ищет гибридно (BM25 + bge-m3) и отвечает утверждениями с дословными цитатами по ID предложений. Python 3.12, uv, SQLite, LLM через OpenAI-совместимый API (локально LM Studio, Qwen3.5-4B). 45 модулей пакета, ~5 тыс. строк кода, 16 тестовых файлов (~2 тыс. строк).

**Точки входа:** `src/research_assistant/cli.py` (`research-assistant <команда>`), веб — `ui/app.py` (запуск через `research-assistant ui`), сборка агента и поиска — `agent/session.py`.

## Системы (подробно — arch/systems.md)

| Система | Где | Суть |
|---|---|---|
| Сбор | `ingest/` | OpenAlex/arXiv → слияние дублей → robots.txt → PDF/HTML |
| Разбор | `parse/` | строки PDF/HTML → разделы → фрагменты ≤220 слов → предложения со страницами |
| Разметка | `nlp/annotate.py`, `entities.py`, `keywords.py` | сущности моделью с проверкой по тексту, YAKE |
| Хранилище | `store/` | SQLite (`Library`, FTS5 по основам слов), векторы `.npz` |
| Поиск | `retrieval/`, `nlp/embed.py` | BM25 + плотный + RRF, перевод запроса, реранкер |
| Агент | `agent/`, `answer/`, `llm.py` | свой цикл, инструменты, бюджет шагов, prefetch, цитаты по ID |
| Оценка | `evaluation/`, `eval/` | метрики поиска, разметки, агента (pass@k, Вильсон) |
| Веб-интерфейс | `ui/` | Streamlit (`research-assistant ui`): вопрос агенту с карточками цитат, поиск, библиотека; добавлен на этапе 8, вне AST-анализа |

Общий фундамент: `nlp/text.py` (анализатор, основы Snowball en/ru), `ingest/models.py` (`PaperRecord`, 18 импортёров), `config.py`, `llm.py`.

## Ключевые инварианты
- Цитата = `sentences.text == passages.text[char_start:char_end]`; модель называет только ID, текст подставляет код (`answer/citations.py`).
- Разбор, FTS5 и упоминания сущностей обновляются в одной транзакции `Library.save_structure`; переразбор меняет ID предложений.
- Версии `PARSER_VERSION` (`parse/pipeline.py`) и `ANNOTATOR_VERSION` (`nlp/annotate.py`) поднимаются при смене правил.

## Риски (проверено на этом этапе)
- `cli.py` — 639 строк, импортирует 20 модулей, растёт на 80–185 строк каждый этап.
- Слои: `store/db.py` и `retrieval/rerank.py` импортируют модуль разметки `nlp/entities.py` ради утилит `contains_stems`/`parse_json`; на уровне каталогов цикл `store ↔ nlp`.
- Без прямых тестов: `agent/session.py`, `agent/prompts.py`.
- Git: 6 коммитов «один этап — один коммит», поэтому связность файлов 1.0 — артефакт процесса, а не архитектуры.

## Не в git (evidence gap для анализа)
`data/` (база, PDF, векторы), `runs/` (журналы и отчёты экспериментов), `.env` — содержимое не анализировалось.

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

- Если задача **меняет интерфейс, добавляет межмодульный вызов, удаляет или переименовывает публичную функцию**:
  → до написания кода запустить `python C:/Users/Azuz/.claude/skills/nexus-mapper/scripts/query_graph.py .nexus-map/raw/ast_nodes.json --impact <файл>` и оценить радиус влияния.
- Если нужно **понять, кто использует файл**:
  → `python C:/Users/Azuz/.claude/skills/nexus-mapper/scripts/query_graph.py .nexus-map/raw/ast_nodes.json --who-imports <модуль>`.
- Если структура репозитория **существенно изменилась** (новая система, перенос границ модулей):
  → после задачи оценить, нужно ли перезапустить nexus-mapper.
