> generated_by: nexus-mapper v2
> verified_at: 2026-11-24
> provenance: на этой машине `.git` нет (проект развёрнут из архива, `MANIFEST.json`: `git_commit` e5651a0, `git_dirty: true`) — `git_detective.py` не запускался (hotspots skipped: no git metadata). Таблица ниже — исторический снимок `raw/git_stats.json` с исходной машины на 2026-11-09 (90 дней, 8 коммитов, 1 автор); интерпретация — ручная

# Git: горячие файлы и связность

## evidence gap
С 2026-11-09 git-история недоступна: изменения 2026-11-14 (сбор по набору запросов, `add`, `read`, `coverage`, таблицы, профили провайдеров) и 2026-11-24 (MCP-сервер, навык) в статистике не отражены. Если здесь создаётся новый репозиторий, его история начнётся с одного коммита — для анализа горячих файлов этого мало; перезапустить `git_detective.py` после нескольких недель коммитов.

## Снимок 2026-11-09 (исходная машина, только коммиты)
8 коммитов: по одному на этап (1–2 … 8) и один коммит карты. Последний — e5651a0 «Этап 8: веб-интерфейс, поиск статей по автору, история чатов» (2026-10-27). Порог риска: <5 изменений — low, 5–15 — medium.

| Файл | Изменений | Риск |
|---|---|---|
| AGENTS.md | 8 | medium |
| PLAN.md, README.md, src/research_assistant/cli.py | 7 | medium |
| pyproject.toml, uv.lock, store/db.py | 5 | medium |
| agent/prompts.py, agent/tools.py, tests/test_agent.py, docs/experiments.md | 4 | low |
| agent/loop.py, agent/session.py, config.py, .env.example, tests/test_config.py | 3 | low |

## Интерпретация (по снимку и по росту файлов)
- **Связность 1.0 у AGENTS/PLAN/README/cli.py/pyproject/uv.lock/db.py — артефакт процесса** «один этап — один коммит». Для границ систем эти пары ничего не говорят.
- **`cli.py` снова растёт:** 677 строк на 2026-11-09 → 916 на 2026-11-24 (команды `add`, `read`, `coverage`, `providers`, `mcp`, `format_agent_result`). Кандидат на разбиение по группам команд, как уже сделано для `ft-*` (`finetune/commands.py`).
- **`store/db.py` (38 импортёров)** — самый большой радиус: перед правкой `query_graph.py --impact src/research_assistant/store/db.py`.
- **`agent/prompts.py` + `agent/tools.py`** — скрытый хотспот: правка `SYSTEM_PROMPT` или `SCHEMAS` меняет `prompt_hash` и инвалидирует датасет `data/finetune/sft` и дообученную модель. Теперь `tools.py` используют ещё и MCP-сервер (копия параметров) и `evaluation/agent.py` (приватный `_call_key` из `loop.py`).
- **Новая зона 2026-11-24** — `mcp_server.py` (422 строки, fan-out 16): связан с агентом, сессией и сбором; ленивая зависимость вверх на `cli.format_agent_result`.
