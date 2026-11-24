# Библиотека для внешних агентов: MCP-сервер и навык

Claude Code, Hermes Agent, Codex, Cursor и другие агенты могут искать по библиотеке и цитировать её так же, как собственный агент проекта: утверждение, ID предложения и дословная цитата, которую подставляет код. Для этого в проекте есть две части:

- **MCP-сервер** `research-assistant mcp` (`src/research_assistant/mcp_server.py`). Он отдаёт агенту инструменты библиотеки по протоколу MCP через stdio. Сервер свой, без SDK и новых зависимостей.
- **Навык** `skills/research-library/SKILL.md`. Он объясняет агенту, когда пользоваться библиотекой, в каком порядке вызывать инструменты и как оформлять цитаты. Формат `SKILL.md` понимают и Claude Code, и Hermes Agent.

## Уровни доступа

| запуск | инструменты | что тратит |
|---|---|---|
| `research-assistant mcp` | `library_overview`, `find_papers`, `search_library`, `read_passage`, `list_entities`, `compare_papers`, `cite` | ничего: только чтение базы, без сети и модели (эмбеддинги запроса — LM Studio, если он настроен) |
| `… mcp --allow-model` | + `ask_agent`, `read_paper`, `check_coverage`; перевод запроса для поиска по словам | ключ провайдера из `.env` |
| `… mcp --allow-write` | + `search_external`, `add_papers` | сеть (OpenAlex, arXiv), меняет библиотеку |

По умолчанию сервер только читает базу. Так безопаснее всего: в выдачу попадает текст статей, то есть непроверенные данные из интернета, а у внешнего агента обычно есть оболочка и сеть. В инструкциях сервера и в навыке записано, что текст статей — это данные, а не команды. Записи в библиотеку и платные вызовы включаются только явными флагами.

Другие флаги:

- `--project-dir <папка>` — где лежат `.env` и `data/`. Если флаг не задан, сервер берёт текущую папку, а когда в ней нет ни `.env`, ни `data/`, — папку, откуда установлен пакет (после `uv sync`).
- `--no-log` — не писать журнал. По умолчанию каждый вызов записывается в `runs/mcp-*.jsonl`: имя, аргументы, ошибка или нет, длина ответа, время.
- Общий `--provider <имя>` выбирает модель для `--allow-model`: `research-assistant --provider anthropic mcp --allow-model`.

## Как сохраняется инвариант цитат

Внешний агент получает из `search_library` предложения с ID вида `[s123]`. Чтобы процитировать их, он вызывает `cite(["s123", …])` и вставляет то, что инструмент вернул. Текст цитаты берётся из `sentences.text`, так что цитата всегда дословная. Несуществующие ID отбрасываются с пометкой. ID, которого в этой сессии не было в выдаче поиска, процитировать можно (он мог прийти из `show --sentences`), но рядом появится предупреждение: агент мог угадать номер.

`SCHEMAS` и `SYSTEM_PROMPT` агента не менялись: сервер берёт копию параметров и пишет для внешнего агента свои описания инструментов. Поэтому `prompt_hash`, датасет и дообученная модель остаются совместимыми. Это проверяет тест.

## Подключение

Ниже везде `uv run --directory <путь к проекту> research-assistant mcp`. Если `uv` нет в PATH, укажите сам исполняемый файл из окружения: `<проект>/.venv/Scripts/research-assistant.exe` на Windows или `<проект>/.venv/bin/research-assistant` на Linux и macOS, с аргументом `mcp`. На Windows задайте `PYTHONIOENCODING=utf-8`.

### Claude Code

В корне проекта уже лежит `.mcp.json`: при первом запуске Claude Code в этой папке предложит включить сервер `research-library`. Подключить сервер для всех папок:

```bash
claude mcp add research-library --scope user -- uv run --directory /путь/к/research-assistant research-assistant mcp
```

Установить навык для всех проектов:

```bash
cp -r skills/research-library ~/.claude/skills/
```

### Hermes Agent

В `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  research-library:
    command: uv
    args: ["run", "--directory", "/путь/к/research-assistant", "research-assistant", "mcp"]
```

Навык:

```bash
cp -r skills/research-library ~/.hermes/skills/
```

Ключи конфига — по документации Hermes той версии, что у вас установлена. Формат `SKILL.md` там такой же, как в Claude Code.

### Codex CLI

В `~/.codex/config.toml`:

```toml
[mcp_servers.research-library]
command = "uv"
args = ["run", "--directory", "/путь/к/research-assistant", "research-assistant", "mcp"]
```

### Cursor, Claude Desktop и другие клиенты

Блок `mcpServers` в формате как в `.mcp.json`: для Cursor — `.cursor/mcp.json`, для Claude Desktop — `claude_desktop_config.json`, с аргументом `--directory`.

## Проверка

```bash
uv run pytest tests/test_mcp_server.py -q
```

Проверить вручную: запустите `research-assistant mcp` и отправьте в stdin строку `initialize`, затем `tools/list` и `tools/call` (по одному объекту JSON-RPC 2.0 в строке). Ответы приходят в stdout, служебные сообщения — в stderr. Удобнее посмотреть сервер в MCP Inspector:

```bash
npx @modelcontextprotocol/inspector uv run research-assistant mcp
```

## Ограничения

- Транспорт только stdio. HTTP-транспорта нет, сервер рассчитан на локальный запуск одним пользователем.
- Запросы обрабатываются по одному. Пока идёт долгий `read_paper` или `add_papers`, остальные вызовы ждут, а отмена (`notifications/cancelled`) работу не прерывает.
- `tools/list` всегда один и тот же: набор инструментов задаётся флагами при запуске.
