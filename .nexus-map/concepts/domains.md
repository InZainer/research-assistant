> generated_by: nexus-mapper v2
> verified_at: 2026-11-24
> provenance: понятия выведены из кода (AST + чтение модулей) и PLAN.md, docs/finetune.md, docs/experiments.md, docs/mcp.md; домены — концептуальная группировка (inferred), своих каталогов в коде у них нет

# Предметные понятия

## Наполнение библиотеки
- **Тема (topic)** — пользовательская тема; статья может принадлежать нескольким темам (`paper_topics`). Широкая тема собирается набором запросов (`--query`, `--queries-file`, каталог `queries/`) с фильтром по словам темы (`--require`, `TermFilter`).
- **Статья (paper / `PaperRecord`)** — запись из источника; дубли склеиваются по DOI, arXiv id, OpenAlex id или нормализованному заголовку. Сбор по теме, по автору (`author_matches`) или одной ссылкой (`add`: DOI, arXiv, адрес страницы).
- **Полный текст** — `fulltext_status`: pending / ok / failed / unavailable; pdf или html (CyberLeninka), PDF со страницы статьи по метатегам. Без полного текста в поиск идёт аннотация из метаданных (`--include-closed`).
- **Раздел (section)** — заголовок + тип (`abstract`, `introduction`, `method`, `experiments`, `results`, `discussion`, `conclusion`, `limitations`, `appendix`, `other`, `body`).
- **Фрагмент (passage)** — единица поиска, ≤220 слов, со страницей; ID `p123`; вид `table` — таблица с числами. **Предложение (sentence)** — единица цитирования; ID `s123`; инвариант `sentences.text == passages.text[char_start:char_end]`.
- **Сущность (entity)** — method / dataset / metric / task; ключ = основы слов; хранится, только если встречается в тексте. **Упоминание** — связь сущности с фрагментом.
- **Версии правил** — `PARSER_VERSION` (2) и `ANNOTATOR_VERSION`: переразбор меняет ID предложений.

## Хранилище и поиск
- **Анализатор** (`nlp/text.analyze`) — токены → стоп-слова → основы Snowball по алфавиту слова. Один для FTS5, ключей сущностей, запросов, `find_papers` и почти-дублей при сплите.
- **Режимы поиска** — `bm25`, `dense`, `hybrid` (RRF, k = 60); `translate`, `rerank`. **Откат на BM25**: нет сервера эмбеддингов → гибрид работает по словам, агент получает заметку «Semantic search is unavailable…» и ищет английскими словами.
- **PassageFilter** — тема, годы, типы разделов, язык, статьи, сущность.

## Ответы с цитатами
- **final_answer** — `answer` + `claims[{text, sentence_ids}]` или `not_found`; принимается, только если все ID агент видел в выдаче (`seen_sentences`).
- **Бюджет** — 12 шагов, ограничение числа поисков/чтений, до 2 напоминаний (nudge). **prefetch** — поиск по самому вопросу до первого шага (лучший режим в E3).
- **Чтение целиком** (`read`) — вся статья окнами: находки с цитатами, «чего нет», расхождения. **Новизна** (`coverage`) — вердикт yes / partial / no по каждой статье темы. Утверждение без верного ID отбрасывается.
- **Провайдер модели** — профиль `LLM_PROVIDER` (локальный LM Studio/Ollama или облачный сервис); облачным моделям — подсказки `answer_hints` и бюджет из `config.py`, а не правка `SYSTEM_PROMPT`.
- **Чат (UI)** — JSON в `data/chats/` с готовыми карточками цитат; открывается без модели.

## Внешние агенты (MCP)
- **MCP-сервер** — `research-assistant mcp`: инструменты библиотеки для Claude Code, Hermes, Codex, Cursor по stdio.
- **Уровень доступа** — чтение (по умолчанию) / `--allow-model` (платные вызовы модели) / `--allow-write` (интернет и пополнение библиотеки).
- **cite** — ID предложений → дословная цитата из базы: внешний агент не набирает цитату сам. ID, которого не было в выдаче, — с предупреждением.
- **Навык** — `skills/research-library/SKILL.md`: порядок вызовов и правила цитирования; формат общий для Claude Code и Hermes.
- **Текст статей — данные** — инструкции внутри статей не выполняются (prompt injection); записано в `instructions` сервера и в навыке.

## Оценка
- **Золотой вопрос** (`GoldQuestion`) — вопрос + пары (статья, дословная цитата); `kind`: fact, entity, compare, paper_lookup, unanswerable, followup, external; `meta`.
- **Успех агента** — строгий: цитата из золотого фрагмента; мягкий: из золотой статьи; на `unanswerable` — честный `not_found`. **pass@1 / pass^k** — с интервалом Вильсона.
- **Метрики поведения** — по журналу: call_validity, redundant_rate, first_answer_accept, abstention P/R, false_not_found, lang_match, unauthorized, nudge/max_steps rate, p50/p95. **error_kind** — json, unknown_tool, bad_args, tool_error, env, answer_rejected.

## Дообучение (дистилляция, отложено)
- **Сплит по статьям** — test = все статьи золотого набора; обучающая траектория не может цитировать тестовую статью.
- **Траектория** — прогон учителя через `run_agent(prefetch)`: ровно `messages` и `tools`, которые видела модель; отсев `accept()`.
- **Пример SFT** — `text` (рендер чат-шаблоном GGUF) + `train_spans` + `meta.prompt_hash`.
- **prompt_hash** — sha1(`SYSTEM_PROMPT` + json(`SCHEMAS`)): отпечаток контракта между промптом, датасетом и обученной моделью. MCP, `read`, `coverage` и `answer_hints` в него не входят.
- **Комплект RTX 5070** — zip с `train_sft.py`, чат-шаблоном, данными SFT и скриптами WSL2; `MANIFEST.json` с sha256; без `.env`, базы и журналов.
