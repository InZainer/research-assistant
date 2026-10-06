# Ассистент исследователя

ИИ-агент, который по заданной теме собирает научные статьи, хранит и размечает их, ищет по ним и отвечает на вопросы точными цитатами со ссылкой на статью, раздел и страницу.

Проект по курсу «Агентный ИИ: теория и инженерия» (MMCS, осень 2026). План и архитектура — в [PLAN.md](PLAN.md).

## Запуск

Нужны Python 3.12+ и [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env    # затем вписать адрес сервиса, ключ и модель
uv run research-assistant check
uv run pytest
```

`check` делает один запрос к модели и показывает, умеет ли она вызывать инструменты и возвращает ли счётчики токенов. Без этого агент работать не будет.

## Модель

Подойдёт любой сервис с OpenAI-совместимым API. Например, локальный LM Studio:

```
OPENAI_BASE_URL=http://localhost:1234/v1
OPENAI_API_KEY=lm-studio
MODEL=<id модели из LM Studio>
```

## Сбор статей

```bash
# английские статьи: OpenAlex + arXiv, по 40 результатов из каждого источника, с 2023 года
uv run research-assistant ingest "speculative decoding" --limit 40 --year-from 2023

# русские статьи: только OpenAlex с фильтром по языку
uv run research-assistant ingest "машинное обучение в медицине" --language ru --limit 30

# что лежит в библиотеке
uv run research-assistant library
uv run research-assistant library --topic "speculative decoding"
```

Статьи из разных источников склеиваются по DOI, идентификатору arXiv, идентификатору OpenAlex и нормализованному заголовку. Полные тексты сохраняются в `data/fulltext/`: PDF или HTML-страница статьи CyberLeninka. Перед скачиванием проверяется robots.txt сайта, паузы между запросами берутся из его Crawl-delay (у arxiv.org это 15 секунд, поэтому сбор 30 статей занимает минут десять). Ответы API кэшируются в `data/cache/`; флаг `--refresh` их обходит, `--retry-failed` повторяет неудачные скачивания, `--no-fulltext` пропускает скачивание.

## Разбор полных текстов

`ingest` сразу разбирает скачанные статьи (флаг `--no-parse` это отключает). Отдельно:

```bash
uv run research-assistant parse                  # разобрать всё, что ещё не разобрано текущей версией
uv run research-assistant parse --force --paper 10
uv run research-assistant show 10                # разделы и фрагменты со страницами
uv run research-assistant show 10 --sentences    # предложения с ID: на них будут ссылаться цитаты
```

Статья делится на разделы с типом (`introduction`, `method`, `results`, `appendix`…), разделы — на фрагменты до 220 слов, фрагменты — на предложения. У каждого предложения есть позиция в тексте фрагмента и номер страницы PDF. Колонтитулы, сноски, формулы, шапка статьи и список литературы отбрасываются. Если полного текста нет, в поиск попадает аннотация из метаданных.

## Поиск

```bash
uv run research-assistant search "how does tree-based verification work" --k 5
uv run research-assistant search "ограничения метода" --section limitations --topic "speculative decoding"
uv run research-assistant search "draft model latency" --mode bm25     # bm25 | dense | hybrid
```

BM25 работает сразу: полнотекстовый индекс SQLite FTS5 строится вместе с разбором, слова приводятся к основам (английский и русский). Для плотного поиска нужна многоязычная модель эмбеддингов — по замерам лучшая из опробованных `bge-m3` (в LM Studio: `lms get https://huggingface.co/ggml-org/bge-m3-Q8_0-GGUF`, затем `EMBED_MODEL=text-embedding-bge-m3` в `.env`). Векторы считаются один раз и потом досчитываются для новых статей:

```bash
uv run research-assistant index
```

По умолчанию поиск гибридный: списки BM25 и плотного поиска сливаются по местам (Reciprocal Rank Fusion). Без эмбеддингов он сам переключается на BM25.

## Оценка поиска

```bash
uv run research-assistant eval-retrieval                 # Recall@5/10, MRR@10, nDCG@10 по режимам
uv run research-assistant eval-retrieval --modes bm25
```

Результаты экспериментов — в [docs/experiments.md](docs/experiments.md). Вопросы лежат в `eval/retrieval_gold.jsonl`: ответ задан названием статьи и дословной цитатой, поэтому набор не ломается при переразборе. Метрики считаются отдельно для групп «язык вопроса→язык статьи» (en→en, ru→en, ru→ru), подробности по каждому вопросу сохраняются в `runs/`.

## Агент

```bash
uv run research-assistant ask "Почему спекулятивное декодирование хуже работает для малоресурсных языков?"
uv run research-assistant chat            # диалог в терминале; пустая строка или /exit — выход
uv run research-assistant chat --offline  # без поиска новых статей в OpenAlex/arXiv
```

Агент сам ищет по библиотеке (в том числе на втором языке), при необходимости читает соседние фрагменты и отвечает утверждениями с дословными цитатами: статья, раздел, страница, ID предложения. Цитаты подставляет код по ID, поэтому они всегда дословные; ответ принимается, только если агент действительно видел процитированные предложения в выдаче. По просьбе пользователя агент может найти новые статьи (`search_external`) и добавить их в библиотеку (`add_papers`). Каждый прогон пишется в `runs/*.jsonl`: шаги, вызовы инструментов, ошибки, токены.

Локальная модель в LM Studio должна быть загружена с достаточным контекстом:

```bash
lms load qwen3.5-4b --context-length 16384 --gpu max -y
```

В `.env`: `MODEL=qwen3.5-4b`, `REASONING_EFFORT=none` (иначе Qwen3.5 тратит сотни токенов на «размышления» на каждом шаге).

## Статус

Готовы этапы 1–5 из 10: каркас, сбор статей, разбор полных текстов, поиск, агент. Ход работы — в разделе «Журнал этапов» файла [PLAN.md](PLAN.md).
