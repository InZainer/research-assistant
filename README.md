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

## Статус

Готовы этапы 1–2 из 10: каркас, проверка провайдера, сбор статей. Ход работы — в разделе «Журнал этапов» файла [PLAN.md](PLAN.md).
