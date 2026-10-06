"""Командная строка проекта: `uv run research-assistant <команда>`."""

import argparse
import json
import sys

from research_assistant.config import Settings, load_settings

CHECK_TOOL = {
    "type": "function",
    "function": {
        "name": "count_papers",
        "description": "Возвращает число статей в библиотеке по заданной теме.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Тема, например 'retrieval-augmented generation'."}
            },
            "required": ["topic"],
        },
    },
}
CHECK_PROMPT = "Сколько статей в библиотеке по теме 'retrieval-augmented generation'? Узнай через инструмент."


def evaluate_check(response) -> dict:
    """Проверяет ответ модели: есть ли вызов инструмента и счётчики usage."""
    message = response.choices[0].message
    calls = message.tool_calls or []
    tool_ok = False
    args = None
    if calls and calls[0].function.name == "count_papers":
        try:
            args = json.loads(calls[0].function.arguments or "{}")
            tool_ok = isinstance(args, dict) and bool(args.get("topic"))
        except json.JSONDecodeError:
            tool_ok = False

    usage = response.usage
    usage_ok = usage is not None and bool(usage.prompt_tokens) and bool(usage.completion_tokens)
    return {
        "tool_calls": tool_ok,
        "tool_args": args,
        "usage": usage_ok,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
    }


def cmd_check(settings: Settings) -> int:
    missing = settings.missing_llm_vars()
    if missing:
        print(f"Не заданы переменные: {', '.join(missing)}. Заполните .env по образцу .env.example.")
        return 2

    from openai import OpenAI

    client = OpenAI(base_url=settings.base_url, api_key=settings.api_key)
    response = client.chat.completions.create(
        model=settings.model,
        messages=[{"role": "user", "content": CHECK_PROMPT}],
        tools=[CHECK_TOOL],
    )
    result = evaluate_check(response)

    print(f"Сервис: {settings.base_url or 'api.openai.com'}, модель: {settings.model}")
    print(f"Вызов инструмента: {'да' if result['tool_calls'] else 'НЕТ'} {result['tool_args'] or ''}")
    print(f"Счётчики usage:    {'да' if result['usage'] else 'НЕТ'} "
          f"(prompt={result['prompt_tokens']}, completion={result['completion_tokens']})")
    return 0 if result["tool_calls"] and result["usage"] else 1


SOURCE_NAMES = ("arxiv", "openalex")


def build_sources(names: list[str], http, settings: Settings) -> list:
    from research_assistant.ingest.sources.arxiv import ArxivSource
    from research_assistant.ingest.sources.openalex import OpenAlexSource

    factories = {
        "arxiv": lambda: ArxivSource(http),
        "openalex": lambda: OpenAlexSource(http, mailto=settings.contact_email, api_key=settings.openalex_api_key),
    }
    return [factories[name]() for name in names]


def cmd_ingest(settings: Settings, args) -> int:
    from research_assistant.ingest.download import FulltextFetcher
    from research_assistant.ingest.http import HttpClient
    from research_assistant.ingest.pipeline import ingest_topic
    from research_assistant.ingest.robots import RobotsPolicy
    from research_assistant.store.db import Library

    names = [n.strip() for n in args.sources.split(",") if n.strip()]
    unknown = [n for n in names if n not in SOURCE_NAMES]
    if unknown or not names:
        print(f"Неизвестные источники: {', '.join(unknown) or '(пусто)'}. Доступны: {', '.join(SOURCE_NAMES)}")
        return 2

    http = HttpClient(settings.user_agent, cache_dir=None if args.refresh else settings.http_cache_dir)
    fetcher = None
    if not args.no_fulltext:
        fetcher = FulltextFetcher(http, RobotsPolicy(http), settings.fulltext_dir)
    try:
        with Library(settings.db_path) as library:
            report = ingest_topic(
                library,
                build_sources(names, http, settings),
                topic=args.topic,
                query=args.query,
                limit=args.limit,
                year_from=args.year_from,
                year_to=args.year_to,
                language=args.language,
                fetcher=fetcher,
                retry_failed=args.retry_failed,
                log=print,
            )
    finally:
        http.close()

    print(f"\nТема «{report.topic}», запрос «{report.query}»")
    print(f"Уникальных статей: {report.unique}; новых: {report.added}; уже были в библиотеке: {report.already_in_library}")
    if report.fulltext:
        print("Полные тексты: " + ", ".join(f"{k} {v}" for k, v in sorted(report.fulltext.items())))
    return 1 if report.errors and not report.found else 0


def cmd_library(settings: Settings, args) -> int:
    from research_assistant.store.db import Library

    if not settings.db_path.exists():
        print("Библиотека пуста. Начните с команды ingest.")
        return 0
    with Library(settings.db_path) as library:
        if args.topic is None:
            summary = library.topic_summary()
            if not summary:
                print("Тем пока нет.")
            for row in summary:
                print(f"{row['name']}: статей {row['papers']}, с полным текстом {row['fulltext']}, на русском {row['russian']}")
            return 0
        papers = library.list_papers(args.topic, limit=args.limit)
        if not papers:
            print(f"По теме «{args.topic}» статей нет.")
        for paper in papers:
            mark = {"ok": paper["fulltext_kind"], "pending": "ожидает"}.get(paper["fulltext_status"], paper["fulltext_status"])
            print(f"#{paper['id']:<4} {paper['year'] or '----'}  [{mark}]  {paper['title'][:90]}  ({paper['sources']})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research-assistant", description="Ассистент исследователя")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="проверить, что модель вызывает инструменты и отдаёт usage")

    ingest = sub.add_parser("ingest", help="найти статьи по теме и добавить в библиотеку")
    ingest.add_argument("topic", help="название темы в библиотеке")
    ingest.add_argument("--query", help="поисковый запрос, если отличается от названия темы")
    ingest.add_argument("--sources", default=",".join(SOURCE_NAMES), help="через запятую: arxiv,openalex")
    ingest.add_argument("--limit", type=int, default=20, help="сколько статей брать из каждого источника")
    ingest.add_argument("--year-from", type=int)
    ingest.add_argument("--year-to", type=int)
    ingest.add_argument("--language", help="код языка, например ru (arXiv при этом пропускается)")
    ingest.add_argument("--no-fulltext", action="store_true", help="не скачивать полные тексты")
    ingest.add_argument("--retry-failed", action="store_true", help="повторить неудачные скачивания")
    ingest.add_argument("--refresh", action="store_true", help="не брать ответы источников из кэша")

    library = sub.add_parser("library", help="показать содержимое библиотеки")
    library.add_argument("--topic", help="показать статьи темы")
    library.add_argument("--limit", type=int, default=50)

    args = parser.parse_args(argv)
    settings = load_settings()
    if args.command == "check":
        return cmd_check(settings)
    if args.command == "ingest":
        return cmd_ingest(settings, args)
    if args.command == "library":
        return cmd_library(settings, args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
