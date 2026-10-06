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
    parsed = None
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
            if not args.no_parse:
                from research_assistant.parse.pipeline import parse_papers

                print()
                print("Разбор полных текстов:")
                parsed = parse_papers(library, topic_id=library.get_or_create_topic(args.topic), log=print)
    finally:
        http.close()

    print(f"\nТема «{report.topic}», запрос «{report.query}»")
    print(f"Уникальных статей: {report.unique}; новых: {report.added}; уже были в библиотеке: {report.already_in_library}")
    if report.fulltext:
        print("Полные тексты: " + ", ".join(f"{k} {v}" for k, v in sorted(report.fulltext.items())))
    if not args.no_parse and parsed:
        print("Разбор: " + ", ".join(f"{k} {v}" for k, v in sorted(parsed.items())))
    return 1 if report.errors and not report.found else 0


def cmd_parse(settings: Settings, args) -> int:
    from research_assistant.parse.pipeline import parse_papers
    from research_assistant.store.db import Library

    if not settings.db_path.exists():
        print("Библиотека пуста. Начните с команды ingest.")
        return 0
    with Library(settings.db_path) as library:
        topic_id = None
        if args.topic:
            row = library.conn.execute("SELECT id FROM topics WHERE name = ?", (" ".join(args.topic.split()),)).fetchone()
            if row is None:
                print(f"Темы «{args.topic}» нет в библиотеке.")
                return 2
            topic_id = row["id"]
        statuses = parse_papers(library, topic_id=topic_id, paper_ids=args.paper, force=args.force, log=print)
    if not statuses:
        print("Разбирать нечего: все статьи уже разобраны текущей версией (используйте --force).")
    else:
        print("Итого: " + ", ".join(f"{k} {v}" for k, v in sorted(statuses.items())))
    return 0


def cmd_show(settings: Settings, args) -> int:
    from research_assistant.store.db import Library

    with Library(settings.db_path) as library:
        paper = library.get_paper(args.paper_id)
        if paper is None:
            print(f"Статьи #{args.paper_id} нет.")
            return 2
        parse = library.get_parse(args.paper_id)
        print(f"#{paper['id']} {paper['title']} ({paper['year']}, {paper['language'] or '?'})")
        if parse is None:
            print("Ещё не разобрана: выполните parse.")
            return 0
        print(f"Разбор: {parse['status']}, парсер {parse['parser']} v{parse['parser_version']}, "
              f"страниц {parse['n_pages']}, аннотация {'есть' if parse['has_abstract'] else 'нет'}"
              + (f", ошибка: {parse['error']}" if parse["error"] else ""))
        for section in library.paper_structure(args.paper_id):
            print()
            print(f"{'#' * section['level']} {section['title'] or '(без заголовка)'}  [{section['kind']}]")
            for passage in section["passages"]:
                where = f"с. {passage['page']}" if passage["page"] else passage["source"]
                mark = " подпись" if passage["kind"] == "caption" else ""
                if args.sentences:
                    print(f"  ¶{passage['id']} ({where}{mark})")
                    for sentence in passage["sentences"]:
                        print(f"    [s{sentence['id']}] {sentence['text']}")
                else:
                    text = passage["text"]
                    cut = text if len(text) <= args.max_chars else text[: args.max_chars] + " …"
                    print(f"  ¶{passage['id']} ({where}{mark}, предл. {len(passage['sentences'])}): {cut}")
    return 0


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


def make_searcher(settings: Settings, library):
    from research_assistant.agent.session import make_searcher as build

    return build(settings, library)


def cmd_index(settings: Settings, args) -> int:
    from research_assistant.nlp.embed import OpenAIEmbedder
    from research_assistant.retrieval.search import build_index
    from research_assistant.store.db import Library
    from research_assistant.store.vectors import VectorIndex, index_path

    if not settings.embed_model:
        print("Не задана EMBED_MODEL: плотный поиск выключен, работает только BM25. См. .env.example.")
        return 2
    embedder = OpenAIEmbedder(settings.embed_model, settings.embed_base_url, settings.embed_api_key)
    index = VectorIndex.load(index_path(settings.vectors_dir, settings.embed_model))
    with Library(settings.db_path) as library:
        stats = build_index(library, embedder, index, log=print)
    print(f"Готово: в индексе {len(index)} векторов (добавлено {stats['added']}, удалено {stats['removed']}), "
          f"модель {settings.embed_model}")
    return 0


def search_filter(args):
    from research_assistant.store.db import PassageFilter

    return PassageFilter(
        topic=args.topic, year_from=args.year_from, year_to=args.year_to,
        section_kinds=args.section, language=args.language,
    )


def cmd_search(settings: Settings, args) -> int:
    from research_assistant.store.db import Library

    with Library(settings.db_path) as library:
        searcher = make_searcher(settings, library)
        mode = args.mode
        if mode != "bm25" and not searcher.dense_ready:
            print("Эмбеддингов нет (EMBED_MODEL пуст или не выполнен index) — ищу только BM25.")
            mode = "bm25"
        hits = searcher.search(args.query, k=args.k, mode=mode, where=search_filter(args))
    if not hits:
        print("Ничего не найдено.")
    for place, hit in enumerate(hits, 1):
        p = hit.passage
        where = f"с. {p['page']}" if p.get("page") else p.get("source", "")
        ranks = ", ".join(f"{name} #{rank}" for name, rank in hit.ranks.items())
        print()
        print(f"{place}. [{hit.score:.3f}; {ranks}] #{p['paper_id']} {p['title'][:80]} ({p['year']})")
        print(f"   {p['section'] or '(без заголовка)'} [{p['section_kind']}], {where}, ¶{hit.passage_id}")
        text = p["text"]
        print("   " + (text if len(text) <= args.max_chars else text[: args.max_chars] + " …"))
    return 0


def cmd_eval_retrieval(settings: Settings, args) -> int:
    import json
    from datetime import datetime

    from research_assistant.evaluation.retrieval import evaluate, format_report, load_questions
    from research_assistant.store.db import Library

    questions = load_questions(args.file)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    with Library(settings.db_path) as library:
        searcher = make_searcher(settings, library)
        if not searcher.dense_ready:
            dropped = [m for m in modes if m != "bm25"]
            modes = [m for m in modes if m == "bm25"]
            if dropped:
                print(f"Эмбеддингов нет — пропускаю режимы: {', '.join(dropped)}")
        report = evaluate(searcher, questions, modes, k=args.k)
    report["embed_model"] = settings.embed_model or None
    report["file"] = str(args.file)
    print(format_report(report))
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    out = settings.runs_dir / f"retrieval-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Подробности по вопросам: {out}")
    return 0


def print_agent_result(result) -> None:
    from research_assistant.answer.citations import format_citation

    print()
    if result.stop_reason != "final_answer":
        reason = {"max_steps": "исчерпан лимит шагов", "no_final_answer": "модель не оформила ответ с цитатами"}
        print(f"[!] {reason.get(result.stop_reason, result.stop_reason)} — ответ без проверенных цитат.")
    print(result.answer or "(пустой ответ)")
    if result.not_found:
        print("В библиотеке не нашлось подтверждающих фрагментов.")
    for number, claim in enumerate(result.claims, 1):
        print()
        print(f"{number}. {claim.text}")
        for citation in claim.citations:
            print(f"   {format_citation(citation)}")
    usage = result.usage
    print()
    print(f"Шагов {result.steps}, вызовов инструментов {result.tool_calls} (ошибок {result.tool_errors}), "
          f"токенов {usage.total} (prompt {usage.prompt_tokens}, completion {usage.completion_tokens}), "
          f"{result.seconds:.1f} с. Журнал: {result.log_path}")


def _agent_or_error(settings: Settings, library, external: bool):
    missing = settings.missing_llm_vars()
    if missing:
        print(f"Не заданы переменные: {', '.join(missing)}. Заполните .env по образцу .env.example.")
        return None
    from research_assistant.agent.session import make_agent

    return make_agent(settings, library, external=external)


def cmd_ask(settings: Settings, args) -> int:
    from datetime import datetime

    from research_assistant.agent.loop import run_agent
    from research_assistant.store.db import Library

    with Library(settings.db_path) as library:
        made = _agent_or_error(settings, library, external=not args.offline)
        if made is None:
            return 2
        ctx, model, http = made
        log_path = settings.runs_dir / f"ask-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
        try:
            result = run_agent(args.question, ctx, model, max_steps=settings.max_steps, log_path=log_path)
        finally:
            http.close()
    print_agent_result(result)
    return 0 if result.stop_reason == "final_answer" else 1


def cmd_chat(settings: Settings, args) -> int:
    from datetime import datetime

    from research_assistant.agent.loop import run_agent
    from research_assistant.store.db import Library

    with Library(settings.db_path) as library:
        made = _agent_or_error(settings, library, external=not args.offline)
        if made is None:
            return 2
        ctx, model, http = made
        log_path = settings.runs_dir / f"chat-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
        history: list[dict] = []
        print(f"Ассистент исследователя (модель {settings.model}). Пустая строка или /exit — выход.")
        try:
            while True:
                try:
                    question = input("\nВы: ").strip()
                except EOFError:
                    break
                if not question or question in ("/exit", "/quit", "выход"):
                    break
                ctx.seen_sentences.clear()  # цитировать можно только то, что найдено для этого вопроса
                result = run_agent(question, ctx, model, max_steps=settings.max_steps, history=history,
                                   log_path=log_path)
                print_agent_result(result)
                # В историю идут только вопрос и итоговый ответ: выдача инструментов прошлых шагов не раздувает контекст.
                history += [{"role": "user", "content": question}, {"role": "assistant", "content": result.answer}]
                history = history[-2 * args.memory:]
        finally:
            http.close()
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
    ingest.add_argument("--no-parse", action="store_true", help="не разбирать полные тексты после скачивания")

    parse = sub.add_parser("parse", help="разобрать полные тексты на разделы, фрагменты и предложения")
    parse.add_argument("--topic", help="только статьи темы")
    parse.add_argument("--paper", type=int, action="append", help="id статьи (можно несколько раз)")
    parse.add_argument("--force", action="store_true", help="переразобрать уже разобранные")

    show = sub.add_parser("show", help="показать разбор статьи")
    show.add_argument("paper_id", type=int)
    show.add_argument("--sentences", action="store_true", help="показать предложения с ID")
    show.add_argument("--max-chars", type=int, default=160, help="обрезать фрагменты до этой длины")

    library = sub.add_parser("library", help="показать содержимое библиотеки")
    library.add_argument("--topic", help="показать статьи темы")
    library.add_argument("--limit", type=int, default=50)

    sub.add_parser("index", help="посчитать эмбеддинги фрагментов для плотного поиска")

    search = sub.add_parser("search", help="найти фрагменты по запросу")
    search.add_argument("query")
    search.add_argument("--k", type=int, default=5, help="сколько фрагментов показать")
    search.add_argument("--mode", choices=("hybrid", "bm25", "dense"), default="hybrid")
    search.add_argument("--topic")
    search.add_argument("--year-from", type=int)
    search.add_argument("--year-to", type=int)
    search.add_argument("--section", action="append", help="тип раздела: method, results, limitations…")
    search.add_argument("--language", help="язык статьи: en, ru")
    search.add_argument("--max-chars", type=int, default=300)

    evaluation = sub.add_parser("eval-retrieval", help="оценить поиск на золотом наборе вопросов")
    evaluation.add_argument("--file", default="eval/retrieval_gold.jsonl")
    evaluation.add_argument("--modes", default="bm25,dense,hybrid")
    evaluation.add_argument("--k", type=int, default=10)

    ask = sub.add_parser("ask", help="задать вопрос агенту: ответ с цитатами из библиотеки")
    ask.add_argument("question")
    ask.add_argument("--offline", action="store_true", help="без внешнего поиска и добавления статей")

    chat = sub.add_parser("chat", help="диалог с агентом в терминале")
    chat.add_argument("--offline", action="store_true", help="без внешнего поиска и добавления статей")
    chat.add_argument("--memory", type=int, default=3, help="сколько прошлых вопросов и ответов помнить")

    args = parser.parse_args(argv)
    settings = load_settings()
    if args.command == "check":
        return cmd_check(settings)
    if args.command == "ingest":
        return cmd_ingest(settings, args)
    if args.command == "library":
        return cmd_library(settings, args)
    if args.command == "parse":
        return cmd_parse(settings, args)
    if args.command == "show":
        return cmd_show(settings, args)
    if args.command == "index":
        return cmd_index(settings, args)
    if args.command == "search":
        return cmd_search(settings, args)
    if args.command == "eval-retrieval":
        return cmd_eval_retrieval(settings, args)
    if args.command == "ask":
        return cmd_ask(settings, args)
    if args.command == "chat":
        return cmd_chat(settings, args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
