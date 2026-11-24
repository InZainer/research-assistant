"""Командная строка проекта: `uv run research-assistant <команда>`."""

import argparse
from pathlib import Path
import json
import os
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


def cmd_providers(settings: Settings) -> int:
    from research_assistant.providers import describe_all

    print(f"Активный: {settings.provider or 'не выбран (переменные OPENAI_BASE_URL, OPENAI_API_KEY, MODEL)'}"
          f" — выбирается LLM_PROVIDER в .env или флагом --provider")
    for row in describe_all():
        mark = "*" if row["active"] else " "
        print(f"{mark} {row['name']:<10} {row['title']}")
        print(f"    адрес: {row['base_url']}")
        print(f"    модель: {row['model']}")
        print(f"    ключ: {row['key']}")
        if row["note"]:
            print(f"    {row['note']}")
    print(f"Эмбеддинги: {settings.embed_model or 'нет (только BM25)'} — {settings.embed_base_url}")
    return 0


def cmd_check(settings: Settings) -> int:
    missing = settings.missing_llm_vars()
    if missing:
        print(f"Не заданы переменные: {', '.join(missing)}. Заполните .env по образцу .env.example.")
        return 2

    from openai import OpenAI

    client = OpenAI(base_url=settings.base_url, api_key=settings.api_key, max_retries=1)
    try:
        response = client.chat.completions.create(
            model=settings.model,
            messages=[{"role": "user", "content": CHECK_PROMPT}],
            tools=[CHECK_TOOL],
        )
    except Exception as error:  # сервер не запущен, неверный ключ или модель — объяснить, а не падать
        from research_assistant.ui.views import friendly_error

        print(f"Провайдер: {settings.provider or '— (OPENAI_*)'}, сервис: {settings.base_url}, модель: {settings.model}")
        print(friendly_error(error, settings.base_url, settings.model))
        return 1
    result = evaluate_check(response)

    print(f"Провайдер: {settings.provider or '— (OPENAI_*)'}, сервис: {settings.base_url or 'api.openai.com'}, "
          f"модель: {settings.model}")
    print(f"Вызов инструмента: {'да' if result['tool_calls'] else 'НЕТ'} {result['tool_args'] or ''}")
    print(f"Счётчики usage:    {'да' if result['usage'] else 'НЕТ'} "
          f"(prompt={result['prompt_tokens']}, completion={result['completion_tokens']})")
    return 0 if result["tool_calls"] and result["usage"] else 1


SOURCE_NAMES = ("arxiv", "openalex")


def build_sources(names: list[str], http, settings: Settings, include_closed: bool = False,
                  country: str | None = None) -> list:
    from research_assistant.ingest.sources.arxiv import ArxivSource
    from research_assistant.ingest.sources.openalex import OpenAlexSource

    factories = {
        "arxiv": lambda: ArxivSource(http),
        "openalex": lambda: OpenAlexSource(http, mailto=settings.contact_email, api_key=settings.openalex_api_key,
                                           open_access_only=not include_closed, country=country),
    }
    return [factories[name]() for name in names]


def read_lines(path: str) -> list[str]:
    """Строки файла со списком (запросы, ссылки): пустые и начинающиеся с # пропускаются."""
    lines = (line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("#")]


# Массовый сбор ходит по сотням сайтов журналов: мёртвый хост не должен держать его по три минуты на повторах.
BULK_BACKOFF = (5,)
BULK_TIMEOUT = 30


def cmd_ingest(settings: Settings, args) -> int:
    from research_assistant.ingest.download import FulltextFetcher
    from research_assistant.ingest.http import HttpClient
    from research_assistant.ingest.pipeline import ingest_topic
    from research_assistant.ingest.relevance import TermFilter
    from research_assistant.ingest.robots import RobotsPolicy
    from research_assistant.store.db import Library

    names = [n.strip() for n in args.sources.split(",") if n.strip()]
    unknown = [n for n in names if n not in SOURCE_NAMES]
    if unknown or not names:
        print(f"Неизвестные источники: {', '.join(unknown) or '(пусто)'}. Доступны: {', '.join(SOURCE_NAMES)}")
        return 2

    try:
        queries = list(args.query or []) + (read_lines(args.queries_file) if args.queries_file else [])
    except OSError as error:
        print(f"Не удалось прочитать файл запросов: {error}")
        return 2
    http = HttpClient(settings.user_agent, cache_dir=None if args.refresh else settings.http_cache_dir,
                      backoff=BULK_BACKOFF, timeout=BULK_TIMEOUT)
    fetcher = None
    parsed = None
    if not args.no_fulltext:
        fetcher = FulltextFetcher(http, RobotsPolicy(http), settings.fulltext_dir)
    try:
        with Library(settings.db_path) as library:
            report = ingest_topic(
                library,
                build_sources(names, http, settings, include_closed=args.include_closed, country=args.country),
                topic=args.topic,
                queries=queries or None,
                relevance=TermFilter.parse(args.require),
                workers=args.workers,
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
            if args.annotate and not settings.missing_llm_vars():
                from research_assistant.llm import ChatModel
                from research_assistant.nlp.annotate import annotate_papers

                print()
                print("Разметка:")
                model = ChatModel(settings.model, settings.base_url, settings.api_key,
                                  reasoning_effort=settings.reasoning_effort)
                annotate_papers(library, model, topic_id=library.get_or_create_topic(args.topic), log=print)
    finally:
        http.close()

    print(f"\nТема «{report.topic}», запрос «{report.query}»")
    print(f"Уникальных статей: {report.unique}; новых: {report.added}; уже были в библиотеке: {report.already_in_library}"
          + (f"; отброшено не по теме: {report.off_topic}" if report.off_topic else ""))
    if report.fulltext:
        print("Полные тексты: " + ", ".join(f"{k} {v}" for k, v in sorted(report.fulltext.items())))
    if not args.no_parse and parsed:
        print("Разбор: " + ", ".join(f"{k} {v}" for k, v in sorted(parsed.items())))
    if any("лимит запросов исчерпан" in error for error in report.errors.values()):
        print("Источник исчерпал лимит запросов. У OpenAlex без ключа бесплатный дневной лимит общий на IP и "
              "сбрасывается в полночь UTC; с бесплатным ключом (OPENALEX_API_KEY в .env) он больше. "
              "Повторите ту же команду позже: уже найденное не потеряется.")
    return 1 if report.errors and not report.found else 0


def cmd_add(settings: Settings, args) -> int:
    from functools import partial

    from research_assistant.ingest.download import FulltextFetcher
    from research_assistant.ingest.http import HttpClient
    from research_assistant.ingest.pipeline import add_references
    from research_assistant.ingest.resolve import resolve_reference
    from research_assistant.ingest.robots import RobotsPolicy
    from research_assistant.ingest.sources.openalex import OpenAlexSource
    from research_assistant.parse.pipeline import parse_papers
    from research_assistant.store.db import Library

    try:
        references = list(args.reference) + (read_lines(args.file) if args.file else [])
    except OSError as error:
        print(f"Не удалось прочитать файл ссылок: {error}")
        return 2
    if not references:
        print("Нет ссылок: укажите DOI, идентификатор arXiv или адрес страницы статьи (или --file).")
        return 2
    http = HttpClient(settings.user_agent, cache_dir=settings.http_cache_dir, backoff=BULK_BACKOFF,
                      timeout=BULK_TIMEOUT)
    robots = RobotsPolicy(http)
    openalex = OpenAlexSource(http, mailto=settings.contact_email, api_key=settings.openalex_api_key,
                              open_access_only=False)
    try:
        with Library(settings.db_path) as library:
            report = add_references(
                library, references, args.topic,
                resolve=partial(resolve_reference, http=http, robots=robots, openalex=openalex),
                fetcher=None if args.no_fulltext else FulltextFetcher(http, robots, settings.fulltext_dir),
                log=print,
            )
            parsed = parse_papers(library, paper_ids=report.paper_ids, log=print) if report.paper_ids else None
    finally:
        http.close()
    print(f"\nТема «{report.topic}»: добавлено {report.added}, уже были {report.already_in_library}, "
          f"не удалось {len(report.errors)}")
    if report.fulltext:
        print("Полные тексты: " + ", ".join(f"{k} {v}" for k, v in sorted(report.fulltext.items())))
    if parsed:
        print("Разбор: " + ", ".join(f"{k} {v}" for k, v in sorted(parsed.items())))
    return 1 if report.errors and not report.unique else 0


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
        entities = library.paper_entities(args.paper_id)
        if entities:
            for entity_type, names in entities.items():
                print(f"{entity_type}: {'; '.join(names)}")
        keywords = library.paper_keywords(args.paper_id)
        if keywords:
            print(f"ключевые слова: {'; '.join(keywords)}")
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


def make_searcher(settings: Settings, library, with_llm: bool = False):
    from research_assistant.agent.session import make_searcher as build

    return build(settings, library, with_llm=with_llm)


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
        section_kinds=args.section, language=args.language, entity=args.entity,
    )


def cmd_search(settings: Settings, args) -> int:
    from research_assistant.store.db import Library

    with Library(settings.db_path) as library:
        searcher = make_searcher(settings, library, with_llm=args.translate or args.rerank)
        mode = args.mode
        if mode != "bm25" and not searcher.dense_ready:
            print("Эмбеддингов нет (EMBED_MODEL пуст или не выполнен index) — ищу только BM25.")
            mode = "bm25"
        hits = searcher.search(args.query, k=args.k, mode=mode, where=search_filter(args),
                               translate=args.translate, rerank=args.rerank)
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
        searcher = make_searcher(settings, library, with_llm=any("+" in m for m in modes))
        if not searcher.dense_ready:
            dropped = [m for m in modes if not m.startswith("bm25")]
            modes = [m for m in modes if m.startswith("bm25")]
            if dropped:
                print(f"Эмбеддингов нет — пропускаю режимы: {', '.join(dropped)}")
        report = evaluate(searcher, questions, modes, k=args.k)
    report["embed_model"] = settings.embed_model or None
    if searcher.reranker is not None:
        report["llm_model"] = settings.model
        report["llm_tokens"] = searcher.reranker.tokens + searcher.translator.tokens
    report["file"] = str(args.file)
    print(format_report(report))
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    out = settings.runs_dir / f"retrieval-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Подробности по вопросам: {out}")
    return 0


def format_agent_result(result) -> str:
    """Ответ агента с цитатами текстом: для терминала и для MCP-инструмента ask_agent."""
    from research_assistant.answer.citations import format_citation

    lines = []
    if result.stop_reason != "final_answer":
        reason = {"max_steps": "исчерпан лимит шагов", "no_final_answer": "модель не оформила ответ с цитатами"}
        lines.append(f"[!] {reason.get(result.stop_reason, result.stop_reason)} — ответ без проверенных цитат.")
    lines.append(result.answer or "(пустой ответ)")
    if result.not_found:
        lines.append("В библиотеке не нашлось подтверждающих фрагментов.")
    for number, claim in enumerate(result.claims, 1):
        lines += ["", f"{number}. {claim.text}"]
        lines += [f"   {format_citation(citation)}" for citation in claim.citations]
    usage = result.usage
    lines += ["", f"Шагов {result.steps}, вызовов инструментов {result.tool_calls} (ошибок {result.tool_errors}), "
                  f"токенов {usage.total} (prompt {usage.prompt_tokens}, completion {usage.completion_tokens}), "
                  f"{result.seconds:.1f} с. Журнал: {result.log_path}"]
    return "\n".join(lines)


def print_agent_result(result) -> None:
    print()
    print(format_agent_result(result))


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
    from research_assistant.agent.session import run_options
    from research_assistant.store.db import Library

    with Library(settings.db_path) as library:
        made = _agent_or_error(settings, library, external=not args.offline)
        if made is None:
            return 2
        ctx, model, http = made
        log_path = settings.runs_dir / f"ask-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
        try:
            # prefetch: поиск по самому вопросу заранее — лучший вариант в эксперименте E3 (docs/experiments.md)
            result = run_agent(args.question, ctx, model, log_path=log_path, prefetch=True,
                               **run_options(settings))
        finally:
            http.close()
    print_agent_result(result)
    return 0 if result.stop_reason == "final_answer" else 1


def cmd_chat(settings: Settings, args) -> int:
    from datetime import datetime

    from research_assistant.agent.loop import run_agent
    from research_assistant.agent.session import run_options
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
                result = run_agent(question, ctx, model, history=history, log_path=log_path, prefetch=True,
                                   **run_options(settings))
                print_agent_result(result)
                # В историю идут только вопрос и итоговый ответ: выдача инструментов прошлых шагов не раздувает контекст.
                history += [{"role": "user", "content": question}, {"role": "assistant", "content": result.answer}]
                history = history[-2 * args.memory:]
        finally:
            http.close()
    return 0


def _chat_model(settings: Settings, max_tokens: int = 4096):
    """Чат-модель для команд read и coverage; None (с сообщением), если она не настроена."""
    missing = settings.missing_llm_vars()
    if missing:
        print(f"Не заданы переменные: {', '.join(missing)}. Заполните .env по образцу .env.example.")
        return None
    from research_assistant.llm import ChatModel

    return ChatModel(settings.model, settings.base_url, settings.api_key,
                     reasoning_effort=settings.reasoning_effort, max_tokens=max_tokens)


def cmd_read(settings: Settings, args) -> int:
    from datetime import datetime

    from research_assistant.agent.reading import format_reading, read_paper
    from research_assistant.store.db import Library

    model = _chat_model(settings)
    if model is None:
        return 2
    code = 0
    with Library(settings.db_path) as library:
        for paper_id in args.paper_id:
            if library.get_paper(paper_id) is None:
                print(f"Статьи #{paper_id} нет.")
                code = 2
                continue
            log_path = settings.runs_dir / f"read-{paper_id}-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
            reading = read_paper(library, model, paper_id, question=args.question, aspects=args.aspect,
                                 window_chars=args.window or settings.read_window_chars, log_path=log_path)
            print(format_reading(reading))
            print()
            if reading.stop_reason != "report":
                code = code or 1
    return code


def cmd_coverage(settings: Settings, args) -> int:
    from datetime import datetime

    from research_assistant.agent.coverage import check_coverage, format_coverage
    from research_assistant.agent.session import make_searcher
    from research_assistant.retrieval.rerank import QueryTranslator
    from research_assistant.store.db import Library

    model = _chat_model(settings, max_tokens=1024)
    if model is None:
        return 2
    with Library(settings.db_path) as library:
        if _topic_id(library, args.topic) is False:
            return 2
        searcher = make_searcher(settings, library)
        searcher.translator = QueryTranslator(model)
        log_path = settings.runs_dir / f"coverage-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
        report = check_coverage(library, searcher, model, args.aspect, topic=args.topic, paper_ids=args.paper,
                                limit=args.limit, expand=not args.no_expand, log_path=log_path, progress=print)
    print()
    print(format_coverage(report))
    return 0


def cmd_ui(args) -> int:
    import subprocess
    from pathlib import Path

    app = Path(__file__).parent / "ui" / "app.py"
    command = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port),
               "--browser.gatherUsageStats", "false"]
    if args.no_browser:
        command += ["--server.headless", "true"]
    return subprocess.call(command)


def cmd_eval_entities(settings: Settings, args) -> int:
    import json
    from datetime import datetime

    from research_assistant.evaluation.entities import evaluate_entities, format_entity_report, load_gold
    from research_assistant.llm import ChatModel

    model_name = args.model or settings.model
    if not model_name or not settings.api_key:
        print("Не задана модель: укажите --model или MODEL и OPENAI_API_KEY в .env.")
        return 2
    model = ChatModel(model_name, settings.base_url, settings.api_key, reasoning_effort=settings.reasoning_effort)
    report = evaluate_entities(model, load_gold(args.file), log=print)
    print(format_entity_report(report))
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    out = settings.runs_dir / f"entities-{model_name.replace('/', '_')}-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Подробности: {out}")
    return 0


def _topic_id(library, topic: str | None) -> int | None | bool:
    """id темы; None — тема не задана; False — темы нет."""
    if not topic:
        return None
    row = library.conn.execute("SELECT id FROM topics WHERE name = ?", (" ".join(topic.split()),)).fetchone()
    if row is None:
        print(f"Темы «{topic}» нет в библиотеке.")
        return False
    return row["id"]


def cmd_annotate(settings: Settings, args) -> int:
    from research_assistant.llm import ChatModel
    from research_assistant.nlp.annotate import annotate_papers
    from research_assistant.store.db import Library

    model_name = args.model or settings.model
    if not model_name or not settings.api_key:
        print("Не задана модель: укажите --model или MODEL и OPENAI_API_KEY в .env.")
        return 2
    model = ChatModel(model_name, settings.base_url, settings.api_key, reasoning_effort=settings.reasoning_effort)
    with Library(settings.db_path) as library:
        topic_id = _topic_id(library, args.topic)
        if topic_id is False:
            return 2
        statuses = annotate_papers(library, model, topic_id=topic_id, paper_ids=args.paper, force=args.force, log=print)
    if not statuses:
        print("Размечать нечего: все разобранные статьи уже размечены текущей версией (используйте --force).")
    else:
        print("Итого: " + ", ".join(f"{k} {v}" for k, v in sorted(statuses.items())))
    return 0


def cmd_entities(settings: Settings, args) -> int:
    from research_assistant.store.db import Library

    with Library(settings.db_path) as library:
        rows = library.entity_stats(args.type, args.topic, limit=args.limit)
    if not rows:
        print("Сущностей нет: выполните annotate.")
    for row in rows:
        print(f"{row['type']:<8} {row['name'][:50]:<50} статей {row['papers']:>3}, упоминаний {row['mentions']:>4}")
    return 0


def cmd_eval_agent(settings: Settings, args) -> int:
    import json
    from datetime import datetime

    from research_assistant.agent.session import make_agent
    from research_assistant.evaluation.agent import evaluate_agent, format_agent_report
    from research_assistant.evaluation.retrieval import load_questions
    from research_assistant.store.db import Library

    questions = load_questions(args.file)
    if args.rescore:
        from research_assistant.evaluation.agent import rescore_log

        with Library(settings.db_path) as library:
            report = rescore_log(library, questions, Path(args.rescore), args.variant)
        print(format_agent_report(report))
        return 0
    if settings.missing_llm_vars():
        print("Не задана модель: заполните MODEL и OPENAI_API_KEY в .env.")
        return 2
    # Отклонённые кандидаты синтетических наборов (data/finetune/questions) не оцениваем.
    questions = [q for q in questions if q.meta.get("accepted", True) is not False]
    # Вопросам external нужен поиск в интернете; остальным его не даём, как в обучающих траекториях.
    skipped = sum((q.kind == "external") != args.external for q in questions)
    questions = [q for q in questions if (q.kind == "external") == args.external]
    if skipped:
        print(f"Пропущено вопросов {'не ' if args.external else ''}типа external: {skipped}"
              + ("" if args.external else " (их оценивает --external)"))
    if args.only:
        questions = [q for q in questions if any(q.id.startswith(prefix) for prefix in args.only.split(","))]
    if args.limit:
        questions = questions[: args.limit]
    with Library(settings.db_path) as library:
        _, model, http = make_agent(settings, library, external=False)
        http.close()

        def fresh_context():
            ctx, _, client = make_agent(settings, library, external=args.external)
            if not args.external:
                client.close()
            return ctx

        report = evaluate_agent(fresh_context, model, questions, variant=args.variant, runs=args.runs,
                                max_steps=settings.max_steps, log_dir=settings.runs_dir, log=print)
    print(format_agent_report(report))
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    out = settings.runs_dir / f"agent-{args.variant}-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Подробности: {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research-assistant", description="Ассистент исследователя")
    parser.add_argument("--provider", help="AI-сервис для чат-модели: lmstudio, inception, openai, anthropic, "
                                           "openrouter, ollama, deepseek, gemini, custom (см. команду providers)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="проверить, что модель вызывает инструменты и отдаёт usage")
    sub.add_parser("providers", help="профили AI-сервисов: адрес, модель, есть ли ключ (значения не печатаются)")

    ingest = sub.add_parser("ingest", help="найти статьи по теме и добавить в библиотеку")
    ingest.add_argument("topic", help="название темы в библиотеке")
    ingest.add_argument("--query", action="append",
                        help="поисковый запрос, если отличается от названия темы; можно несколько раз — "
                             "синонимы и варианты на русском и английском складываются в одну тему")
    ingest.add_argument("--queries-file", help="файл с запросами, по одному в строке (# — комментарий)")
    ingest.add_argument("--sources", default=",".join(SOURCE_NAMES), help="через запятую: arxiv,openalex")
    ingest.add_argument("--limit", type=int, default=20,
                        help="сколько статей брать из каждого источника на каждый запрос (OpenAlex — до 2000)")
    ingest.add_argument("--require", action="append",
                        help="группа слов темы через «|»: хотя бы одно должно быть в названии или аннотации; "
                             "флаг можно повторить — тогда нужны слова из каждой группы "
                             "(--require \"ресторан|кафе|restaurant\" --require \"нейросеть|machine learning\")")
    ingest.add_argument("--country", help="код страны организаций авторов для OpenAlex, например RU: работы "
                                          "российских авторов находятся и по английским запросам")
    ingest.add_argument("--include-closed", action="store_true",
                        help="брать из OpenAlex и статьи без открытого доступа: в библиотеку попадёт аннотация")
    ingest.add_argument("--year-from", type=int)
    ingest.add_argument("--year-to", type=int)
    ingest.add_argument("--language", help="код языка, например ru (arXiv при этом пропускается)")
    ingest.add_argument("--no-fulltext", action="store_true", help="не скачивать полные тексты")
    ingest.add_argument("--workers", type=int, default=6,
                        help="потоков скачивания полных текстов; один сайт всегда качает один поток")
    ingest.add_argument("--retry-failed", action="store_true", help="повторить неудачные скачивания")
    ingest.add_argument("--refresh", action="store_true", help="не брать ответы источников из кэша")
    ingest.add_argument("--no-parse", action="store_true", help="не разбирать полные тексты после скачивания")
    ingest.add_argument("--annotate", action="store_true", help="разметить новые статьи моделью из .env")

    add = sub.add_parser("add", help="добавить статьи по DOI, идентификатору arXiv или адресу страницы")
    add.add_argument("reference", nargs="*", help="DOI, arXiv id или адрес страницы статьи (например, CyberLeninka)")
    add.add_argument("--topic", required=True, help="тема библиотеки, в которую добавить")
    add.add_argument("--file", help="файл со ссылками, по одной в строке (# — комментарий)")
    add.add_argument("--no-fulltext", action="store_true", help="не скачивать полные тексты")

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
    search.add_argument("--entity", help="только фрагменты, где упоминается сущность, например MT-Bench")
    search.add_argument("--translate", action="store_true", help="добавить BM25 по английскому переводу запроса")
    search.add_argument("--rerank", action="store_true", help="переупорядочить кандидатов чат-моделью")
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

    read = sub.add_parser("read", help="прочитать статью целиком: что в ней есть, чего нет, расхождения")
    read.add_argument("paper_id", type=int, nargs="+", help="id статьи (можно несколько)")
    read.add_argument("--question", help="вопрос к статье; без него — общий отчёт о прочтении")
    read.add_argument("--aspect", action="append",
                      help="что проверить в статье (можно несколько раз): по каждому — есть с цитатой или нет")
    read.add_argument("--window", type=int, help="символов текста на один вызов модели (по умолчанию из настроек)")

    coverage = sub.add_parser("coverage", help="проверить по каждой статье темы, рассмотрен ли в ней аспект")
    coverage.add_argument("aspect", help="что ищем, например «расстановка мебели в непрямоугольном зале»")
    coverage.add_argument("--topic", help="тема библиотеки; без неё — вся библиотека")
    coverage.add_argument("--paper", type=int, action="append", help="только эти статьи (можно несколько раз)")
    coverage.add_argument("--limit", type=int, default=60, help="сколько самых близких статей проверить моделью")
    coverage.add_argument("--no-expand", action="store_true", help="не переформулировать аспект моделью")

    ui = sub.add_parser("ui", help="веб-интерфейс: вопрос агенту, поиск, библиотека (Streamlit)")
    ui.add_argument("--port", type=int, default=8501)
    ui.add_argument("--no-browser", action="store_true", help="не открывать браузер")

    mcp = sub.add_parser("mcp", help="MCP-сервер (stdio) для внешних агентов: Claude Code, Hermes, Codex, Cursor")
    mcp.add_argument("--project-dir", help="папка проекта с .env и data/; по умолчанию текущая или папка установки")
    mcp.add_argument("--allow-model", action="store_true",
                     help="добавить ask_agent, read_paper, check_coverage: вызывают модель из .env и тратят ключ")
    mcp.add_argument("--allow-write", action="store_true",
                     help="добавить search_external и add_papers: поиск в интернете и пополнение библиотеки")
    mcp.add_argument("--no-log", action="store_true", help="не писать журнал вызовов runs/mcp-*.jsonl")

    eval_entities = sub.add_parser("eval-entities", help="оценить извлечение сущностей на золотом наборе")
    eval_entities.add_argument("--file", default="eval/entities_gold.jsonl")
    eval_entities.add_argument("--model", help="чат-модель; по умолчанию MODEL из .env")

    annotate = sub.add_parser("annotate", help="разметить статьи: сущности (модель) и ключевые слова (YAKE)")
    annotate.add_argument("--topic")
    annotate.add_argument("--paper", type=int, action="append")
    annotate.add_argument("--force", action="store_true", help="переразметить уже размеченные")
    annotate.add_argument("--model", help="чат-модель; по умолчанию MODEL из .env")

    entities = sub.add_parser("entities", help="сущности библиотеки по числу статей")
    entities.add_argument("--type", choices=("method", "dataset", "metric", "task"))
    entities.add_argument("--topic")
    entities.add_argument("--limit", type=int, default=25)

    eval_agent = sub.add_parser("eval-agent", help="оценить агента на золотых вопросах (pass@1, pass^k)")
    eval_agent.add_argument("--variant", choices=("agent", "single", "quotes", "prefetch"), default="agent",
                            help="agent — итеративный; single — один поиск и ответ (E3); quotes — цитаты текстом (E2)")
    eval_agent.add_argument("--runs", type=int, default=1, help="прогонов каждого вопроса (для pass^k)")
    eval_agent.add_argument("--file", default="eval/retrieval_gold.jsonl")
    eval_agent.add_argument("--only", help="префиксы id вопросов через запятую, например en,xl")
    eval_agent.add_argument("--limit", type=int, help="первые N вопросов")
    eval_agent.add_argument("--rescore", help="пересчитать метрики по журналу runs/eval-agent-*.jsonl без модели")
    eval_agent.add_argument("--external", action="store_true",
                            help="оценить только вопросы типа external (с поиском новых статей в интернете)")

    from research_assistant.finetune.commands import COMMANDS as FT_COMMANDS
    from research_assistant.finetune.commands import add_parsers as add_finetune_parsers

    add_finetune_parsers(sub)

    args = parser.parse_args(argv)
    if args.provider:
        from research_assistant.providers import get_provider

        try:
            get_provider(args.provider)
        except ValueError as error:
            print(f"Ошибка: {error}")
            return 2
        # Через окружение — чтобы выбор увидел и процесс веб-интерфейса (команда ui запускает Streamlit).
        os.environ["LLM_PROVIDER"] = args.provider.lower()
    if args.command == "mcp":
        from research_assistant.mcp_server import project_root

        # .env и data/ ищутся от текущей папки: внешний агент может запустить сервер откуда угодно.
        os.chdir(project_root(args.project_dir))
    settings = load_settings()
    if args.command == "providers":
        return cmd_providers(settings)
    if args.command == "check":
        return cmd_check(settings)
    if args.command == "ingest":
        return cmd_ingest(settings, args)
    if args.command == "add":
        return cmd_add(settings, args)
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
    if args.command == "annotate":
        return cmd_annotate(settings, args)
    if args.command == "entities":
        return cmd_entities(settings, args)
    if args.command == "eval-entities":
        return cmd_eval_entities(settings, args)
    if args.command == "eval-agent":
        return cmd_eval_agent(settings, args)
    if args.command == "ask":
        return cmd_ask(settings, args)
    if args.command == "chat":
        return cmd_chat(settings, args)
    if args.command == "read":
        return cmd_read(settings, args)
    if args.command == "coverage":
        return cmd_coverage(settings, args)
    if args.command == "ui":
        return cmd_ui(args)
    if args.command == "mcp":
        from research_assistant.mcp_server import run_stdio

        return run_stdio(settings, allow_model=args.allow_model, allow_write=args.allow_write, log=not args.no_log)
    if args.command in FT_COMMANDS:
        from research_assistant.finetune.commands import run as run_finetune

        return run_finetune(settings, args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
