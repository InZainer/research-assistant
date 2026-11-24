"""Команды CLI для датасета дообучения: ft-split, ft-questions, ft-collect, ft-export (docs/finetune.md).

Вынесены из cli.py, чтобы он не рос: cli.py только добавляет парсеры и передаёт управление сюда.
"""

import argparse
import json
from pathlib import Path

from research_assistant.config import Settings

COMMANDS = ("ft-split", "ft-questions", "ft-collect", "ft-export")

# Заказ кандидатов по умолчанию; после проверок принимается около 70%. Пилот — с --limit.
DEFAULT_COUNTS = {
    "train": {"fact_en": 300, "fact_xl": 300, "fact_ru": 250, "entity": 100, "paper_lookup": 80,
              "unanswerable": 150, "followup": 100, "external": 40, "compare": 80},
    "val": {"fact_en": 30, "fact_xl": 30, "fact_ru": 30, "entity": 10, "paper_lookup": 10,
            "unanswerable": 20, "followup": 10, "external": 5, "compare": 10},
    "test": {"fact_en": 50, "fact_xl": 50, "fact_ru": 50, "entity": 20, "paper_lookup": 15,
             "unanswerable": 40, "followup": 20, "external": 10, "compare": 15},
}


def add_parsers(sub) -> None:
    split = sub.add_parser("ft-split", help="дообучение: разбить статьи на train/val/test (без модели)")
    split.add_argument("--gold", default="eval/retrieval_gold.jsonl", help="золото: его статьи целиком уходят в test")
    split.add_argument("--val-papers", type=int, default=6)
    split.add_argument("--seed", type=int, default=0)

    questions = sub.add_parser("ft-questions", help="дообучение: синтез вопросов учителем из .env")
    questions.add_argument("--split", choices=("train", "val", "test"), required=True)
    questions.add_argument("--kinds", help="заказ по типам: fact_en=300,unanswerable=150 (по умолчанию — план сплита)")
    questions.add_argument("--per-paper-max", type=int, default=30)
    questions.add_argument("--limit", type=int, help="обработать не больше N новых кандидатов (пилот)")
    questions.add_argument("--seed", type=int, default=0)
    questions.add_argument("--gold", default="eval/retrieval_gold.jsonl")
    questions.add_argument("--restart", action="store_true", help="начать файл вопросов заново")
    questions.add_argument("--dry-run", action="store_true", help="только план выборки, без модели")

    collect = sub.add_parser("ft-collect", help="дообучение: траектории учителя по принятым вопросам")
    collect.add_argument("--split", choices=("train", "val", "test"), required=True)
    collect.add_argument("--samples", type=int, default=2)
    collect.add_argument("--temperatures", default="0.2,0.7", help="температуры сэмплов через запятую")
    collect.add_argument("--kinds", help="только эти типы вопросов, через запятую")
    collect.add_argument("--limit", type=int, help="первые N принятых вопросов")

    export = sub.add_parser("ft-export", help="дообучение: отбор траекторий и примеры SFT в data/finetune/sft")
    export.add_argument("--max-tokens", type=int, default=10240)
    export.add_argument("--recovery-share", type=float, default=0.1)


def run(settings: Settings, args: argparse.Namespace) -> int:
    from research_assistant.finetune.questions import TeacherConfigError

    handlers = {"ft-split": _split, "ft-questions": _questions, "ft-collect": _collect, "ft-export": _export}
    try:
        return handlers[args.command](settings, args)
    except (TeacherConfigError, ValueError, FileNotFoundError) as error:
        print(f"Ошибка: {error}")
        return 2


def parse_counts(raw: str | None, split: str) -> dict[str, int]:
    if not raw:
        return dict(DEFAULT_COUNTS[split])
    counts = {}
    for part in raw.split(","):
        name, _, number = part.partition("=")
        if not number.strip().isdigit():
            raise ValueError(f"заказ типов задаётся как kind=число через запятую, получено {part!r}")
        counts[name.strip()] = int(number)
    return counts


def finetune_dir(settings: Settings) -> Path:
    return settings.data_dir / "finetune"


def _splits_path(settings: Settings) -> Path:
    return finetune_dir(settings) / "splits.json"


def _load_splits(settings: Settings) -> dict:
    from research_assistant.finetune.splits import load_splits

    if not _splits_path(settings).exists():
        raise FileNotFoundError(f"нет {_splits_path(settings)}: сначала research-assistant ft-split")
    return load_splits(_splits_path(settings))


def _teacher(settings: Settings):
    from research_assistant.finetune.trajectories import is_local_url
    from research_assistant.llm import ChatModel

    missing = settings.missing_llm_vars()
    if missing:
        raise ValueError(f"не заданы {', '.join(missing)} в .env")
    if is_local_url(settings.base_url):
        raise ValueError(f"учитель должен быть облачной моделью, а адрес чат-модели {settings.base_url} — это "
                         "ученик в LM Studio. Уберите переопределение переменных окружения.")
    return ChatModel(settings.model, settings.base_url, settings.api_key, reasoning_effort=settings.reasoning_effort)


def _split(settings: Settings, args) -> int:
    from research_assistant.evaluation.retrieval import load_questions
    from research_assistant.finetune.splits import gold_paper_ids, save_splits, split_papers
    from research_assistant.store.db import Library

    gold = load_questions(args.gold)
    with Library(settings.db_path) as library:
        _, missing = gold_paper_ids(library, gold)
        splits = split_papers(library, gold, val_papers=args.val_papers, seed=args.seed, gold_file=args.gold)
    if missing:
        print(f"Статей из золота нет в библиотеке: {', '.join(sorted(set(missing)))}")
    save_splits(splits, _splits_path(settings))
    print("Статей: " + ", ".join(f"{name} {len(splits[name])}" for name in ("train", "val", "test"))
          + f". Записано в {_splits_path(settings)}")
    return 0


def _questions(settings: Settings, args) -> int:
    from research_assistant.agent.session import make_searcher
    from research_assistant.evaluation.retrieval import load_questions
    from research_assistant.finetune.questions import (
        gold_neighbourhood, load_held_out, plan_sampling, synthesize,
    )
    from research_assistant.store.db import Library

    splits = _load_splits(settings)
    counts = parse_counts(args.kinds, args.split)
    gold = load_questions(args.gold) if args.split == "test" else ()
    out = finetune_dir(settings) / "questions" / f"{args.split}.jsonl"
    with Library(settings.db_path) as library:
        if args.dry_run:
            exclude = gold_neighbourhood(library, gold) if args.split == "test" else set()
            test_file = finetune_dir(settings) / "questions" / "test.jsonl"
            subjects = load_held_out(test_file).subjects if args.split != "test" and test_file.exists() else ()
            plan = plan_sampling(library, splits[args.split], counts, split=args.split,
                                 per_paper_max=args.per_paper_max, seed=args.seed, exclude_passages=exclude,
                                 exclude_subjects=subjects)
            by_kind: dict[str, int] = {}
            for candidate in plan:
                by_kind[candidate.kind] = by_kind.get(candidate.kind, 0) + 1
            print(f"План {args.split}: {len(plan)} кандидатов из {len(splits[args.split])} статей")
            for kind, wanted in counts.items():
                print(f"  {kind}: {by_kind.get(kind, 0)} из {wanted}")
            return 0
        model = _teacher(settings)
        searcher = make_searcher(settings, library)
        stats = synthesize(library, searcher, model, splits[args.split], counts, seed=args.seed, out_path=out,
                           resume=not args.restart, split=args.split, gold_questions=gold,
                           per_paper_max=args.per_paper_max, limit=args.limit)
    print(json.dumps({k: v for k, v in stats.items() if k != "reasons"}, ensure_ascii=False))
    if stats["reasons"]:
        print("Причины отказа: " + ", ".join(f"{k} {v}" for k, v in stats["reasons"].items()))
    print(f"Вопросы: {out}")
    return 0


def _collect(settings: Settings, args) -> int:
    from research_assistant.agent.session import make_agent
    from research_assistant.evaluation.retrieval import load_questions
    from research_assistant.finetune.trajectories import collect
    from research_assistant.store.db import Library

    _load_splits(settings)
    path = finetune_dir(settings) / "questions" / f"{args.split}.jsonl"
    questions = [q for q in load_questions(path) if q.meta.get("accepted")]
    if args.kinds:
        wanted = {k.strip() for k in args.kinds.split(",")}
        questions = [q for q in questions if q.kind in wanted]
    if args.limit:
        questions = questions[: args.limit]
    temperatures = tuple(float(t) for t in args.temperatures.split(","))
    model = _teacher(settings)
    out = finetune_dir(settings) / "trajectories" / f"{args.split}.jsonl"
    totals = {"written": 0, "skipped": 0, "errors": 0}
    with Library(settings.db_path) as library:
        # Внешний поиск нужен только вопросам external: иначе учитель может пойти в интернет на обычный вопрос.
        for external in (False, True):
            group = [q for q in questions if (q.kind == "external") == external]
            if not group:
                continue

            def fresh_context(external=external):
                ctx, _, http = make_agent(settings, library, external=external)
                if not external:
                    http.close()  # без внешних источников клиент не нужен; с ними им пользуется search_external
                return ctx

            counts = collect(group, fresh_context, model, out, samples=args.samples, temperatures=temperatures,
                             max_steps=settings.max_steps, log_path=settings.runs_dir / f"teacher-{args.split}.jsonl")
            for key in totals:
                totals[key] += counts.get(key, 0)
    print(f"Траектории {args.split}: записано {totals['written']}, уже было {totals['skipped']}, "
          f"ошибок {totals['errors']}. Файл: {out}")
    return 0


def _export(settings: Settings, args) -> int:
    from research_assistant.finetune.export import export

    splits = _load_splits(settings)
    stats = export(finetune_dir(settings) / "trajectories", finetune_dir(settings) / "questions", finetune_dir(settings), splits=splits,
                   max_tokens=args.max_tokens, recovery_share=args.recovery_share)
    for name, split in stats["splits"].items():
        tokens = split["tokens"]
        print(f"{name}: примеров {split['examples']} из {split['questions']} вопросов, трудных {split['hard']}, "
              f"токенов {tokens}")
    if stats["rejection_reasons"]:
        print("Причины отказа: " + ", ".join(f"{k} {v}" for k, v in list(stats["rejection_reasons"].items())[:12]))
    print(f"Примеры: {finetune_dir(settings) / 'sft'}")
    return 0
