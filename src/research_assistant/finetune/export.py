"""Экспорт траекторий учителя в примеры SFT: сообщения, отрендеренный текст и границы потерь.

Текст рендерится шаблоном чата из GGUF (qwen3_5_chat_template.jinja) так же, как это делает токенайзер HF,
с enable_thinking=False — посимвольно как в LM Studio. Границы потерь (train_spans) — символьные [start, end)
по ходам ассистента с weight=1: от конца «<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n» до «<|im_end|>»
включительно. Считаются инкрементальным рендером: префикс с generation prompt должен совпасть с полным текстом.
"""

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from research_assistant.agent.prompts import SYSTEM_PROMPT
from research_assistant.agent.tools import FINAL_ANSWER
from research_assistant.evaluation.retrieval import GoldQuestion, load_questions
from research_assistant.finetune.trajectories import (
    ANSWER_ACCEPTED,
    PREFETCH_CALL_ID,
    accept,
    choose_best,
    failed_calls,
    is_recovery,
    load_records,
)

TEMPLATE_PATH = Path(__file__).with_name("qwen3_5_chat_template.jinja")
CHARS_PER_TOKEN = 3.4  # калибровка по usage.prompt в журналах Qwen
ASSISTANT_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
IM_END = "<|im_end|>"
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)
HISTOGRAM_BIN = 1024


# --- рендер шаблоном из GGUF ---

def _tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False) -> str:
    """Фильтр tojson как у HF: без HTML-экранирования jinja и с кириллицей как есть."""
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


@lru_cache(maxsize=1)
def _template():
    # То же окружение, что в transformers (apply_chat_template): песочница, trim_blocks и lstrip_blocks.
    from jinja2.exceptions import TemplateError
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise TemplateError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = raise_exception
    return env.from_string(TEMPLATE_PATH.read_text(encoding="utf-8"))


def render(messages: list[dict], tools: list[dict] | None, add_generation_prompt: bool = False) -> str:
    return _template().render(messages=messages, tools=tools or None, add_generation_prompt=add_generation_prompt,
                              enable_thinking=False)


def est_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def strip_think(content: str) -> str:
    """Убирает размышления: шаблон сам ставит пустой <think></think>, как в инференсе без размышлений."""
    content = THINK_RE.sub("", content or "")
    if "</think>" in content:
        content = content.split("</think>")[-1]
    if "<think>" in content:
        content = content.split("<think>")[0]
    return content.strip()


# --- пример SFT ---

def _parse_calls(calls: list[dict]) -> tuple[list[dict] | None, int]:
    """Аргументы → словари (шаблон Qwen3.5 выводит <parameter=…> только для словаря). None — пример не годится.

    Второе значение — сколько раз убран not_found=false у final_answer. Шаблон пишет булево как «False»
    (фильтр string), и если парсер LM Studio оставит его строкой, args.get("not_found") станет истинным:
    верный ответ примется как not_found. false — значение по умолчанию, поэтому без него смысл тот же."""
    out, dropped = [], 0
    for call in calls:
        name = call["function"]["name"]
        args = call["function"].get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except json.JSONDecodeError:
                return None, dropped
        if not isinstance(args, dict):
            return None, dropped
        if name == FINAL_ANSWER and args.get("not_found") is False:
            args = {k: v for k, v in args.items() if k != "not_found"}
            dropped += 1
        if name == FINAL_ANSWER and args.get("not_found") is True and args.get("claims"):
            # При not_found утверждения игнорируются (loop.check_final_answer): ученик учится их не писать.
            args = {**args, "claims": []}
        out.append({"id": call.get("id"), "type": "function", "function": {"name": name, "arguments": args}})
    return out, dropped


def _spans(messages: list[dict], tools: list[dict], text: str) -> list[list[int]]:
    """Границы ходов с weight=1. Если частичный рендер — не префикс полного, шаблон ведёт себя не так,
    как мы думаем, и маска была бы неверной: ValueError."""
    spans = []
    for i, message in enumerate(messages):
        if message.get("role") != "assistant" or message.get("weight") != 1:
            continue
        prefix = render(messages[:i], tools, add_generation_prompt=True)
        turn = render(messages[: i + 1], tools)
        if not (turn.startswith(prefix) and text.startswith(turn) and turn.endswith(IM_END + "\n")):
            raise ValueError(f"частичный рендер хода {i} не совпадает с полным текстом")
        spans.append([len(prefix), len(turn) - len("\n")])
    return spans


def build_example(record: dict, question: GoldQuestion, split: str) -> tuple[dict | None, str | None]:
    """(пример, None) или (None, причина отказа)."""
    from jinja2.exceptions import TemplateError

    source = record.get("messages") or []
    if not source:
        return None, "no_messages"
    failed = {i for i, _, _ in failed_calls(source)}  # ход с ошибкой учит ошибке: без веса
    messages = [dict(m) for m in source]
    # Подсказку учителю (trajectories.TEACHER_HINTS) убираем: ученик учится на входе деплоя — штатном промпте.
    if messages[0].get("role") != "system" or not messages[0].get("content", "").startswith(SYSTEM_PROMPT):
        return None, "foreign_system_prompt"
    messages[0]["content"] = SYSTEM_PROMPT
    if messages[-1].get("role") == "tool" and messages[-1].get("content") == ANSWER_ACCEPTED:
        messages.pop()  # пример заканчивается вызовом final_answer
    users = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if not users:
        return None, "no_user"
    dropped_false = 0
    for i, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        message["content"] = strip_think(message.get("content") or "")
        calls = message.get("tool_calls") or []
        if calls:
            message["tool_calls"], dropped = _parse_calls(calls)
            if message["tool_calls"] is None:
                return None, "bad_arguments"
            dropped_false += dropped
        prefetch = any(c.get("id") == PREFETCH_CALL_ID for c in calls)
        # Без веса: история диалога (до последнего user), поиск шага 0 (его делает код) и ход с ошибкой.
        message["weight"] = 0 if i < users[-1] or prefetch or i in failed else 1
    tools = record.get("tools") or []
    try:
        text = render(messages, tools)
        spans = _spans(messages, tools, text)
    except (TemplateError, ValueError):
        return None, "render_mismatch"
    result = record.get("result") or {}
    example = {
        "id": f"{record['qid']}-s{record['sample']}", "split": split, "kind": record.get("kind") or question.kind,
        "messages": messages, "tools": tools, "text": text, "train_spans": spans,
        "meta": {
            "qid": record["qid"], "sample": record["sample"], "prompt_hash": record.get("prompt_hash"),
            "est_tokens": est_tokens(text), "steps": result.get("steps"), "teacher": record.get("model"),
            "temperature": record.get("temperature"), "language": question.language,
            "parser_version": record.get("parser_version"), "gold_rank": question.meta.get("gold_rank"),
            "gold_hit": result.get("gold_hit"), "recovery": is_recovery(record),
            "dropped_not_found_false": dropped_false,
        },
    }
    return example, None


def to_sft(record: dict, question: GoldQuestion, split: str) -> dict | None:
    """Пример SFT из записи траектории; None, если аргументы вызова не словарь или рендер не сошёлся."""
    return build_example(record, question, split)[0]


# --- проверка примера ---

def _order_problems(messages: list[dict]) -> list[str]:
    """system → (user, assistant)* история → user → (assistant с вызовами → tool…)* → assistant последним."""
    roles = [m.get("role") for m in messages]
    if not roles or roles[0] != "system" or roles.count("system") != 1:
        return ["system"]
    if "user" not in roles:
        return ["no_user"]
    last_user = len(roles) - 1 - roles[::-1].index("user")
    problems = []
    history = messages[1:last_user]
    expected = ["user", "assistant"] * (len(history) // 2)
    if [m["role"] for m in history] != expected or any(m.get("tool_calls") for m in history):
        problems.append("history_order")
    episode = messages[last_user + 1:]
    j = 0
    while j < len(episode):
        message = episode[j]
        if message["role"] != "assistant":
            problems.append(f"order:{last_user + 1 + j}")
            break
        calls = message.get("tool_calls") or []
        if j == len(episode) - 1:
            break
        replies = episode[j + 1: j + 1 + len(calls)]
        if not calls or len(replies) != len(calls) or any(r["role"] != "tool" for r in replies):
            problems.append(f"order:{last_user + 1 + j}")
            break
        if [r.get("tool_call_id") for r in replies] != [c.get("id") for c in calls]:
            problems.append(f"tool_call_id:{last_user + 1 + j}")
        j += 1 + len(calls)
    last = messages[-1]
    final_calls = [c["function"]["name"] for c in last.get("tool_calls") or []]
    if last["role"] != "assistant" or FINAL_ANSWER not in final_calls:
        problems.append("last_not_final_answer")
    return problems


def validate_example(example: dict) -> list[str]:
    """Проблемы примера (пустой список — пример корректен)."""
    messages = example.get("messages") or []
    problems = _order_problems(messages)
    assistants = [m for m in messages if m.get("role") == "assistant"]
    for m in assistants:
        if "weight" not in m:
            problems.append("missing_weight")
        if "<think>" in (m.get("content") or "") or "</think>" in (m.get("content") or ""):
            problems.append("think_leak")
        if any(not isinstance(c["function"].get("arguments"), dict) for c in m.get("tool_calls") or []):
            problems.append("arguments_not_dict")
    text, spans = example.get("text") or "", example.get("train_spans") or []
    if len(spans) != sum(1 for m in assistants if m.get("weight") == 1):
        problems.append("span_count")
    previous_end = 0
    for start, end in spans:
        if not 0 <= previous_end <= start < end <= len(text):
            problems.append("span_bounds")
            continue
        previous_end = end
        if not text[start:end].endswith(IM_END):
            problems.append("span_end")
        if not text[:start].endswith(ASSISTANT_PREFIX):
            problems.append("span_start")
        if "<think>" in text[start:end]:
            problems.append("think_leak")
    return list(dict.fromkeys(problems))


# --- экспорт ---

def _latest(records: list[dict]) -> dict[tuple[str, int], dict]:
    """Последняя запись на (qid, sample); удачная запись важнее более поздней ошибки."""
    latest: dict[tuple[str, int], dict] = {}
    for record in records:
        key = (record["qid"], record["sample"])
        if key not in latest or not record.get("error") or latest[key].get("error"):
            latest[key] = record
    return latest


def _token_stats(values: list[int]) -> dict:
    if not values:
        return {"mean": 0, "p50": 0, "p95": 0, "max": 0, "histogram": {}}
    ordered = sorted(values)
    histogram = Counter(v // HISTOGRAM_BIN * HISTOGRAM_BIN for v in ordered)
    return {
        "mean": round(sum(ordered) / len(ordered)), "p50": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)], "max": ordered[-1],
        "histogram": {f"{lo}-{lo + HISTOGRAM_BIN - 1}": histogram[lo] for lo in sorted(histogram)},
    }


def _test_papers(splits, trajectories_dir: Path) -> set[int] | None:
    """Статьи test: из переданного содержимого splits.json или из файла рядом с trajectories_dir.
    None — неизвестны (файла нет)."""
    if isinstance(splits, dict):
        return set(splits.get("test") or [])
    path = trajectories_dir.parent / "splits.json"  # data/finetune/splits.json рядом с trajectories/
    return set(json.loads(path.read_text(encoding="utf-8")).get("test") or []) if path.exists() else None


def _rejection(record: dict, split: str, reasons: list[str]) -> dict:
    result = record.get("result") or {}
    return {"qid": record.get("qid"), "sample": record.get("sample"), "split": split, "kind": record.get("kind"),
            "reasons": reasons, "stop_reason": result.get("stop_reason"), "steps": result.get("steps"),
            "answer": result.get("answer"), "error": record.get("error")}


def _allowed_recovery(plain: int, share: float) -> float:
    """Сколько recovery-примеров можно при plain обычных: recovery / (recovery + plain) ≤ share."""
    return math.inf if share >= 1 else math.floor(share * plain / (1 - share))


def _apply_recovery_cap(chosen: dict[str, dict], fallback: dict[str, dict | None], share: float) -> set[str]:
    """Вопросы, у которых выбранный recovery-сэмпл не помещается в долю share.

    Отклонённый recovery заменяется лучшим обычным сэмплом того же вопроса, если такой есть. Места
    сначала получают вопросы без замены: иначе они пропали бы совсем."""
    recovery = [qid for qid, r in chosen.items() if is_recovery(r)]
    without_fallback = [qid for qid in recovery if fallback[qid] is None]
    with_fallback = [qid for qid in recovery if fallback[qid] is not None]
    plain = len(chosen) - len(recovery) + len(with_fallback)  # пока все замены считаются обычными примерами
    kept = set()
    for qid in without_fallback + with_fallback:
        after = plain - (fallback[qid] is not None)
        if len(kept) + 1 <= _allowed_recovery(after, share):
            kept.add(qid)
            plain = after
    return set(recovery) - kept


def export_split(records: list[dict], questions: dict[str, GoldQuestion], split: str, max_tokens: int,
                 recovery_share: float, test_papers: set[int]) -> tuple[list[dict], list[dict], list[str], int]:
    """(примеры, отклонённые, трудные вопросы, вопросы без примера из-за доли recovery) одного сплита.

    Трудный — учитель провалил все сэмплы. Принятые, но не выбранные сэмплы идут в отклонённые с not_chosen."""
    by_question: dict[str, list[dict]] = defaultdict(list)
    for record in _latest(records).values():
        by_question[record["qid"]].append(record)
    rejected, hard = [], []
    candidates: dict[str, dict[int, tuple[dict, dict]]] = {}
    leak_papers = set() if split == "test" else test_papers
    for qid in sorted(by_question):
        question = questions.get(qid)
        accepted = {}
        for record in sorted(by_question[qid], key=lambda r: r["sample"]):
            if question is None:
                reasons = ["unknown_question"]
            elif question.meta.get("accepted") is False:
                reasons = ["question_rejected"]
            else:
                reasons = accept(record, question, set(), leak_papers)
            if not reasons:
                example, why = build_example(record, question, split)
                reasons = [why] if example is None else validate_example(example)
                if not reasons and example["meta"]["est_tokens"] > max_tokens:
                    reasons = ["too_long"]
                if not reasons:
                    accepted[record["sample"]] = (record, example)
                    continue
            rejected.append(_rejection(record, split, reasons))
        if accepted:
            candidates[qid] = accepted
        else:
            hard.append(qid)
    chosen = {qid: choose_best([r for r, _ in c.values()]) for qid, c in candidates.items()}
    fallback = {qid: choose_best([r for r, _ in c.values() if not is_recovery(r)]) for qid, c in candidates.items()}
    # Восстановление после ошибки полезно, но в малой доле: иначе ученик учится сначала ошибаться.
    capped = _apply_recovery_cap(chosen, fallback, recovery_share)
    examples, lost = [], 0
    for qid, best in chosen.items():
        if qid in capped:
            rejected.append(_rejection(best, split, ["recovery_cap"]))
            best = fallback[qid]
        if best is None:
            lost += 1
        else:
            examples.append(candidates[qid][best["sample"]][1])
        for sample, (record, _) in sorted(candidates[qid].items()):
            if record is not best and not (qid in capped and record is chosen[qid]):
                rejected.append(_rejection(record, split, ["not_chosen"]))
    return sorted(examples, key=lambda e: e["id"]), rejected, hard, lost


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def export(
    trajectories_dir: str | Path,
    questions_dir: str | Path,
    out_dir: str | Path,
    splits=("train", "val"),
    max_tokens: int = 10240,
    recovery_share: float = 0.1,
    log: Callable[[str], None] = print,
) -> dict:
    """Пишет out_dir/sft/<split>.jsonl, out_dir/sft/stats.json и out_dir/rejected.jsonl (out_dir — data/finetune).

    splits — имена сплитов или содержимое splits.json (тогда экспортируются train и val, а статьи test
    берутся оттуда); иначе статьи test читаются из splits.json рядом с trajectories_dir. Нет splits.json при
    экспорте train или val — ValueError: утечку из теста не проверить. Возвращает stats."""
    trajectories_dir, questions_dir, out_dir = Path(trajectories_dir), Path(questions_dir), Path(out_dir)
    test_papers = _test_papers(splits, trajectories_dir)
    names = [s for s in ("train", "val") if s in splits] if isinstance(splits, dict) else list(splits)
    if test_papers is None:
        if {"train", "val"} & set(names):
            raise ValueError(f"Статьи test неизвестны: нет {trajectories_dir.parent / 'splits.json'}. Без них "
                             "утечку цитат из теста в train и val не проверить. Запустите ft-split или передайте "
                             "splits={'test': []}, если проверка заведомо не нужна.")
        test_papers = set()
    stats = {"max_tokens": max_tokens, "recovery_share": recovery_share, "splits": {}, "rejection_reasons": {},
             "hard": {}}
    all_rejected, reasons = [], Counter()
    for split in names:
        questions_path = questions_dir / f"{split}.jsonl"
        questions = {q.id: q for q in load_questions(questions_path)} if questions_path.exists() else {}
        records = load_records(trajectories_dir / f"{split}.jsonl")
        examples, rejected, hard, lost = export_split(records, questions, split, max_tokens, recovery_share,
                                                      test_papers)
        _write_jsonl(out_dir / "sft" / f"{split}.jsonl", examples)
        all_rejected += rejected
        reasons.update(reason for row in rejected for reason in row["reasons"])
        stats["splits"][split] = {
            "examples": len(examples), "questions": len({r["qid"] for r in records}), "hard": len(hard),
            "recovery": sum(1 for e in examples if e["meta"]["recovery"]), "recovery_capped_questions": lost,
            "dropped_not_found_false": sum(e["meta"]["dropped_not_found_false"] for e in examples),
            "by_kind": dict(Counter(e["kind"] for e in examples)),
            "by_language": dict(Counter(e["meta"]["language"] for e in examples)),
            "tokens": _token_stats([e["meta"]["est_tokens"] for e in examples]),
        }
        stats["hard"][split] = hard
        log(f"{split}: примеров {len(examples)}, отклонено траекторий {len(rejected)}, трудных вопросов {len(hard)}")
    stats["rejection_reasons"] = dict(reasons.most_common())
    _write_jsonl(out_dir / "rejected.jsonl", all_rejected)
    (out_dir / "sft").mkdir(parents=True, exist_ok=True)
    (out_dir / "sft" / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats
