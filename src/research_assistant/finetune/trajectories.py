"""Траектории учителя для дообучения: сбор прогонов агента и отсев (rejection sampling).

Сбор идёт тем же путём, что и в деплое: run_agent с prefetch, свежий ToolContext на каждый прогон.
В запись попадает ровно то, что видела модель (messages, tools), и сводка результата. Доказательства
(какие статьи и фрагменты процитированы, где золото) считаются по библиотеке сразу при сборе, поэтому
отсев и экспорт работают без базы. Отсев — accept(): причины списком, пустой список — траектория годится
(docs/finetune.md, «Траектория» и «Решения»).
"""

import hashlib
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from urllib.parse import urlsplit

from research_assistant.agent import tools as agent_tools
from research_assistant.agent.loop import REPEATED_CALL, AgentResult, run_agent
from research_assistant.agent.prompts import NUDGE_FINAL_ANSWER, SYSTEM_PROMPT
from research_assistant.agent.tools import FINAL_ANSWER, SCHEMAS, ToolContext
from research_assistant.evaluation.agent import papers_of
from research_assistant.evaluation.retrieval import GoldQuestion, resolve
from research_assistant.llm import ChatModel
from research_assistant.nlp.text import detect_language
from research_assistant.parse.pipeline import PARSER_VERSION
from research_assistant.store.db import Library

PREFETCH_CALL_ID = "single_shot_search"  # поиск шага 0, его делает код, а не модель
ANSWER_ACCEPTED = "Answer accepted."
SEMANTIC_UNAVAILABLE = "[Semantic search is unavailable"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}

# Пороги отсева: учитель в среднем отвечает за 1.6 шага, длинные траектории учат блуждать.
MAX_STEPS = 4
MAX_TOOL_CALLS = 8
MAX_ABSTAIN_SEARCHES = 3  # на неотвечаемый вопрос — не больше трёх поисков до not_found
GOLD_KINDS = {"fact", "entity", "paper_lookup", "followup"}
# Виды ошибок как в agent/loop.py (error_kind); там вид пишется только в журнал, поэтому здесь он
# восстанавливается по началу наблюдения. Порядок важен: сначала частные префиксы, «Error:» — последним.
ERROR_PREFIXES = (
    ("Error: arguments are not valid JSON", "json"),
    ("Error: unknown tool", "unknown_tool"),
    ("Error: arguments must be a JSON object", "bad_args"),
    ("Error: wrong arguments for", "bad_args"),
    ("Error in ", "env"),
    ("Error:", "tool_error"),
)
# Recovery — только после ошибки, которую модель исправляет сама: неверные аргументы, отказ инструмента,
# отклонённый final_answer. Битый JSON и неизвестный инструмент учат ломать формат, env — не вина модели:
# такие траектории отбрасываются целиком.
RECOVERY_KINDS = {"bad_args", "tool_error", "answer_rejected"}


# --- общие помощники по сообщениям ---

def prompt_hash() -> str:
    """Отпечаток SYSTEM_PROMPT и SCHEMAS; берётся из agent.tools, если там он есть."""
    shared = getattr(agent_tools, "prompt_hash", None)
    if callable(shared):
        return shared()
    text = SYSTEM_PROMPT + json.dumps(SCHEMAS, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def count_nudges(messages: list[dict]) -> int:
    return sum(1 for m in messages if m.get("role") == "user" and m.get("content") == NUDGE_FINAL_ANSWER)


def call_pairs(messages: list[dict]) -> list[tuple[int, dict, dict | None]]:
    """(индекс хода ассистента, вызов, сообщение инструмента с ответом на него или None).

    Ответы сопоставляются по порядку: run_agent пишет их сразу за ходом, вызов за вызовом."""
    pairs = []
    for i, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        following = messages[i + 1:]
        for n, call in enumerate(message.get("tool_calls") or []):
            reply = following[n] if n < len(following) and following[n].get("role") == "tool" else None
            pairs.append((i, call, reply))
    return pairs


def error_kind(name: str, observation: str) -> str | None:
    """Вид ошибки вызова (json | unknown_tool | bad_args | env | tool_error | answer_rejected) или None."""
    for prefix, kind in ERROR_PREFIXES:
        if observation.startswith(prefix):
            return kind
    if name == FINAL_ANSWER and not observation.startswith(ANSWER_ACCEPTED):
        return "answer_rejected"
    return None


def is_error_observation(name: str, observation: str) -> bool:
    """Наблюдение — ошибка: «Error…» от цикла или отклонённый final_answer."""
    return error_kind(name, observation) is not None


def failed_calls(messages: list[dict]) -> list[tuple[int, str, str]]:
    """(индекс хода, имя инструмента, наблюдение) для вызовов модели, закончившихся ошибкой."""
    out = []
    for i, call, reply in call_pairs(messages):
        name = call["function"]["name"]
        if reply is not None and is_error_observation(name, reply.get("content") or ""):
            out.append((i, name, reply["content"]))
    return out


def error_kinds(record: dict) -> list[str]:
    """Виды ошибок траектории по порядку: из записи (пишется при сборе), для старых записей — по сообщениям."""
    stored = (record.get("result") or {}).get("error_kinds")
    if isinstance(stored, list):
        return stored
    return [error_kind(name, observation) for _, name, observation in failed_calls(record.get("messages") or [])]


def model_calls(messages: list[dict]) -> list[dict]:
    """Вызовы, которые сделала модель: без поиска шага 0."""
    return [call for _, call, _ in call_pairs(messages) if call.get("id") != PREFETCH_CALL_ID]


def final_arguments(messages: list[dict]) -> dict:
    """Аргументы последнего вызова final_answer (словарь; битый JSON — пустой словарь)."""
    for call in reversed(model_calls(messages)):
        if call["function"]["name"] == FINAL_ANSWER:
            args = call["function"]["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args or "{}")
                except json.JSONDecodeError:
                    return {}
            return args if isinstance(args, dict) else {}
    return {}


# --- сбор ---

def is_local_url(url: str | None) -> bool:
    host = (urlsplit(str(url)).hostname or "") if url else ""
    return host in LOCAL_HOSTS or host.endswith(".localhost")


def check_teacher(model: ChatModel) -> None:
    """Учитель — облачная модель. Локальный адрес — это ученик в LM Studio: на его ответах учить нельзя."""
    base_url = getattr(getattr(model, "client", None), "base_url", None)
    if base_url is not None and is_local_url(str(base_url)):
        raise ValueError(f"Сбор траекторий: модель по адресу {base_url} локальная — это ученик, а не учитель. "
                         "Выберите облачного учителя: LLM_PROVIDER=inception в .env.")


def load_records(path: str | Path) -> list[dict]:
    """Записи JSONL; оборванная последняя строка (сбор прервали) пропускается."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def done_keys(path: str | Path) -> set[tuple[str, int]]:
    """Готовые пары (qid, sample). Записи с ошибкой API не считаются: при возобновлении их повторяем."""
    return {(r["qid"], r["sample"]) for r in load_records(path) if not r.get("error")}


def _in_list(column: str, ids: list[int]) -> str:
    return f"{column} IN ({','.join('?' * len(ids))})"


def evidence(library: Library, question: GoldQuestion, citations: list[int]) -> dict:
    """Что процитировано и где золото: id фрагментов и статей. Хранится в записи, чтобы отсев не ходил в базу."""
    cited_passages = set()
    if citations:
        rows = library.conn.execute(f"SELECT DISTINCT passage_id FROM sentences WHERE {_in_list('id', citations)}",
                                    citations)
        cited_passages = {r[0] for r in rows}
    gold_passages = resolve(library, question) if question.relevant else set()
    return {
        "cited_passages": sorted(cited_passages), "cited_papers": sorted(papers_of(library, cited_passages)),
        "gold_passages": sorted(gold_passages), "gold_papers": sorted(papers_of(library, gold_passages)),
        "gold_hit": bool(cited_passages & gold_passages),
    }


def summarize(result: AgentResult, library: Library, question: GoldQuestion) -> dict:
    citations = [c.sentence_id for claim in result.claims for c in claim.citations]
    return {
        "stop_reason": result.stop_reason, "steps": result.steps, "tool_calls": result.tool_calls,
        "tool_errors": result.tool_errors, "not_found": result.not_found, "answer": result.answer,
        "citations": citations,
        "claims": [{"text": claim.text, "sentence_ids": [c.sentence_id for c in claim.citations]}
                   for claim in result.claims],
        "usage": {"prompt": result.usage.prompt_tokens, "completion": result.usage.completion_tokens},
        "seconds": round(result.seconds, 2), "nudges": count_nudges(result.messages),
        "error_kinds": [error_kind(name, obs) for _, name, obs in failed_calls(result.messages)],
        **evidence(library, question, citations),
    }


# Подсказка видна только учителю: без неё Mercury 3 отвечал по-русски на 6 из 10 английских вопросов (E5), а
# с ней — 10 из 10 по-английски. При экспорте системный промпт заменяется штатным: ученик получает вход деплоя.
TEACHER_HINTS = {
    "en": ("The user's question is in English. Finish with the final_answer tool as usual; inside it, write the "
           "answer and every claim in English, without markdown."),
    "ru": ("Вопрос пользователя на русском. Заверши ответ вызовом final_answer, как обычно; внутри пиши ответ и "
           "каждое утверждение по-русски, без markdown-разметки."),
}


def teacher_hint(question: GoldQuestion) -> str:
    return TEACHER_HINTS.get(question.language, TEACHER_HINTS["en"])


def run_one(question: GoldQuestion, sample: int, ctx: ToolContext, model: ChatModel, max_steps: int = 12,
            log_path: Path | None = None, phash: str | None = None) -> dict:
    """Один прогон учителя → запись траектории. Ошибка API (после повторов клиента) — запись с error."""
    record = {
        "qid": question.id, "sample": sample, "kind": question.kind, "split": question.meta.get("split"),
        "model": model.model, "temperature": model.temperature, "prompt_hash": phash or prompt_hash(),
        "parser_version": PARSER_VERSION, "dense_search": ctx.searcher.dense_ready,
        "messages": [], "tools": [], "result": None, "error": None,
    }
    try:
        result = run_agent(question.question, ctx, model, max_steps=max_steps, prefetch=True,
                           history=question.meta.get("history") or None, log_path=log_path,
                           extra_system=teacher_hint(question))
    except Exception as exc:  # сеть, лимиты, 5xx: вопрос не роняет весь сбор
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    record.update(messages=result.messages, tools=result.tools, result=summarize(result, ctx.library, question))
    return record


def collect(
    questions: Iterable[GoldQuestion],
    make_context: Callable[[], ToolContext],
    model: ChatModel,
    out_path: str | Path,
    samples: int = 2,
    temperatures: tuple[float, ...] = (0.2, 0.7),
    max_steps: int = 12,
    resume: bool = True,
    log: Callable[[str], None] = print,
    log_path: Path | None = None,
) -> dict:
    """Траектории учителя: samples прогонов на вопрос, только дозапись в out_path.

    Температура сэмпла i — temperatures[i] (последняя — для остальных). resume — пропускать готовые
    (qid, sample). Возвращает счётчики {"written", "skipped", "errors"}."""
    check_teacher(model)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(out_path) if resume else set()
    counts = {"written": 0, "skipped": 0, "errors": 0}
    phash = prompt_hash()
    original_temperature = model.temperature
    warned = False
    try:
        for question in questions:
            for sample in range(samples):
                if (question.id, sample) in done:
                    counts["skipped"] += 1
                    continue
                model.temperature = temperatures[min(sample, len(temperatures) - 1)]
                ctx = make_context()
                if not ctx.searcher.dense_ready and not warned:
                    log("Внимание: плотный поиск не настроен (нет bge-m3) — наблюдения будут только по BM25.")
                    warned = True
                record = run_one(question, sample, ctx, model, max_steps, log_path, phash)
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts["written"] += 1
                if record["error"]:
                    counts["errors"] += 1
                    log(f"{question.id} s{sample}: ошибка — {record['error']}")
                else:
                    r = record["result"]
                    log(f"{question.id} s{sample}: {r['stop_reason']}, шагов {r['steps']}, вызовов {r['tool_calls']}, "
                        f"{r['seconds']:.1f} с")
    finally:
        model.temperature = original_temperature
    return counts


# --- отсев ---

def is_recovery(record: dict) -> bool:
    """Ровно одна ошибка вида из RECOVERY_KINDS, и после неё ответ принят. Таких в train не больше доли
    recovery_share — export ограничивает; ход с ошибкой идёт без веса."""
    result = record.get("result") or {}
    if result.get("stop_reason") != "final_answer" or result.get("tool_errors") != 1:
        return False
    kinds = error_kinds(record)
    return len(kinds) == 1 and kinds[0] in RECOVERY_KINDS


def _kind_reasons(kind: str, result: dict, messages: list[dict]) -> list[str]:
    not_found = bool(result.get("not_found"))
    cited, gold = set(result.get("cited_papers") or []), set(result.get("gold_papers") or [])
    names = [c["function"]["name"] for c in model_calls(messages)]
    if kind in GOLD_KINDS:
        if not_found:
            return ["not_found"]
        if not gold:
            return ["gold_unresolved"]
        return [] if cited & gold else ["wrong_paper"]
    if kind == "compare":
        if not_found:
            return ["not_found"]
        if len(gold) < 2:
            return ["gold_unresolved"]
        return [] if gold <= cited else ["missing_paper"]
    if kind == "unanswerable":
        # Утверждения при not_found=true программа игнорирует, а экспорт их убирает — пример годится.
        reasons = [] if not_found else ["answered_unanswerable"]
        if sum(1 for n in names if n != FINAL_ANSWER) > MAX_ABSTAIN_SEARCHES:
            reasons.append("too_many_searches")
        return reasons
    if kind == "external":
        reasons = [] if not_found else ["expected_not_found"]
        if "search_external" not in names:
            reasons.append("no_search_external")
        return reasons
    return [f"unknown_kind:{kind}"]


def accept(record: dict, question: GoldQuestion, test_sentence_ids: set[int] = frozenset(),
           test_papers: set[int] = frozenset()) -> list[str]:
    """Причины отказа (пустой список — траектория годится).

    test_sentence_ids / test_papers — предложения и статьи теста: для train и val нельзя их цитировать
    (для самого test передавайте пустые множества). Recovery — не причина: см. is_recovery()."""
    if record.get("error"):
        return ["error"]
    result = record.get("result") or {}
    messages = record.get("messages") or []
    kind = record.get("kind") or question.kind
    reasons = []
    if result.get("stop_reason") != "final_answer":
        reasons.append(f"stop_reason:{result.get('stop_reason')}")
    if count_nudges(messages):
        reasons.append("nudge")
    if (result.get("steps") or 0) > MAX_STEPS:
        reasons.append("too_many_steps")
    if (result.get("tool_calls") or 0) > MAX_TOOL_CALLS:
        reasons.append("too_many_tool_calls")
    if result.get("tool_errors") and not is_recovery(record):
        reasons.append("tool_errors")
    observations = [m.get("content") or "" for m in messages if m.get("role") == "tool"]
    if any(SEMANTIC_UNAVAILABLE in o for o in observations):
        reasons.append("semantic_search_unavailable")
    if any(o.startswith(REPEATED_CALL) for o in observations):
        reasons.append("repeated_call")  # повтор вызова — ровно то, от чего отучаем ученика
    names = [c["function"]["name"] for c in model_calls(messages)]
    if "add_papers" in names:
        reasons.append("add_papers")
    if "search_external" in names and kind != "external":
        reasons.append("search_external")
    # Ответ на external — список английских названий статей: язык по нему не определить.
    if kind != "external":
        texts = [result.get("answer") or ""] + [c.get("text") or "" for c in result.get("claims") or []]
        if any(detect_language(t) != question.language for t in texts if t.strip()):
            reasons.append("language")
    if set(result.get("citations") or []) & set(test_sentence_ids) or \
            set(result.get("cited_papers") or []) & set(test_papers):
        reasons.append("test_leak")
    return reasons + _kind_reasons(kind, result, messages)


def tokens_of(record: dict) -> int:
    usage = (record.get("result") or {}).get("usage") or {}
    return (usage.get("prompt") or 0) + (usage.get("completion") or 0)


def choose_best(records: list[dict]) -> dict | None:
    """Лучшая из принятых траекторий вопроса: с золотым фрагментом, без восстановления после ошибки,
    затем меньше шагов, затем меньше токенов."""
    if not records:
        return None
    return min(records, key=lambda r: (not (r.get("result") or {}).get("gold_hit"), is_recovery(r),
                                       (r.get("result") or {}).get("steps") or 0, tokens_of(r), r.get("sample", 0)))
