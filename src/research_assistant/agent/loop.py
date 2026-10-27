"""Агентный цикл без фреймворка: модель → вызовы инструментов → наблюдения → … → final_answer.

- Лимит шагов: один шаг — один вызов модели.
- Ошибка инструмента (неизвестное имя, битые аргументы, исключение) возвращается модели как наблюдение.
- Ответ принимается только через final_answer, и только если все ID предложений агент видел в выдаче.
- Каждый шаг пишется в JSONL: вызов модели с usage, вызов инструмента с аргументами и временем.
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from research_assistant.agent.prompts import NUDGE_FINAL_ANSWER, SYSTEM_PROMPT
from research_assistant.agent.tools import (
    FINAL_ANSWER,
    FINAL_ANSWER_QUOTES,
    FUNCTIONS,
    ToolContext,
    ToolError,
    available_schemas,
    search_library,
    wants_to_add,
)
from research_assistant.answer.citations import Citation, load_citations, parse_sentence_id
from research_assistant.llm import ChatModel, Usage

MAX_OBSERVATION_CHARS = 6000
MAX_NUDGES = 2
# После стольких поисков и чтений агенту остаётся только final_answer: маленькие модели иначе
# бесконечно «уточняют» и не отвечают (так выглядел первый прогон на Qwen3.5-4B).
MAX_RESEARCH_CALLS = 6
BUDGET_WARNING_STEPS = 3


@dataclass
class Claim:
    text: str
    citations: list[Citation]


@dataclass
class AgentResult:
    question: str
    answer: str
    claims: list[Claim] = field(default_factory=list)
    quotes: list[tuple[str, str]] = field(default_factory=list)  # режим E2: (утверждение, цитата текстом от модели)
    not_found: bool = False
    stop_reason: str = "final_answer"  # final_answer | max_steps | no_final_answer
    steps: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    usage: Usage = field(default_factory=Usage)
    seconds: float = 0.0
    log_path: str | None = None


class RunLog:
    """Журнал прогона: одна строка JSON на событие. listener получает те же события сразу —
    так веб-интерфейс показывает, что агент делает прямо сейчас."""

    def __init__(self, path: Path | None, listener: Callable[[str, dict], None] | None = None):
        self.path = path
        self.listener = listener
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **data) -> None:
        if self.listener is not None:
            self.listener(event, data)
        if self.path is None:
            return
        record = {"ts": datetime.now().isoformat(timespec="milliseconds"), "event": event, **data}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _truncate(text: str, limit: int = MAX_OBSERVATION_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n… [truncated, {len(text) - limit} more characters]"


def _assistant_message(message) -> dict:
    """Сообщение ассистента в виде dict для следующего запроса (с вызовами инструментов, если были)."""
    out = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        out["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in message.tool_calls
        ]
    return out


def check_final_answer(ctx: ToolContext, args: dict) -> tuple[list[Claim], str | None]:
    """Проверяет ответ. Возвращает (утверждения с цитатами, текст ошибки для модели или None)."""
    if not isinstance(args.get("answer"), str) or not args["answer"].strip():
        return [], "final_answer.answer must be a non-empty string"
    if args.get("not_found"):
        return [], None
    raw_claims = args.get("claims") or []
    if not isinstance(raw_claims, list) or not raw_claims:
        return [], "final_answer needs at least one claim with sentence_ids, or not_found=true"
    parsed: list[tuple[str, list[int]]] = []
    problems = []
    for claim in raw_claims:
        if not isinstance(claim, dict) or not str(claim.get("text", "")).strip():
            problems.append("every claim needs a non-empty text")
            continue
        ids = [parse_sentence_id(raw) for raw in claim.get("sentence_ids") or []]
        if not ids or None in ids:
            problems.append(f'claim "{claim["text"][:60]}" needs sentence ids like "s123"')
            continue
        unseen = [i for i in ids if i not in ctx.seen_sentences]
        if unseen:
            problems.append(f"ids {['s%d' % i for i in unseen]} were not in your tool results")
            continue
        parsed.append((claim["text"].strip(), ids))
    if problems:
        return [], "Fix final_answer: " + "; ".join(problems)
    citations = load_citations(ctx.library, [i for _, ids in parsed for i in ids])
    return [Claim(text, [citations[i] for i in ids if i in citations]) for text, ids in parsed], None


def _budget_note(steps_left: int, research_left: int) -> str:
    """Подсказка о бюджете в конце наблюдения: модель видит, сколько ей осталось."""
    left = min(steps_left, research_left)
    if left <= 0:
        return "\n[No more searching: call final_answer now with the sentence ids you already have.]"
    if left <= BUDGET_WARNING_STEPS:
        return f"\n[{left} more tool calls at most. If you already have supporting sentences, call final_answer now.]"
    return ""


def check_final_answer_quotes(args: dict) -> tuple[list[tuple[str, str]], str | None]:
    """Режим E2: цитаты пишет модель. Проверяется только форма ответа, дословность — после прогона."""
    if not isinstance(args.get("answer"), str) or not args["answer"].strip():
        return [], "final_answer.answer must be a non-empty string"
    if args.get("not_found"):
        return [], None
    quotes = [(str(c.get("text", "")).strip(), str(c.get("quote", "")).strip())
              for c in args.get("claims") or [] if isinstance(c, dict)]
    quotes = [(text, quote) for text, quote in quotes if text and quote]
    if not quotes:
        return [], "final_answer needs claims with a text and a quote, or not_found=true"
    return quotes, None


def run_agent(
    question: str,
    ctx: ToolContext,
    model: ChatModel,
    max_steps: int = 12,
    history: list[dict] | None = None,
    log_path: Path | None = None,
    answer_mode: str = "ids",
    single_shot: bool = False,
    prefetch: bool = False,
    on_event: Callable[[str, dict], None] | None = None,
) -> AgentResult:
    """answer_mode="quotes" — цитаты текстом от модели (эксперимент E2);
    single_shot=True — один поиск по вопросу и сразу ответ, без итераций (базовая линия для E3);
    prefetch=True — тот же поиск по вопросу заранее, но инструменты остаются: агент может ответить сразу
    или искать дальше.
    on_event(событие, данные) — те же события, что пишутся в журнал (start, llm, tool, nudge, finish)."""
    started = time.perf_counter()
    log = RunLog(log_path, on_event)
    result = AgentResult(question=question, answer="", log_path=str(log_path) if log_path else None)
    ctx.allow_add = wants_to_add(question)  # качать статьи в библиотеку — только если об этом просят
    tools = available_schemas(ctx)
    if answer_mode == "quotes":
        tools = [t for t in tools if t["function"]["name"] != FINAL_ANSWER] + [FINAL_ANSWER_QUOTES]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *(history or []), {"role": "user", "content": question}]
    log.write("start", question=question, model=model.model, tools=[t["function"]["name"] for t in tools],
              answer_mode=answer_mode, single_shot=single_shot, prefetch=prefetch)
    nudges = 0
    research_calls = 0
    final_only = [t for t in tools if t["function"]["name"] == FINAL_ANSWER]
    if single_shot or prefetch:
        # Поиск по самому вопросу делает код: при single_shot модели остаётся только ответить.
        observation = search_library(ctx, question, k=6)
        call_id = "single_shot_search"
        messages.append({"role": "assistant", "content": "", "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": "search_library", "arguments": json.dumps({"query": question, "k": 6}, ensure_ascii=False)},
        }]})
        messages.append({"role": "tool", "tool_call_id": call_id, "content": _truncate(observation)})
        log.write("tool", step=0, name="search_library", arguments=question, error=None, seconds=0,
                  observation=_truncate(observation))

    for step in range(1, max_steps + 1):
        result.steps = step
        # На последнем шаге и после исчерпания бюджета исследования модель может только ответить.
        must_answer = single_shot or step == max_steps or research_calls >= MAX_RESEARCH_CALLS
        response = model.complete(messages, final_only if must_answer else tools)
        result.usage.add(response.usage)
        message = response.choices[0].message
        calls = message.tool_calls or []
        log.write(
            "llm", step=step, content=message.content,
            tool_calls=[{"name": c.function.name, "arguments": c.function.arguments} for c in calls],
            usage={"prompt": getattr(response.usage, "prompt_tokens", None),
                   "completion": getattr(response.usage, "completion_tokens", None)},
            finish_reason=response.choices[0].finish_reason,
        )
        messages.append(_assistant_message(message))

        if not calls:
            if nudges < MAX_NUDGES:
                nudges += 1
                messages.append({"role": "user", "content": NUDGE_FINAL_ANSWER})
                log.write("nudge", step=step)
                continue
            result.answer, result.stop_reason = (message.content or "").strip(), "no_final_answer"
            break

        finished = False
        for call in calls:
            result.tool_calls += 1
            name = call.function.name
            tool_started = time.perf_counter()
            error = None
            try:
                args = json.loads(call.function.arguments or "{}")
                if not isinstance(args, dict):
                    raise ToolError("arguments must be a JSON object")
                if name == FINAL_ANSWER and answer_mode == "quotes":
                    quotes, error = check_final_answer_quotes(args)
                    if error is None:
                        result.answer, result.quotes = args["answer"].strip(), quotes
                        result.not_found = bool(args.get("not_found"))
                        observation, finished = "Answer accepted.", True
                    else:
                        observation = error
                elif name == FINAL_ANSWER:
                    claims, error = check_final_answer(ctx, args)
                    if error is None:
                        result.answer, result.claims = args["answer"].strip(), claims
                        result.not_found = bool(args.get("not_found"))
                        observation, finished = "Answer accepted.", True
                    else:
                        observation = error
                elif name in FUNCTIONS:
                    research_calls += 1
                    observation = FUNCTIONS[name](ctx, **args)
                else:
                    raise ToolError(f"unknown tool {name}; available: {', '.join(t['function']['name'] for t in tools)}")
            except json.JSONDecodeError as exc:
                error = observation = f"Error: arguments are not valid JSON ({exc.msg}). Call {name} again."
            except TypeError as exc:
                error = observation = f"Error: wrong arguments for {name}: {exc}"
            except ToolError as exc:
                error = observation = f"Error: {exc}"
            except Exception as exc:  # среда: база, сеть, модель эмбеддингов — модель должна узнать об этом
                error = observation = f"Error in {name}: {type(exc).__name__}: {exc}"
            if error:
                result.tool_errors += 1
            observation = _truncate(observation)
            if not finished:
                observation += _budget_note(max_steps - step, MAX_RESEARCH_CALLS - research_calls)
            log.write("tool", step=step, name=name, arguments=call.function.arguments, error=error,
                      seconds=round(time.perf_counter() - tool_started, 3), observation=observation)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": observation})
        if finished:
            break
    else:
        result.stop_reason = "max_steps"

    result.seconds = time.perf_counter() - started
    log.write(
        "finish", stop_reason=result.stop_reason, steps=result.steps, tool_calls=result.tool_calls,
        tool_errors=result.tool_errors, not_found=result.not_found, answer=result.answer,
        citations=[c.sentence_id for claim in result.claims for c in claim.citations],
        usage={"prompt": result.usage.prompt_tokens, "completion": result.usage.completion_tokens,
               "calls": result.usage.calls},
        seconds=round(result.seconds, 2),
    )
    return result
