"""Вспомогательные команды скриптов комплекта RTX 5070.

    python train/bundle_tools.py pick-max-len --train data/train.jsonl --model unsloth/Qwen3.5-4B
        печатает наименьшую длину из --candidates, при которой остаётся не меньше --min-kept примеров;
    python train/bundle_tools.py loss-report --out ~/ft/runs/smoke
        сводка loss по trainer_state.json последнего чекпоинта: первые и последние значения, NaN.

Чистые функции (choose_max_len, loss_summary) не импортируют torch и transformers.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def choose_max_len(lengths, candidates, min_kept):
    """Наименьший порог, при котором примеров не длиннее него хотя бы min_kept (иначе — наибольший).

    lengths — длины примеров в токенах; min_kept ограничивается числом примеров.
    """
    ordered = sorted(int(c) for c in candidates)
    need = max(1, min(int(min_kept), len(lengths)))
    for candidate in ordered:
        if sum(1 for n in lengths if n <= candidate) >= need:
            return candidate
    return ordered[-1]


def loss_summary(log_history):
    """Сводка по log_history Trainer: число записей, первый/последний loss, NaN, eval loss."""
    losses = [entry["loss"] for entry in log_history if isinstance(entry.get("loss"), (int, float))]
    evals = [entry["eval_loss"] for entry in log_history if isinstance(entry.get("eval_loss"), (int, float))]
    grads = [entry["grad_norm"] for entry in log_history if isinstance(entry.get("grad_norm"), (int, float))]
    bad = any(not math.isfinite(x) for x in losses + evals + grads)
    head = losses[: max(1, len(losses) // 4)]
    tail = losses[-max(1, len(losses) // 4):]
    first = sum(head) / len(head) if head else None
    last = sum(tail) / len(tail) if tail else None
    return {
        "records": len(losses),
        "first": first,
        "last": last,
        "decreasing": bool(losses) and len(losses) > 1 and last < first,
        "nan": bad,
        "eval_loss": evals[-1] if evals else None,
    }


def latest_state(out_dir):
    """trainer_state.json самого позднего чекпоинта в out_dir (None, если чекпоинтов нет)."""
    states = []
    for path in Path(out_dir).glob("checkpoint-*/trainer_state.json"):
        try:
            states.append((int(path.parent.name.split("-")[-1]), path))
        except ValueError:
            continue
    return max(states)[1] if states else None


def cmd_pick_max_len(args):
    sys.path.insert(0, str(HERE))
    import train_sft as ts
    from transformers import AutoTokenizer

    with contextlib.redirect_stdout(sys.stderr):  # stdout — только ответ для скрипта
        tokenizer = ts.text_tokenizer(AutoTokenizer.from_pretrained(args.model))
        examples = ts.encode_rows(tokenizer, ts.read_rows(args.train))
    lengths = [len(e["input_ids"]) for e in examples]
    chosen = choose_max_len(lengths, args.candidates, args.min_kept)
    kept, dropped, _ = ts.filter_by_length(examples, chosen)
    print(f"--max-len {chosen}: останется {len(kept)} из {len(examples)} (отброшено {dropped})", file=sys.stderr)
    print(chosen)
    return 0


def cmd_loss_report(args):
    path = latest_state(args.out)
    if path is None:
        print(f"В {args.out} нет checkpoint-*/trainer_state.json — сводки loss нет")
        return 1
    summary = loss_summary(json.loads(path.read_text(encoding="utf-8")).get("log_history", []))
    if not summary["records"]:
        print(f"{path}: нет записей loss")
        return 1
    print(f"loss ({path.parent.name}): начало {summary['first']:.4f} → конец {summary['last']:.4f}, "
          f"записей {summary['records']}" + (f", eval loss {summary['eval_loss']:.4f}" if summary["eval_loss"] is not None else ""))
    if summary["nan"]:
        print("ПЛОХО: в loss или grad_norm есть NaN/inf")
        return 2
    print("loss падает — конвейер обучения работает" if summary["decreasing"]
          else "ВНИМАНИЕ: loss не падает — проверьте lr и участки с потерями в выводе выше")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="bundle_tools.py", description="вспомогательные команды комплекта")
    sub = parser.add_subparsers(dest="command", required=True)
    pick = sub.add_parser("pick-max-len", help="подобрать --max-len по длинам примеров (нужен токенайзер)")
    pick.add_argument("--train", required=True)
    pick.add_argument("--model", default="unsloth/Qwen3.5-4B")
    pick.add_argument("--candidates", type=int, nargs="+", default=[4096, 6144, 8192])
    pick.add_argument("--min-kept", type=int, default=4)
    report = sub.add_parser("loss-report", help="сводка loss по последнему чекпоинту")
    report.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return cmd_pick_max_len(args) if args.command == "pick-max-len" else cmd_loss_report(args)


if __name__ == "__main__":
    sys.exit(main())
