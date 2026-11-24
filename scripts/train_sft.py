"""LoRA SFT для Qwen3.5-4B на траекториях учителя (data/finetune/sft/*.jsonl).

Скрипт вне пакета: torch/unsloth/transformers нужны только на машине обучения
(RTX 5070 через WSL2 или Docker unsloth/unsloth, либо Colab для пробного запуска).
Чистые функции наверху модуля не импортируют torch — их проверяет tests/test_train_script.py.

Вход — строки JSONL с полями "text" (диалог, отрендеренный шаблоном LM Studio)
и "train_spans" ([start, end) в символах — ходы ассистента с потерями).
Чат-шаблон повторно не применяется: обучаемся на готовых input_ids/labels.

Примеры:
    python scripts/train_sft.py --train data/finetune/sft/train.jsonl --val data/finetune/sft/val.jsonl --dry-run
    python scripts/train_sft.py --train ... --val ... --out runs/sft/r16 --merge-out runs/sft/r16-merged
    python scripts/train_sft.py --merge-from runs/sft/r16/checkpoint-400 --merge-out runs/sft/r16-merged
    python scripts/train_sft.py --check-gguf qwen3.5-4b-ra-Q4_K_M.gguf
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

IGNORE_INDEX = -100
DEFAULT_MODEL = "unsloth/Qwen3.5-4B"
REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_TEMPLATE = REPO_ROOT / "src" / "research_assistant" / "finetune" / "qwen3_5_chat_template.jinja"
# вне репозитория (комплект для RTX 5070, Colab) шаблон лежит рядом со скриптом
DEFAULT_TEMPLATE = (_REPO_TEMPLATE if _REPO_TEMPLATE.exists()
                    else Path(__file__).resolve().with_name("qwen3_5_chat_template.jinja"))

# Полное внимание и MLP плюс проекции Gated DeltaNet (линейное внимание Qwen3.5).
ATTENTION_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
DELTANET_MODULES = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")
MLP_MODULES = ("gate_proj", "up_proj", "down_proj")
LORA_TARGETS = ATTENTION_MODULES + DELTANET_MODULES + MLP_MODULES
VISION_MARKERS = ("visual", "vision")


# --- маска потерь ---

def labels_from_spans(offsets, spans, input_ids):
    """Метки для токенов: id внутри обучаемых отрезков, IGNORE_INDEX вне их.

    offsets — (start, end) каждого токена в символах текста, spans — [start, end) ходов ассистента.
    Токен обучаемый, если он начинается внутри отрезка: start <= s < end.
    Отсюда правила для токенов на границе (детерминированные):
    - токен целиком внутри отрезка — обучаемый;
    - токен начинается до отрезка и заходит в него (левая граница) — НЕ обучаемый:
      его начало — текст шаблона, который на инференсе уже стоит в промпте, модель его не генерирует;
    - токен начинается внутри и выходит за правый край — обучаемый: его начало модель генерирует сама
      (на практике правый край — спецтокен <|im_end|>, его не склеить с соседями);
    - токен нулевой ширины (s == e) — обучаемый, если start <= s < end.
    """
    if len(offsets) != len(input_ids):
        raise ValueError(f"offsets ({len(offsets)}) и input_ids ({len(input_ids)}) разной длины")
    ordered = sorted((int(a), int(b)) for a, b in spans)
    labels = []
    for token_id, (s, _e) in zip(input_ids, offsets):
        inside = any(a <= s < b for a, b in ordered)
        labels.append(int(token_id) if inside else IGNORE_INDEX)
    return labels


def edge_tokens(offsets, spans):
    """Токены на границах отрезков: (левая граница, правая граница) — для отчёта."""
    left = right = 0
    for s, e in offsets:
        for a, b in spans:
            if s < a < e:
                left += 1
            if s < b < e:
                right += 1
    return left, right


def trainable_segments(labels):
    """Непрерывные участки обучаемых токенов: [(i, j)), чтобы показать их декодированными."""
    segments, start = [], None
    for i, label in enumerate(labels):
        if label != IGNORE_INDEX and start is None:
            start = i
        elif label == IGNORE_INDEX and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(labels)))
    return segments


# --- длина примеров ---

def drop_or_truncate(example, max_len):
    """Пример длиннее max_len не режется посреди хода: он отбрасывается (None).

    Исключение: если за пределами max_len нет обучаемых токенов (например, хвостовой «\\n» после
    <|im_end|>), хвост отрезается — ни одного хода ассистента это не задевает.
    """
    ids = example["input_ids"]
    if len(ids) <= max_len:
        return example
    if all(label == IGNORE_INDEX for label in example["labels"][max_len:]):
        cut = {key: value[:max_len] if isinstance(value, list) else value for key, value in example.items()}
        if any(label != IGNORE_INDEX for label in cut["labels"]):
            return cut
    return None


DROP_WARN_SHARE = 0.5


def drop_warning(dropped, total, max_len):
    """Предупреждение, если фильтр длины отбросил больше половины примеров (None — всё в порядке)."""
    if total and dropped / total > DROP_WARN_SHARE:
        return (f"  ВНИМАНИЕ: при --max-len {max_len} отброшено {dropped} из {total} ({dropped / total:.0%}) — "
                "увеличьте --max-len (ориентир — p50/p90 выше)")
    return None


def filter_by_length(examples, max_len):
    """Отбор по длине; возвращает (оставшиеся, число отброшенных, число обрезанных)."""
    kept, dropped, truncated = [], 0, 0
    for example in examples:
        result = drop_or_truncate(example, max_len)
        if result is None:
            dropped += 1
            continue
        if len(result["input_ids"]) < len(example["input_ids"]):
            truncated += 1
        kept.append(result)
    return kept, dropped, truncated


def percentile(values, q):
    """Перцентиль по ближайшему рангу (без numpy)."""
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(q / 100 * len(ordered)) - 1)
    return ordered[index]


def length_stats(lengths, thresholds=(2048, 4096, 8192, 10240, 16384)):
    """Сводка длин в токенах: перцентили и сколько примеров длиннее каждого порога."""
    return {
        "count": len(lengths),
        "min": min(lengths, default=0),
        "p50": percentile(lengths, 50),
        "p90": percentile(lengths, 90),
        "p95": percentile(lengths, 95),
        "max": max(lengths, default=0),
        "over": {str(t): sum(1 for n in lengths if n > t) for t in thresholds},
    }


# --- батчи ---

@dataclass
class PadCollator:
    """Дополняет input_ids/labels/attention_mask справа до самого длинного примера в батче.

    Работает со списками; to_tensor (например, torch.tensor) подставляется только в main.
    """

    pad_token_id: int
    label_pad_id: int = IGNORE_INDEX
    pad_to_multiple_of: int | None = None
    to_tensor: object = None

    def __call__(self, features):
        width = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            width = math.ceil(width / self.pad_to_multiple_of) * self.pad_to_multiple_of
        batch = {"input_ids": [], "labels": [], "attention_mask": []}
        for f in features:
            ids = list(f["input_ids"])
            pad = width - len(ids)
            mask = list(f.get("attention_mask") or [1] * len(ids))
            batch["input_ids"].append(ids + [self.pad_token_id] * pad)
            batch["labels"].append(list(f["labels"]) + [self.label_pad_id] * pad)
            batch["attention_mask"].append(mask + [0] * pad)
        if self.to_tensor is not None:
            batch = {key: self.to_tensor(value) for key, value in batch.items()}
        return batch


# --- данные ---

def read_rows(path, limit=None):
    """Строки SFT из JSONL; limit — первые N (для пробного запуска)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def check_row(row):
    """Список проблем строки: нет текста, отрезки вне текста или пересекаются."""
    problems = []
    text, spans = row.get("text"), row.get("train_spans")
    if not isinstance(text, str) or not text:
        return ["нет text"]
    if not spans:
        return ["нет train_spans"]
    if not isinstance(spans, list):
        return ["train_spans не список"]
    for span in spans:
        # отрезок — пара целых [start, end); иначе распаковка ниже уронила бы весь прогон
        if (not isinstance(span, (list, tuple)) or len(span) != 2
                or not all(isinstance(x, int) and not isinstance(x, bool) for x in span)):
            return [f"битый отрезок {span!r}: нужна пара целых [start, end)"]
    previous_end = 0
    for a, b in sorted(spans):
        if not 0 <= a < b <= len(text):
            problems.append(f"отрезок [{a}, {b}) вне текста длины {len(text)}")
        if a < previous_end:
            problems.append(f"отрезок [{a}, {b}) пересекается с предыдущим")
        previous_end = max(previous_end, b)
    return problems


def prompt_hashes(rows):
    """Хеши промпта в данных: больше одного — данные собраны с разными SYSTEM_PROMPT/SCHEMAS."""
    return sorted({(row.get("meta") or {}).get("prompt_hash") or "?" for row in rows})


def text_tokenizer(tokenizer):
    """У VL-чекпоинта Unsloth возвращает процессор; offset_mapping умеет только его токенайзер."""
    return getattr(tokenizer, "tokenizer", tokenizer)


def encode_row(tokenizer, row):
    """Токенизирует готовый текст (без спецтокенов сверху) и строит метки по offset_mapping."""
    encoded = tokenizer(row["text"], return_offsets_mapping=True, add_special_tokens=False)
    ids = list(encoded["input_ids"])
    offsets = [tuple(o) for o in encoded["offset_mapping"]]
    labels = labels_from_spans(offsets, row["train_spans"], ids)
    left, right = edge_tokens(offsets, row["train_spans"])
    return {"id": row.get("id"), "input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids),
            "est_tokens": (row.get("meta") or {}).get("est_tokens"), "edges": (left, right)}


def encode_rows(tokenizer, rows):
    """Кодирует строки; пропускает битые с сообщением и печатает счётчики границ."""
    examples, bad = [], 0
    for row in rows:
        problems = check_row(row)
        if problems:
            bad += 1
            print(f"  пропуск {row.get('id')}: {'; '.join(problems)}")
            continue
        examples.append(encode_row(tokenizer, row))
    left = sum(e["edges"][0] for e in examples)
    right = sum(e["edges"][1] for e in examples)
    if left or right:
        print(f"  токенов на границе отрезков: слева {left} (без потерь), справа {right} (с потерями)")
    if bad:
        print(f"  битых строк: {bad}")
    return examples


def model_features(example):
    """Только поля, которые уходят в модель."""
    return {key: example[key] for key in ("input_ids", "labels", "attention_mask")}


def describe(name, examples, max_len):
    """Печатает длины, долю обучаемых токенов и калибровку est_tokens (len(text) / 3.4)."""
    lengths = [len(e["input_ids"]) for e in examples]
    stats = length_stats(lengths, thresholds=sorted({2048, 4096, 8192, 10240, 16384, max_len}))
    trainable = sum(sum(1 for x in e["labels"] if x != IGNORE_INDEX) for e in examples)
    print(f"{name}: {stats['count']} примеров, токенов min {stats['min']} / p50 {stats['p50']} / "
          f"p90 {stats['p90']} / p95 {stats['p95']} / max {stats['max']}")
    print("  длиннее порога: " + ", ".join(f">{t}: {n}" for t, n in stats["over"].items()))
    total = sum(lengths) or 1
    print(f"  обучаемых токенов: {trainable} ({trainable / total:.1%})")
    pairs = [(e["est_tokens"], len(e["input_ids"])) for e in examples if e.get("est_tokens")]
    if pairs:
        ratio = sum(n for _, n in pairs) / sum(est for est, _ in pairs)
        print(f"  токенов на est_tokens: {ratio:.3f} (>1 — оценка занижена)")
    return stats


def show_sample(tokenizer, example, limit=600):
    """Декодирует обучаемые участки первого примера: глазами проверить, что маска на ответах."""
    print(f"Обучаемые участки примера {example.get('id')}:")
    for i, j in trainable_segments(example["labels"]):
        text = tokenizer.decode(example["input_ids"][i:j])
        shown = text if len(text) <= limit else text[: limit // 2] + " … " + text[-limit // 2:]
        print(f"  [{i}:{j}] {shown!r}")


# --- LoRA и точность ---

def lora_target_regex(targets=LORA_TARGETS):
    """Регулярка для PEFT: нужные проекции во всех слоях, кроме визуальной башни."""
    names = "|".join(targets)
    markers = "|".join(VISION_MARKERS)
    return rf"^(?!.*(?:{markers})).*\.(?:{names})$"


def lora_coverage(module_names, targets=LORA_TARGETS):
    """Какие проекции получили LoRA: счётчики, отсутствующие проекции, модули визуальной башни."""
    counts = {t: 0 for t in targets}
    visual = []
    for name in module_names:
        suffix = name.rsplit(".", 1)[-1]
        if suffix in counts:
            counts[suffix] += 1
        if any(marker in name for marker in VISION_MARKERS):
            visual.append(name)
    missing = [t for t, n in counts.items() if n == 0]
    return {"counts": counts, "missing": missing, "visual": visual}


def native_bf16(capability):
    """bf16 без эмуляции — только с Ampere (sm_80).

    torch.cuda.is_bf16_supported() в torch >= 2.3 по умолчанию учитывает эмуляцию и на T4 (sm_75)
    отвечает True, поэтому решаем по compute capability.
    """
    return bool(capability) and capability[0] >= 8


def pick_precision(requested, bf16_supported, capability=None):
    """bf16 на Ampere и новее (RTX 5070 — sm_120), иначе float32.

    fp16 не предлагается: на Qwen3.5 он даёт NaN, поэтому на T4 (sm_75) Unsloth и так переходит на float32.
    """
    if requested in ("bf16", "fp32"):
        if requested == "bf16" and not bf16_supported:
            raise SystemExit("bf16 не поддерживается этой видеокартой — запускайте с --fp32")
        return requested
    major = capability[0] if capability else 0
    return "bf16" if bf16_supported and major >= 8 else "fp32"


def warmup_steps(total_steps, ratio):
    """Шагов разогрева: доля от всех шагов, минимум 1."""
    return max(1, math.ceil(total_steps * ratio))


def total_steps(examples, batch, grad_accum, epochs, max_steps):
    """Число шагов оптимизатора: max_steps для пробного запуска или эпохи × шагов в эпохе."""
    if max_steps and max_steps > 0:
        return max_steps
    per_epoch = math.ceil(examples / (batch * grad_accum))
    return max(1, math.ceil(per_epoch * epochs))


# --- чат-шаблон в выгрузке ---

def install_chat_template(model_dir, template_path):
    """Кладёт шаблон LM Studio в слитую модель, чтобы convert_hf_to_gguf.py записал его в GGUF.

    Пишется chat_template.jinja (его читают transformers v5 и свежий llama.cpp) и, если есть,
    поле chat_template в tokenizer_config.json — чтобы оба источника совпадали.
    """
    model_dir, template = Path(model_dir), Path(template_path).read_text(encoding="utf-8")
    (model_dir / "chat_template.jinja").write_text(template, encoding="utf-8")
    config_path = model_dir / "tokenizer_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["chat_template"] = template
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    stale = model_dir / "chat_template.json"  # старый формат перекрыл бы .jinja
    if stale.exists():
        stale.unlink()
    return template


def same_template(found, expected):
    """Совпадение шаблонов без учёта хвостовых пробелов и переводов строк."""
    return (found or "").rstrip() == (expected or "").rstrip()


# --- CLI ---

def build_parser():
    parser = argparse.ArgumentParser(
        prog="train_sft.py",
        description="LoRA SFT Qwen3.5-4B на траекториях учителя (16-битная LoRA, без повторного чат-шаблона)",
    )
    data = parser.add_argument_group("данные")
    data.add_argument("--train", help="sft/train.jsonl")
    data.add_argument("--val", help="sft/val.jsonl (для eval loss и выбора лучшего чекпоинта)")
    data.add_argument("--limit-train", type=int, help="первые N обучающих примеров (пробный запуск)")
    data.add_argument("--limit-val", type=int, help="первые N примеров val")
    data.add_argument("--max-len", type=int, default=4096,
                      help="максимум токенов в примере; длиннее — отбрасываются (по умолчанию 4096)")
    data.add_argument("--dry-run", action="store_true",
                      help="только токенизировать и показать статистику длин (нужен transformers, не GPU)")
    data.add_argument("--tokenizer", help="путь или id токенайзера для --dry-run (по умолчанию --model)")

    model = parser.add_argument_group("модель и LoRA")
    model.add_argument("--model", default=DEFAULT_MODEL, help=f"базовая модель (по умолчанию {DEFAULT_MODEL})")
    model.add_argument("--loader", choices=("vision", "language"), default="vision",
                       help="FastVisionModel (как в ноутбуке Unsloth для Qwen3.5, башня зрения заморожена) "
                            "или FastLanguageModel")
    model.add_argument("--rank", type=int, default=16)
    model.add_argument("--alpha", type=int, default=16)
    model.add_argument("--dropout", type=float, default=0.0)

    train = parser.add_argument_group("обучение")
    train.add_argument("--out", default="runs/sft/qwen3.5-4b-ra", help="папка чекпоинтов и адаптера")
    train.add_argument("--epochs", type=float, default=2.0)
    train.add_argument("--lr", type=float, default=1e-4, help="скорость обучения (перебор 5e-5..2e-4)")
    train.add_argument("--warmup", type=float, default=0.04, help="доля шагов разогрева")
    train.add_argument("--batch", type=int, default=1, help="примеров на шаг на устройство")
    train.add_argument("--grad-accum", type=int, default=16, help="накопление градиента (эффективный батч 16)")
    train.add_argument("--max-steps", type=int, default=-1, help="ограничить число шагов (пробный запуск)")
    train.add_argument("--eval-steps", type=int, default=50)
    train.add_argument("--save-steps", type=int, default=50, help="кратно --eval-steps")
    train.add_argument("--logging-steps", type=int, default=5)
    train.add_argument("--optim", default="adamw_8bit")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--resume", nargs="?", const="last",
                       help="продолжить с чекпоинта: путь или без значения — последний в --out")
    precision = train.add_mutually_exclusive_group()
    precision.add_argument("--bf16", dest="precision", action="store_const", const="bf16")
    precision.add_argument("--fp32", dest="precision", action="store_const", const="fp32")
    parser.set_defaults(precision="auto")

    export = parser.add_argument_group("выгрузка")
    export.add_argument("--merge-out", help="сохранить слитую 16-битную модель сюда (для GGUF)")
    export.add_argument("--merge-from", help="не обучать, а слить готовый адаптер или чекпоинт в --merge-out")
    export.add_argument("--chat-template", default=str(DEFAULT_TEMPLATE),
                        help="шаблон LM Studio, который кладётся в слитую модель")
    export.add_argument("--check-gguf", help="сравнить чат-шаблон в GGUF с --chat-template и выйти")
    return parser


def validate_args(parser, args):
    if args.check_gguf:
        return
    if args.merge_from:
        if not args.merge_out:
            parser.error("--merge-from требует --merge-out")
        return
    if not args.train:
        parser.error("нужен --train")
    if args.eval_steps <= 0 or args.save_steps <= 0:
        parser.error("--eval-steps и --save-steps должны быть больше нуля")
    if args.val and args.save_steps % args.eval_steps:
        parser.error("--save-steps должен быть кратен --eval-steps (лучший чекпоинт выбирается по eval loss)")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    if args.check_gguf:
        return check_gguf(args.check_gguf, args.chat_template)
    if args.merge_from:
        return merge_only(args)
    train_rows = read_rows(args.train, args.limit_train)
    val_rows = read_rows(args.val, args.limit_val) if args.val else []
    hashes = prompt_hashes(train_rows + val_rows)
    print(f"Строк: train {len(train_rows)}, val {len(val_rows)}; prompt_hash: {', '.join(hashes)}")
    if len(hashes) > 1:
        print("  ВНИМАНИЕ: в данных несколько prompt_hash — примеры собраны с разными промптами")
    if args.dry_run:
        return dry_run(args, train_rows, val_rows)
    return train(args, train_rows, val_rows)


def dry_run(args, train_rows, val_rows):
    """Токенизация без модели и GPU: подобрать --max-len по распределению длин."""
    from transformers import AutoTokenizer

    tokenizer = text_tokenizer(AutoTokenizer.from_pretrained(args.tokenizer or args.model))
    for name, rows in (("train", train_rows), ("val", val_rows)):
        if not rows:
            continue
        examples = encode_rows(tokenizer, rows)
        describe(name, examples, args.max_len)
        kept, dropped, truncated = filter_by_length(examples, args.max_len)
        print(f"  при --max-len {args.max_len}: останется {len(kept)}, отброшено {dropped}, обрезан хвост {truncated}")
        warning = drop_warning(dropped, len(examples), args.max_len)
        if warning:
            print(warning)
        if name == "train" and kept:
            show_sample(tokenizer, kept[0])
    return 0


def load_model(args, dtype, max_len, model_name=None):
    """Загрузка через Unsloth в 16 битах (QLoRA для Qwen3.5 не рекомендуется)."""
    import unsloth  # noqa: F401 — Unsloth должен импортироваться раньше transformers
    from unsloth import FastLanguageModel, FastVisionModel

    loader = FastVisionModel if args.loader == "vision" else FastLanguageModel
    model, tokenizer = loader.from_pretrained(
        model_name or args.model,
        max_seq_length=max_len,
        dtype=dtype,
        load_in_4bit=False,
    )
    return loader, model, tokenizer


def add_lora(args, loader, model):
    """LoRA на внимание, Gated DeltaNet и MLP языковой части; башня зрения не трогается."""
    common = dict(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none",
                  use_gradient_checkpointing="unsloth", random_state=args.seed)
    if args.loader == "vision":
        flags = dict(finetune_vision_layers=False, finetune_language_layers=True,
                     finetune_attention_modules=True, finetune_mlp_modules=True)
        try:
            peft_model = loader.get_peft_model(model, target_modules=lora_target_regex(), **flags, **common)
        except (AssertionError, TypeError, ValueError) as error:
            # некоторые версии Unsloth принимают только список имён; башню зрения тогда отсекает проверка ниже
            print(f"  регулярка target_modules не принята ({error}); пробую список имён")
            peft_model = loader.get_peft_model(model, target_modules=list(LORA_TARGETS), **flags, **common)
        model = peft_model
    else:
        model = loader.get_peft_model(model, target_modules=list(LORA_TARGETS), **common)
    names = [name for name, module in model.named_modules() if hasattr(module, "lora_A")]
    coverage = lora_coverage(names)
    print("LoRA: " + ", ".join(f"{t} {n}" for t, n in coverage["counts"].items()))
    if coverage["visual"]:
        raise SystemExit(f"LoRA попала в визуальную башню: {coverage['visual'][:3]}…")
    if coverage["missing"]:
        raise SystemExit(f"LoRA не нашла проекции {coverage['missing']} — проверьте имена модулей модели")
    if hasattr(loader, "for_training"):
        loader.for_training(model)
    return model


def train(args, train_rows, val_rows):
    import torch

    capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    bf16_ok = native_bf16(capability)
    precision = pick_precision(args.precision, bf16_ok, capability)
    if capability and capability[0] < 8:
        print(f"ВНИМАНИЕ: GPU sm_{capability[0]}{capability[1]} (T4?) — Qwen3.5 учится во float32 "
              "примерно в 10 раз медленнее; годится только для пробного запуска")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    print(f"Точность: {precision}")

    loader, model, tokenizer = load_model(args, dtype, args.max_len)
    text_tok = text_tokenizer(tokenizer)

    datasets = {}
    for name, rows in (("train", train_rows), ("val", val_rows)):
        if not rows:
            continue
        examples = encode_rows(text_tok, rows)
        describe(name, examples, args.max_len)
        kept, dropped, truncated = filter_by_length(examples, args.max_len)
        print(f"  {name}: оставлено {len(kept)}, отброшено длиннее {args.max_len}: {dropped}, обрезан хвост {truncated}")
        warning = drop_warning(dropped, len(examples), args.max_len)
        if warning:
            print(warning)
        datasets[name] = kept
    if not datasets.get("train"):
        raise SystemExit("нет обучающих примеров после фильтра длины")
    show_sample(text_tok, datasets["train"][0])

    model = add_lora(args, loader, model)
    trainer = make_trainer(args, model, text_tok, datasets, precision)
    resume = True if args.resume == "last" else args.resume
    result = trainer.train(resume_from_checkpoint=resume)

    out = Path(args.out)
    adapter_dir = out / "adapter"
    model.save_pretrained(str(adapter_dir))  # при --val это лучший по eval loss (load_best_model_at_end)
    tokenizer.save_pretrained(str(adapter_dir))
    summary = {"args": vars(args), "precision": precision, "train_examples": len(datasets["train"]),
               "val_examples": len(datasets.get("val", [])), "prompt_hashes": prompt_hashes(train_rows),
               "best_checkpoint": trainer.state.best_model_checkpoint, "best_eval_loss": trainer.state.best_metric,
               "train_loss": result.training_loss}
    (out / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Адаптер: {adapter_dir}; лучший чекпоинт: {trainer.state.best_model_checkpoint}")
    if args.merge_out:
        merge(model, tokenizer, args.merge_out, args.chat_template)
    return 0


def make_trainer(args, model, tokenizer, datasets, precision):
    """Обычный Trainer на готовых input_ids/labels: SFTTrainer переприменил бы чат-шаблон."""
    import torch
    from transformers import Trainer, TrainingArguments

    steps = total_steps(len(datasets["train"]), args.batch, args.grad_accum, args.epochs, args.max_steps)
    has_val = bool(datasets.get("val"))
    print(f"Шагов оптимизатора: {steps}, разогрев {warmup_steps(steps, args.warmup)}")
    training_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps(steps, args.warmup),
        weight_decay=0.01,
        optim=args.optim,
        bf16=precision == "bf16",
        fp16=False,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if has_val else "no",
        eval_steps=args.eval_steps if has_val else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,  # три чекпоинта всего; лучший по eval loss среди них Trainer не удаляет
        load_best_model_at_end=has_val,
        metric_for_best_model="eval_loss" if has_val else None,
        greater_is_better=False if has_val else None,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
    )
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    collator = PadCollator(pad_token_id=pad_id, to_tensor=lambda rows: torch.tensor(rows, dtype=torch.long))
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=[model_features(e) for e in datasets["train"]],
        eval_dataset=[model_features(e) for e in datasets["val"]] if has_val else None,
        data_collator=collator,
    )


def merge(model, tokenizer, merge_out, template_path):
    """Слитая 16-битная модель плюс шаблон LM Studio — вход для convert_hf_to_gguf.py."""
    model.save_pretrained_merged(merge_out, tokenizer, save_method="merged_16bit")
    install_chat_template(merge_out, template_path)
    print(f"Слитая модель: {merge_out} (чат-шаблон: {template_path})")


def merge_only(args):
    """Слить выбранный чекпоинт (лучший по метрикам агента, а не только по eval loss)."""
    import torch

    bf16_ok = native_bf16(torch.cuda.get_device_capability() if torch.cuda.is_available() else None)
    dtype = torch.bfloat16 if bf16_ok else torch.float32
    _loader, model, tokenizer = load_model(args, dtype, args.max_len, model_name=args.merge_from)
    merge(model, tokenizer, args.merge_out, args.chat_template)
    return 0


def check_gguf(path, template_path):
    """Сверяет tokenizer.chat_template в GGUF с шаблоном LM Studio (нужен пакет gguf из llama.cpp)."""
    from gguf import GGUFReader

    field = GGUFReader(path).fields.get("tokenizer.chat_template")
    found = bytes(field.parts[field.data[0]]).decode("utf-8") if field else None
    expected = Path(template_path).read_text(encoding="utf-8")
    if same_template(found, expected):
        print("Чат-шаблон в GGUF совпадает с шаблоном LM Studio")
        return 0
    print("Чат-шаблон в GGUF ОТЛИЧАЕТСЯ" if found else "В GGUF нет tokenizer.chat_template")
    return 1


if __name__ == "__main__":
    sys.exit(main())
