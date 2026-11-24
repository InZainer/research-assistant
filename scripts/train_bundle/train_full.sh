#!/usr/bin/env bash
# Полное обучение с параметрами из docs/finetune.md: LoRA r=16, α=16, lr 1e-4, 2 эпохи, косинус,
# разогрев 4%, эффективный батч 16 (1 × 16), eval/сохранение каждые 50 шагов.
# Запуск: bash train/train_full.sh [доп. аргументы train_sft.py]
#   MAX_LEN=8192 LR=2e-4 bash train/train_full.sh      — другая длина и lr (перебор: 5e-5, 1e-4, 2e-4)
#   RESUME=1 bash train/train_full.sh                  — продолжить после обрыва с последнего чекпоинта
#   MERGE=0 bash train/train_full.sh                   — не сливать (выберете чекпоинт по метрикам агента)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MAX_LEN="${MAX_LEN:-4096}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-1e-4}"
RANK="${RANK:-16}"
ALPHA="${ALPHA:-16}"
BATCH="${BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
EVAL_STEPS="${EVAL_STEPS:-50}"
SAVE_STEPS="${SAVE_STEPS:-50}"
RUN="${RUN:-r${RANK}-lr${LR}-len${MAX_LEN}}"
OUT="${OUT:-$FT_HOME/runs/$RUN}"
MERGE="${MERGE:-1}"
MERGED="${MERGED:-$FT_HOME/merged-$RUN}"
TRAIN="$DATA/train.jsonl"
VAL="$DATA/val.jsonl"

need_python
has_rows "$TRAIN" || die "пустой $TRAIN — пересоберите комплект"
TRAIN_N="$(count_rows "$TRAIN")"
if [ "$TRAIN_N" -lt 200 ]; then
    echo "ВНИМАНИЕ: в train всего $TRAIN_N примеров — похоже на пилотные данные."
    echo "          Полный прогон имеет смысл на полном датасете (пересоберите комплект после ft-export)."
fi

ARGS=(--train "$TRAIN" --max-len "$MAX_LEN" --model "$BASE_MODEL"
      --epochs "$EPOCHS" --lr "$LR" --rank "$RANK" --alpha "$ALPHA" --batch "$BATCH" --grad-accum "$GRAD_ACCUM"
      --eval-steps "$EVAL_STEPS" --save-steps "$SAVE_STEPS" --out "$OUT" --chat-template "$TEMPLATE")
if has_rows "$VAL"; then
    ARGS+=(--val "$VAL")
else
    echo "ВНИМАНИЕ: val пуст — нет eval loss; в adapter/ попадёт последний шаг, а не лучший"
fi
[ "${RESUME:-0}" = "1" ] && ARGS+=(--resume)
[ "$MERGE" = "1" ] && ARGS+=(--merge-out "$MERGED")

mkdir -p "$OUT"
LOG="$OUT/train_full.log"
say "Обучение: $RUN → $OUT (журнал $LOG)"
start_vram_log "$OUT/vram.csv"
STARTED=$(date +%s)
set +e
"$PY" "$TRAIN_DIR/train_sft.py" "${ARGS[@]}" "$@" 2>&1 | tee -a "$LOG"
STATUS=${PIPESTATUS[0]}
set -e
ELAPSED=$(( $(date +%s) - STARTED ))
stop_vram_log | tee -a "$LOG"
echo "Время: $((ELAPSED / 3600)) ч $(((ELAPSED % 3600) / 60)) мин" | tee -a "$LOG"
[ "$STATUS" -eq 0 ] || die "обучение завершилось с кодом $STATUS; продолжить: RESUME=1 bash train/train_full.sh"

"$PY" "$TRAIN_DIR/bundle_tools.py" loss-report --out "$OUT" | tee -a "$LOG" || true
echo "Сводка: $OUT/train_summary.json"
if [ "$MERGE" = "1" ]; then
    echo "Дальше: MERGED=$MERGED NAME=qwen3.5-4b-ra bash train/convert_gguf.sh"
else
    echo "Слить выбранный чекпоинт:"
    echo "    $PY train/train_sft.py --merge-from $OUT/adapter --merge-out $MERGED --chat-template train/qwen3_5_chat_template.jinja"
fi
