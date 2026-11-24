#!/usr/bin/env bash
# Сквозной пробный запуск: длины примеров → ~30 шагов обучения → слияние в 16-битную модель.
# Проверяет конвейер, а не качество: на пилотных данных прироста метрик не ждём.
# Запуск: bash train/smoke.sh    Переменные: MAX_LEN (auto|4096|8192…), STEPS, GRAD_ACCUM, OUT, MERGED
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

STEPS="${STEPS:-30}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_LEN="${MAX_LEN:-auto}"
OUT="${OUT:-$FT_HOME/runs/smoke}"
MERGED="${MERGED:-$FT_HOME/merged-smoke}"
TRAIN="$DATA/train.jsonl"
VAL="$DATA/val.jsonl"

need_python
has_rows "$TRAIN" || die "пустой $TRAIN — пересоберите комплект"
mkdir -p "$OUT"
LOG="$OUT/smoke.log"
echo "Журнал: $LOG"

VAL_ARGS=()
if has_rows "$VAL"; then
    VAL_ARGS=(--val "$VAL" --limit-val 8)
    echo "val: $(count_rows "$VAL") примеров (для пробы берутся первые 8)"
else
    echo "val пуст — пробный запуск без --val (eval loss не считается, сохраняется последний шаг)"
fi

say "1. Длины примеров и выбор --max-len"
if [ "$MAX_LEN" = "auto" ]; then
    # наименьшая из 4096/6144/8192, при которой останется хотя бы 4 примера
    MAX_LEN="$("$PY" "$TRAIN_DIR/bundle_tools.py" pick-max-len --train "$TRAIN" --model "$BASE_MODEL")"
fi
echo "--max-len $MAX_LEN"
"$PY" "$TRAIN_DIR/train_sft.py" --train "$TRAIN" "${VAL_ARGS[@]}" --max-len "$MAX_LEN" --dry-run \
    --model "$BASE_MODEL" --chat-template "$TEMPLATE" 2>&1 | tee "$LOG"

say "2. Обучение: $STEPS шагов, накопление $GRAD_ACCUM, --max-len $MAX_LEN (первый раз скачивает модель ~9 ГБ)"
start_vram_log "$OUT/vram.csv"
STARTED=$(date +%s)
set +e
"$PY" "$TRAIN_DIR/train_sft.py" --train "$TRAIN" "${VAL_ARGS[@]}" --max-len "$MAX_LEN" \
    --model "$BASE_MODEL" --max-steps "$STEPS" --grad-accum "$GRAD_ACCUM" --batch 1 \
    --logging-steps 1 --eval-steps "$STEPS" --save-steps "$STEPS" \
    --out "$OUT" --merge-out "$MERGED" --chat-template "$TEMPLATE" "$@" 2>&1 | tee -a "$LOG"
STATUS=${PIPESTATUS[0]}
set -e
ELAPSED=$(( $(date +%s) - STARTED ))
stop_vram_log | tee -a "$LOG"
echo "Время обучения и слияния: $((ELAPSED / 60)) мин $((ELAPSED % 60)) с" | tee -a "$LOG"
[ "$STATUS" -eq 0 ] || die "обучение завершилось с кодом $STATUS (OOM — см. «Неполадки» в README_RTX5070.md)"

say "3. Итог"
"$PY" "$TRAIN_DIR/bundle_tools.py" loss-report --out "$OUT" | tee -a "$LOG" || true
[ -f "$MERGED/config.json" ] || die "нет слитой модели в $MERGED"
[ -f "$MERGED/chat_template.jinja" ] || die "в $MERGED нет chat_template.jinja"
du -sh "$MERGED" | tee -a "$LOG"
cat <<EOF | tee -a "$LOG"

Пробный запуск прошёл: адаптер $OUT/adapter, слитая модель $MERGED.
Дальше — GGUF и квантование:
    MERGED=$MERGED NAME=qwen3.5-4b-ra-smoke bash train/convert_gguf.sh
EOF
