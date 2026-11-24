#!/usr/bin/env bash
# Слитая 16-битная модель → GGUF (bf16) → Q4_K_M → сверка чат-шаблона с шаблоном LM Studio.
# Запуск: MERGED=~/ft/merged-smoke NAME=qwen3.5-4b-ra-smoke bash train/convert_gguf.sh
#   LLAMA_CPP=~/llama.cpp      — где лежит (или куда склонировать) llama.cpp
#   UPDATE_LLAMA=1             — обновить уже склонированный llama.cpp (git pull)
#   LLAMA_QUANTIZE=/путь/к/llama-quantize — готовый бинарник вместо сборки
#   OUTTYPE=f16                — тип промежуточного GGUF (по умолчанию bf16: веса уже в bf16, без потерь)
#   WIN_DIR=/mnt/c/Users/<user>/Downloads — скопировать итоговый Q4_K_M туда
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MERGED="${MERGED:-${1:-$FT_HOME/merged-smoke}}"
NAME="${NAME:-qwen3.5-4b-ra}"
OUTTYPE="${OUTTYPE:-bf16}"
LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"
GGUF_DIR="${GGUF_DIR:-$FT_HOME/gguf}"
GGUF_16="$GGUF_DIR/$NAME-$OUTTYPE.gguf"
GGUF_Q4="$GGUF_DIR/$NAME-Q4_K_M.gguf"

need_python
[ -f "$MERGED/config.json" ] || die "нет слитой модели $MERGED (сначала smoke.sh или train_full.sh)"
mkdir -p "$GGUF_DIR"

say "1. llama.cpp: $LLAMA_CPP"
if [ ! -d "$LLAMA_CPP/.git" ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP"
elif [ "${UPDATE_LLAMA:-0}" = "1" ]; then
    git -C "$LLAMA_CPP" pull --ff-only
fi
git -C "$LLAMA_CPP" log -1 --format='коммит %h от %cs' || true
# qwen35 поддерживается с февраля 2026 (PR #19435); конвертер регистрирует Qwen3_5ForConditionalGeneration
if ! grep -rqs "Qwen3_5ForConditionalGeneration" "$LLAMA_CPP"/convert_hf_to_gguf.py "$LLAMA_CPP"/conversion 2>/dev/null; then
    die "llama.cpp слишком старый для qwen35: UPDATE_LLAMA=1 bash train/convert_gguf.sh"
fi

# Полный requirements-convert_hf_to_gguf.txt из llama.cpp НЕ ставим: он закрепляет CPU-сборку torch
# и сломал бы окружение обучения. Конвертеру хватает torch/transformers отсюда, numpy, sentencepiece,
# protobuf и gguf-py из того же llama.cpp (так версия gguf совпадает с конвертером).
"$PY" -m pip install -q "$LLAMA_CPP/gguf-py" 2>/dev/null || uv pip install -q "$LLAMA_CPP/gguf-py" \
    || echo "ВНИМАНИЕ: gguf-py не поставился — используется пакет gguf из requirements-train.txt"

say "2. llama-quantize"
QUANTIZE="${LLAMA_QUANTIZE:-}"
if [ -z "$QUANTIZE" ] && command -v llama-quantize >/dev/null 2>&1; then
    QUANTIZE="$(command -v llama-quantize)"
fi
if [ -z "$QUANTIZE" ]; then
    QUANTIZE="$LLAMA_CPP/build/bin/llama-quantize"
    if [ ! -x "$QUANTIZE" ] || [ "${UPDATE_LLAMA:-0}" = "1" ]; then
        # CUDA для квантования не нужна: собираем только CPU-бинарник
        cmake -S "$LLAMA_CPP" -B "$LLAMA_CPP/build" -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
        cmake --build "$LLAMA_CPP/build" --target llama-quantize -j "$(nproc)" >/dev/null
    fi
fi
[ -x "$QUANTIZE" ] || die "нет llama-quantize ($QUANTIZE)"
echo "$QUANTIZE"

say "3. Конвертация в GGUF ($OUTTYPE): $GGUF_16"
# только текстовая часть: mmproj (башня зрения) агенту не нужен
"$PY" "$LLAMA_CPP/convert_hf_to_gguf.py" "$MERGED" --outfile "$GGUF_16" --outtype "$OUTTYPE"

say "4. Квантование Q4_K_M: $GGUF_Q4"
"$QUANTIZE" "$GGUF_16" "$GGUF_Q4" Q4_K_M

say "5. Чат-шаблон в GGUF против шаблона LM Studio"
"$PY" "$TRAIN_DIR/train_sft.py" --check-gguf "$GGUF_Q4" --chat-template "$TEMPLATE" \
    || die "чат-шаблон в GGUF отличается от шаблона LM Studio — в LM Studio модель будет видеть другой промпт"

ls -lh "$GGUF_16" "$GGUF_Q4"
sha256sum "$GGUF_Q4" | tee "$GGUF_Q4.sha256"
if [ -n "${WIN_DIR:-}" ]; then
    cp "$GGUF_Q4" "$GGUF_Q4.sha256" "$WIN_DIR"/
    echo "Скопировано в $WIN_DIR"
fi
cat <<EOF

Готово: $GGUF_Q4
Перенесите этот файл (и .sha256 рядом) на ноутбук с LM Studio — шаг 4 в README_RTX5070.md.
Промежуточный $GGUF_16 можно удалить, если место нужнее.
EOF
