# Общие переменные скриптов комплекта. Подключается через `source`, сам не запускается.
# Любую переменную можно переопределить перед запуском: FT_HOME=/data/ft bash train/smoke.sh

TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$(dirname "$TRAIN_DIR")"
DATA="$BUNDLE/data"
TEMPLATE="$TRAIN_DIR/qwen3_5_chat_template.jinja"

# Рабочая папка: venv, чекпоинты, слитая модель, GGUF. Держите её в файловой системе Linux
# (~/ft), а не на /mnt/c — так в разы быстрее.
FT_HOME="${FT_HOME:-$HOME/ft}"
VENV="${VENV:-$FT_HOME/.venv}"
BASE_MODEL="${BASE_MODEL:-unsloth/Qwen3.5-4B}"
export PYTHONUNBUFFERED=1

# В Docker-контейнере (FT_NO_VENV=1) используется Python образа; в WSL — venv из setup_wsl.sh.
if [ -z "${FT_NO_VENV:-}" ] && [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi
PY="${PY:-python}"

say() { printf '\n=== %s\n' "$*"; }
die() { printf 'ОШИБКА: %s\n' "$*" >&2; exit 1; }

# Файл существует и в нём есть хотя бы одна непустая строка.
has_rows() { [ -s "$1" ] && grep -q '[^[:space:]]' "$1"; }

count_rows() { if [ -f "$1" ]; then grep -c '[^[:space:]]' "$1" || true; else echo 0; fi; }

need_python() {
    command -v "$PY" >/dev/null 2>&1 || die "нет $PY: сначала bash train/setup_wsl.sh (или запуск в Docker)"
    "$PY" -c "import torch, transformers" 2>/dev/null \
        || die "в $PY нет torch/transformers: сначала bash train/setup_wsl.sh"
}

# Пиковая занятость видеопамяти по nvidia-smi (МиБ, вся GPU, включая рабочий стол Windows).
start_vram_log() {
    VRAM_LOG="$1"
    : > "$VRAM_LOG"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -l 5 >> "$VRAM_LOG" 2>/dev/null &
        VRAM_PID=$!
    else
        VRAM_PID=""
    fi
}

stop_vram_log() {
    if [ -n "${VRAM_PID:-}" ]; then
        kill "$VRAM_PID" 2>/dev/null || true
        wait "$VRAM_PID" 2>/dev/null || true
    fi
    if [ -s "${VRAM_LOG:-}" ]; then
        awk -F', *' '$1+0 > max { max = $1+0; total = $2+0 } END { if (max) printf "Пик видеопамяти (nvidia-smi): %d из %d МиБ\n", max, total }' "$VRAM_LOG"
    fi
}
