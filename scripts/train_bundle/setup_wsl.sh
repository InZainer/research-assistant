#!/usr/bin/env bash
# Установка стека обучения в WSL2 Ubuntu 24.04 для RTX 5070 (sm_120, CUDA 12.8+).
# Запуск: bash train/setup_wsl.sh    (повторный запуск безопасен: venv переиспользуется)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

say "1. Видеокарта"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    die "nvidia-smi не найден. Поставьте в Windows драйвер NVIDIA R570+ и выполните в PowerShell: wsl --update; wsl --shutdown"
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || die "nvidia-smi не видит GPU"
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | cut -d. -f1)"
if [ "${DRIVER:-0}" -lt 570 ] 2>/dev/null; then
    die "драйвер $DRIVER старше R570: для RTX 5070 (CUDA 12.8) обновите драйвер NVIDIA в Windows"
fi

say "2. Системные пакеты (git, cmake, компилятор — для llama.cpp и ядер Unsloth)"
MISSING=""
for tool in git cmake gcc g++ make curl python3; do
    command -v "$tool" >/dev/null 2>&1 || MISSING="$MISSING $tool"
done
python3 -c "import venv, ensurepip" 2>/dev/null || MISSING="$MISSING python3-venv"
if [ -n "$MISSING" ]; then
    echo "Не хватает:$MISSING — ставлю через apt (спросит пароль sudo)"
    sudo apt-get update
    sudo apt-get install -y git cmake build-essential curl unzip python3 python3-venv python3-dev
fi

say "3. Виртуальное окружение: $VENV"
mkdir -p "$FT_HOME"
if command -v uv >/dev/null 2>&1; then
    [ -d "$VENV" ] || uv venv -p 3.12 "$VENV"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    PIP=(uv pip install)
    FREEZE=(uv pip freeze)
else
    echo "uv не найден — использую python3 -m venv (uv быстрее: curl -LsSf https://astral.sh/uv/install.sh | sh)"
    [ -d "$VENV" ] || python3 -m venv "$VENV"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    python -m pip install -q --upgrade pip
    PIP=(python -m pip install)
    FREEZE=(python -m pip freeze)
fi
python --version

say "4. torch 2.8.0 под CUDA 12.8 (cu128)"
"${PIP[@]}" "torch==2.8.0" "torchvision==0.23.0" --index-url https://download.pytorch.org/whl/cu128

say "5. Unsloth 2026.11.3, transformers 5.2.0, trl 0.22.2 и остальное (train/requirements-train.txt)"
# Только PyPI: torch 2.8.0+cu128 уже стоит (шаг 4) и удовлетворяет torch==2.8.0 из файла.
# Индекс PyTorch здесь не подключается — иначе uv взял бы из него старые certifi/setuptools.
"${PIP[@]}" -r "$TRAIN_DIR/requirements-train.txt"

say "6. Быстрые ядра Gated DeltaNet (unsloth install-kernels)"
if ! unsloth install-kernels; then
    echo "ВНИМАНИЕ: ядра не поставились. Обучение всё равно пойдёт, но медленнее (запасной путь на чистом PyTorch)."
fi

say "7. Проверка"
FREEZE_FILE="$FT_HOME/env-freeze.txt"
"${FREEZE[@]}" > "$FREEZE_FILE" || true
echo "Полный список версий пакетов: $FREEZE_FILE (приложите его к отчёту о неполадке)"
python - <<'PY'
import sys
from importlib.metadata import version

import torch
import transformers

print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| transformers", transformers.__version__)
print("unsloth", version("unsloth"), "| unsloth_zoo", version("unsloth_zoo"), "| trl", version("trl"),
      "| torchao", version("torchao"))
cuda = tuple(int(part) for part in (torch.version.cuda or "0.0").split(".")[:2])
if cuda < (12, 8):
    sys.exit(f"ОШИБКА: torch собран под CUDA {torch.version.cuda}, а RTX 5070 нужна 12.8+ — "
             "удалите ~/ft/.venv и запустите setup_wsl.sh заново")
if not torch.cuda.is_available():
    sys.exit("ОШИБКА: torch не видит CUDA — проверьте драйвер в Windows и `nvidia-smi` в WSL")
cap = torch.cuda.get_device_capability(0)
bf16 = torch.cuda.is_bf16_supported()
print("GPU:", torch.cuda.get_device_name(0), "| compute capability", cap, "| bf16", bf16)
if cap != (12, 0):
    print(f"ВНИМАНИЕ: ожидалась RTX 5070 с (12, 0), а здесь {cap}")
if cap[0] < 8 or not bf16:
    sys.exit("ОШИБКА: нет нативного bf16 — эта видеокарта не подходит для полного обучения")
if not transformers.__version__.startswith("5."):
    sys.exit("ОШИБКА: для Qwen3.5 нужен transformers v5")
x = torch.ones(1024, 1024, device="cuda", dtype=torch.bfloat16)
print("Проверочное умножение bf16 на GPU:", float((x @ x).mean()))
import unsloth  # noqa: E402,F401 — печатает свою сводку (версии, Triton, GPU)
print("Готово: окружение подходит для обучения")
PY

echo
echo "Окружение: $VENV (скрипты комплекта подключают его сами)."
echo "Дальше: bash train/smoke.sh"
