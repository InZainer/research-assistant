#!/usr/bin/env bash
# Альтернатива setup_wsl.sh: готовый образ unsloth/unsloth (подходит и для RTX 50xx).
# Нужен Docker Desktop с бэкендом WSL2 и поддержкой GPU (или Docker Engine + NVIDIA Container Toolkit).
# Запуск из WSL: bash train/run_docker.sh — поднимает контейнер и открывает в нём bash.
# НЕ ПРОВЕРЕНО на RTX 5070: содержимое образа (версии torch/transformers/Unsloth) не сверялось.
# Основной путь — setup_wsl.sh; Docker — запасной.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

IMAGE="${IMAGE:-unsloth/unsloth}"
NAME="${CONTAINER:-unsloth-ra}"
command -v docker >/dev/null 2>&1 || die "docker не найден: установите Docker Desktop и включите интеграцию с WSL"
mkdir -p "$FT_HOME/hf-cache"

if docker container inspect "$NAME" >/dev/null 2>&1; then
    say "Контейнер $NAME уже есть — запускаю (новые параметры docker run к нему не применяются: docker rm -f $NAME)"
    docker start "$NAME" >/dev/null
else
    say "Создаю контейнер $NAME из $IMAGE (первый раз скачивает образ на несколько ГБ)"
    # Комплект — /workspace/bundle, рабочая папка (чекпоинты, модель, GGUF) — /workspace/ft.
    # HF_HOME в /workspace/ft/hf-cache: модель (~9 ГБ) скачивается один раз и переживает пересоздание контейнера.
    # --ipc=host и ulimit — как в инструкции Unsloth по Docker (общая память загрузчика данных, pinned memory).
    # FT_NO_VENV=1: внутри контейнера используется Python образа, а не venv из WSL.
    # Порты наружу не пробрасываются: Jupyter образа не нужен.
    docker run -d --name "$NAME" --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864         -v "$BUNDLE":/workspace/bundle         -v "$FT_HOME":/workspace/ft         -e FT_HOME=/workspace/ft -e HF_HOME=/workspace/ft/hf-cache -e FT_NO_VENV=1 -e PYTHONUNBUFFERED=1         "$IMAGE" >/dev/null
fi

# Образ работает от непривилегированного пользователя unsloth: обычная установка может упасть на правах
# или на «externally-managed environment». Пробуем обычную, затем --user; неудача — не повод не открывать
# shell: сообщаем, что поставить вручную.
container_pip() {
    docker exec "$NAME" python -m pip install -q "$@"         || docker exec "$NAME" python -m pip install -q --user "$@"         || { echo "ВНИМАНИЕ: в контейнере не поставилось: $* — поставьте вручную внутри контейнера"; return 1; }
}

say "Проверка GPU и версий в контейнере"
docker exec "$NAME" nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv     || echo "ВНИМАНИЕ: nvidia-smi в контейнере не видит GPU — см. «Неполадки» в README_RTX5070.md"
docker exec "$NAME" python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| sm', torch.cuda.get_device_capability(0))"     || echo "ВНИМАНИЕ: torch в контейнере не видит CUDA"

# Qwen3.5 нужен transformers v5: если в образе старее — ставим проверенный набор.
if ! docker exec "$NAME" python -c "import sys, transformers; sys.exit(0 if transformers.__version__.startswith('5.') else 1)"; then
    say "В образе transformers старше v5 — ставлю transformers 5.2.0 и trl 0.22.2"
    container_pip "transformers==5.2.0" "trl==0.22.2"         || echo "Без transformers v5 Qwen3.5 не загрузится: внутри контейнера выполните pip install --user transformers==5.2.0 trl==0.22.2"
fi
container_pip gguf sentencepiece protobuf || true

cat <<EOF

Контейнер готов. Внутри него:
    cd /workspace/bundle
    bash train/smoke.sh
Остановить: docker stop $NAME    Удалить: docker rm -f $NAME
EOF
docker exec -it -w /workspace/bundle "$NAME" bash
