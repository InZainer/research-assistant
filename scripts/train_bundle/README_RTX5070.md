# Дообучение Qwen3.5-4B на ПК с RTX 5070

Комплект для LoRA-дообучения агента «Ассистент исследователя». Обучение идёт на ПК с RTX 5070 (WSL2 или Docker). Готовая модель в GGUF переносится на ноутбук с LM Studio, там же её и оценивают.

| | |
|---|---|
| Версия комплекта | `{{VERSION}}` |
| Исходный коммит | `{{GIT_COMMIT}}` |
| Режим | **{{MODE}}** — {{MODE_TEXT}} |
| Данные | train: {{TRAIN_EXAMPLES}} примеров, val: {{VAL_EXAMPLES}} |
| Длина примеров (оценка) | {{TOKENS_LINE}} |
| `prompt_hash` данных | `{{PROMPT_HASHES}}` |
| Базовая модель | `unsloth/Qwen3.5-4B` (HF: `Qwen/Qwen3.5-4B`), в LM Studio — `qwen3.5-4b` |

{{MODE_NOTE}}

## Что внутри

```
rtx5070_train_bundle/
  README_RTX5070.md          эта инструкция
  MANIFEST.json              список файлов с sha256 и размерами, версия, сводка данных
  data/train.jsonl           примеры SFT (текст диалога + границы ходов ассистента)
  data/val.jsonl             валидация (может быть пустой — тогда без eval loss)
  data/stats.json            статистика экспорта: kind, язык, гистограмма длин
  train/train_sft.py         скрипт обучения/слияния/проверки GGUF (копия scripts/train_sft.py)
  train/qwen3_5_chat_template.jinja   чат-шаблон LM Studio, кладётся в слитую модель
  train/requirements-train.txt        закреплённый стек: torch 2.8 cu128, transformers 5.2.0, trl 0.22.2, unsloth 2026.11.3
  train/setup_wsl.sh         установка окружения в WSL2 Ubuntu 24.04
  train/run_docker.sh        запасной путь: контейнер unsloth/unsloth (не проверен)
  train/smoke.sh             пробный прогон: длины → ~30 шагов → слияние
  train/train_full.sh        полное обучение (параметры из docs/finetune.md)
  train/convert_gguf.sh      слитая модель → GGUF → Q4_K_M → сверка чат-шаблона
  train/bundle_tools.py      подбор --max-len и сводка loss (вызывается скриптами)
  train/verify_bundle.py     проверка sha256 по MANIFEST.json
  train/common.sh            общие переменные скриптов
  notebook/finetune_qwen35_colab.ipynb   запасной путь: пробный прогон в Colab T4
```

Ключей API в комплекте нет, и они не нужны: модель скачивается с Hugging Face без токена.

## Требования к ПК

- Windows 11 (или Windows 10 22H2+) с WSL2 и Ubuntu 24.04.
- Драйвер NVIDIA **R570 или новее**, поставленный в Windows. RTX 5070 — это Blackwell (sm_120), ей нужна CUDA 12.8+. Внутри WSL драйвер и CUDA Toolkit ставить **не нужно**.
- RTX 5070 12 ГБ. LoRA 4B в bf16 занимает около 10 ГБ, поэтому браузеры и игры на время обучения лучше закрыть.
- Около **40 ГБ** свободного места в WSL. Модель занимает ~9 ГБ, слитая модель ещё ~9 ГБ, GGUF bf16 ~8 ГБ, Q4_K_M ~2.7 ГБ, плюс окружение ~10 ГБ.
- Оперативная память: для слияния нужно 16 ГБ, доступных WSL (см. «Неполадки»).
- Интернет: базовая модель (~9 ГБ), пакеты Python (~5 ГБ), llama.cpp.

Установить WSL (PowerShell от администратора, один раз, потом перезагрузка):

```powershell
wsl --install -d Ubuntu-24.04
wsl --update
```

## 1. Копирование и распаковка

Распаковывайте **в файловую систему Linux** (`~/`), а не в `/mnt/c/...`: чтение данных и запись чекпоинтов через `/mnt/c` в разы медленнее.

Скопируйте `rtx5070_train_bundle.zip` на ПК, например в `C:\Users\<user>\Downloads`. Дальше в Ubuntu (WSL):

```bash
cp /mnt/c/Users/<user>/Downloads/rtx5070_train_bundle.zip ~/
cd ~
python3 -m zipfile -e rtx5070_train_bundle.zip .     # или: sudo apt install -y unzip && unzip rtx5070_train_bundle.zip
cd ~/rtx5070_train_bundle
python3 train/verify_bundle.py
```

`verify_bundle.py` должен вывести «Комплект цел». Скрипты запускаются как `bash train/<скрипт>.sh`, поэтому флаг исполнения не нужен.

> **Переводы строк.** Файлы `.sh` записаны с LF и работают в Linux. Не открывайте и не сохраняйте их в редакторах Windows, которые меняют переводы строк на CRLF (Блокнот, некоторые настройки VS Code и git). Иначе bash выдаст ошибку вида `$'\r': command not found`. Если такое случилось, см. «Неполадки».

## 2. Окружение

### Вариант А: WSL2 (основной)

```bash
cd ~/rtx5070_train_bundle
bash train/setup_wsl.sh
```

Скрипт делает следующее:

- проверяет `nvidia-smi` и версию драйвера;
- ставит через apt git, cmake и компилятор (спросит пароль sudo);
- создаёт окружение `~/ft/.venv` (через `uv`, если он есть, иначе `python3 -m venv`);
- ставит torch 2.8.0 из индекса cu128, затем с PyPI `train/requirements-train.txt` (unsloth и unsloth_zoo 2026.11.3, transformers 5.2.0, trl 0.22.2, torchao 0.16.0);
- выполняет `unsloth install-kernels`;
- записывает версии всех пакетов в `~/ft/env-freeze.txt` — приложите этот файл, если что-то пойдёт не так;
- проверяет, что `torch.cuda.get_device_capability()` равно `(12, 0)` и bf16 работает на GPU.

В конце должно быть «Готово: окружение подходит для обучения». Остальные скрипты подключают `~/ft/.venv` сами.

`triton` отдельно не обновляйте: torch 2.8 ставит свою версию, и она уже поддерживает Blackwell.

### Вариант Б: Docker (не проверен)

Запасной путь, если WSL-установка не удалась. **Этот вариант не тестировался:** версии torch, transformers и Unsloth внутри образа `unsloth/unsloth` не сверялись с проверенным стеком. Нужен Docker Desktop с бэкендом WSL2, поддержкой GPU и включённой интеграцией с Ubuntu.

```bash
cd ~/rtx5070_train_bundle
bash train/run_docker.sh
```

Скрипт поднимает контейнер `unsloth-ra` из образа `unsloth/unsloth`. Комплект монтируется в `/workspace/bundle`, а `~/ft` — в `/workspace/ft`; кеш Hugging Face лежит в `~/ft/hf-cache`, так что модель не скачивается заново после пересоздания контейнера. Контейнер запускается с `--ipc=host --ulimit memlock=-1 --ulimit stack=67108864`, как в инструкции Unsloth. Скрипт проверяет GPU и версию transformers (нужна v5) и открывает bash внутри контейнера. Если доустановка пакетов в образ не удалась (образ работает от пользователя `unsloth` без root), скрипт не падает, а пишет, что поставить вручную. Контейнер, созданный старой версией скрипта, удалите (`docker rm -f unsloth-ra`): новые параметры к нему не применяются. Дальше те же команды, что ниже: `bash train/smoke.sh` и так далее.

Шаг GGUF (`convert_gguf.sh`) надёжнее выполнять в варианте А: там точно есть cmake и компилятор.

## 3. Шаг 1 — длины примеров и `--max-len`

`smoke.sh` делает это сам, но посмотреть можно и отдельно. Нужен только токенайзер, GPU не требуется:

```bash
cd ~/rtx5070_train_bundle
source ~/ft/.venv/bin/activate
python train/train_sft.py --train data/train.jsonl --max-len 8192 --dry-run
```

Если `data/val.jsonl` не пуст, добавьте `--val data/val.jsonl`. В выводе будут:

- перцентили длин (p50/p90/p95/max);
- сколько примеров длиннее 2048/4096/8192;
- доля обучаемых токенов;
- декодированные участки с потерями первого примера. Там должны быть только ходы ассистента: `<tool_call>…</tool_call><|im_end|>`.

Как выбирать `--max-len`:

- **4096** — если она отбрасывает не больше ~5% train и среди отброшенных нет перекоса по `kind`;
- **8192** — если 4096 отбрасывает много. Это на грани 12 ГБ: сначала пробный прогон, смотрим пик памяти;
- меньше 4096 не брать: системный промпт со схемами инструментов сам занимает около 3000 токенов.

В данных этого комплекта по оценке экспорта {{TOKENS_LINE}}. Если p50 больше 4096, то при 4096 большая часть примеров отбросится. `smoke.sh` в режиме `MAX_LEN=auto` берёт наименьшую длину из 4096/6144/8192, при которой остаётся хотя бы 4 примера.

## 4. Шаг 2 — пробный прогон (smoke)

```bash
cd ~/rtx5070_train_bundle
bash train/smoke.sh
```

Что делает скрипт:

1. Подбирает `--max-len` и печатает статистику длин.
2. Обучает 30 шагов (`--grad-accum 4`, без `--val`, если val пуст). При первом запуске скачивает модель (~9 ГБ в `~/.cache/huggingface`).
3. Сливает LoRA в 16-битную модель `~/ft/merged-smoke` с чат-шаблоном LM Studio.
4. Печатает пик видеопамяти, время и сводку loss.

Журнал пишется в `~/ft/runs/smoke/smoke.log`.

Признаки успеха:

- строка `Точность: bf16`;
- строка `LoRA:` с ненулевыми счётчиками у всех проекций: `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj` (Gated DeltaNet), q/k/v/o и MLP. Если проекция не нашлась, скрипт остановится сам;
- «Обучаемые участки примера …» показывают только ходы ассистента (`<tool_call>…`), а не системный промпт и не выдачу поиска;
- loss падает (итоговая строка «loss падает — конвейер обучения работает»), `nan` нет;
- «Пик видеопамяти (nvidia-smi): … из ~12000 МиБ». Это вся GPU, включая рабочий стол Windows. Запишите число: от него зависит, пробовать ли 8192;
- в конце «Пробный запуск прошёл» и путь к слитой модели.

**Время заранее неизвестно**: замеров Qwen3.5-4B на длинном контексте на RTX 5070 нет. Скрипт печатает фактическое время, запишите его: по нему оценивается полный прогон (время ≈ пропорционально числу шагов × накопление × длина).

Попробовать 8192 после удачного прогона на меньшей длине:

```bash
MAX_LEN=8192 OUT=~/ft/runs/smoke-8k MERGED=~/ft/merged-smoke-8k bash train/smoke.sh
```

## 5. Шаг 3 — слияние, GGUF и квантование

Слияние уже сделано в `smoke.sh` (`--merge-out`). Конвертация:

```bash
cd ~/rtx5070_train_bundle
MERGED=~/ft/merged-smoke NAME=qwen3.5-4b-ra-smoke bash train/convert_gguf.sh
```

Что делает скрипт:

- клонирует свежий llama.cpp в `~/llama.cpp` (qwen35 поддерживается с февраля 2026) и проверяет, что конвертер знает `Qwen3_5ForConditionalGeneration`;
- собирает `llama-quantize` (только CPU);
- выполняет `convert_hf_to_gguf.py` → `~/ft/gguf/qwen3.5-4b-ra-smoke-bf16.gguf`, затем `llama-quantize` → `~/ft/gguf/qwen3.5-4b-ra-smoke-Q4_K_M.gguf` (~2.7 ГБ);
- сверяет чат-шаблон внутри GGUF с шаблоном LM Studio (`train_sft.py --check-gguf`). Ожидается «Чат-шаблон в GGUF совпадает с шаблоном LM Studio».

Готовый `llama-quantize` можно указать через `LLAMA_QUANTIZE=/путь/llama-quantize`. Тип промежуточного файла задаётся через `OUTTYPE=f16`; по умолчанию bf16 — без потерь, веса и так в bf16. Сразу скопировать результат в Windows:

```bash
WIN_DIR=/mnt/c/Users/<user>/Downloads MERGED=~/ft/merged-smoke NAME=qwen3.5-4b-ra-smoke bash train/convert_gguf.sh
```

Башня зрения (`mmproj`) агенту не нужна и не конвертируется.

## 6. Шаг 4 — перенос на ноутбук и LM Studio

Перенесите `qwen3.5-4b-ra-smoke-Q4_K_M.gguf` и `.sha256` рядом с ним на ноутбук: флешкой, по сети или через облако, но только своё, без публичной ссылки. Проверить и импортировать (PowerShell на ноутбуке):

```powershell
cd $HOME\Downloads
(Get-FileHash .\qwen3.5-4b-ra-smoke-Q4_K_M.gguf -Algorithm SHA256).Hash.ToLower()   # сравнить с .sha256
lms import .\qwen3.5-4b-ra-smoke-Q4_K_M.gguf      # спросит издателя и имя, например local / qwen3.5-4b-ra-smoke
lms ls                                            # ключ импортированной модели
lms unload --all
lms load <ключ из lms ls> --identifier {{MODEL_ID}} --context-length 16384 --gpu max -y
lms load text-embedding-bge-m3                    # без эмбеддингов плотный поиск отключится
```

Id модели для API — `{{MODEL_ID}}` (значение `--identifier`). Базовая модель остаётся `qwen3.5-4b`. На ноутбуке мало памяти, поэтому сначала `lms unload --all`, а потом загружать одну модель.

## 7. Шаг 5 — оценка на ноутбуке

В папке репозитория `research-assistant` на ноутбуке. Флаг `--provider lmstudio` переключает чат-модель на LM Studio без правки `.env`, а `LLM_LMSTUDIO_MODEL` задаёт id модели.

```powershell
$env:LLM_LMSTUDIO_MODEL = "{{MODEL_ID}}"
uv run research-assistant --provider lmstudio eval-agent --variant prefetch      # золото, 30 вопросов
uv run research-assistant --provider lmstudio eval-agent --variant agent         # чистый агент
uv run research-assistant eval-agent --rescore runs\eval-agent-<...>.jsonl       # таблица поведения по журналу
Remove-Item Env:LLM_LMSTUDIO_MODEL                                                # вернуть qwen3.5-4b
```

Базовая модель для сравнения — те же команды с `$env:MODEL = "qwen3.5-4b"`. Ориентиры из `docs/experiments.md` (E5, Qwen3.5-4B, новый промпт):

| вариант | точно по фрагменту | по статье | шагов | токенов |
|---|---|---|---|---|
| `prefetch` | 0.80 | 0.97 | 2.6 | 14.8k |
| `agent` | 0.70 | 0.90 | 3.1 | 11.9k |

После smoke-прогона на {{TRAIN_EXAMPLES}} пилотных примерах **прироста не ожидается**. Цифры около базовых означают, что конвейер не сломал модель. Резкое падение — повод проверить формат: `call_validity`, `<tool_call>` в ответах, предупреждение о `prompt_hash`. Предупреждения о несовпадении `prompt_hash` быть не должно.

Для полного прогона — `--runs 3` и синтетический test (`--file data/finetune/questions/test.jsonl`), проверка забывания (5–6 общих запросов без инструментов). Подробно — в `docs/finetune.md`, «Оценка» и «Проверка забывания».

## Полный прогон (когда будет полный датасет)

1. На ноутбуке: `uv run research-assistant ft-export`, затем пересобрать комплект (см. ниже) и перенести новый zip.
2. На ПК: распаковать в новую папку, выполнить `verify_bundle.py`, повторить `smoke.sh` (быстрая проверка новых данных), затем:

```bash
cd ~/rtx5070_train_bundle
bash train/train_full.sh                              # r=16, α=16, lr 1e-4, 2 эпохи, батч 1×16, --max-len 4096
MAX_LEN=8192 bash train/train_full.sh                 # если пилот показал, что 8192 помещается
LR=5e-5 bash train/train_full.sh; LR=2e-4 bash train/train_full.sh   # перебор lr, у каждого своя папка
RESUME=1 bash train/train_full.sh                     # продолжить после обрыва
```

Каждый прогон пишет в `~/ft/runs/r16-lr<LR>-len<MAX_LEN>/`. Там же `train_summary.json` (лучший чекпоинт по eval loss) и сливает лучший адаптер в `~/ft/merged-<прогон>`. Дальше:

```bash
MERGED=~/ft/merged-r16-lr1e-4-len4096 NAME=qwen3.5-4b-ra bash train/convert_gguf.sh
```

На ноутбуке `--identifier qwen3.5-4b-ra`. Окончательный выбор из 2–3 кандидатов делается по метрикам агента на вопросах val, а не только по eval loss.

## Пересборка комплекта (на ноутбуке)

```powershell
uv run python scripts/make_train_bundle.py
```

Результат — `dist/rtx5070_train_bundle.zip`. Сборка откажется работать (код 2), если `data/finetune/sft/train.jsonl` пуст. Если пуст val, сборка предупредит и запишет режим `smoke` в `MANIFEST.json`.

## Неполадки

**Нехватка видеопамяти (`CUDA out of memory`).** Попробуйте по порядку:

- уменьшить длину (`MAX_LEN=4096`);
- закрыть программы, которые держат GPU (браузер, игры, второй экземпляр LM Studio);
- для полного прогона оставить `BATCH=1` и при желании поменять накопление: `GRAD_ACCUM=8` даёт меньший эффективный батч, на память почти не влияет.

QLoRA (4 бита) для Qwen3.5 Unsloth не рекомендует.

**Слияние убито (`Killed`) или WSL «падает».** Не хватает оперативной памяти WSL: по умолчанию ей доступна половина ОЗУ. Создайте в Windows `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=24GB
swap=16GB
```

Затем выполните `wsl --shutdown` в PowerShell.

**`loss = nan` или `grad_norm` nan/inf.** Убедитесь, что точность `bf16`, а не fp16. Уменьшите lr (`LR=5e-5`). Проверьте, что данные собраны одним `prompt_hash` (скрипт предупреждает). Если NaN появляются на первых шагах, попробуйте `--fp32` для диагностики: это медленно.

**Предупреждение о медленном пути / fallback ядер Gated DeltaNet.** Значит, `unsloth install-kernels` не поставил быстрые ядра (fla, causal-conv1d). Обучение идёт верно, но медленнее. Повторите `unsloth install-kernels` в `~/ft/.venv`. Упаковка (packing) скриптом не используется, так что ничего не ломается.

**GPU не видна в WSL** (`nvidia-smi: command not found`, `torch.cuda.is_available() == False`). Проверьте:

- драйвер NVIDIA R570+ поставлен в Windows (не в WSL);
- выполнено `wsl --update` и `wsl --shutdown` в PowerShell;
- `nvidia-smi` в Ubuntu лежит в `/usr/lib/wsl/lib/`.

Не ставьте Linux-драйвер NVIDIA внутрь WSL.

**`$'\r': command not found` или `bad interpreter`.** У `.sh` переводы строк CRLF. Исправить:

```bash
sed -i 's/\r$//' train/*.sh
```

Затем перепроверьте `python3 train/verify_bundle.py`: хеши изменённых файлов не совпадут, это ожидаемо. Лучше распаковать архив заново, не открывая скрипты в Windows.

**llama.cpp слишком старый для qwen35** («llama.cpp слишком старый», `Model Qwen3_5ForConditionalGeneration is not supported`). Обновите: `UPDATE_LLAMA=1 bash train/convert_gguf.sh`. Если конвертеру не хватает модуля Python, поставьте именно его в `~/ft/.venv`. Полный `requirements-convert_hf_to_gguf.txt` из llama.cpp **не ставьте**: он закрепляет CPU-сборку torch и ломает окружение обучения.

**`--check-gguf`: шаблон ОТЛИЧАЕТСЯ.** Значит, в слитой модели не тот `chat_template.jinja`. Слейте заново через `train_sft.py --merge-from … --merge-out …` (шаблон `train/qwen3_5_chat_template.jinja` подставляется сам) и повторите конвертацию.

**Docker: `could not select device driver "" with capabilities: [[gpu]]`.** В Docker Desktop включите бэкенд WSL2 и проверьте: `docker run --rm --gpus all ubuntu nvidia-smi`. Если в `/workspace/ft` нет прав на запись, выполните в WSL `chmod -R a+rwX ~/ft`.

## Условия использования и приватность

- Данные — траектории учителя Mercury 3 (Inception), полученные через API. По условиям Inception разрешено только **внутреннее** использование в курсовом проекте.
- **Не публикуйте** датасет, адаптеры, слитую модель и GGUF: никаких `push_to_hub`, публичных репозиториев и открытых ссылок.
- В `data/*.jsonl` лежат **тексты научных статей** из библиотеки проекта. Не выкладывайте архив и не оставляйте его в общих папках. После работы его можно удалить с ПК: `rm -rf ~/rtx5070_train_bundle ~/rtx5070_train_bundle.zip`.
- Ключей API, `.env`, базы `library.sqlite`, полных текстов из `data/fulltext`, журналов `runs/` и истории чатов в комплекте нет.
