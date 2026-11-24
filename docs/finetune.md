# Дообучение Qwen3.5-4B: датасет, метрики, обучение

Цель: локальная Qwen3.5-4B (Q4_K_M в LM Studio) должна вести себя как агент на Mercury 3. То есть отвечать сразу, когда ответ уже есть в выдаче, искать словами вопроса, не зацикливаться, честно отвечать `not_found` и отвечать на языке вопроса. Способ: дистилляция траекторий учителя (Mercury 3) с LoRA SFT. Диагностика ошибок — в `docs/experiments.md` (E5).

## Решения

- **Промпт и схемы инструментов замораживаются до сбора данных.** Ученик привязывается к `SYSTEM_PROMPT` и `SCHEMAS`. Хеш `prompt_hash` = sha1(`SYSTEM_PROMPT` + json(`SCHEMAS`, sort_keys)) пишется в каждый пример и в журнал. Если хеш изменится, `eval-agent` предупредит.
- **Режим — без размышлений.** Это LM Studio с `REASONING_EFFORT=none` и `enable_thinking=false`: перед ответом идёт пустой блок `<think>\n\n</think>\n\n`.
- **Шаблон чата берётся из GGUF.** Это `src/research_assistant/finetune/qwen3_5_chat_template.jinja`, извлечённый из `Qwen3.5-4B-Q4_K_M.gguf`: ровно то, что применяет LM Studio. Обучающий текст рендерится им же (jinja2), поэтому он совпадает с инференсом посимвольно.
- **Аргументы вызовов — словари.** Шаблон Qwen3.5 выводит `<parameter=…>` только для `arguments`-словаря. JSON-строку из API перед экспортом разбираем; если это не словарь, пример отбрасывается.
- **Маска потерь считается при экспорте.** Для каждого хода ассистента с `weight=1` записываются символьные границы `[start, end)` в отрендеренном тексте: от конца `<|im_start|>assistant\n<think>\n\n</think>\n\n` до `<|im_end|>` включительно. Скрипт обучения переводит их в токены через `offset_mapping`. Без веса (`weight=0`) остаются:
  - поиск, который код сделал по самому вопросу на шаге 0 (`prefetch`): модель его не генерирует;
  - ходы истории диалога.
- **Траектории с напоминанием (`nudge`) исключаются.** Напоминание — это сообщение `user` внутри эпизода. Оно меняет последний пользовательский запрос, и предыдущие ходы рендерятся без `<think>`.
- **Последнее сообщение инструмента «Answer accepted.» удаляется.** Пример заканчивается вызовом `final_answer`.
- **Сплит по статьям.** В test идут все статьи, на которые ссылается `eval/retrieval_gold.jsonl` (24 статьи), целиком. В val — около 6 других статей (стратификация по теме и языку, `seed=0`), остальное — train.
  - Обучающую траекторию отбрасываем, если она **цитирует** предложение тестовой статьи. В наблюдениях тестовые фрагменты допустимы: поиск идёт по всей библиотеке, как в деплое, а потерь на наблюдениях нет.
  - Почти-дубли (Жаккар по основам `analyze()` ≥ 0.6) с вопросами test и золота удаляются из train и val.
- **Учитель** — модель из `.env` (Mercury 3). Сбор отказывается работать, если адрес чат-модели — localhost (профиль `lmstudio` или `ollama`): это ученик. Нужен bge-m3 в LM Studio. Траектории, в наблюдениях которых есть `[Semantic search is unavailable`, отбрасываются.
- **Обучение** — LoRA в 16 битах (bf16) на RTX 5070 12 ГБ через WSL2 или Docker `unsloth/unsloth`. QLoRA для Qwen3.5 Unsloth не рекомендует. На T4 Qwen3.5 учится только во float32 (в fp16 появляются NaN), примерно в 10 раз медленнее, поэтому Colab T4 — только для пробного запуска. `target_modules` должен включать проекции Gated DeltaNet (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`), а не только q/k/v/o.

- **Условия Inception** (Terms of Use, проверено 2026-11-09). Отдельного запрета на обучение моделей на выходах нет: выходы можно использовать «for any purpose not otherwise restricted». Ограничения такие:
  - нельзя использовать сервис так, что это «directly or indirectly competes… with Inception»;
  - использование только «internal, personal».

  Поэтому дообученная модель и датасет траекторий остаются внутри курсового проекта и не публикуются как замена Mercury. В отчёте указываем, что данные получены через API Mercury 3. В настройках платформы Inception стоит выключить «Improve the model for everyone»: в запросах уходят тексты статей.
- **Цена.** mercury-3 при запуске: $0.10 за 1 млн входных токенов и $0.35 за 1 млн выходных. Весь план — порядка 40 млн токенов (90% входных), около $5. Лимит Free — 1 млн входных токенов в минуту.

## Файлы

Всё с текстами статей лежит в `data/finetune/` (не в git):

```
data/finetune/
  splits.json                 {"test": [paper_id…], "val": […], "train": […], "seed": 0, "gold_file": …}
  questions/<split>.jsonl     вопросы-кандидаты (формат ниже), включая отклонённые с reasons
  trajectories/<split>.jsonl  записи траекторий учителя (формат ниже), только дозапись
  sft/<split>.jsonl           примеры для обучения (формат ниже)
  sft/stats.json              числа по kind, split, языку; гистограмма длины
  rejected.jsonl              отклонённые траектории с причинами
```

### Вопрос (`questions/*.jsonl`)

Формат `GoldQuestion` из `evaluation/retrieval.py` плюс два поля:

```json
{"id": "tr-fact-000123", "question": "…", "language": "ru", "topic": "speculative decoding", "status": "draft",
 "relevant": [{"paper": "<точное название статьи>", "quote": "<дословное предложение или предложения>"}],
 "kind": "fact", "meta": {"split": "train", "paper_id": 17, "passage_id": 30511, "sentence_ids": [94398],
 "source_language": "en", "gold_rank": 3, "reasons": [], "accepted": true, "history": []}}
```

`kind` ∈ `fact` (en→en, ru→en, ru→ru по `language` и `meta.source_language`), `entity`, `compare`, `paper_lookup`, `unanswerable` (`relevant` = []), `followup` (`meta.history` = [{"role": "user"|"assistant", "content": …}] — предыдущий обмен, как в `cmd_chat`), `external`.

### Траектория (`trajectories/*.jsonl`)

```json
{"qid": "…", "sample": 0, "kind": "fact", "model": "mercury-3", "temperature": 0.2, "prompt_hash": "…",
 "parser_version": 7, "messages": […AgentResult.messages…], "tools": […AgentResult.tools…],
 "result": {"stop_reason": "final_answer", "steps": 1, "tool_calls": 1, "tool_errors": 0, "not_found": false,
            "answer": "…", "citations": [94398], "usage": {"prompt": 0, "completion": 0}, "seconds": 4.1,
            "nudges": 0},
 "error": null}
```

### Пример SFT (`sft/*.jsonl`)

```json
{"id": "tr-fact-000123-s0", "split": "train", "kind": "fact",
 "messages": [{"role": "system", "content": "…"}, {"role": "user", "content": "…"},
              {"role": "assistant", "content": "", "tool_calls": [{"id": "…", "type": "function",
                "function": {"name": "search_library", "arguments": {"query": "…", "k": 6}}}], "weight": 0},
              {"role": "tool", "tool_call_id": "…", "content": "…"},
              {"role": "assistant", "content": "", "tool_calls": [… "final_answer" …], "weight": 1}],
 "tools": […],
 "text": "<|im_start|>system\n…",
 "train_spans": [[1234, 1567]],
 "meta": {"qid": "…", "prompt_hash": "…", "est_tokens": 4100, "steps": 1, "teacher": "mercury-3"}}
```

`est_tokens` = ceil(len(text) / 3.4): калибровка по `usage.prompt` в журналах Qwen.

## Метрики поведения (`evaluation/agent.py`, считаются и при `--rescore`)

| метрика | определение |
|---|---|
| `call_validity` | доля вызовов модели (без шага 0) без ошибок формата: невалидный JSON, неизвестный инструмент, неверные аргументы |
| `tool_error_rate` | доля вызовов с ошибкой инструмента (`Error: …`) |
| `redundant_rate` | доля точных повторов вызовов (имя + аргументы без `k`, запрос без учёта регистра), включая повтор поиска шага 0 |
| `first_answer_accept` | первый вызов `final_answer` принят |
| `abstention_precision`, `abstention_recall` | по вопросам `kind=unanswerable`: доля верных `not_found` среди всех `not_found`; доля неотвечаемых вопросов с `not_found` |
| `false_not_found` | `not_found` на вопросе с ответом |
| `lang_match` | язык ответа (`detect_language`) совпадает с `question.language` |
| `unauthorized` | вызов `add_papers` без `wants_to_add(question)` |
| `nudge_rate`, `max_steps_rate`, `budget_exhausted` | доли прогонов с напоминанием, с остановкой по лимиту шагов, с исчерпанным бюджетом поиска |
| `read_passage_per_q`, `parallel_call_rate` | среднее число `read_passage` на вопрос; доля шагов с несколькими вызовами |
| `steps`, `tokens`, `seconds` p50/p95 | время недостоверно, если GPU делят несколько сессий |

Для `kind=unanswerable` успех означает `not_found`. Группа вопроса без `relevant` — `ru→none` или `en→none`.

## Команды

```bash
uv run research-assistant ft-split                       # splits.json, без модели
uv run research-assistant ft-questions --split train --limit 50 --dry-run   # план выборки без модели
uv run research-assistant ft-questions --split train     # синтез вопросов учителем (.env)
uv run research-assistant ft-collect --split train --samples 2   # траектории учителя, возобновляется
uv run research-assistant ft-export                      # sft/{train,val}.jsonl + stats.json
```

Оценка ученика до и после обучения — одним инструментом: `eval-agent --variant prefetch --runs 3` на `eval/retrieval_gold.jsonl` и на тестовых вопросах `data/finetune/questions/test.jsonl`, с моделью из переменных окружения.

## Обучение и возврат в LM Studio

Обучение — вне пакета: torch, unsloth и transformers ставятся только на машине обучения, в `pyproject.toml` их нет. Один скрипт на все пути — `scripts/train_sft.py`:

- токенизирует готовый `text` (`add_special_tokens=False`) и переводит `train_spans` в метки через `offset_mapping`. Токен учится, если начинается внутри отрезка. Токен, который заходит в отрезок слева, остаётся без потерь: его начало — текст шаблона, модель его не генерирует;
- примеры длиннее `--max-len` отбрасывает, а не режет посреди хода. Отрезается только хвост без потерь;
- учит обычным `Trainer` на готовых `input_ids`/`labels`. `SFTTrainer` не используется: он применил бы чат-шаблон повторно;
- печатает декодированные участки с потерями первого примера. Там должны быть только ходы ассистента: `<tool_call>…</tool_call><|im_end|>`.

Ноутбук `notebooks/finetune_qwen35_colab.ipynb` запускает тот же скрипт в Colab.

### Среда на RTX 5070 (основной путь)

RTX 5070 — это sm_120, ей нужны CUDA 12.8+ и torch со сборкой cu128. Нативный Windows для Unsloth не подходит, поэтому WSL2 или Docker. Драйвер NVIDIA ставится только в Windows, внутри WSL он не нужен.

WSL2 (Ubuntu), стек из ноутбука Unsloth для Qwen3.5:

```bash
wsl --install -d Ubuntu-24.04            # в PowerShell, один раз; дальше — в Ubuntu
nvidia-smi                               # видна RTX 5070
curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p ~/ft && cd ~/ft && uv venv -p 3.12 && source .venv/bin/activate
uv pip install "torch==2.8.0" "torchvision==0.23.0" --index-url https://download.pytorch.org/whl/cu128
uv pip install "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git" \
               "unsloth @ git+https://github.com/unslothai/unsloth.git"
uv pip install "transformers==5.2.0" "trl==0.22.2" peft accelerate datasets bitsandbytes
unsloth install-kernels
python -c "import torch; print(torch.cuda.get_device_capability(), torch.cuda.is_bf16_supported())"   # (12, 0) True
REPO=/mnt/c/Users/<user>/Desktop/SFEDU_MASTERS/Urushkin_NLP/research-assistant
```

Скрипт и данные читаются прямо из репозитория на `/mnt/c`. Чекпоинты лучше писать в файловую систему Linux (`~/ft/runs`): так быстрее.

Docker (Docker Desktop с бэкендом WSL2 и поддержкой GPU):

```bash
docker run -d --name unsloth --gpus all -v "$REPO":/workspace/work unsloth/unsloth
docker exec -it unsloth bash
pip install -U "transformers==5.2.0" "trl==0.22.2"   # если в образе версии старее
cd /workspace/work
```

### Пробный запуск в Colab T4

На T4 Qwen3.5 учится только во float32. Скрипт сам выбирает точность (`--bf16`/`--fp32` — вручную) и предупреждает. Поэтому в Colab — только 30 шагов с `--max-len 4096 --limit-train 200`, ячейки 1–5 ноутбука. Меньше 4096 не брать: системный промпт со схемами инструментов сам по себе около 3000 токенов, и при 2048 фильтр длины отбросит все примеры. Если отброшено больше половины, скрипт предупреждает — тогда длину берём не меньше p50 из `--dry-run`. Что проверить в выводе:

- строка `LoRA:` — ненулевые счётчики у всех проекций: `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`, q/k/v/o и MLP. Если проекции не нашлись или LoRA попала в башню зрения, скрипт останавливается;
- loss падает, `nan` нет;
- участки с потерями — только ходы ассистента.

### Пилот: выбор `--max-len`

```bash
python $REPO/scripts/train_sft.py --train $REPO/data/finetune/sft/train.jsonl \
    --val $REPO/data/finetune/sft/val.jsonl --max-len 8192 --dry-run
```

`--dry-run` загружает только токенайзер, GPU не нужен. Он печатает перцентили длин, сколько примеров длиннее 2048/4096/8192/10240, долю обучаемых токенов и поправку к `est_tokens`. Длину выбираем так:

- берём 4096, если она отбрасывает не больше ~5% train и среди отброшенных нет перекоса по `kind` (длинные — обычно `compare` и многошаговые, см. `sft/stats.json`);
- иначе 8192. Тогда — 30 шагов с `--max-steps 30 --max-len 8192` и смотрим память в `nvidia-smi`;
- при нехватке памяти возвращаемся к 4096.

LoRA 4B в bf16 занимает около 10 ГБ.

### Полное обучение

LoRA r = 16, α = 16, dropout 0, lr 1e-4, 2 эпохи, косинус, разогрев 4%, эффективный батч 16 (1 × 16 накоплений), gradient checkpointing `unsloth`, без упаковки (packing):

```bash
python $REPO/scripts/train_sft.py --train $REPO/data/finetune/sft/train.jsonl \
    --val $REPO/data/finetune/sft/val.jsonl --max-len 4096 \
    --epochs 2 --lr 1e-4 --rank 16 --alpha 16 --batch 1 --grad-accum 16 \
    --eval-steps 50 --save-steps 50 --out ~/ft/runs/r16-lr1e-4
# обрыв: та же команда с --resume (последний чекпоинт в --out) или --resume <путь к checkpoint-N>
```

- Перебор lr: 5e-5, 1e-4, 2e-4, каждый в своей `--out`.
- Каждые `--eval-steps` считается eval loss на val. Хранятся три чекпоинта (`save_total_limit=3`), лучший по eval loss среди них не удаляется; в конце в `adapter/` сохраняется лучший по eval loss.
- В `train_summary.json` — параметры, число примеров, `prompt_hash`, лучший чекпоинт и его loss. Если в данных больше одного `prompt_hash`, скрипт предупреждает.
- Eval loss только сужает выбор. Из 2–3 близких кандидатов окончательно выбираем по метрикам агента на вопросах val (`data/finetune/questions/val.jsonl`). Золото и test остаются для итогового сравнения, иначе выбор подгоняется под тест.

### Слияние, GGUF и квантование

```bash
# слить выбранный чекпоинт (или сразу при обучении: --merge-out)
python $REPO/scripts/train_sft.py --merge-from ~/ft/runs/r16-lr1e-4/adapter --merge-out ~/ft/merged
# свежий llama.cpp: qwen35 поддерживается с февраля 2026
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
uv pip install -r ~/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
cmake -S ~/llama.cpp -B ~/llama.cpp/build && cmake --build ~/llama.cpp/build --target llama-quantize -j
python ~/llama.cpp/convert_hf_to_gguf.py ~/ft/merged --outfile ~/ft/qwen3.5-4b-ra-bf16.gguf --outtype bf16
~/llama.cpp/build/bin/llama-quantize ~/ft/qwen3.5-4b-ra-bf16.gguf ~/ft/qwen3.5-4b-ra-Q4_K_M.gguf Q4_K_M
# чат-шаблон в GGUF должен совпасть с шаблоном LM Studio (нужен пакет gguf)
uv pip install gguf
python $REPO/scripts/train_sft.py --check-gguf ~/ft/qwen3.5-4b-ra-Q4_K_M.gguf
cp ~/ft/qwen3.5-4b-ra-Q4_K_M.gguf /mnt/c/Users/<user>/Downloads/
```

- Слияние сохраняет модель в 16 битах (`save_method="merged_16bit"`) и кладёт в неё `qwen3_5_chat_template.jinja`: в `chat_template.jinja` и в `tokenizer_config.json`. Так `convert_hf_to_gguf.py` запишет в GGUF ровно шаблон LM Studio, а не шаблон из HF.
- Контроль `--check-gguf` на исходном `~/.lmstudio/models/local/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf` тоже должен показать совпадение.
- Есть альтернатива — `model.save_pretrained_gguf` из Unsloth, но для VL-чекпоинта с MTP она менее надёжна.
- Для агента нужна только текстовая часть, `mmproj` не нужен.

### LM Studio

```powershell
lms import C:\Users\<user>\Downloads\qwen3.5-4b-ra-Q4_K_M.gguf   # спросит издателя и имя, например local/qwen3.5-4b-ra
lms ls                                                          # ключ импортированной модели
lms unload --all
lms load <ключ> --identifier qwen3.5-4b-ra --context-length 16384 --gpu max -y
lms load text-embedding-bge-m3                                  # без эмбеддингов плотный поиск отключится
```

### Оценка

Модель выбирается профилем провайдера (`src/research_assistant/providers.py`): флаг `--provider lmstudio` переключает на LM Studio без правки `.env`, а `LLM_LMSTUDIO_MODEL` задаёт id модели. Сравниваем три модели на одних командах: базовую `qwen3.5-4b`, дообученную `qwen3.5-4b-ra` и Mercury 3 (потолок учителя, профиль `inception`):

```powershell
$env:LLM_LMSTUDIO_MODEL = "qwen3.5-4b-ra"
uv run research-assistant --provider lmstudio eval-agent --variant prefetch --runs 3                                          # золото, 30 вопросов
uv run research-assistant --provider lmstudio eval-agent --variant prefetch --runs 3 --file data/finetune/questions/test.jsonl # синтетический test
uv run research-assistant eval-agent --rescore runs/eval-agent-<...>.jsonl                                                     # таблица поведения по журналу
Remove-Item Env:LLM_LMSTUDIO_MODEL                                                                                             # вернуть qwen3.5-4b
```

В `questions/test.jsonl` лежат и отклонённые кандидаты (`meta.accepted: false`). Если `eval-agent` их не пропускает сам, сначала отберите принятые в отдельный файл.

Что сравниваем:

- pass@1 по статье и по фрагменту, pass^3 на золоте и на test с доверительными интервалами. На 30 вопросах интервал ±0.15, поэтому вывод делаем по test;
- таблицу поведения (метрики выше): `call_validity`, `redundant_rate`, `first_answer_accept`, `abstention_precision`/`recall` и `false_not_found`, `lang_match`, `nudge_rate`, `max_steps_rate`, `steps` и `tokens` p50/p95;
- разбивку по `kind` и группам языков (en→en, ru→en, ru→ru, *→none).

Предупреждения о несовпадении `prompt_hash` быть не должно. Если оно есть, промпт или схемы изменились после сбора данных.

### Проверка забывания

Агентные метрики не покажут, что модель разучилась говорить вне инструментов. Поэтому 5–6 общих запросов без инструментов задаём базовой и дообученной модели (чат LM Studio, размышления выключены):

- объяснение («Объясни простыми словами, что такое градиентный спуск»);
- письмо («Напиши короткое письмо коллеге о переносе встречи»);
- перевод абзаца на английский;
- арифметика в одну строку;
- один запрос на английском.

Плохие признаки: `<tool_call>` или «sentence ID» в обычном ответе, ответ не на языке вопроса, сломанная грамматика, заметно более короткие ответы. Ответы обеих моделей сохраняем рядом в отчёте.

### Пакет для RTX 5070

`uv run python scripts/make_train_bundle.py` собирает `dist/rtx5070_train_bundle.zip`. Внутри:
- `README_RTX5070.md` — пошаговая инструкция;
- `train/` — скрипт обучения, закреплённые зависимости, `setup_wsl.sh`, `smoke.sh`, `train_full.sh`, `convert_gguf.sh`, `run_docker.sh` и шаблон чата;
- `data/` — `train.jsonl`, `val.jsonl` и `stats.json`;
- `MANIFEST.json` с sha256 и `verify_bundle.py` для проверки целостности.

В пакет не попадают `.env`, ключи, база, полные тексты и журналы. Сборщик сам проверяет архив на секреты. Если val пуст, пакет собирается в режиме smoke: на нём проверяется только конвейер (обучение → слияние → GGUF → LM Studio), без прироста качества. После новой генерации данных пакет пересобирается той же командой.

### Публикация

Условия Inception (см. «Решения») разрешают только внутреннее использование. Поэтому:

- дообученную модель, адаптеры, GGUF и датасет не выкладываем: никаких `push_to_hub`, публичных репозиториев и ссылок с Drive;
- в git — только код и сводные метрики;
- в отчёте указываем, что траектории получены через API Mercury 3.
