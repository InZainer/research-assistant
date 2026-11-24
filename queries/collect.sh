#!/usr/bin/env bash
# Сбор тем про общепит. Запуск из корня проекта:
#   bash queries/collect.sh ai          — «ИИ в общепите»
#   bash queries/collect.sh marketing   — «маркетинг и соцсети общепита»
#   bash queries/collect.sh ru          — работы российских авторов по обеим темам (фильтр OpenAlex по стране)
# Разбор отложен (--no-parse): после сбора выполнить `research-assistant parse`, затем `index`.
# Темы можно собирать параллельно (разные процессы), но одну и ту же тему — только одним процессом:
# иначе оба скачают одни и те же полные тексты.
set -u
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
RA="${RA:-uv run research-assistant}"
LIMIT="${LIMIT:-60}"

FOOD='ресторан|ресторатор|общественное питание|общепит|кафе|кофейня|столовая|фастфуд|быстрое питание|доставка еды|предприятие питания|HoReCa|фудтех|restaurant|restaurateur|food service|foodservice|catering|cafe|coffee shop|fast food|food delivery|canteen|dining|foodtech'
AI='искусственный интеллект|ИИ|нейросеть|нейронная сеть|машинное обучение|глубокое обучение|чат-бот|чатбот|большие данные|компьютерное зрение|робот|роботизация|цифровизация|цифровая трансформация|цифровые технологии|автоматизация|прогнозирование|рекомендательная система|инновации|инновационные технологии|artificial intelligence|AI|machine learning|deep learning|neural network|language model|LLM|ChatGPT|generative|chatbot|robot|computer vision|recommender|recommendation|forecasting|prediction|predictive|data mining|big data|sentiment analysis|natural language processing|optimization|digitalization|digital transformation|digital technologies|automation|innovation|smart'
MARKETING='маркетинг|маркетинговый|продвижение|социальные сети|соцсети|SMM|реклама|бренд|лояльность|отзывы|контент|инфлюенсер|блогер|ВКонтакте|Instagram|Telegram|TikTok|потребительское поведение|marketing|social media|social networks|advertising|brand|loyalty|online reviews|reviews|user-generated content|influencer|word of mouth|eWOM|Facebook|Yelp|TripAdvisor|engagement|promotion|consumer behavior'

case "${1:-}" in
  ai)
    $RA ingest "ИИ в общепите" --queries-file queries/ai_foodservice.txt --limit "$LIMIT" \
        --require "$FOOD" --require "$AI" --include-closed --no-parse ;;
  marketing)
    $RA ingest "маркетинг и соцсети общепита" --queries-file queries/restaurant_marketing.txt --limit "$LIMIT" \
        --require "$FOOD" --require "$MARKETING" --include-closed --no-parse ;;
  ru)
    # Российские авторы: их статьи в OpenAlex часто записаны с английскими названием и аннотацией («public
    # catering»), поэтому русским запросом не находятся. Ищем общие слова темы с фильтром по стране.
    $RA ingest "ИИ в общепите" --queries-file queries/russia_foodservice.txt --limit 200 --country RU \
        --sources openalex --require "$FOOD" --require "$AI" --include-closed --no-parse
    $RA ingest "маркетинг и соцсети общепита" --queries-file queries/russia_foodservice.txt --limit 200 --country RU \
        --sources openalex --require "$FOOD" --require "$MARKETING" --include-closed --no-parse ;;
  *)
    echo "Что собирать: ai | marketing | ru"; exit 2 ;;
esac
