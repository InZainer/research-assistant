"""Страница Streamlit: вопрос по статьям (с историей чатов), поиск статей, поиск цитат.

Запуск: uv run research-assistant ui (или uv run streamlit run src/research_assistant/ui/app.py).
База открывается заново на каждый перезапуск скрипта: соединение SQLite нельзя делить между потоками Streamlit.
"""

import streamlit as st

from research_assistant.agent.loop import run_agent
from research_assistant.agent.session import make_agent, make_searcher
from research_assistant.agent.tools import SECTION_KINDS
from research_assistant.config import load_settings
from research_assistant.ingest.http import HttpClient
from research_assistant.ingest.models import PaperRecord
from research_assistant.ingest.pipeline import search_sources
from research_assistant.ingest.sources.arxiv import ArxivSource
from research_assistant.ingest.sources.openalex import OpenAlexSource
from research_assistant.retrieval.papers import PaperHit, find_papers
from research_assistant.store.db import Library, PassageFilter
from research_assistant.ui.chats import Chat, ChatStore
from research_assistant.ui.views import (
    ENTITY_LABELS,
    SECTION_LABELS,
    AnswerView,
    answer_view,
    describe_event,
    external_card,
    friendly_error,
    highlight,
    hit_rows,
    link_label,
    md_escape,
    paper_card,
)

# Варианты переключателей — сами подписи, без format_func: так их понимает и streamlit.testing.
CHAT, PAPERS, QUOTES = "💬 Вопрос по статьям", "📄 Статьи", "🔍 Поиск цитат"
LANGUAGES = {"Любой": None, "Английский": "en", "Русский": "ru"}
SEARCH_MODES = {"Смешанный (рекомендуется)": "hybrid", "Точные слова": "bm25", "По смыслу": "dense"}
# Примеры под демонстрационную библиотеку (темы «speculative decoding» и «машинное обучение в медицине»).
EXAMPLES = [
    "Почему ускорение спекулятивного декодирования падает при слабой черновой модели?",
    "Какие датасеты используют для оценки спекулятивного декодирования?",
    "Как искусственный интеллект помогает в диагностике заболеваний?",
    "Какие статьи в библиотеке написал Tri Dao?",
]

st.set_page_config(page_title="Ассистент исследователя", page_icon="📚", layout="wide")
settings = load_settings()
store = ChatStore(settings.data_dir / "chats")
state = st.session_state
state.setdefault("chat_id", None)


def library_summary() -> list[dict]:
    if not settings.db_path.exists():
        return []
    with Library(settings.db_path) as library:
        return library.topic_summary()


def start_new_chat() -> None:
    state.chat_id = None
    state.page = CHAT


def open_chat(chat_id: str) -> None:
    state.chat_id = chat_id
    state.page = CHAT


def delete_chat(chat_id: str) -> None:
    store.delete(chat_id)
    state.chat_id = None


def ask_example(question: str) -> None:
    state.pending_question = question


# --- боковая панель ---

with st.sidebar:
    st.title("📚 Ассистент исследователя")
    st.caption("Отвечает на вопросы по собранным научным статьям и подтверждает ответ дословными цитатами "
               "с номером страницы.")
    page = st.radio("Раздел", [CHAT, PAPERS, QUOTES], key="page", label_visibility="collapsed")

    st.divider()
    st.button("➕ Новый чат", on_click=start_new_chat, width="stretch")
    chats = store.list()
    if chats:
        st.caption("История чатов")
        for saved in chats[:40]:
            current = saved.id == state.chat_id and page == CHAT
            st.button(saved.title, key=f"chat-{saved.id}", on_click=open_chat, args=(saved.id,), width="stretch",
                      type="primary" if current else "tertiary", help=f"Начат {saved.created.replace('T', ' ')}")
    else:
        st.caption("Здесь появятся ваши чаты: каждый диалог сохраняется автоматически.")

    st.divider()
    with st.expander("⚙️ Настройки"):
        st.toggle("Разрешить искать новые статьи в интернете", key="external", value=False,
                  help="Ассистент сможет найти статьи в OpenAlex и arXiv и добавить их в библиотеку, если вы "
                       "попросите. Добавление долгое: скачивание, разбор и разметка каждой статьи.")
        st.caption(f"Модель: {settings.model or 'не задана'}  \nЭмбеддинги: {settings.embed_model or 'нет'}  \n"
                   f"Чаты хранятся в {settings.data_dir / 'chats'}, подробные журналы шагов — в {settings.runs_dir}.")


# --- вопрос по статьям ---

def citation_block(card) -> None:
    with st.container(border=True):
        st.markdown(f"«{md_escape(card.quote)}»")
        st.caption(f"📄 {md_escape(card.where)}")
        if card.link:
            st.link_button(link_label(card.link, card.page), card.link, icon="🔗")
        if card.context:
            with st.expander("Показать абзац целиком"):
                before, sentence, after = card.context
                st.markdown(f"{md_escape(before)}**{md_escape(sentence)}**{md_escape(after)}")


MAX_ADD = 5  # за раз: каждая статья скачивается, разбирается и размечается моделью


def pick_and_add(records: list[PaperRecord], key: str) -> None:
    """Карточки статей из интернета с галочками и кнопка «Добавить выбранные» — решение за пользователем."""
    with Library(settings.db_path) as library:
        cards = [external_card(library, r) for r in records]
    st.markdown(f"**Найдено в интернете: {len(cards)}.** Отметьте статьи, которые хотите добавить (до {MAX_ADD} за раз).")
    for card in cards:
        with st.container(border=True):
            st.markdown(f"**{md_escape(card['title'])}**")
            meta = [card["authors"], str(card["year"] or ""), card["language"], card["venue"]]
            st.caption(" · ".join(md_escape(m) for m in meta if m))
            if card["in_library"]:
                st.markdown(":green-badge[уже в библиотеке]")
            else:
                st.checkbox("Добавить в библиотеку", key=f"{key}-add-{card['id']}")
                if not card["has_pdf"]:
                    st.caption("Ссылки на PDF нет: возможно, в библиотеку попадёт только аннотация.")
            if card["abstract"]:
                with st.expander("Аннотация"):
                    st.markdown(md_escape(card["abstract"]))
            if card["link"]:
                st.link_button("Открыть источник", card["link"], icon="🔗")

    chosen = [r for r in records if state.get(f"{key}-add-{r.external_id}")]
    left, right = st.columns(2)
    topic = left.selectbox("В какую тему добавить", [*[t["name"] for t in library_summary()], "Новая тема…"], key=f"{key}-topic")
    if topic == "Новая тема…":
        topic = right.text_input("Название новой темы", key=f"{key}-new-topic").strip()
    disabled = not chosen or len(chosen) > MAX_ADD or not topic
    if len(chosen) > MAX_ADD:
        st.warning(f"Выбрано {len(chosen)} статей — добавляйте не больше {MAX_ADD} за раз.")
    if st.button(f"Добавить выбранные ({len(chosen)})", disabled=disabled, icon="⬇️", key=f"{key}-add"):
        with st.status(f"Добавляю {len(chosen)} статей в тему «{topic}»…", expanded=True) as status:
            status.write("Скачиваю полные тексты, разбираю на разделы, считаю эмбеддинги и размечаю — "
                         "это может занять несколько минут.")
            try:
                with Library(settings.db_path) as library:
                    ctx, _, http = make_agent(settings, library, external=True)
                    try:
                        report = ctx.add_papers_fn(chosen, topic)
                    finally:
                        http.close()
            except Exception as error:
                status.update(label="Не получилось добавить", state="error")
                st.error(friendly_error(error, settings.base_url, settings.model))
                return
            status.update(label="Готово", state="complete")
            st.code(report, language=None)
        for r in chosen:
            state.pop(f"{key}-add-{r.external_id}", None)
        st.success("Статьи добавлены — их уже видно во вкладке «В библиотеке» раздела «Статьи», и ассистент может их цитировать.")


def show_answer(view: AnswerView, index: int = 0) -> None:
    with st.chat_message("user"):
        st.markdown(md_escape(view.question))
    with st.chat_message("assistant"):
        if not view.verified:
            st.warning("Ассистент не смог подтвердить этот ответ цитатами из статей — относитесь к нему осторожно.")
        st.markdown(md_escape(view.answer))
        if view.not_found and not view.found_papers:
            st.info("В библиотеке не нашлось мест, которые отвечают на этот вопрос. Попробуйте переформулировать "
                    "его или в настройках разрешите искать новые статьи.")
        if view.claims:
            st.markdown("**Откуда это известно**")
        for number, (claim, cards) in enumerate(view.claims, 1):
            st.markdown(f"**{number}.** {md_escape(claim)}")
            for card in cards:
                citation_block(card)
        # Переключатель, а не expander: внутри карточек свои expander-ы, а вкладывать их Streamlit не даёт.
        if view.found_papers and st.toggle(f"Показать найденные в интернете статьи ({len(view.found_papers)})",
                                           value=index == state.get("last_index"), key=f"answer{index}-show"):
            pick_and_add([PaperRecord(**r) for r in view.found_papers], f"answer{index}")
        if view.steps:
            with st.expander(f"Как ассистент искал ответ ({view.stats})"):
                for step in view.steps:
                    st.markdown(f"- {md_escape(step)}")


def welcome() -> None:
    st.markdown("### Задайте вопрос по статьям из библиотеки")
    st.markdown(
        "Ассистент сам найдёт нужные места в статьях и коротко ответит. Каждое утверждение он подтвердит "
        "**дословной цитатой**: видно, из какой статьи, раздела и страницы она взята, а PDF открывается сразу "
        "на этой странице. Можно спрашивать по-русски даже про английские статьи."
    )
    summary = library_summary()
    if summary:
        topics = ", ".join(f"«{t['name']}» ({t['papers']})" for t in summary)
        st.caption(f"Сейчас в библиотеке {sum(t['papers'] for t in summary)} статей по темам: {topics}.")
    st.markdown("**Попробуйте спросить:**")
    for i, example in enumerate(EXAMPLES):
        st.button(example, key=f"example-{i}", on_click=ask_example, args=(example,), icon="💡")
    st.caption("Ответ обычно занимает 20–60 секунд: модель работает локально на вашем компьютере.")


def run_question(chat: Chat, question: str) -> None:
    with st.chat_message("user"):
        st.markdown(md_escape(question))
    with st.chat_message("assistant"):
        status = st.status("Ассистент ищет ответ в статьях…", expanded=True)
        steps: list[str] = []

        def on_event(event: str, data: dict) -> None:
            text = describe_event(event, data)
            if text:
                steps.append(text)
                status.markdown(f"- {md_escape(text)}")

        try:
            with Library(settings.db_path) as library:
                ctx, model, http = make_agent(settings, library, external=state.get("external", False))
                try:
                    # prefetch — как в ask/chat: лучший вариант в эксперименте E3 (docs/experiments.md)
                    result = run_agent(question, ctx, model, max_steps=settings.max_steps, history=chat.history(),
                                       log_path=settings.runs_dir / f"ui-{chat.id}.jsonl", prefetch=True,
                                       on_event=on_event)
                finally:
                    http.close()
                view = answer_view(library, result, steps, found=list(ctx.candidates.values()))
        except Exception as error:  # модель недоступна, сеть и т. п. — объяснить, а не уронить страницу
            status.update(label="Не получилось", state="error", expanded=False)
            st.error(friendly_error(error, settings.base_url, settings.model))
            return
        status.update(label=f"Готово за {view.seconds:.0f} с", state="complete", expanded=False)
    chat.messages.append(view)
    store.save(chat)
    state.chat_id = chat.id
    st.rerun()


if page == CHAT:
    chat = store.load(state.chat_id) if state.chat_id else None
    missing = settings.missing_llm_vars()
    if missing:
        st.error(f"Не настроена модель: в файле .env нет {', '.join(missing)}. Скопируйте .env.example в .env "
                 "и заполните его.")
    # Поле ввода всё равно прижато к низу страницы; читаем его первым, чтобы не показывать приветствие над новым вопросом.
    question = st.chat_input("Спросите что-нибудь о статьях…", disabled=bool(missing))
    question = question or state.pop("pending_question", None)
    if chat is not None and chat.messages:
        st.subheader(chat.title)
        state.last_index = len(chat.messages) - 1  # найденные статьи раскрыты только у последнего ответа
        for index, view in enumerate(chat.messages):
            show_answer(view, index)
        if not question:
            st.button("🗑 Удалить этот чат", on_click=delete_chat, args=(chat.id,), type="tertiary")
    elif not question:
        welcome()
    if question and not missing:
        run_question(chat or store.new(question), question)


# --- статьи ---

@st.dialog("Статья", width="large")
def paper_dialog(paper_id: int) -> None:
    with Library(settings.db_path) as library:
        paper = library.get_paper(paper_id)
        card = paper_card(library, PaperHit(dict(paper), 0.0))
        st.markdown(f"#### {md_escape(card['title'])}")
        st.caption(" · ".join(filter(None, [", ".join(card["all_authors"]), str(card["year"] or ""),
                                            card["venue"], card["language"]])))
        if card["link"]:
            st.link_button("Открыть источник", card["link"])
        if card["topics"]:
            st.markdown("**Темы:** " + md_escape(", ".join(card["topics"])))
        for entity_type, names in library.paper_entities(paper_id).items():
            st.markdown(f"**{ENTITY_LABELS.get(entity_type, entity_type).capitalize()}:** {md_escape('; '.join(names))}")
        keywords = library.paper_keywords(paper_id)
        if keywords:
            st.markdown("**Ключевые слова:** " + md_escape("; ".join(keywords)))
        if card["abstract"]:
            st.markdown("**Аннотация**")
            st.markdown(md_escape(card["abstract"]))
        structure = library.paper_structure(paper_id)
        if structure:
            st.markdown("**Текст по разделам**")
        else:
            st.info("Полного текста нет: доступны только сведения из каталога.")
        for section in structure:
            title = section["title"] or SECTION_LABELS.get(section["kind"], "Без заголовка")
            with st.expander(title):
                for passage in section["passages"]:
                    if passage["page"]:
                        st.caption(f"с. {passage['page']}")
                    st.markdown(md_escape(passage["text"]))


def external_search(query: str, author: str, year_range, language: str | None, topics: list[str]) -> None:
    """Поиск в OpenAlex и arXiv по тем же полям и добавление выбранных статей в библиотеку.
    year_range — (None, None), если пользователь не трогал ползунок: годы библиотеки не должны отсекать классику."""
    st.caption("Ищет в OpenAlex и arXiv по полям «Автор» и «Слова» из формы выше. Найденное можно добавить в "
               "библиотеку: статья скачается, разберётся на разделы и будет доступна ассистенту.")
    if not (query.strip() or author.strip()):
        st.info("Заполните поле «Автор» или «Слова» в форме выше, например автор Geoffrey Hinton.")
        return
    if st.button("Искать в OpenAlex и arXiv", type="primary", icon="🌐"):
        with st.spinner("Ищу в OpenAlex и arXiv… (arXiv иногда отвечает медленно)"):
            http = HttpClient(settings.user_agent, cache_dir=settings.http_cache_dir)
            try:
                sources = [OpenAlexSource(http, mailto=settings.contact_email, api_key=settings.openalex_api_key),
                           ArxivSource(http)]
                found = search_sources(sources, query.strip(), limit=15, year_from=year_range[0],
                                       year_to=year_range[1], language=language, author=author.strip() or None)
            finally:
                http.close()
        state.web_results = {"records": found.records, "errors": found.errors}
    found = state.get("web_results")
    if not found:
        return
    for name, error in found["errors"].items():
        st.warning(f"Источник {name} не ответил ({error.split(':')[0]}), показываю результаты остальных.")
    records = found["records"]
    if not records:
        st.info("Ничего не нашлось. Проверьте написание имени (латиницей, как в статьях) или уберите часть слов.")
        return
    pick_and_add(records, "web")


if page == PAPERS:
    st.header("📄 Статьи")
    st.caption("Найдите статью по автору, по словам из названия или аннотации. Слова можно писать в любой форме: "
               "«декодирования» найдёт «декодирование».")
    summary = library_summary()
    if not summary:
        st.info('Библиотека пуста. Соберите статьи командой: uv run research-assistant ingest "<тема>"')
    else:
        cols = st.columns(4)
        cols[0].metric("Статей", sum(t["papers"] for t in summary))
        cols[1].metric("Тем", len(summary))
        cols[2].metric("С полным текстом", sum(t["fulltext"] for t in summary))
        cols[3].metric("На русском", sum(t["russian"] for t in summary))

        with Library(settings.db_path) as library:
            years = [r[0] for r in library.conn.execute("SELECT DISTINCT year FROM papers WHERE year IS NOT NULL")]
        with st.form("papers"):
            left, right = st.columns([3, 2])
            query = left.text_input("Слова из названия или аннотации", placeholder="например: черновая модель, diagnosis")
            author = right.text_input("Автор", placeholder="например: Tri Dao или Барсуков")
            left, middle, right = st.columns(3)
            topic = left.selectbox("Тема", ["Все темы", *[t["name"] for t in summary]])
            language = middle.selectbox("Язык статьи", list(LANGUAGES))
            year_range = (right.slider("Годы", min(years), max(years), (min(years), max(years)))
                          if years and min(years) < max(years) else (None, None))
            semantic = st.checkbox("Добавить похожие по смыслу, в том числе на другом языке", value=True)
            st.form_submit_button("Найти", type="primary")

        with Library(settings.db_path) as library:
            searcher = make_searcher(settings, library) if semantic and query.strip() else None
            try:
                hits = find_papers(library, query, author, None if topic == "Все темы" else topic,
                                   year_range[0], year_range[1], LANGUAGES[language], searcher=searcher, limit=60)
            except Exception as error:  # сервер эмбеддингов недоступен — ищем без смысла
                st.warning(f"Поиск по смыслу недоступен ({type(error).__name__}), показываю совпадения по словам.")
                hits = find_papers(library, query, author, None if topic == "Все темы" else topic,
                                   year_range[0], year_range[1], LANGUAGES[language], limit=60)
            cards = [paper_card(library, hit) for hit in hits]

        searching = bool(query.strip() or author.strip())
        local_tab, web_tab = st.tabs([f"📚 В библиотеке ({len(cards)})", "🌐 Новые статьи в интернете"])
        with local_tab:
            st.markdown(f"**Найдено статей: {len(cards)}**" if searching else f"**Все статьи ({len(cards)})**")
            if searching and not cards:
                st.info("Ничего не нашлось. Попробуйте написать фамилию автора латиницей или кириллицей, убрать часть "
                        "слов или выбрать «Все темы».")
            for card in cards:
                with st.container(border=True):
                    st.markdown(f"**{highlight(card['title'], query)}**")
                    meta = [card["authors"], str(card["year"] or "год не указан"), card["language"], card["venue"]]
                    st.caption(" · ".join(md_escape(m) for m in meta if m))
                    badges = [":green-badge[полный текст]" if card["fulltext"] else ":gray-badge[только аннотация]"]
                    badges += [f":blue-badge[{m}]" for m in card["matched"]]
                    st.markdown(" ".join(badges))
                    if card["abstract"]:
                        with st.expander("Аннотация"):
                            st.markdown(highlight(card["abstract"], query))
                    with st.container(horizontal=True):
                        if st.button("Подробнее", key=f"paper-{card['id']}", icon="📖"):
                            paper_dialog(card["id"])
                        if card["link"]:
                            st.link_button("Открыть источник", card["link"], icon="🔗")
        with web_tab:
            web_years = (None, None) if not years or year_range == (min(years), max(years)) else year_range
            external_search(query, author, web_years, LANGUAGES[language], [t["name"] for t in summary])

        with st.expander("Какие методы, датасеты и метрики встречаются в статьях"):
            label = st.radio("Тип", list(ENTITY_LABELS.values()), horizontal=True, label_visibility="collapsed")
            kind = next(k for k, v in ENTITY_LABELS.items() if v == label)
            with Library(settings.db_path) as library:
                stats = library.entity_stats(kind, topic=None if topic == "Все темы" else topic, limit=30)
            if stats:
                st.dataframe([{"название": s["name"], "статей": s["papers"], "упоминаний в тексте": s["mentions"]}
                              for s in stats], hide_index=True, width="stretch")
            else:
                st.caption("Статьи ещё не размечены: uv run research-assistant annotate")


# --- поиск цитат ---

if page == QUOTES:
    st.header("🔍 Поиск цитат")
    st.caption("Находит конкретные абзацы в текстах статей — когда нужно быстро увидеть, где говорится о чём-то, "
               "без ответа ассистента.")
    topics = [t["name"] for t in library_summary()]
    with st.form("quotes"):
        query = st.text_input("Что найти", placeholder="например: почему падает ускорение при слабой черновой модели")
        with st.expander("Уточнить поиск"):
            left, middle, right = st.columns(3)
            topic = left.selectbox("Тема", ["Все темы", *topics])
            language = middle.selectbox("Язык статьи", list(LANGUAGES))
            k = right.slider("Сколько абзацев показать", 3, 30, 10)
            left, right = st.columns(2)
            section_labels = left.multiselect("Только разделы", [SECTION_LABELS[k] for k in SECTION_KINDS],
                                              placeholder="любые")
            sections = [k for k in SECTION_KINDS if SECTION_LABELS[k] in section_labels]
            entity = right.text_input("Упоминается метод, датасет или метрика", placeholder="например: MT-Bench")
            mode_label = st.radio("Как искать", list(SEARCH_MODES), horizontal=True,
                                  help="Смешанный объединяет поиск по словам и по смыслу. «По смыслу» находит "
                                       "абзацы на другом языке.")
            translate = st.checkbox("Также искать по английскому переводу запроса (переводит модель, +5–10 с)")
        submitted = st.form_submit_button("Найти", type="primary")

    if submitted and query.strip():
        where = PassageFilter(
            topic=None if topic == "Все темы" else topic,
            section_kinds=sections or None,
            language=LANGUAGES[language],
            entity=entity.strip() or None,
        )
        mode = SEARCH_MODES[mode_label]
        with Library(settings.db_path) as library, st.spinner("Ищу…"):
            searcher = make_searcher(settings, library, with_llm=translate)
            if mode != "bm25" and not searcher.dense_ready:
                st.info("Поиск по смыслу не настроен (нет эмбеддингов) — ищу по словам.")
                mode = "bm25"
            try:
                hits = searcher.search(query, k=k, mode=mode, where=where,
                                       translate=translate and searcher.translator is not None)
                rows = hit_rows(library, hits)
            except Exception as error:
                st.error(friendly_error(error, settings.embed_base_url or settings.base_url, settings.embed_model))
                rows = []
        if not rows:
            st.info("Ничего не нашлось. Попробуйте другие слова или уберите фильтры.")
        for row in rows:
            with st.container(border=True):
                st.markdown(f"**{md_escape(row['title'])}** ({row['year'] or '—'})")
                where_text = [row["section"], f"с. {row['page']}" if row["page"] else "из аннотации"]
                st.caption(" · ".join(md_escape(w) for w in where_text if w))
                st.markdown(highlight(row["text"], query))
                if row["link"]:
                    st.link_button(link_label(row["link"], row["page"]), row["link"], icon="🔗")
    elif submitted:
        st.info("Введите, что найти.")
