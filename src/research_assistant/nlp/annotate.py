"""Разметка статей библиотеки: сущности (чат-модель по аннотации и заключению) и ключевые слова (YAKE).

Модель вызывается один раз на статью, а не на каждый фрагмент: найденные сущности затем привязываются
ко всем фрагментам статьи, где они встречаются (store.db._link_mentions). Так фильтр «фрагменты про MT-Bench»
работает по всей статье при 1 вызове модели вместо сотни.
"""

from collections import Counter
from collections.abc import Callable

from research_assistant.nlp.entities import extract_entities
from research_assistant.nlp.keywords import extract_keywords
from research_assistant.store.db import Library

# Увеличивать при изменении промпта или правил: тогда annotate переразметит старые статьи.
ANNOTATOR_VERSION = 2
MAX_INPUT_WORDS = 900
# Что показываем модели и сколько фрагментов каждого раздела: датасеты и метрики чаще названы в экспериментах,
# чем в аннотации, поэтому начало экспериментов и результатов тоже идёт на вход.
INPUT_SECTIONS = (("abstract", 3), ("conclusion", 2), ("experiments", 2), ("results", 2))


def annotation_text(library: Library, paper) -> str:
    """Заголовок, аннотация, заключение и начало экспериментов; без них — начало текста. До MAX_INPUT_WORDS слов."""
    texts = []
    for kind, limit in INPUT_SECTIONS:
        rows = library.conn.execute(
            """SELECT p.text FROM passages p JOIN sections s ON s.id = p.section_id
               WHERE p.paper_id = ? AND s.kind = ? AND p.kind = 'text' ORDER BY p.order_idx LIMIT ?""",
            (paper["id"], kind, limit),
        ).fetchall()
        texts += [r["text"] for r in rows]
    if not texts:
        texts = [r["text"] for r in library.conn.execute(
            "SELECT text FROM passages WHERE paper_id = ? AND kind = 'text' ORDER BY order_idx LIMIT 6", (paper["id"],)
        )]
    words = " ".join([paper["title"], *texts]).split()
    return " ".join(words[:MAX_INPUT_WORDS])


def keyword_text(library: Library, paper) -> str:
    rows = library.conn.execute(
        "SELECT text FROM passages WHERE paper_id = ? AND kind = 'text' ORDER BY order_idx", (paper["id"],)
    ).fetchall()
    # Точка между частями: иначе YAKE склеивает конец заголовка с началом текста в одну фразу.
    return ". ".join([paper["title"], *(r["text"] for r in rows)])


def annotate_paper(library: Library, model, paper) -> dict:
    keywords = extract_keywords(keyword_text(library, paper))
    try:
        extraction = extract_entities(model, annotation_text(library, paper))
    except Exception as exc:  # сбой модели на одной статье не останавливает разметку библиотеки
        library.save_annotation(paper["id"], {}, keywords, status="failed", annotator=model.model,
                                version=ANNOTATOR_VERSION, error=f"{type(exc).__name__}: {exc}")
        return {"status": "failed", "error": str(exc), "entities": 0, "mentions": 0, "dropped": 0}
    mentions = library.save_annotation(
        paper["id"], extraction.entities, keywords, status="ok", annotator=model.model,
        version=ANNOTATOR_VERSION, dropped=len(extraction.dropped),
    )
    return {
        "status": "ok", "entities": sum(len(v) for v in extraction.entities.values()), "mentions": mentions,
        "dropped": len(extraction.dropped), "tokens": extraction.tokens,
    }


def annotate_papers(
    library: Library,
    model,
    topic_id: int | None = None,
    paper_ids: list[int] | None = None,
    force: bool = False,
    log: Callable[[str], None] = lambda message: None,
) -> Counter:
    papers = library.papers_to_annotate(ANNOTATOR_VERSION, topic_id=topic_id, paper_ids=paper_ids, force=force)
    statuses = Counter()
    for number, paper in enumerate(papers, 1):
        outcome = annotate_paper(library, model, paper)
        statuses[outcome["status"]] += 1
        detail = outcome.get("error") or (
            f"сущностей {outcome['entities']} (отброшено {outcome['dropped']}), упоминаний {outcome['mentions']}"
        )
        log(f"[{number}/{len(papers)}] {outcome['status']}: {paper['title'][:60]} — {detail}")
    return statuses
