"""Сборка агента из настроек: библиотека, поиск, внешние источники, добавление статей."""

from collections.abc import Callable

from research_assistant.agent.tools import ToolContext
from research_assistant.config import Settings
from research_assistant.ingest.download import FulltextFetcher
from research_assistant.ingest.http import HttpClient
from research_assistant.ingest.models import PaperRecord
from research_assistant.ingest.robots import RobotsPolicy
from research_assistant.ingest.sources.arxiv import ArxivSource
from research_assistant.ingest.sources.openalex import OpenAlexSource
from research_assistant.llm import ChatModel
from research_assistant.nlp.annotate import annotate_papers
from research_assistant.parse.pipeline import parse_papers
from research_assistant.retrieval.search import Searcher, build_index
from research_assistant.store.db import Library
from research_assistant.store.vectors import VectorIndex, index_path


def make_searcher(settings: Settings, library: Library, with_llm: bool = False) -> Searcher:
    """with_llm — подключить перевод запроса и реранкер на чат-модели из .env (если она задана)."""
    searcher = Searcher(library)
    if settings.embed_model:
        from research_assistant.nlp.embed import OpenAIEmbedder

        searcher.embedder = OpenAIEmbedder(settings.embed_model, settings.embed_base_url, settings.embed_api_key)
        searcher.index = VectorIndex.load(index_path(settings.vectors_dir, settings.embed_model))
    if with_llm and not settings.missing_llm_vars():
        from research_assistant.retrieval.rerank import LLMReranker, QueryTranslator

        model = ChatModel(settings.model, settings.base_url, settings.api_key, reasoning_effort=settings.reasoning_effort)
        searcher.reranker, searcher.translator = LLMReranker(model), QueryTranslator(model)
    return searcher


def make_add_papers(
    library: Library, searcher: Searcher, fetcher: FulltextFetcher, model: ChatModel | None = None
) -> Callable[[list[PaperRecord], str], str]:
    """Добавление статей из диалога: запись в библиотеку, полный текст, разбор, эмбеддинги, разметка."""

    def add(records: list[PaperRecord], topic: str) -> str:
        topic_id = library.get_or_create_topic(topic)
        paper_ids = []
        for record in records:
            paper_id, _ = library.upsert_paper(record)
            library.link_topic(paper_id, topic_id, "agent")
            paper = library.get_paper(paper_id)
            if paper["fulltext_status"] in ("pending", "failed"):
                fetched = fetcher.fetch(paper)
                library.set_fulltext(paper_id, fetched.status, fetched.path, fetched.kind, fetched.error)
            paper_ids.append(paper_id)
        parse_papers(library, paper_ids=paper_ids)
        if searcher.dense_ready:
            build_index(library, searcher.embedder, searcher.index)
        if model is not None:
            annotate_papers(library, model, paper_ids=paper_ids)
        lines = [f'Added to topic "{topic}":']
        for paper_id in paper_ids:
            paper = library.get_paper(paper_id)
            parse = library.get_parse(paper_id)
            passages = library.conn.execute("SELECT COUNT(*) FROM passages WHERE paper_id = ?", (paper_id,)).fetchone()[0]
            lines.append(f"- paper {paper_id}: {paper['title']} — full text {paper['fulltext_status']}, "
                         f"parse {parse['status'] if parse else '-'}, {passages} passages")
        return "\n".join(lines)

    return add


def make_agent(settings: Settings, library: Library, external: bool = True) -> tuple[ToolContext, ChatModel, HttpClient]:
    searcher = make_searcher(settings, library)
    http = HttpClient(settings.user_agent, cache_dir=settings.http_cache_dir)
    sources = None
    add_fn = None
    if external:
        sources = [
            OpenAlexSource(http, mailto=settings.contact_email, api_key=settings.openalex_api_key),
            ArxivSource(http),
        ]
    model = ChatModel(settings.model, settings.base_url, settings.api_key, reasoning_effort=settings.reasoning_effort)
    if external:
        add_fn = make_add_papers(library, searcher, FulltextFetcher(http, RobotsPolicy(http), settings.fulltext_dir), model)
    ctx = ToolContext(library=library, searcher=searcher, sources=sources, add_papers_fn=add_fn)
    return ctx, model, http
