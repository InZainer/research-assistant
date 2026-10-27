"""Запись о статье из внешнего источника и ключи для поиска дублей."""

import re
import unicodedata
from dataclasses import dataclass, field

# Новый формат arXiv (2301.12345) и старый (cs.CL/0612001, math.GT/0309136).
_ARXIV_ID = r"(\d{4}\.\d{4,5}|[a-z][a-z\-]*(?:\.[a-z]{2})?/\d{7})"
ARXIV_ID_RE = re.compile(rf"^{_ARXIV_ID}(?:v\d+)?$", re.I)
ARXIV_URL_RE = re.compile(rf"arxiv\.org/(?:abs|pdf)/{_ARXIV_ID}(?:v\d+)?", re.I)
ARXIV_DOI_RE = re.compile(rf"^10\.48550/arxiv\.{_ARXIV_ID}(?:v\d+)?$", re.I)
DOI_PREFIX_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)", re.I)

# Короче этого нормализованные заголовки не используются как ключ: «Introduction» не должен склеивать разные статьи.
MIN_TITLE_KEY_LEN = 20


def normalize_doi(raw: str | None) -> str | None:
    if not raw:
        return None
    doi = DOI_PREFIX_RE.sub("", raw.strip()).lower()
    return doi if doi.startswith("10.") else None


def normalize_arxiv_id(raw: str | None) -> str | None:
    """Достаёт идентификатор arXiv без версии из id, ссылки или DOI вида 10.48550/arXiv.X."""
    if not raw:
        return None
    raw = raw.strip()
    if raw.lower().startswith("arxiv:"):
        raw = raw[6:]
    for pattern in (ARXIV_URL_RE, ARXIV_DOI_RE):
        match = pattern.search(raw)
        if match:
            return match.group(1).lower()
    doi = normalize_doi(raw)
    if doi:
        match = ARXIV_DOI_RE.match(doi)
        return match.group(1).lower() if match else None
    match = ARXIV_ID_RE.match(raw)
    return match.group(1).lower() if match else None


def normalize_title(title: str) -> str:
    text = unicodedata.normalize("NFKC", title).casefold().replace("ё", "е")
    return " ".join(re.sub(r"[\W_]+", " ", text).split())


def author_words(name: str) -> list[str]:
    """Слова имени без инициалов из одной буквы: «А. Барсуков» и «Барсуков А.» дают одно и то же."""
    return [w for w in normalize_title(name).split() if len(w) > 1]


def author_matches(authors: list[str], query: str) -> bool:
    """Все слова запроса — начала слов в имени одного автора: «Dao» и «tri dao» находят «Tri Dao»,
    «Geoffrey Hinton» находит «Geoffrey E. Hinton»."""
    words = author_words(query)
    if not words:
        return True
    for author in authors:
        tokens = normalize_title(author).split()
        if all(any(t.startswith(w) for t in tokens) for w in words):
            return True
    return False


def title_key(title: str) -> str | None:
    norm = normalize_title(title)
    return norm if len(norm) >= MIN_TITLE_KEY_LEN else None


@dataclass
class PaperRecord:
    title: str
    sources: list[str]
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    abstract: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    openalex_id: str | None = None
    venue: str | None = None
    language: str | None = None
    landing_url: str | None = None
    pdf_url: str | None = None

    def __post_init__(self):
        self.title = " ".join(self.title.split())
        self.doi = normalize_doi(self.doi)
        self.arxiv_id = normalize_arxiv_id(self.arxiv_id)
        if self.arxiv_id is None:
            for candidate in (self.doi, self.landing_url, self.pdf_url):
                self.arxiv_id = normalize_arxiv_id(candidate)
                if self.arxiv_id:
                    break

    @property
    def title_key(self) -> str | None:
        return title_key(self.title)

    def keys(self) -> list[tuple[str, str]]:
        """Ключи для поиска дублей, от самого надёжного к наименее надёжному."""
        keys = []
        if self.doi:
            keys.append(("doi", self.doi))
        if self.arxiv_id:
            keys.append(("arxiv", self.arxiv_id))
        if self.openalex_id:
            keys.append(("openalex", self.openalex_id))
        if self.title_key:
            keys.append(("title", self.title_key))
        return keys

    @property
    def external_id(self) -> str:
        """Короткий идентификатор для показа пользователю и агенту."""
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id}"
        if self.doi:
            return f"doi:{self.doi}"
        if self.openalex_id:
            return f"openalex:{self.openalex_id}"
        return f"title:{normalize_title(self.title)[:60]}"
