"""Нарезка абзацев на фрагменты для поиска и предложения для цитат."""

from bisect import bisect_right
from dataclasses import dataclass, field

from research_assistant.parse.sentences import split_sentences
from research_assistant.parse.structure import Document, Paragraph

# Длинный абзац режется по границам предложений: фрагменты по 100–250 слов удобны и для BM25, и для эмбеддингов.
MAX_PASSAGE_WORDS = 220


@dataclass
class SentenceOut:
    start: int  # смещение в тексте фрагмента
    end: int
    page: int | None


@dataclass
class PassageOut:
    text: str
    page: int | None
    kind: str
    sentences: list[SentenceOut]


@dataclass
class SectionOut:
    title: str
    kind: str
    level: int
    passages: list[PassageOut] = field(default_factory=list)


def page_at(pages: list[tuple[int, int | None]], offset: int) -> int | None:
    index = bisect_right([start for start, _ in pages], offset) - 1
    return pages[max(index, 0)][1]


def paragraph_passages(paragraph: Paragraph, max_words: int = MAX_PASSAGE_WORDS) -> list[PassageOut]:
    text = paragraph.text
    spans = split_sentences(text) or [(0, len(text))]
    chunks: list[list[tuple[int, int]]] = [[]]
    words = 0
    for start, end in spans:
        n = len(text[start:end].split())
        if chunks[-1] and words + n > max_words:
            chunks.append([])
            words = 0
        chunks[-1].append((start, end))
        words += n

    passages = []
    for chunk in chunks:
        base, stop = chunk[0][0], chunk[-1][1]
        sentences = [SentenceOut(s - base, e - base, page_at(paragraph.pages, s)) for s, e in chunk]
        passages.append(PassageOut(text[base:stop], sentences[0].page, paragraph.kind, sentences))
    return passages


def document_passages(document: Document) -> list[SectionOut]:
    out = []
    for section in document.sections:
        result = SectionOut(section.title, section.kind, section.level)
        for paragraph in section.paragraphs:
            result.passages.extend(paragraph_passages(paragraph))
        out.append(result)
    return out
