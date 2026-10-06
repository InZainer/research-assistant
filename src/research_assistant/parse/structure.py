"""Сборка документа из строк: заголовки, разделы, абзацы со страницами.

Правила подобраны по реальным статьям из библиотеки: LaTeX-препринты arXiv, двухколоночные статьи ACL,
статьи ICLR (заголовки капсом без жирного), русские сборники конференций (статья начинается посреди
страницы после чужой) и страницы CyberLeninka.
"""

import re
from dataclasses import dataclass, field

from research_assistant.ingest.models import normalize_title
from research_assistant.parse.units import Unit

MAX_HEADING_WORDS = 15
HEADING_SIZE_RATIO = 1.15
MIN_PARAGRAPH_WORDS = 6
MIN_BLOCK_LETTERS = 0.55  # меньше — формула или обрывок рисунка
FRONT_MATTER_PAGES = 2
FRONT_MATTER_UNITS = 60  # для HTML, где страниц нет
TITLE_LOOKAHEAD = 40  # сколько строк после заголовка статьи искать аннотацию или введение
FOOTNOTE_ZONE = 0.5  # сноски — в нижней половине страницы
MIN_BODY_WORDS = 150  # после стольких слов основного текста новая аннотация — уже следующая статья сборника

NUMBERING_RE = re.compile(r"^(?:(\d+(?:\.\d+)*)\.?|([A-H](?:\.\d+)*)\.?|([IVX]+)\.)\s+(?=\S)")
NUMBERING_ONLY_RE = re.compile(r"^(?:\d+(?:\.\d+)*\.?|[A-H](?:\.\d+)*\.?|[IVX]+\.?)$")
DECIMAL_NUMBERING_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s")
LETTER_NUMBERED_RE = re.compile(r"^[A-H](?:\.\d+)*\.?\s")
CAPTION_RE = re.compile(r"^(?:figure|fig\.|table|tab\.|algorithm|рис\.|рисунок|таблица|табл\.)\s*[A-Z]?\d+\s*[.:|]", re.I)
SKIP_BLOCK_RE = re.compile(r"^(?:keywords|key words|index terms|ключевые слова)\b", re.I)
FOOTNOTE_RE = re.compile(r"^(?:\d{1,2}[A-ZА-Я]|[∗*†‡§¶])")
EMAIL_RE = re.compile(r"@|email\s+protected", re.I)
ABSTRACT_RUNIN_RE = re.compile(r"^(?:abstract|аннотация|резюме)\s*[.:—–-]", re.I)
SENTENCE_END_RE = re.compile(r"[.!?…][\"'»”’)\]]*\d{0,2}$")
WORD_RE = re.compile(r"\w+(?:-\w+)*")
BULLETS = "•▪◦●○■-–—"
CODE_CHARS = set("=;{}<>")

KNOWN_HEADINGS = {
    "abstract", "introduction", "related work", "background", "method", "methods", "methodology",
    "approach", "experiments", "experimental setup", "evaluation", "results", "discussion",
    "conclusion", "conclusions", "limitations", "references", "bibliography", "acknowledgements",
    "acknowledgments", "appendix",
    "аннотация", "резюме", "введение", "обзор литературы", "методы", "методология", "материалы и методы",
    "эксперименты", "результаты", "обсуждение", "результаты и обсуждение", "заключение", "выводы",
    "литература", "список литературы", "библиографический список", "список использованных источников",
    "благодарности", "приложение",
}

# Порядок важен: первое совпадение определяет тип раздела.
SECTION_KINDS = [
    ("abstract", r"^(abstract|аннотация|резюме)$"),
    ("references", r"^(references|bibliography|literature cited|литература|список литературы"
                   r"|список использованных источников|библиографический список|список источников)$"),
    ("acknowledgements", r"^(acknowledge?ments?|благодарност)"),
    ("appendix", r"^(appendix|appendices|supplementary|приложени)"),
    ("introduction", r"(introduction|введение)"),
    ("related_work", r"(related work|background|prior work|literature review|обзор литературы|обзор)"),
    ("limitations", r"(limitation|ограничени)"),
    ("conclusion", r"(conclusion|concluding|future work|заключение|выводы)"),
    ("results", r"(result|результат)"),
    ("experiments", r"(experiment|evaluation|setup|эксперимент)"),
    ("discussion", r"(discussion|analysis|обсуждение|анализ)"),
    ("method", r"(method|approach|model|framework|algorithm|метод|подход|модел|алгоритм)"),
]


@dataclass
class Paragraph:
    text: str
    pages: list[tuple[int, int | None]]  # (смещение в text, номер страницы с этого места)
    kind: str = "text"  # text | caption


@dataclass
class Section:
    title: str
    kind: str
    level: int
    paragraphs: list[Paragraph] = field(default_factory=list)
    numbered: bool = False


@dataclass
class Document:
    sections: list[Section]
    has_abstract: bool


def heading_name(text: str) -> str:
    """Заголовок без нумерации и финального двоеточия/точки, в нижнем регистре."""
    text = NUMBERING_RE.sub("", text.strip())
    return " ".join(text.casefold().replace("ё", "е").rstrip(":.").split())


def is_known_heading(text: str) -> bool:
    return any(part.strip() in KNOWN_HEADINGS for part in heading_name(text).split("/"))


def section_kind(title: str) -> str:
    names = [part.strip() for part in heading_name(title).split("/")]
    for kind, pattern in SECTION_KINDS:
        if any(re.search(pattern, name) for name in names):
            return kind
    return "other"


def heading_level(title: str) -> int:
    match = NUMBERING_RE.match(title.strip())
    if not match:
        return 1 if is_known_heading(title) else 2
    number = match.group(1) or match.group(2) or match.group(3)
    return number.count(".") + 1


def letters_ratio(text: str) -> float:
    visible = [c for c in text if not c.isspace()]
    return sum(c.isalpha() for c in visible) / len(visible) if visible else 0.0


def is_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 3 and sum(c.isupper() for c in letters) >= 0.8 * len(letters)


def heading_candidate(unit: Unit) -> bool:
    """Может ли строка быть заголовком или его частью (например, номер «1» отдельной строкой)."""
    text = unit.text.strip()
    styled = unit.bold or unit.size >= HEADING_SIZE_RATIO
    if NUMBERING_ONLY_RE.match(text):
        return styled or unit.page is not None  # номер раздела отдельной строкой
    words = text.split()
    if not words or len(words) > MAX_HEADING_WORDS or letters_ratio(text) < 0.5:
        return False
    if text[0].islower() or text[0] in BULLETS or text.endswith((",", ";", "-")) or CODE_CHARS & set(text):
        return False
    known = is_known_heading(text)
    if text.endswith(".") and not known:
        return False
    if CAPTION_RE.match(text) or SKIP_BLOCK_RE.match(text):
        return False
    caps = is_caps(text) and len(words) >= 2
    return styled or known or caps


def ends_sentence(text: str) -> bool:
    return bool(SENTENCE_END_RE.search(text.rstrip()))


def continues(previous: str, text: str) -> bool:
    """Продолжает ли новый блок незаконченный абзац (перенос между колонками, страницами, формулами)."""
    if previous.rstrip().endswith((":", "-", "‐")):
        return True
    if ends_sentence(previous):
        return False
    first = text.lstrip()[:1]
    return bool(first) and not first.isupper() and first not in BULLETS


def join_lines(left: str, right: str, vocabulary: set[str]) -> str:
    """Склеивает строки. Перенос «decod-» + «ing» снимается, если слово без дефиса встречается
    в документе или дефисный вариант не встречается; «draft-» + «then-verify» остаётся с дефисом."""
    hyphen = re.search(r"(\w+)[-‐]$", left)
    tail = re.match(r"\w+(?:-\w+)*", right)  # составное слово целиком: «then-verify»
    if hyphen and tail and right[:1].islower():
        head, rest = hyphen.group(1).casefold(), tail.group(0).casefold()
        if f"{head}-{rest}" in vocabulary and f"{head}{rest}" not in vocabulary:
            return left + right
        return left[:-1] + right
    if hyphen and tail:
        return left + right  # «Llama-» + «2», «GPT-» + «4»
    return f"{left} {right}"


class _Builder:
    def __init__(self, vocabulary: set[str]):
        self.vocabulary = vocabulary
        self.sections: list[Section] = []
        self.current: Paragraph | None = None  # последний текстовый абзац, который можно продолжить
        self.last_block: int | None = None
        self.runin_abstract = False  # раздел открыт абзацем «Аннотация: …», а не заголовком

    def section(self, title: str, kind: str, level: int, numbered: bool = False) -> None:
        if kind == "other" and level > 1:
            parent = next((s for s in reversed(self.sections) if s.level < level), None)
            if parent is not None and parent.numbered:
                kind = parent.kind  # «4.2 Speedup» внутри «4 Experiments»
        self.sections.append(Section(title=title, kind=kind, level=level, numbered=numbered))
        self.current = None
        self.last_block = None
        self.runin_abstract = False

    def has_body(self) -> bool:
        """Был ли уже основной текст статьи (а не строка-другая об авторах между аннотациями)."""
        words = sum(len(p.text.split()) for s in self.sections if s.kind != "abstract" for p in s.paragraphs)
        return words >= MIN_BODY_WORDS

    def _target(self) -> Section:
        if not self.sections:
            self.sections.append(Section(title="", kind="body", level=1))
        return self.sections[-1]

    def add(self, unit: Unit, caption: bool = False) -> None:
        section = self._target()
        same_block = unit.block == self.last_block
        if caption:
            section.paragraphs.append(Paragraph(unit.text, [(0, unit.page)], kind="caption"))
        elif self.current is not None and (same_block or continues(self.current.text, unit.text)):
            paragraph = self.current
            joined = join_lines(paragraph.text, unit.text, self.vocabulary)
            offset = len(joined) - len(unit.text)
            paragraph.text = joined
            if paragraph.pages[-1][1] != unit.page:
                paragraph.pages.append((offset, unit.page))
        else:
            if self.runin_abstract and section.paragraphs:
                # аннотация-абзац закончилась, дальше основной текст
                self.section("", "body", 1)
                section = self.sections[-1]
            self.current = Paragraph(unit.text, [(0, unit.page)])
            section.paragraphs.append(self.current)
        self.last_block = unit.block

    def extend_caption(self, unit: Unit) -> None:
        caption = self._target().paragraphs[-1]
        caption.text = join_lines(caption.text, unit.text, self.vocabulary)


def _is_paper_title(name: str, title_key: str | None) -> bool:
    """Строка — заголовок статьи целиком, его часть (перенос на две строки) или он вместе с аффилиацией."""
    if not title_key or not name:
        return False
    if name == title_key or (len(name.split()) >= 3 and name in title_key):
        return True
    return len(title_key.split()) >= 4 and title_key in name


def _group_headings(units: list[Unit], title_key: str | None) -> list[tuple[str, int, int]]:
    """Находит заголовки: (текст, индекс первой строки, индекс после последней).

    Заголовок самой статьи возвращается с пустым текстом: он отмечает начало статьи, но разделом не становится.
    """
    groups = []
    i = 0
    while i < len(units):
        if not heading_candidate(units[i]):
            i += 1
            continue
        j = i + 1
        html = units[i].page is None
        while j < len(units) and heading_candidate(units[j]) and (html or units[j].block == units[i].block):
            j += 1
        text = " ".join(u.text for u in units[i:j])
        words = len(text.split())
        # Жирная строка, за которой в том же блоке идёт ещё жирный текст, — начало жирного абзаца.
        bold_paragraph = (
            not html and j < len(units) and units[j].block == units[i].block
            and (units[j].bold or units[j].size >= HEADING_SIZE_RATIO)
            and not is_known_heading(text)
        )
        alone_number = NUMBERING_ONLY_RE.match(text)
        if _is_paper_title(normalize_title(text), title_key):
            groups.append(("", i, j))  # заголовок статьи, даже слипшийся с аффилиацией
        elif not bold_paragraph and not alone_number and words <= MAX_HEADING_WORDS and letters_ratio(text) >= 0.5:
            groups.append((text, i, j))
        i = j
    return groups


def _front_matter_end(units: list[Unit], headings: dict[int, tuple[str, int]]) -> int:
    """Индекс, с которого начинается статья.

    Если в начале есть заголовок статьи — ищем аннотацию или введение сразу после него (так статья
    находится внутри сборника). Иначе — аннотацию, а без неё введение на первых страницах.
    """
    def early(index: int) -> bool:
        page = units[index].page
        return index < FRONT_MATTER_UNITS if page is None else page <= FRONT_MATTER_PAGES

    def opening(index: int) -> bool:
        if index in headings and headings[index][0]:
            return section_kind(headings[index][0]) in ("abstract", "introduction")
        return bool(ABSTRACT_RUNIN_RE.match(units[index].text))

    # Заголовок статьи может встретиться и в оглавлении выпуска: берём то вхождение,
    # за которым действительно идёт аннотация или введение.
    title_positions = [i for i, (text, _) in headings.items() if not text]
    for title_at in title_positions:
        window = range(title_at, min(len(units), title_at + TITLE_LOOKAHEAD))
        found = next((i for i in window if opening(i)), None)
        if found is not None:
            return found
    if title_positions:
        return title_positions[-1]

    for index, unit in enumerate(units):
        if not early(index):
            break
        if (index in headings and section_kind(headings[index][0]) == "abstract") or ABSTRACT_RUNIN_RE.match(unit.text):
            return index
    for index in headings:
        if early(index) and section_kind(headings[index][0]) == "introduction":
            return index
    return 0


def _junk_blocks(units: list[Unit], heading_lines: set[int]) -> set[int]:
    """Блоки-формулы, обрывки рисунков и сноски внизу страницы."""
    blocks: dict[int, list[Unit]] = {}
    for index, unit in enumerate(units):
        if index not in heading_lines:
            blocks.setdefault(unit.block, []).append(unit)
    junk = set()
    for block, members in blocks.items():
        text = " ".join(u.text for u in members)
        first = members[0]
        if letters_ratio(text) < MIN_BLOCK_LETTERS:
            junk.add(block)
        elif len(text.split()) < 4 and not ends_sentence(text) and not CAPTION_RE.match(text):
            junk.add(block)
        elif first.y is not None and first.y > FOOTNOTE_ZONE and FOOTNOTE_RE.match(first.text):
            junk.add(block)
    return junk


def build_document(units: list[Unit], title: str | None = None) -> Document:
    title_key = normalize_title(title) if title else None
    vocabulary = {w for u in units for w in WORD_RE.findall(u.text.casefold())}
    groups = _group_headings(units, title_key)
    headings = {start: (text, end) for text, start, end in groups}
    heading_lines = {i for _, start, end in groups for i in range(start, end)}
    junk = _junk_blocks(units, heading_lines)
    start = _front_matter_end(units, headings)

    builder = _Builder(vocabulary)
    has_abstract = False
    skipping = False  # внутри списка литературы
    after_references = False  # всё после литературы — приложения
    skip_block: int | None = None
    caption_block: int | None = None
    index = start
    while index < len(units):
        unit = units[index]
        if index in headings:
            text, end = headings[index]
            index = end
            if not text:
                continue  # заголовок статьи
            kind = section_kind(text)
            if kind == "references":
                skipping = after_references = True
                continue
            if skipping:
                # Буквенная нумерация («A Implementation details») — приложение только после литературы:
                # до неё «A Simple Baseline» — обычный заголовок.
                if kind != "appendix" and not LETTER_NUMBERED_RE.match(text):
                    continue
                kind = "appendix"
            if kind == "abstract" and builder.has_body():
                break  # вторая аннотация после основного текста — началась следующая статья сборника
            skipping = False
            if after_references:
                kind = "appendix"
            has_abstract = has_abstract or kind == "abstract"
            numbered = bool(DECIMAL_NUMBERING_RE.match(text) or LETTER_NUMBERED_RE.match(text) and after_references)
            level = heading_level(text)
            if not numbered and not is_known_heading(text):
                # безымянный подзаголовок вкладывается в последний нумерованный раздел
                parent = next((s for s in reversed(builder.sections) if s.numbered), None)
                level = parent.level + 1 if parent else level
            builder.section(text, kind, level, numbered=numbered)
            continue
        index += 1
        if skipping or unit.block in junk or unit.block == skip_block:
            continue
        if SKIP_BLOCK_RE.match(unit.text):
            skip_block = unit.block  # ключевые слова до конца блока
            if builder.sections and builder.sections[-1].kind == "abstract":
                builder.section("", "body", 1)  # ключевые слова закрывают аннотацию
            continue
        if EMAIL_RE.search(unit.text):
            continue
        if ABSTRACT_RUNIN_RE.match(unit.text) and builder.last_block != unit.block:
            if builder.has_body():
                break
            builder.section("Abstract", "abstract", 1)
            builder.runin_abstract = True
            has_abstract = True
        if unit.block == caption_block:
            builder.extend_caption(unit)
            continue
        if CAPTION_RE.match(unit.text) and unit.block != builder.last_block:
            caption_block = unit.block
            builder.add(unit, caption=True)
            continue
        builder.add(unit)

    for section in builder.sections:
        section.paragraphs = [
            p for p in section.paragraphs
            if p.kind == "caption" or len(p.text.split()) >= MIN_PARAGRAPH_WORDS or ends_sentence(p.text)
        ]
    # Пустой раздел оставляем, если за ним есть текст: «4 Experiments» → «4.1 Settings» → абзацы.
    sections = list(builder.sections)
    while sections and not sections[-1].paragraphs:
        sections.pop()
    sections = [s for i, s in enumerate(sections) if s.paragraphs or (i + 1 < len(sections) and sections[i + 1].level > s.level)]
    return Document(sections=sections, has_abstract=has_abstract)
