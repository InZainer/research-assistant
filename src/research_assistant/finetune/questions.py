"""Синтез вопросов учителем по статьям сплита (docs/finetune.md, раздел «Вопрос»).

Учитель видит фрагмент с метками [s<id>] и называет только id предложений-ответов, а цитату собирает код из
текста фрагмента. Поэтому золото дословно по построению и находится через evaluation.retrieval.resolve.
Каждый кандидат проходит автоматические проверки; отклонённые тоже пишутся в файл с причинами (reasons),
чтобы было видно, что и почему отсеяно. Запись идёт дозаписью, повторный запуск пропускает готовые id.
"""

import hashlib
import json
import math
import random
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from research_assistant.agent.tools import wants_to_add
from research_assistant.answer.citations import parse_sentence_id
from research_assistant.evaluation.retrieval import GoldItem, GoldQuestion, resolve
from research_assistant.finetune.splits import jaccard, near_duplicate
from research_assistant.llm import Usage
from research_assistant.nlp.entities import contains_stems, parse_json
from research_assistant.nlp.text import TOKEN_RE, analyze, detect_language
from research_assistant.store.db import PASSAGE_JOIN, Library

# Вес типа раздела при выборке фрагментов: в аннотации и результатах факты плотнее, чем в обзоре и приложениях.
SECTION_WEIGHTS = {
    "abstract": 1.0, "results": 1.0, "experiments": 1.0, "limitations": 1.0, "conclusion": 1.0, "discussion": 1.0,
    "method": 0.8, "introduction": 0.8,
    "related_work": 0.5, "body": 0.5,
    "other": 0.3, "appendix": 0.3,
}
EXCLUDED_SECTIONS = {"acknowledgements", "references"}
MIN_PASSAGE_CHARS = 200
QUOTE_MIN, QUOTE_MAX = 30, 400

KINDS = ("fact_en", "fact_xl", "fact_ru", "entity", "paper_lookup", "unanswerable", "followup", "external", "compare")
# Подтипы fact различаются языками вопроса и статьи; в записи kind = fact, язык статьи — в meta.source_language.
RECORD_KIND = {"fact_en": "fact", "fact_xl": "fact", "fact_ru": "fact"}
PASSAGE_KINDS = {"fact_en": "en", "fact_xl": "en", "fact_ru": "ru", "unanswerable": None, "followup": None}
QUESTION_LANGUAGE = {"fact_en": "en", "fact_xl": "ru", "fact_ru": "ru"}
SPLIT_PREFIX = {"train": "tr", "val": "va", "test": "te"}

NOT_SELF_CONTAINED_RE = re.compile(r"this paper|the authors|in this work|данн(ой|ая) (стать|работ)|в этой работе", re.I)
COPY_JACCARD = 0.5  # больше — вопрос пересказывает цитату теми же словами
COPY_NGRAM = 6  # общая последовательность из стольких слов — копия
GOLD_DUPLICATE = 0.5
CANDIDATE_DUPLICATE = 0.7
RANK_K = 20  # глубина поиска для meta.gold_rank
JUDGE_K = 10  # сколько результатов поиска видит судья неотвечаемого вопроса
JUDGE_CHARS = 700
RETRIES = 3

# Сбои сети и перегрузка сервиса: повторяем. Ошибки самого запроса (400, 413, 422) — причина отказа кандидата.
# Любой другой код (401 ключ, 403 доступ, 404 модель…) — ошибка настройки: синтез останавливается, иначе
# один запуск с неверным .env записал бы отказ на каждый кандидат.
TRANSIENT_ERRORS = {"APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError",
                    "ConnectionError", "TimeoutError", "ReadTimeout", "ConnectTimeout"}
TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504}
REQUEST_STATUS = {400, 413, 422}
CONFIG_ERRORS = {"AuthenticationError", "PermissionDeniedError", "NotFoundError"}
RETRYABLE_REASONS = ("api_error:", "error:")  # такие записи при возобновлении пересчитываются
HELD_OUT_DUPLICATE = 0.6  # почти-дубль вопроса test в train и val (docs/finetune.md)

OFFTOPIC_FIELDS = [
    "astronomy", "organic chemistry", "macroeconomics", "geology", "materials science", "ornithology",
    "climate modelling", "civil engineering", "marine biology", "number theory", "archaeology", "soil science",
]
EXTERNAL_TEMPLATES = {
    "ru": ["Найди в интернете новые статьи про {s}", "Поищи во внешних источниках свежие публикации о {s}",
           "Какие новые работы по теме «{s}» вышли недавно? Посмотри в arXiv и OpenAlex."],
    "en": ["Find new papers about {s} on the internet", "Search external sources for recent publications on {s}",
           "Are there any recent papers on {s}? Look them up on arXiv or OpenAlex."],
}
EXTERNAL_ENTITIES = 20  # сколько самых частых методов сплита идут темами внешних запросов


# --- промпты учителя (инструкции по-английски: так их надёжнее выполняет любая модель) ---

LANGUAGE_RULES = {
    "en": "Write the question in English.",
    "ru": "Write the question in Russian.",
    "xl": "Write the question in Russian although the passage is in English. Names of methods, models, datasets "
          "and metrics may stay in Latin script (e.g. EAGLE, KV cache, BLEU).",
}

QUESTION_RULES = """Rules for the question:
- Self-contained: a reader who has not seen the passage understands it. Name the method, model, task or subject \
explicitly. Never write "this paper", "the authors", "in this work", "данная статья", "в этой работе".
- Do not copy the sentence: paraphrase and ask about the fact; do not reuse any sequence of five or more words.
- A natural question a researcher would ask a research assistant. One question, without the answer.
- Ask about ONE thing: no compound questions joined with "and", "и", "а также", "and why". At most 25 words.
- {language}"""

INTRO = ("You write test questions for a research assistant that answers from a library of scientific papers.\n"
         "Every sentence of a passage starts with a tag [s<id>].\n")
RETURN_QUESTION = '\nReturn only JSON: {"question": "...", "sentence_ids": [<id>, ...]}'

FACT_PROMPT = (INTRO + "You get one passage of a paper. Write ONE question about one fact that is stated in a "
               "single sentence of the passage (two adjacent sentences at most); list their ids.\n" + QUESTION_RULES + RETURN_QUESTION)

ENTITY_PROMPT = (INTRO + 'You get one passage of a paper that mentions the {type} "{name}". Write ONE question whose '
                 'answer is this {type}: for example, which {type}s are used for or in <the method, task or subject, '
                 'named explicitly>. Do not mention "{name}" in the question. List the ids of the sentences that '
                 'mention "{name}" and answer the question.\n' + QUESTION_RULES + RETURN_QUESTION)

PAPER_LOOKUP_PROMPTS = {
    "author": "Refer to the paper by the surname of one of its authors (e.g. \"What did <Surname> et al. propose "
              "for ...?\", \"Что предложили <Фамилия> и соавторы для ...?\").",
    "title": "Refer to the paper by its title or the distinctive words of its title (e.g. \"What is proposed in "
             "the paper «<title>»?\", \"Что предложено в статье «<title>»?\").",
}
PAPER_LOOKUP_PROMPT = (INTRO + "You get the title, the authors and the abstract of a paper. Write ONE question that "
                       "asks what the paper proposes, finds or studies. {how} Add a hint of the subject so the "
                       "question is not vague. The answer must be in the abstract; list the ids of the answering "
                       "sentences.\n" + QUESTION_RULES + RETURN_QUESTION)

UNANSWERABLE_PROMPTS = {
    "detail": (INTRO + "You get one passage of a paper. Write ONE plausible question on the same subject that a "
               "researcher might ask, but that the passage does NOT answer and that a library of papers on this "
               "topic is unlikely to answer: ask about a specific detail (an exact number, a hardware setup, a "
               "dataset, a comparison with a named system, a release date) that is not given in the passage. "
               "No false premises.\n"),
    "offtopic": ("You write test questions for a research assistant whose library covers only these topics: "
                 "{topics}. Write ONE realistic research question about {field}, a field these topics do not "
                 "cover.\n"),
}
UNANSWERABLE_RETURN = '\nReturn only JSON: {"question": "..."}'

FOLLOWUP_PROMPT = (INTRO + "You get one passage of a paper. Write a two-turn conversation:\n"
                   "- first_question: a self-contained question about the method or subject of the passage, "
                   "answered by the passage;\n"
                   "- first_answer: a short answer to it (one or two sentences), as an assistant would reply;\n"
                   "- question: a follow-up that only makes sense after the first exchange: it refers back with a "
                   "pronoun or an ellipsis (e.g. \"And what are its limitations?\", \"А на каких данных это "
                   "проверяли?\") and is answered by OTHER sentences of the passage; list their ids.\n"
                   "Do not copy sentences of the passage. Never write \"this paper\", \"the authors\", "
                   "\"in this work\". {language} All three texts use this language.\n"
                   'Return only JSON: {"first_question": "...", "first_answer": "...", "question": "...", '
                   '"sentence_ids": [<id>, ...]}')

COMPARE_PROMPT = (INTRO + 'You get two passages from two different papers on the same topic; both mention "{name}". '
                  'Write ONE question that compares the two papers\' methods or findings with respect to "{name}", '
                  "so that answering needs both passages. Name both methods or papers explicitly (by method name or "
                  "a short form of the title). sentence_ids must include at least one sentence from each passage.\n"
                  + QUESTION_RULES + RETURN_QUESTION)

CHECK_PROMPT = ("You check test questions for a research assistant. You get a passage of a scientific paper, where "
                "every sentence starts with a tag [s<id>], and a question (sometimes after a previous exchange). "
                "List the ids of the sentences that answer the question. If the passage does not answer it, return "
                "an empty list.\n"
                'Return only JSON: {"sentence_ids": [<id>, ...]}')

JUDGE_PROMPT = ("You check that a question cannot be answered from a library of scientific papers. You get the "
                "question and the top search results, each starting with a tag [p<id>]. List the ids of the results "
                "that answer the question fully or in substantial part; being on the same topic is not enough.\n"
                'Return only JSON: {"answering_ids": [<id>, ...]}')


@dataclass
class Candidate:
    """Запланированный вопрос. kind — подтип из KINDS; question и sentence_ids заполняет учитель."""

    id: str
    kind: str
    split: str
    paper_ids: list[int]
    passage_ids: list[int]
    language: str
    source_language: str | None = None
    topic: str | None = None
    question: str = ""
    sentence_ids: list[int] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)  # тип и имя сущности, вариант вопроса и т. п.


class TransientError(Exception):
    """Сервис учителя недоступен и после повторов: кандидата не пишем, следующий запуск его повторит."""


class TeacherError(Exception):
    """Постоянная ошибка API на этом запросе (например, 400): кандидат отклоняется с причиной api_error."""


class TeacherConfigError(Exception):
    """Учитель отвергает любой запрос (неверный ключ, нет доступа, нет модели): синтез останавливается."""


@dataclass
class HeldOut:
    """Принятые вопросы test: тексты (dedup_text) и ключи тем внешних запросов — для отсева из train и val."""

    texts: list[str] = field(default_factory=list)
    subjects: set[str] = field(default_factory=set)


# --- выборка фрагментов ---

def _eligible_passages(library: Library, paper_ids, exclude=()) -> list[dict]:
    """Фрагменты, по которым можно задавать вопросы: текст (не подпись), не благодарности и не список
    литературы, не короче MIN_PASSAGE_CHARS, есть предложения. Аннотации из метаданных подходят."""
    paper_ids = list(paper_ids)
    if not paper_ids:
        return []
    rows = library.conn.execute(
        f"""SELECT p.id, p.paper_id, p.order_idx, p.source, p.text, s.title AS section, s.kind AS section_kind,
                   pa.title, pa.language AS paper_language
            FROM {PASSAGE_JOIN}
            WHERE p.paper_id IN ({','.join('?' * len(paper_ids))}) AND p.kind = 'text' AND length(p.text) >= ?
              AND EXISTS (SELECT 1 FROM sentences t WHERE t.passage_id = p.id)
            ORDER BY p.paper_id, p.order_idx""",
        [*paper_ids, MIN_PASSAGE_CHARS],
    ).fetchall()
    excluded = set(exclude)
    passages = []
    for row in rows:
        if row["section_kind"] in EXCLUDED_SECTIONS or row["id"] in excluded or looks_like_references(row["text"]):
            continue
        passages.append({**dict(row), "language": detect_language(row["text"]),
                         "weight": SECTION_WEIGHTS.get(row["section_kind"], 0.3)})
    return passages


REFERENCE_MARKERS = re.compile(r"doi[:.]|\b(?:19|20)\d{2}[a-z]?\)|\d+\(\d+\)|\bpp?\.\s*\d|\bvol\.|\bjournal\b"
                               r"|\bproceedings\b|arxiv:\d|\bт\.\s*\d|\bс\.\s*\d+\s*[–-]", re.I)


def looks_like_references(text: str) -> bool:
    """Хвост списка литературы, который парсер не отрезал: вопрос «в каком журнале…» по нему бесполезен."""
    return len(REFERENCE_MARKERS.findall(text)) >= 4


def paper_quota(n_passages: int, per_paper_max: int = 30) -> int:
    """Сколько фрагментов брать из статьи: корень сглаживает разницу между статьёй на 20 и на 950 фрагментов."""
    return min(per_paper_max, round(3 * math.sqrt(n_passages)))


def _sample_paper(passages: list[dict], quota: int, rng: random.Random) -> list[dict]:
    """Взвешенная выборка без возвращения (ключ u^(1/w)); соседние фрагменты (±1 по порядку) не берём."""
    keyed = sorted(passages, key=lambda p: (-(rng.random() ** (1.0 / p["weight"])), p["id"]))
    chosen, taken = [], set()
    for passage in keyed:
        if len(chosen) >= quota:
            break
        if passage["order_idx"] in taken or {passage["order_idx"] - 1, passage["order_idx"] + 1} & taken:
            continue
        chosen.append(passage)
        taken.add(passage["order_idx"])
    return sorted(chosen, key=lambda p: p["order_idx"])


def _sample(eligible: list[dict], per_paper_max: int, seed: int) -> list[dict]:
    by_paper: dict[int, list[dict]] = {}
    for passage in eligible:
        by_paper.setdefault(passage["paper_id"], []).append(passage)
    sampled = []
    for paper_id in sorted(by_paper):
        rng = random.Random(f"{seed}:{paper_id}")  # своя последовательность у статьи: сплит не сдвигает выборку
        sampled += _sample_paper(by_paper[paper_id], paper_quota(len(by_paper[paper_id]), per_paper_max), rng)
    return sampled


def sample_passages(library: Library, paper_ids, per_paper_max: int = 30, seed: int = 0, exclude=()) -> list[dict]:
    """Фрагменты для вопросов: на статью min(per_paper_max, round(3·√n)), веса по типу раздела, без соседей."""
    return _sample(_eligible_passages(library, paper_ids, exclude), per_paper_max, seed)


# --- план выборки ---

def _interleave(groups: dict[str, list]) -> list:
    """Перемешивает списки пропорционально длине: план с --limit N содержит все типы, а не только первый."""
    items = [(index / len(group), order, item) for order, (_, group) in enumerate(sorted(groups.items()))
             for index, item in enumerate(group)]
    return [item for _, _, item in sorted(items, key=lambda x: (x[0], x[1]))]


def _question_language(source_language: str, rng: random.Random) -> str:
    """Для вопросов не-fact: к русской статье — по-русски, к английской — поровну en и ru (межъязыковой поиск)."""
    return "ru" if source_language == "ru" else rng.choice(["en", "ru"])


class _Planner:
    def __init__(self, library: Library, paper_ids, split: str, per_paper_max: int, seed: int, exclude):
        self.library = library
        self.paper_ids = sorted(set(paper_ids))
        self.split = split
        self.prefix = SPLIT_PREFIX.get(split, split[:2])
        self.rng = random.Random(seed)
        self.eligible = _eligible_passages(library, self.paper_ids, exclude)
        self.by_id = {p["id"]: p for p in self.eligible}
        self.sampled = _sample(self.eligible, per_paper_max, seed)
        self.topics = {pid: (library.paper_topics(pid) or [None])[0] for pid in self.paper_ids}

    def make(self, kind: str, anchor: str, passages: list[dict], language: str, **meta) -> Candidate:
        return Candidate(
            id=f"{self.prefix}-{kind.replace('_', '-')}-{anchor}", kind=kind, split=self.split,
            paper_ids=[p["paper_id"] for p in passages], passage_ids=[p["id"] for p in passages],
            language=language, source_language=passages[0]["language"] if passages else None,
            topic=self.topics.get(passages[0]["paper_id"]) if passages else None, meta=meta,
        )

    def passage_kinds(self, counts: dict[str, int]) -> list[Candidate]:
        """Один фрагмент — один вопрос: fact_en/fact_xl берут английские фрагменты, fact_ru — русские,
        unanswerable и followup — любые. Слоты типов чередуются, поэтому нехватку делят все типы."""
        slots = _interleave({kind: [kind] * counts.get(kind, 0) for kind in PASSAGE_KINDS})
        pool = list(self.sampled)
        self.rng.shuffle(pool)
        used, plan, unanswerable = set(), [], 0
        for kind in slots:
            need = PASSAGE_KINDS[kind]
            passage = next((p for p in pool if p["id"] not in used and need in (None, p["language"])), None)
            if passage is None:
                continue
            used.add(passage["id"])
            meta = {}
            language = QUESTION_LANGUAGE.get(kind) or _question_language(passage["language"], self.rng)
            if kind == "unanswerable":
                # Каждый четвёртый неотвечаемый — вне тем библиотеки, остальные — о детали, которой нет.
                variant = "offtopic" if unanswerable % 4 == 3 else "detail"
                meta = {"variant": variant, **({"field": self.rng.choice(OFFTOPIC_FIELDS)} if variant == "offtopic" else {})}
                unanswerable += 1
            plan.append(self.make(kind, f"{passage['id']:06d}", [passage], language, **meta))
        return plan

    def _entity_mentions(self) -> list[dict]:
        """Сущности статей сплита с лучшим подходящим фрагментом-упоминанием (по весу раздела, затем по порядку)."""
        if not self.paper_ids:
            return []
        rows = self.library.conn.execute(
            f"""SELECT e.id, e.paper_id, e.type, e.name, e.key, m.passage_id FROM entities e
                JOIN entity_mentions m ON m.entity_id = e.id
                WHERE e.paper_id IN ({','.join('?' * len(self.paper_ids))}) ORDER BY e.id, m.passage_id""",
            self.paper_ids,
        ).fetchall()
        best: dict[int, dict] = {}
        for row in rows:
            passage = self.by_id.get(row["passage_id"])
            if passage is None:
                continue
            current = best.get(row["id"])
            rank = (-passage["weight"], passage["order_idx"])
            if current is None or rank < current["rank"]:
                best[row["id"]] = {**dict(row), "passage": passage, "rank": rank}
        return [best[i] for i in sorted(best)]

    def entity(self, count: int) -> list[Candidate]:
        by_paper: dict[str, list] = {}
        for mention in self._entity_mentions():
            by_paper.setdefault(str(mention["paper_id"]), []).append(mention)
        for mentions in by_paper.values():
            self.rng.shuffle(mentions)
        plan = []
        for mention in _interleave(by_paper)[:count]:
            passage = mention["passage"]
            plan.append(self.make("entity", f"{mention['id']:06d}", [passage],
                                  _question_language(passage["language"], self.rng),
                                  entity_type=mention["type"], entity=mention["name"]))
        return plan

    def paper_lookup(self, count: int) -> list[Candidate]:
        """По аннотации: сначала вопрос «по автору» к каждой статье, затем «по названию»."""
        rounds = {"author": [], "title": []}
        for paper_id in self.paper_ids:
            abstract = next((p for p in self.eligible if p["paper_id"] == paper_id and p["section_kind"] == "abstract"),
                            None)
            if abstract is None:
                continue
            row = self.library.get_paper(paper_id)
            authors = json.loads(row["authors"] or "[]")
            if authors:
                rounds["author"].append((paper_id, "author", abstract, authors))
            rounds["title"].append((paper_id, "title", abstract, authors))
        for items in rounds.values():
            self.rng.shuffle(items)
        plan = []
        for paper_id, variant, abstract, authors in (rounds["author"] + rounds["title"])[:count]:
            plan.append(self.make("paper_lookup", f"{paper_id:06d}-{variant}", [abstract],
                                  _question_language(abstract["language"], self.rng), variant=variant,
                                  authors=authors[:6]))
        return plan

    def compare(self, count: int) -> list[Candidate]:
        """Пары статей одной темы с общей сущностью (тот же тип и ключ основ)."""
        shared: dict[tuple[str, str], dict[int, dict]] = {}
        for mention in self._entity_mentions():
            shared.setdefault((mention["type"], mention["key"]), {}).setdefault(mention["paper_id"], mention)
        pairs: dict[tuple[int, int], list] = {}
        for key in sorted(shared):
            papers = sorted(shared[key])
            for i, a in enumerate(papers):
                for b in papers[i + 1:]:
                    if self.topics.get(a) is not None and self.topics.get(a) == self.topics.get(b):
                        pairs.setdefault((a, b), []).append(key)
        ordered = sorted(pairs)
        self.rng.shuffle(ordered)
        cap = max(2, math.ceil(2 * count / max(len(self.paper_ids), 1)))  # статья — не больше чем в cap парах
        used: Counter = Counter()
        plan = []
        for a, b in ordered:
            if len(plan) >= count:
                break
            if used[a] >= cap or used[b] >= cap:
                continue
            key = self.rng.choice(pairs[(a, b)])
            first, second = shared[key][a], shared[key][b]
            passages = [first["passage"], second["passage"]]
            languages = [p["language"] for p in passages]
            language = "ru" if languages == ["ru", "ru"] else self.rng.choice(["en", "ru"])
            plan.append(self.make("compare", f"{a:06d}-{b:06d}", passages, language, entity_type=key[0],
                                  entity=first["name"], source_languages=languages))
            used[a] += 1
            used[b] += 1
        return plan

    def external(self, count: int, exclude_subjects=()) -> list[Candidate]:
        """Просьбы найти новые статьи: темы сплита и самые частые методы. Вопрос собирает код, без учителя.

        Шаблонов мало, поэтому дубли считаются по теме: одна тема — не больше одного запроса на язык,
        варианты написания (регистр, словоформы) — одна тема. exclude_subjects — ключи тем из test."""
        # Тема записывается как "" вместо None: иначе кортежи не сравниваются при сортировке.
        subjects = {(topic, topic) for topic in self.topics.values() if topic}
        if self.paper_ids:
            rows = self.library.conn.execute(
                f"""SELECT MIN(name) AS name, MIN(paper_id) AS paper_id, COUNT(DISTINCT paper_id) AS n FROM entities
                    WHERE type = 'method' AND paper_id IN ({','.join('?' * len(self.paper_ids))})
                    GROUP BY key ORDER BY n DESC, name LIMIT ?""",
                [*self.paper_ids, EXTERNAL_ENTITIES],
            ).fetchall()
            subjects |= {(row["name"], self.topics.get(row["paper_id"]) or "") for row in rows
                         if not _is_abbreviation(row["name"])}
        excluded = set(exclude_subjects)
        # Английский запрос с русской темой выглядел бы неестественно (и не прошёл бы проверку языка).
        options = sorted((subject, topic, language, template) for subject, topic in subjects
                         if subject_key(subject) not in excluded
                         for language, templates in EXTERNAL_TEMPLATES.items() for template in templates
                         if language == "ru" or detect_language(subject) == "en")
        self.rng.shuffle(options)
        plan, seen = [], set()
        for subject, topic, language, template in options:
            if len(plan) >= count:
                break
            if (subject_key(subject), language) in seen:
                continue
            seen.add((subject_key(subject), language))
            question = template.format(s=subject)
            anchor = hashlib.sha1(question.encode("utf-8")).hexdigest()[:10]
            candidate = Candidate(id=f"{self.prefix}-external-{anchor}", kind="external", split=self.split,
                                  paper_ids=[], passage_ids=[], language=language, topic=topic or None,
                                  question=question,
                                  meta={"subject": subject})
            plan.append(candidate)
        return plan


def subject_key(subject: str) -> str:
    """Ключ темы внешнего запроса: основы analyze() без порядка («Speculative Decoding» = «speculative decoding»)."""
    stems = sorted(set(analyze(subject)))
    return " ".join(stems) if stems else subject.casefold().strip()


def _is_abbreviation(name: str) -> bool:
    """Короткая аббревиатура (SD, ASD): как тема поиска в интернете она бессмысленна."""
    letters = re.sub(r"\W", "", name)
    return len(letters) <= 3 and letters.isupper()


def plan_sampling(library: Library, split_papers, kinds_counts: dict[str, int], *, split: str = "train",
                  per_paper_max: int = 30, seed: int = 0, exclude_passages=(),
                  exclude_subjects=()) -> list[Candidate]:
    """Детерминированный план кандидатов (без вызова модели): из него же работает synthesize и --dry-run.

    exclude_subjects — ключи subject_key тем внешних запросов, уже занятых в test (HeldOut.subjects).
    Число кандидатов типа может быть меньше заказанного: у сплита кончились фрагменты, сущности или пары."""
    unknown = sorted(set(kinds_counts) - set(KINDS))
    if unknown:
        raise ValueError(f"неизвестные типы вопросов: {', '.join(unknown)}; доступны {', '.join(KINDS)}")
    planner = _Planner(library, split_papers, split, per_paper_max, seed, exclude_passages)
    groups: dict[str, list] = {"passage": planner.passage_kinds(kinds_counts)}
    for kind in ("entity", "paper_lookup", "compare"):
        if kinds_counts.get(kind):
            groups[kind] = getattr(planner, kind)(kinds_counts[kind])
    if kinds_counts.get("external"):
        groups["external"] = planner.external(kinds_counts["external"], exclude_subjects)
    return _interleave(groups)


def summarize_plan(plan: list[Candidate], kinds_counts: dict[str, int] | None = None) -> dict:
    """Числа для --dry-run: по типам (заказано и запланировано), по статьям и по языкам вопросов."""
    by_kind = Counter(c.kind for c in plan)
    by_paper = Counter(pid for c in plan for pid in c.paper_ids)
    return {
        "total": len(plan),
        "by_kind": {kind: {"planned": by_kind.get(kind, 0), "requested": (kinds_counts or {}).get(kind)}
                    for kind in KINDS if by_kind.get(kind) or (kinds_counts or {}).get(kind)},
        "by_paper": dict(sorted(by_paper.items())),
        "by_language": dict(sorted(Counter(c.language for c in plan).items())),
    }


def format_plan(summary: dict) -> str:
    lines = [f"Кандидатов в плане: {summary['total']}", f"{'тип':<14} {'заказано':>9} {'в плане':>8}"]
    for kind, row in summary["by_kind"].items():
        requested = "-" if row["requested"] is None else row["requested"]
        lines.append(f"{kind:<14} {requested:>9} {row['planned']:>8}")
    lines.append("Языки вопросов: " + ", ".join(f"{k} {v}" for k, v in summary["by_language"].items()))
    papers = summary["by_paper"]
    if papers:
        lines.append(f"Статей: {len(papers)}, вопросов на статью: от {min(papers.values())} до {max(papers.values())}")
    return "\n".join(lines)


# --- запросы к учителю ---

def _sentences(library: Library, passage_id: int) -> list:
    return library.conn.execute(
        "SELECT id, order_idx, char_start, char_end, text FROM sentences WHERE passage_id = ? ORDER BY order_idx",
        (passage_id,),
    ).fetchall()


def tagged_passage(library: Library, passage_id: int) -> str:
    """Фрагмент по предложению на строке с метками [s<id>] — как в выдаче search_library."""
    return "\n".join(f"[s{s['id']}] {s['text']}" for s in _sentences(library, passage_id))


def _passage_header(library: Library, passage_id: int) -> str:
    row = library.conn.execute(
        f"SELECT s.title AS section, s.kind, pa.title FROM {PASSAGE_JOIN} WHERE p.id = ?", (passage_id,)
    ).fetchone()
    return f"Paper: {row['title']}\nSection: {row['section']} ({row['kind']})"


def _language_rule(candidate: Candidate) -> str:
    if candidate.language == "ru" and candidate.source_language == "en":
        return LANGUAGE_RULES["xl"]
    return LANGUAGE_RULES[candidate.language]


def build_prompt(library: Library, candidate: Candidate) -> tuple[str, str]:
    """(system, user) запроса к учителю на синтез вопроса."""
    rule = _language_rule(candidate)
    kind = candidate.kind
    if kind == "compare":
        system = COMPARE_PROMPT.replace("{name}", candidate.meta["entity"])
        parts = [f"Passage {i}.\n{_passage_header(library, pid)}\n\n{tagged_passage(library, pid)}"
                 for i, pid in enumerate(candidate.passage_ids, 1)]
        return system.replace("{language}", rule), "\n\n".join(parts)
    passage_id = candidate.passage_ids[0]
    user = f"{_passage_header(library, passage_id)}\n\nPassage:\n{tagged_passage(library, passage_id)}"
    if kind.startswith("fact"):
        system = FACT_PROMPT
    elif kind == "entity":
        system = ENTITY_PROMPT.replace("{type}", candidate.meta["entity_type"]).replace("{name}", candidate.meta["entity"])
    elif kind == "paper_lookup":
        system = PAPER_LOOKUP_PROMPT.replace("{how}", PAPER_LOOKUP_PROMPTS[candidate.meta["variant"]])
        user = f"Authors: {', '.join(candidate.meta.get('authors') or []) or '-'}\n{user}"
    elif kind == "unanswerable":
        variant = candidate.meta.get("variant", "detail")
        system = UNANSWERABLE_PROMPTS[variant] + "{language}" + UNANSWERABLE_RETURN
        if variant == "offtopic":
            topics = sorted({t for (t,) in library.conn.execute("SELECT name FROM topics")})
            system = system.replace("{topics}", "; ".join(topics) or "-").replace("{field}", candidate.meta["field"])
            user = "Write the question."
    elif kind == "followup":
        system = FOLLOWUP_PROMPT
    else:
        raise ValueError(f"для типа {kind} учитель не нужен")
    return system.replace("{language}", rule), user


def _is_transient(exc: Exception) -> bool:
    return (any(cls.__name__ in TRANSIENT_ERRORS for cls in type(exc).__mro__)
            or getattr(exc, "status_code", None) in TRANSIENT_STATUS)


def _is_config_error(exc: Exception) -> bool:
    """Ошибка настройки, а не запроса: известные классы SDK или любой код ответа вне REQUEST_STATUS."""
    if any(cls.__name__ in CONFIG_ERRORS for cls in type(exc).__mro__):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and status not in REQUEST_STATUS


def ask_teacher(model, system: str, user: str, usage: Usage | None = None, retries: int = RETRIES,
                sleep: Callable[[float], None] = time.sleep) -> str:
    """Текст ответа учителя. Временные сбои повторяются с паузой 2, 4, 8… с; исчерпаны — TransientError.

    Ошибка запроса (400, 413, 422) — TeacherError, ошибка настройки (401, 403, 404…) — TeacherConfigError."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for attempt in range(retries + 1):
        try:
            response = model.complete(messages)
        except Exception as exc:
            if not _is_transient(exc):
                if _is_config_error(exc):
                    raise TeacherConfigError(f"{type(exc).__name__}: {str(exc)[:200]}") from exc
                raise TeacherError(f"{type(exc).__name__}: {str(exc)[:200]}") from exc
            if attempt == retries:
                raise TransientError(f"{type(exc).__name__}: {exc}") from exc
            sleep(2 ** (attempt + 1))
            continue
        if usage is not None:
            usage.add(getattr(response, "usage", None))
        content = response.choices[0].message.content or ""
        # Mercury изредка отвечает пустой строкой (в пилоте 2 из 70 вызовов): это сбой, а не отказ от вопроса.
        if content.strip() or attempt == retries:
            return content
        sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")


def _ids(raw) -> list[int] | None:
    """Список id из ответа учителя (числа, "s12", "[s12]", "p12"); None — если это не список id."""
    if not isinstance(raw, list):
        return None
    ids = []
    for item in raw:
        value = parse_sentence_id(re.sub(r"^\s*\[?p", "", str(item), flags=re.I)) if not isinstance(item, int) else item
        if value is None:
            return None
        ids.append(value)
    return list(dict.fromkeys(ids))


def fill_candidate(candidate: Candidate, content: str) -> list[str]:
    """Разбирает JSON учителя в кандидата. Возвращает причины отказа (битый JSON, пустой вопрос)."""
    try:
        data = parse_json(content)
    except (ValueError, json.JSONDecodeError) as exc:
        return [f"bad_json: {str(exc)[:120]}"]
    question = data.get("question")
    if not isinstance(question, str) or not question.strip():
        return ["bad_json: нет вопроса"]
    candidate.question = " ".join(question.split())
    if candidate.kind == "unanswerable":
        return []
    ids = _ids(data.get("sentence_ids"))
    if ids is None:
        return ["bad_json: sentence_ids — не список id"]
    candidate.sentence_ids = ids
    if candidate.kind == "followup":
        first_q, first_a = data.get("first_question"), data.get("first_answer")
        if not all(isinstance(x, str) and x.strip() for x in (first_q, first_a)):
            return ["bad_json: нет первого вопроса или ответа"]
        candidate.history = [{"role": "user", "content": " ".join(first_q.split())},
                             {"role": "assistant", "content": " ".join(first_a.split())}]
    return []


# --- золото и проверки ---

def build_relevant(library: Library, candidate: Candidate) -> list[GoldItem]:
    """Золото из выбранных предложений: по куску текста фрагмента на каждую непрерывную серию предложений.

    Кусок — срез passages.text от начала первого до конца последнего предложения серии, то есть дословный
    текст фрагмента; resolve() найдёт его. Чужие id (не из фрагментов кандидата) сюда не попадают."""
    chosen = set(candidate.sentence_ids)
    items = []
    for passage_id in candidate.passage_ids:
        row = library.conn.execute(
            "SELECT p.text, pa.title FROM passages p JOIN papers pa ON pa.id = p.paper_id WHERE p.id = ?", (passage_id,)
        ).fetchone()
        run: list = []
        for sentence in [*_sentences(library, passage_id), None]:
            if sentence is not None and sentence["id"] in chosen:
                run.append(sentence)
                continue
            if run:
                items.append(GoldItem(row["title"], row["text"][run[0]["char_start"]:run[-1]["char_end"]]))
                run = []
    return items


def _words(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold().replace("ё", "е"))


def shares_ngram(a: str, b: str, n: int = COPY_NGRAM) -> bool:
    left, right = _words(a), _words(b)
    grams = {tuple(left[i:i + n]) for i in range(len(left) - n + 1)}
    return any(tuple(right[i:i + n]) in grams for i in range(len(right) - n + 1))


def dedup_text(candidate: Candidate) -> str:
    """Текст для поиска почти-дублей: у followup короткий второй вопрос сравнивается вместе с первым."""
    return " ".join([*(m["content"] for m in candidate.history if m["role"] == "user"), candidate.question])


def _passage_of_sentences(library: Library, ids: list[int]) -> dict[int, int]:
    if not ids:
        return {}
    rows = library.conn.execute(
        f"SELECT id, passage_id FROM sentences WHERE id IN ({','.join('?' * len(ids))})", ids
    ).fetchall()
    return {r["id"]: r["passage_id"] for r in rows}


def static_reasons(library: Library, candidate: Candidate, relevant: list[GoldItem], gold_questions=(),
                   others=(), held_out: HeldOut | None = None) -> list[str]:
    """Проверки без модели. others — тексты (dedup_text) уже принятых кандидатов, held_out — вопросы test.

    Внешние запросы собраны из немногих шаблонов, их тексты похожи при разных темах; они сравниваются
    по теме (subject_key) ещё при планировании, а здесь — только с золотом."""
    reasons = []
    question = candidate.question
    if not question.strip():
        return ["empty_question"]
    texts = [question] + [m["content"] for m in candidate.history if m["role"] == "user"]
    if any(detect_language(t) != candidate.language for t in texts):
        reasons.append(f"wrong_language: ожидался {candidate.language}")
    if any(NOT_SELF_CONTAINED_RE.search(t) for t in texts):
        reasons.append("not_self_contained")
    if candidate.kind == "external" and wants_to_add(question):
        reasons.append("asks_to_add")
    if candidate.kind not in ("unanswerable", "external"):
        reasons += _gold_reasons(library, candidate, relevant)
    text = dedup_text(candidate)
    if any(jaccard(text, gold.question) >= GOLD_DUPLICATE for gold in gold_questions):
        reasons.append("near_duplicate_gold")
    if candidate.kind == "external":
        if held_out is not None and subject_key(candidate.meta.get("subject", "")) in held_out.subjects:
            reasons.append("near_duplicate_test")
        return reasons
    if held_out is not None and any(near_duplicate(text, t, HELD_OUT_DUPLICATE) for t in held_out.texts):
        reasons.append("near_duplicate_test")
    if any(jaccard(text, other) >= CANDIDATE_DUPLICATE for other in others):
        reasons.append("near_duplicate_candidate")
    return reasons


def _gold_reasons(library: Library, candidate: Candidate, relevant: list[GoldItem]) -> list[str]:
    reasons = []
    owners = _passage_of_sentences(library, candidate.sentence_ids)
    if not candidate.sentence_ids:
        return ["no_sentence_ids"]
    if any(owners.get(sid) not in candidate.passage_ids for sid in candidate.sentence_ids):
        reasons.append("foreign_sentence_ids")
    for paper in dict.fromkeys(item.paper for item in relevant):
        length = sum(len(item.quote) for item in relevant if item.paper == paper)
        if length < QUOTE_MIN:
            reasons.append("quote_too_short")
        elif length > QUOTE_MAX:
            reasons.append("quote_too_long")
    if candidate.kind == "compare" and set(candidate.passage_ids) - set(owners.values()):
        reasons.append("compare_needs_both_papers")
    probe = GoldQuestion(id=candidate.id, question=candidate.question, language=candidate.language, relevant=relevant)
    if not relevant or not set(candidate.passage_ids) <= resolve(library, probe):
        reasons.append("unresolved_quote")
    quote = " ".join(item.quote for item in relevant)
    if candidate.language == candidate.source_language and quote:
        if jaccard(candidate.question, quote) > COPY_JACCARD or shares_ngram(candidate.question, quote):
            reasons.append("copies_quote")
    if candidate.kind == "entity":
        name = candidate.meta.get("entity", "")
        if not contains_stems(analyze(quote), analyze(name)):
            reasons.append("entity_not_in_quote")
        if contains_stems(analyze(candidate.question), analyze(name)):
            reasons.append("entity_in_question")
    return sorted(set(reasons), key=reasons.index)


def answerable_reasons(library: Library, model, candidate: Candidate, usage: Usage | None = None,
                       retries: int = RETRIES, sleep=time.sleep) -> list[str]:
    """Второй вызов учителя видит только фрагмент(ы) и вопрос и должен выбрать пересекающиеся предложения."""
    parts = [f"Passage:\n{tagged_passage(library, pid)}" for pid in candidate.passage_ids]
    history = "".join(f"{m['role']}: {m['content']}\n" for m in candidate.history)
    user = "\n\n".join(parts) + (f"\n\nPrevious exchange:\n{history}" if history else "") + f"\nQuestion: {candidate.question}"
    try:
        data = parse_json(ask_teacher(model, CHECK_PROMPT, user, usage, retries, sleep))
    except (ValueError, json.JSONDecodeError):
        return ["answerability_bad_json"]
    ids = _ids(data.get("sentence_ids"))
    if ids is None:
        return ["answerability_bad_json"]
    return [] if set(ids) & set(candidate.sentence_ids) else ["not_answerable"]


def library_answers_reasons(searcher, model, candidate: Candidate, usage: Usage | None = None,
                            retries: int = RETRIES, sleep=time.sleep) -> list[str]:
    """Для unanswerable: судья смотрит топ-JUDGE_K гибридного поиска; ни один результат не должен отвечать."""
    hits = searcher.search(candidate.question, k=JUDGE_K)
    candidate.meta["judge_hits"] = [h.passage_id for h in hits]
    if searcher.last_warning:
        candidate.meta["search_warning"] = searcher.last_warning
    if not hits:
        return []
    results = "\n\n".join(f"[p{h.passage_id}] {h.passage.get('title', '')}: {h.passage.get('text', '')[:JUDGE_CHARS]}"
                          for h in hits)
    try:
        data = parse_json(ask_teacher(model, JUDGE_PROMPT, f"Question: {candidate.question}\n\nResults:\n{results}",
                                      usage, retries, sleep))
    except (ValueError, json.JSONDecodeError):
        return ["judge_bad_json"]
    ids = _ids(data.get("answering_ids"))
    if ids is None:
        return ["judge_bad_json"]
    answering = sorted(set(ids) & set(candidate.meta["judge_hits"]))
    candidate.meta["answering_ids"] = answering
    return ["answered_by_library"] if answering else []


def check_candidate(library: Library, candidate: Candidate, relevant: list[GoldItem] | None = None, *,
                    model=None, searcher=None, gold_questions=(), others=(), held_out: HeldOut | None = None,
                    usage: Usage | None = None, retries: int = RETRIES, sleep=time.sleep) -> list[str]:
    """Все проверки кандидата → список причин отказа (пустой — принят).

    Сначала проверки без модели; вызовы учителя (отвечаемость, судья для unanswerable) — только если они
    прошли и переданы model (и searcher для судьи): незачем платить за кандидата, который уже отсеян."""
    if relevant is None:
        relevant = build_relevant(library, candidate)
    reasons = static_reasons(library, candidate, relevant, gold_questions, others, held_out)
    if reasons or model is None:
        return reasons
    if candidate.kind == "unanswerable":
        return library_answers_reasons(searcher, model, candidate, usage, retries, sleep) if searcher else []
    if candidate.kind == "external":
        return []
    return answerable_reasons(library, model, candidate, usage, retries, sleep)


def gold_rank(searcher, question: str, passage_ids, k: int = RANK_K) -> int | None:
    """Место (с 1) первого золотого фрагмента в поиске по вопросу; None — не нашёлся в первых k."""
    hits = searcher.search(question, k=k)
    targets = set(passage_ids)
    return next((place for place, hit in enumerate(hits, 1) if hit.passage_id in targets), None)


def to_record(candidate: Candidate, relevant: list[GoldItem], reasons: list[str], rank: int | None) -> dict:
    """Запись в формате GoldQuestion (+ kind и meta), её читает evaluation.retrieval.load_questions."""
    meta = {
        "split": candidate.split, "subkind": candidate.kind,
        "paper_id": candidate.paper_ids[0] if candidate.paper_ids else None,
        "passage_id": candidate.passage_ids[0] if candidate.passage_ids else None,
        "sentence_ids": candidate.sentence_ids, "source_language": candidate.source_language,
        "gold_rank": rank, "reasons": reasons, "accepted": not reasons, "history": candidate.history,
        **candidate.meta,
    }
    if len(candidate.paper_ids) > 1:
        meta["paper_ids"], meta["passage_ids"] = candidate.paper_ids, candidate.passage_ids
    return {
        "id": candidate.id, "question": candidate.question, "language": candidate.language, "topic": candidate.topic,
        "status": "draft", "relevant": [{"paper": item.paper, "quote": item.quote} for item in relevant],
        "kind": RECORD_KIND.get(candidate.kind, candidate.kind), "meta": meta,
    }


# --- синтез ---

def _record_text(record: dict) -> str:
    """dedup_text по записи файла: вопросы пользователя из истории и сам вопрос."""
    history = (record.get("meta") or {}).get("history") or []
    users = [m.get("content") or "" for m in history if isinstance(m, dict) and m.get("role") == "user"]
    return " ".join([*users, record.get("question") or ""])


def _retryable(record: dict) -> bool:
    """Отказ из-за сбоя (ошибка API или кода), а не из-за вопроса: при возобновлении кандидат пересчитывается."""
    return any(str(r).startswith(RETRYABLE_REASONS) for r in (record.get("meta") or {}).get("reasons") or [])


def read_records(path: str | Path) -> list[dict]:
    """Записи файла вопросов. Битые строки (оборванные при аварийной остановке) и не-записи пропускаются."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and isinstance(record.get("id"), str):
            records.append(record)
    return records


def load_held_out(path: str | Path) -> HeldOut:
    """Принятые вопросы test.jsonl: тексты для Жаккара и ключи тем внешних запросов."""
    held_out = HeldOut()
    for record in read_records(path):
        meta = record.get("meta") or {}
        if not meta.get("accepted"):
            continue
        if record.get("kind") == "external":
            if meta.get("subject"):
                held_out.subjects.add(subject_key(meta["subject"]))
        else:
            held_out.texts.append(_record_text(record))
    return held_out


def mark_held_out_duplicates(records: list[dict], held_out: HeldOut) -> int:
    """Отклоняет принятые записи train/val, похожие на вопросы test: test мог быть синтезирован позже них.

    Меняет записи на месте (reasons += near_duplicate_test, accepted = False), возвращает их число."""
    marked = 0
    for record in records:
        meta = record.get("meta")
        if not isinstance(meta, dict) or not meta.get("accepted"):
            continue
        if record.get("kind") == "external":
            duplicate = subject_key(meta.get("subject") or "") in held_out.subjects
        else:
            duplicate = any(near_duplicate(_record_text(record), t, HELD_OUT_DUPLICATE) for t in held_out.texts)
        if duplicate:
            meta["reasons"] = [*(meta.get("reasons") or []), "near_duplicate_test"]
            meta["accepted"] = False
            marked += 1
    return marked


def _rewrite(path: Path, records: list[dict]) -> None:
    """Перезаписывает файл (через временный), если он изменился: оборванная последняя строка отбрасывается,
    иначе следующая дозапись склеилась бы с ней в одну битую строку."""
    text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    if path.read_text(encoding="utf-8") == text:
        return
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def gold_neighbourhood(library: Library, gold_questions) -> set[int]:
    """Золотые фрагменты и их соседи ±1: синтетический тест не должен повторять исходные 30 вопросов."""
    found = set()
    for question in gold_questions:
        found |= resolve(library, question)
    if not found:
        return set()
    rows = library.conn.execute(
        f"SELECT paper_id, order_idx FROM passages WHERE id IN ({','.join('?' * len(found))})", sorted(found)
    ).fetchall()
    near = set()
    for row in rows:
        near |= {r[0] for r in library.conn.execute(
            "SELECT id FROM passages WHERE paper_id = ? AND order_idx BETWEEN ? AND ?",
            (row["paper_id"], row["order_idx"] - 1, row["order_idx"] + 1))}
    return near


def process_candidate(library: Library, searcher, model, candidate: Candidate, *, gold_questions=(), others=(),
                      held_out: HeldOut | None = None, usage: Usage | None = None, retries: int = RETRIES,
                      sleep=time.sleep) -> dict:
    """Синтез, проверки и ранг золота для одного кандидата → запись. TransientError и TeacherConfigError
    пробрасываются: кандидат не виноват, записывать отказ нельзя."""
    reasons: list[str] = []
    try:
        if candidate.kind != "external":
            system, user = build_prompt(library, candidate)
            reasons = fill_candidate(candidate, ask_teacher(model, system, user, usage, retries, sleep))
        relevant = build_relevant(library, candidate) if candidate.passage_ids else []
        if not reasons:
            reasons = check_candidate(library, candidate, relevant, model=model, searcher=searcher,
                                      gold_questions=gold_questions, others=others, held_out=held_out, usage=usage,
                                      retries=retries, sleep=sleep)
        rank = None
        if candidate.question and relevant and searcher is not None:
            rank = gold_rank(searcher, candidate.question, candidate.passage_ids)
            if searcher.last_warning:
                candidate.meta["search_warning"] = searcher.last_warning
    except (TransientError, TeacherConfigError):
        raise
    except TeacherError as exc:
        relevant, rank = [], None
        reasons = reasons + [f"api_error: {exc}"]
    except Exception as exc:  # один кандидат не должен ронять синтез сотен остальных
        relevant, rank = [], None
        reasons = reasons + [f"error: {type(exc).__name__}: {str(exc)[:200]}"]
    return to_record(candidate, relevant, reasons, rank)


def _default_held_out(out_path: Path, split: str, log: Callable[[str], None]) -> HeldOut | None:
    """Вопросы test для отсева из train и val: соседний файл test.jsonl, если он уже есть."""
    if split not in ("train", "val"):
        return None
    test_path = out_path.with_name("test.jsonl")
    if not test_path.exists():
        log(f"Нет {test_path}: почти-дубли вопросов test не отсеяны. После синтеза test запустите {split} "
            "ещё раз: уже записанные дубли будут отклонены")
        return None
    return load_held_out(test_path)


def synthesize(library: Library, searcher, model, split_papers, kinds_counts: dict[str, int], seed: int = 0,
               out_path: str | Path = "data/finetune/questions/train.jsonl", resume: bool = True,
               log: Callable[[str], None] = print, *, split: str | None = None, gold_questions=(),
               held_out: HeldOut | None = None, per_paper_max: int = 30, limit: int | None = None,
               retries: int = RETRIES, sleep: Callable[[float], None] = time.sleep) -> dict:
    """Синтезирует вопросы по плану plan_sampling и дописывает записи в out_path (и принятые, и отклонённые).

    split по умолчанию — имя файла (questions/train.jsonl → train). resume=False начинает файл заново.
    Для test нужно золото: его фрагменты и их соседи исключаются. Для train и val held_out по умолчанию
    читается из соседнего test.jsonl: почти-дубли его вопросов отклоняются (и среди уже записанных тоже).
    Записи с отказом из-за сбоя (api_error, error) при возобновлении пересчитываются.
    Ошибка настройки учителя (401, 403, 404…) останавливает синтез исключением TeacherConfigError.
    limit — сколько новых кандидатов обработать."""
    out_path = Path(out_path)
    split = split or out_path.stem
    if split == "test" and not gold_questions:
        raise ValueError("для test нужно золото (gold_questions): иначе вопросы попадут на золотые фрагменты")
    exclude = gold_neighbourhood(library, gold_questions) if split == "test" else set()
    if held_out is None:
        held_out = _default_held_out(out_path, split, log)
    plan = plan_sampling(library, split_papers, kinds_counts, split=split, per_paper_max=per_paper_max, seed=seed,
                         exclude_passages=exclude, exclude_subjects=held_out.subjects if held_out else ())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not resume and out_path.exists():
        out_path.unlink()
    records = read_records(out_path)
    marked = mark_held_out_duplicates(records, held_out) if held_out else 0
    retried = sum(_retryable(r) for r in records)
    records = [r for r in records if not _retryable(r)]
    if out_path.exists():
        _rewrite(out_path, records)
    done = {r["id"] for r in records}
    others = [_record_text(r) for r in records if (r.get("meta") or {}).get("accepted")]
    todo = [c for c in plan if c.id not in done]
    if limit is not None:
        todo = todo[:limit]
    usage = Usage()
    stats = Counter(planned=len(plan), skipped=sum(c.id in done for c in plan), retried=retried,
                    marked_near_test=marked)
    if marked:
        log(f"Отклонено как почти-дубли вопросов test: {marked}")
    log(f"План: {len(plan)} кандидатов, уже готово {stats['skipped']}, к обработке {len(todo)}"
        + (f" (из них после сбоя {retried})" if retried else ""))
    for number, candidate in enumerate(todo, 1):
        try:
            record = process_candidate(library, searcher, model, candidate, gold_questions=gold_questions,
                                       others=others, held_out=held_out, usage=usage, retries=retries, sleep=sleep)
        except TransientError as exc:
            stats["api_errors"] += 1
            log(f"[{number}/{len(todo)}] {candidate.id}: учитель недоступен ({exc}), пропущен до следующего запуска")
            continue
        except TeacherConfigError as exc:
            log(f"[{number}/{len(todo)}] {candidate.id}: учитель отклоняет запросы ({exc}). Синтез остановлен: "
                "проверьте ключ и модель провайдера: uv run research-assistant providers")
            raise
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        stats["written"] += 1
        reasons = record["meta"]["reasons"]
        if reasons:
            stats["rejected"] += 1
            stats.update(f"reason:{r.split(':')[0]}" for r in reasons)
        else:
            stats["accepted"] += 1
            others.append(dedup_text(candidate))
        verdict = "принят" if not reasons else "отклонён: " + ", ".join(reasons)
        log(f"[{number}/{len(todo)}] {candidate.id}: {verdict}")
    keys = ("planned", "skipped", "written", "accepted", "rejected", "api_errors", "retried", "marked_near_test")
    result = {key: stats[key] for key in keys}
    result["reasons"] = {k.split(":", 1)[1]: v for k, v in sorted(stats.items()) if k.startswith("reason:")}
    result["usage"] = {"prompt": usage.prompt_tokens, "completion": usage.completion_tokens, "calls": usage.calls}
    return result
