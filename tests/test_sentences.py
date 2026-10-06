import pytest

from research_assistant.parse.sentences import split_sentences


def sentences(text):
    return [text[a:b] for a, b in split_sentences(text)]


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "We follow Leviathan et al. (2023). Speedups reach 2.5x, e.g. on Llama-2 (Fig. 3). "
            "Results in Tab. 1 hold [12]. Next sentence.",
            ["We follow Leviathan et al. (2023).", "Speedups reach 2.5x, e.g. on Llama-2 (Fig. 3).",
             "Results in Tab. 1 hold [12].", "Next sentence."],
        ),
        (
            "As shown in Sec. 4.2, the draft model (cf. Eq. 5) is small, i.e. 68M params. Prof. Smith agrees.",
            ["As shown in Sec. 4.2, the draft model (cf. Eq. 5) is small, i.e. 68M params.", "Prof. Smith agrees."],
        ),
        (
            "Мы используем метод из работы [5], т.е. классификатор. Точность 92.3 %, см. рис. 2. "
            "Результаты приведены в табл. 3. Итог.",
            ["Мы используем метод из работы [5], т.е. классификатор.", "Точность 92.3 %, см. рис. 2.",
             "Результаты приведены в табл. 3.", "Итог."],
        ),
        (
            "Нейросети применяются широко [Барский, 2007; Хайкин, 2019]. Далее текст.",
            ["Нейросети применяются широко [Барский, 2007; Хайкин, 2019].", "Далее текст."],
        ),
        (
            "As argued before (Smith et al., 2020; Rowe, 2019). Next.",
            ["As argued before (Smith et al., 2020; Rowe, 2019).", "Next."],
        ),
    ],
)
def test_scientific_abbreviations_and_citations(text, expected):
    assert sentences(text) == expected


def test_weak_abbreviation_glues_only_before_number_or_lowercase():
    assert sentences("Опубликовано в 2018 г. в журнале. Конец.") == ["Опубликовано в 2018 г. в журнале.", "Конец."]
    assert sentences("See p. 5 for details. Then more.") == ["See p. 5 for details.", "Then more."]
    # «в 2018 г. Следующее» razdel сам не разрывает: известное ограничение, не правим.


def test_offsets_slice_exact_text():
    text = "  First sentence here.   Second one (see Fig. 2) follows!  Third?"
    for start, end in split_sentences(text):
        assert text[start:end] == text[start:end].strip()
    assert sentences(text) == ["First sentence here.", "Second one (see Fig. 2) follows!", "Third?"]


def test_unclosed_bracket_glue_is_limited():
    text = "Broken (" + " word" * 130 + ". Next sentence starts. Another one."
    assert len(sentences(text)) >= 2
