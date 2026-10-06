"""Строки PDF через PyMuPDF с отсевом колонтитулов, сносок, подписей на рисунках и повёрнутого текста."""

import re
from collections import Counter
from pathlib import Path

import pymupdf

from research_assistant.parse.units import Unit, clean_text, header_key

# Лигатуры (ﬁ, ﬂ) раскрываются, потому что флаг TEXT_PRESERVE_LIGATURES не передаётся.
TEXT_FLAGS = pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP
BOLD_FONT_RE = re.compile(r"bold|black|heavy|semibold|demi|cmbx|\bbx|-bd|,bd", re.I)
# Строки мельче основного текста — сноски, подписи осей, текст на рисунках. Аннотации и подписи
# к рисункам обычно не мельче 0.9 от основного размера, сноски — около 0.8.
MIN_SIZE_RATIO = 0.85
# Верх и низ страницы, где ищем повторяющиеся колонтитулы.
MARGIN_RATIO = 0.08
PAGE_NUMBER_RE = re.compile(r"^[\W_]*\d{1,4}[\W_]*$")


def _is_bold(span: dict) -> bool:
    return bool(span["flags"] & pymupdf.TEXT_FONT_BOLD) or bool(BOLD_FONT_RE.search(span["font"]))


def _line_features(line: dict) -> tuple[str, float, bool]:
    spans = [s for s in line["spans"] if s["text"].strip()]
    text = clean_text("".join(s["text"] for s in line["spans"]))
    if not spans:
        return text, 0.0, False
    chars = sum(len(s["text"].strip()) for s in spans)
    size = sum(s["size"] * len(s["text"].strip()) for s in spans) / chars
    return text, size, all(_is_bold(s) for s in spans)


def pdf_units(path: str | Path) -> tuple[list[Unit], int]:
    """Возвращает строки основного текста и число страниц."""
    raw = []  # (page, block, text, size, bold, y0, y1, page_height)
    with pymupdf.open(path) as doc:
        n_pages = doc.page_count
        block_id = 0
        for page in doc:
            height = page.rect.height
            for block in page.get_text("dict", flags=TEXT_FLAGS)["blocks"]:
                if block["type"] != 0:
                    continue
                block_id += 1
                for line in block["lines"]:
                    dx, dy = line["dir"]
                    if abs(dy) > 0.1 or dx <= 0:
                        continue  # повёрнутый текст: штамп arXiv на полях, подписи осей
                    text, size, bold = _line_features(line)
                    if text:
                        raw.append((page.number + 1, block_id, text, size, bold, line["bbox"][1], line["bbox"][3], height))

    if not raw:
        return [], n_pages

    weights = Counter()
    for _, _, text, size, *_ in raw:
        weights[round(size * 2) / 2] += len(text)
    body_size = weights.most_common(1)[0][0]

    in_margin = [y1 < h * MARGIN_RATIO or y0 > h * (1 - MARGIN_RATIO) for *_, y0, y1, h in raw]
    margin_pages: dict[str, set[int]] = {}
    for (page, _, text, *_), margin in zip(raw, in_margin):
        key = header_key(text)
        if margin and key:
            margin_pages.setdefault(key, set()).add(page)
    repeated = {key for key, pages in margin_pages.items() if len(pages) >= 2}

    units = []
    for (page, block, text, size, bold, y0, _, height), margin in zip(raw, in_margin):
        if size < body_size * MIN_SIZE_RATIO:
            continue
        # Номер страницы ищем только на полях: внутри страницы «1» отдельной строкой — номер раздела.
        if margin and (PAGE_NUMBER_RE.match(text) or header_key(text) in repeated):
            continue
        units.append(Unit(text=text, page=page, block=block, bold=bold, size=size / body_size, y=y0 / height))
    return units, n_pages
