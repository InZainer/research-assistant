"""Строки PDF через PyMuPDF с отсевом колонтитулов, сносок, подписей на рисунках и повёрнутого текста.

Таблицы с числами восстанавливаются по геометрии: строки блока на одной высоте — ячейки одной строки таблицы.
Встроенный page.find_tables() не используется: на страницах с векторными рисунками (карты) он работает минутами.
"""

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


# Строки одной строки таблицы стоят на одной высоте: допуск — доля высоты строки.
ROW_TOLERANCE = 0.45
CELL_SEPARATOR = " | "
MIN_TABLE_ROWS = 2  # строк с числами, чтобы блок считался таблицей
MIN_TABLE_SHARE = 0.4  # доля таких строк в блоке: в прозе числа в конце строки случайны


def _numeric(cell: str) -> bool:
    """Ячейка-число: «0.73», «35%», «1,234», «1.91×», «0.81±0.02», «12.3M»."""
    chars = [c for c in cell if not c.isspace()]
    digits = sum(c.isdigit() for c in chars)
    return digits > 0 and digits * 2 >= len(chars)


def _row_cells(texts: list[str]) -> list[str]:
    """Ячейки строки таблицы. PyMuPDF отдаёт ячейки отдельными строками на одной высоте; если вся строка
    пришла одним куском («Density 0.60 0.79 0.73»), числа в её конце — отдельные ячейки."""
    if len(texts) > 1:
        return texts
    tokens = texts[0].split()
    tail = 0
    while tail < len(tokens) and _numeric(tokens[-1 - tail]):
        tail += 1
    if tail < 2:
        return texts
    label = " ".join(tokens[:len(tokens) - tail])
    return ([label] if label else []) + tokens[len(tokens) - tail:]


def table_rows(lines: list[tuple[str, float, float, float]]) -> list[str] | None:
    """lines — строки одного блока PDF: (текст, x0, y0, y1). Возвращает строки таблицы с ячейками через « | »
    или None, если блок — не таблица с числами.

    Зачем: в обычном разборе строки блока склеиваются в абзац, и таблица превращается в «Density 0.60 0.79
    0.73 Entropy 0.65 …» без границ строк и столбцов, а блок из одних чисел отбрасывается как формула.
    Точные значения метрик из такого текста брать нельзя."""
    if len(lines) < MIN_TABLE_ROWS:
        return None
    ordered = sorted(lines, key=lambda line: ((line[2] + line[3]) / 2, line[1]))
    rows: list[list[tuple[str, float, float, float]]] = []
    for line in ordered:
        center = (line[2] + line[3]) / 2
        if rows:
            last = rows[-1][-1]
            if abs(center - (last[2] + last[3]) / 2) <= ROW_TOLERANCE * max(last[3] - last[2], 1.0):
                rows[-1].append(line)
                continue
        rows.append([line])
    cells = [_row_cells([text for text, *_ in sorted(row, key=lambda line: line[1])]) for row in rows]
    data_rows = sum(1 for row in cells if sum(_numeric(c) for c in row) >= 2)
    if data_rows < MIN_TABLE_ROWS or data_rows < MIN_TABLE_SHARE * len(cells):
        return None
    return [CELL_SEPARATOR.join(row) for row in cells]


def pdf_units(path: str | Path) -> tuple[list[Unit], int]:
    """Возвращает строки основного текста (и строки таблиц с kind="table") и число страниц."""
    raw = []  # (page, block, text, size, bold, y0, y1, page_height, x0)
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
                        raw.append((page.number + 1, block_id, text, size, bold, line["bbox"][1], line["bbox"][3],
                                    height, line["bbox"][0]))

    if not raw:
        return [], n_pages

    weights = Counter()
    for _, _, text, size, *_ in raw:
        weights[round(size * 2) / 2] += len(text)
    body_size = weights.most_common(1)[0][0]

    # Таблицы ищем до отсева мелкого шрифта: их часто набирают мельче основного текста.
    by_block: dict[int, list[tuple[str, float, float, float]]] = {}
    for _, block, text, _, _, y0, y1, _, x0 in raw:
        by_block.setdefault(block, []).append((text, x0, y0, y1))
    tables = {block: rows for block, rows in ((b, table_rows(lines)) for b, lines in by_block.items()) if rows}

    in_margin = [y1 < h * MARGIN_RATIO or y0 > h * (1 - MARGIN_RATIO) for *_, y0, y1, h, _ in raw]
    margin_pages: dict[str, set[int]] = {}
    for (page, _, text, *_), margin in zip(raw, in_margin):
        key = header_key(text)
        if margin and key:
            margin_pages.setdefault(key, set()).add(page)
    repeated = {key for key, pages in margin_pages.items() if len(pages) >= 2}

    units = []
    emitted: set[int] = set()
    for (page, block, text, size, bold, y0, _, height, _), margin in zip(raw, in_margin):
        if block in tables:
            if block not in emitted:
                emitted.add(block)
                units.extend(Unit(text=row, page=page, block=block, size=1.0, y=y0 / height, kind="table")
                             for row in tables[block])
            continue
        if size < body_size * MIN_SIZE_RATIO:
            continue
        # Номер страницы ищем только на полях: внутри страницы «1» отдельной строкой — номер раздела.
        if margin and (PAGE_NUMBER_RE.match(text) or header_key(text) in repeated):
            continue
        units.append(Unit(text=text, page=page, block=block, bold=bold, size=size / body_size, y=y0 / height))
    return units, n_pages
