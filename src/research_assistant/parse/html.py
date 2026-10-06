"""Абзацы страницы статьи CyberLeninka: текст лежит в <div class="ocr"> по одному абзацу в <p>.

Номеров страниц в HTML нет, поэтому page = None. Колонтитулы сборника («European science № 4 (53) ■ 18»)
отсеиваются как повторяющиеся строки.
"""

import re
from collections import Counter

from bs4 import BeautifulSoup

from research_assistant.parse.units import Unit, clean_text, header_key

TABLE_CELL_RE = re.compile(r"^[\d\s.,%±+\-–—()]*$")


def _is_table_row(raw: str) -> bool:
    cells = [c.strip() for c in raw.split("\t") if c.strip()]
    if len(cells) < 3:
        return False
    numeric = sum(bool(TABLE_CELL_RE.match(c)) for c in cells)
    return numeric * 2 >= len(cells)


def cyberleninka_units(html: str) -> list[Unit]:
    soup = BeautifulSoup(html, "html.parser")
    container = soup.find("div", class_="ocr")
    if container is None:
        return []
    raw_paragraphs = [p.get_text(" ") for p in container.find_all("p")]
    texts = [(clean_text(raw), _is_table_row(raw)) for raw in raw_paragraphs]
    counts = Counter(header_key(text) for text, _ in texts if text and len(text) < 80)

    units = []
    for index, (text, table_row) in enumerate(texts):
        if not text or table_row:
            continue
        key = header_key(text) if len(text) < 80 else None
        if key and counts[key] >= 2:
            continue  # колонтитул сборника
        units.append(Unit(text=text, page=None, block=index, bold=False, size=1.0))
    return units
