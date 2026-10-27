"""История чатов веб-интерфейса: один чат — один JSON-файл в data/chats/.

В файле лежат готовые к показу ответы (AnswerView с карточками цитат), поэтому старый чат открывается
без модели и без пересчёта. Если статью потом переразобрать, ID предложений в старых чатах устареют,
но текст цитат и ссылки останутся такими, какими их видел пользователь.
"""

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from research_assistant.ui.views import AnswerView, CitationCard

TITLE_CHARS = 60
CHAT_ID_RE = re.compile(r"^[\w-]+$")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class Chat:
    id: str
    title: str
    created: str
    updated: str
    messages: list[AnswerView] = field(default_factory=list)

    def history(self, memory: int = 3) -> list[dict]:
        """Память агента как в команде chat: последние memory пар «вопрос — итоговый ответ»."""
        pairs = []
        for view in self.messages[-memory:] if memory > 0 else []:
            pairs += [{"role": "user", "content": view.question}, {"role": "assistant", "content": view.answer}]
        return pairs


def _view_from_dict(data: dict) -> AnswerView:
    claims = [
        (text, [CitationCard(**{**card, "context": tuple(card["context"]) if card.get("context") else None})
                for card in cards])
        for text, cards in data.get("claims", [])
    ]
    return AnswerView(**{**data, "claims": claims})


class ChatStore:
    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def _path(self, chat_id: str) -> Path:
        if not CHAT_ID_RE.match(chat_id):  # id приходит из состояния страницы: не даём выйти из каталога
            raise ValueError(f"bad chat id {chat_id!r}")
        return self.directory / f"{chat_id}.json"

    def new(self, first_question: str) -> Chat:
        title = " ".join(first_question.split())
        if len(title) > TITLE_CHARS:
            title = title[:TITLE_CHARS].rstrip() + "…"
        chat_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        now = _now()
        return Chat(chat_id, title or "Без названия", now, now)

    def save(self, chat: Chat) -> None:
        chat.updated = _now()
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(chat.id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(chat), ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)  # целиком или никак: обрыв посреди записи не портит чат

    def load(self, chat_id: str) -> Chat | None:
        path = self._path(chat_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return Chat(data["id"], data["title"], data["created"], data["updated"],
                    [_view_from_dict(m) for m in data.get("messages", [])])

    def list(self) -> list[Chat]:
        """Чаты, последние изменённые первыми. Повреждённые файлы пропускаются."""
        if not self.directory.exists():
            return []
        chats = [chat for path in self.directory.glob("*.json") if (chat := self.load(path.stem)) is not None]
        return sorted(chats, key=lambda c: c.updated, reverse=True)

    def delete(self, chat_id: str) -> None:
        self._path(chat_id).unlink(missing_ok=True)
