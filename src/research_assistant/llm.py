"""Чат-модель через OpenAI-совместимый API и учёт токенов из usage."""

from dataclasses import dataclass


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, usage) -> None:
        self.calls += 1
        if usage is not None:
            self.prompt_tokens += usage.prompt_tokens or 0
            self.completion_tokens += usage.completion_tokens or 0


class ChatModel:
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        client=None,
    ):
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=base_url, api_key=api_key)
        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, messages: list[dict], tools: list[dict] | None = None):
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if self.reasoning_effort:
            # В LM Studio reasoning_effort=none отключает «размышления» Qwen3.5: иначе короткие шаги
            # агента тратят сотни токенов на рассуждение.
            kwargs["extra_body"] = {"reasoning_effort": self.reasoning_effort}
        return self.client.chat.completions.create(**kwargs)
