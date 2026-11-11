"""Provider-agnostic LLM + embedding interface.

The rest of the app only ever talks to ``LLMProvider``. Swapping Gemini for OpenAI, or for the
deterministic ``FakeProvider`` used in tests and the offline demo, is a config change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

# The answer prompt marks every retrieved chunk with this header so that (a) the model can cite
# it and (b) FakeProvider can parse the context back out without any model at all.
CONTEXT_HEADER = "### Context "
QUESTION_HEADER = "### Question"
REFUSAL_TEXT = "I don't know — please contact HR."


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int


class LLMProvider(ABC):
    name: str
    chat_model: str
    embed_model: str
    embed_dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts``; returns one unit-length vector per input, in order."""

    @abstractmethod
    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> Completion:
        """Single-shot chat completion."""

    @abstractmethod
    def stream(self, system: str, user: str, *, temperature: float = 0.0) -> Iterator[str]:
        """Yield the completion text incrementally."""

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
