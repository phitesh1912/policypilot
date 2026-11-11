"""Token budget packing and cost estimation."""

from __future__ import annotations

from dataclasses import dataclass

from policypilot.models import RetrievedChunk, TokenUsage


def pack(hits: list[RetrievedChunk], budget_tokens: int) -> tuple[list[RetrievedChunk], int]:
    """Greedy first-fit in rank order: take each chunk if it still fits, otherwise skip it and keep
    going (a later, smaller chunk may still fit). Returns (selected, tokens_used).

    Greedy-by-rank is the right objective here: rank already encodes relevance, and the context
    window is a hard cap, so we never want to drop a rank-1 chunk to fit two rank-8 chunks.
    """
    selected: list[RetrievedChunk] = []
    used = 0
    for h in hits:
        if h.chunk.token_count <= budget_tokens - used:
            selected.append(h)
            used += h.chunk.token_count
    return selected, used


@dataclass(frozen=True)
class Pricing:
    """USD per 1M tokens."""

    input_per_1m: float
    output_per_1m: float
    embed_per_1m: float

    def cost(self, usage: TokenUsage) -> float:
        return round(
            usage.prompt * self.input_per_1m / 1e6
            + usage.completion * self.output_per_1m / 1e6
            + usage.embedding * self.embed_per_1m / 1e6,
            8,
        )
