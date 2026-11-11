"""Pure metric functions, unit-tested in tests/test_eval_metrics.py."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from policypilot.retrieval.bm25 import tokenize

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
_TAG_RE = re.compile(r"\[[a-z0-9-]+:[a-z0-9-]+\]")


def section_id(chunk_id: str) -> str:
    """'doc:section-2' -> 'doc:section' (windowed chunks share their section's identity)."""
    return re.sub(r"-\d+$", "", chunk_id)


def hit_at_k(ranked: Sequence[str], expected: set[str], k: int) -> float:
    return 1.0 if any(section_id(c) in expected for c in ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[str], expected: set[str]) -> float:
    for i, c in enumerate(ranked, start=1):
        if section_id(c) in expected:
            return 1.0 / i
    return 0.0


def citation_precision(cited: Sequence[str], expected: set[str]) -> float | None:
    if not cited:
        return None
    return sum(1 for c in cited if section_id(c) in expected) / len(cited)


def heuristic_faithfulness(answer: str, cited_texts: Sequence[str]) -> float:
    """Offline stand-in for an LLM judge. Every number in the answer must appear verbatim in the
    cited context; if the answer has no numbers, at least half of its content tokens must."""
    body = _TAG_RE.sub("", answer)
    context = " ".join(cited_texts)
    numbers = _NUM_RE.findall(body)
    if numbers:
        return 1.0 if all(n in context for n in numbers) else 0.0
    toks = set(tokenize(body))
    if not toks:
        return 1.0
    ctx = set(tokenize(context))
    return 1.0 if len(toks & ctx) / len(toks) >= 0.5 else 0.0


@dataclass
class Aggregate:
    n_answerable: int = 0
    n_unanswerable: int = 0
    hit5: float = 0.0
    mrr: float = 0.0
    false_refusals: int = 0
    correct_refusals: int = 0
    refusal_by_reason: dict[str, list[int]] = field(default_factory=dict)  # reason -> [ok, total]
    faithful: list[float] = field(default_factory=list)
    cite_prec: list[float] = field(default_factory=list)
    total_cost: float = 0.0
    total_latency_ms: float = 0.0

    def summary(self) -> dict[str, float | int | None]:
        n_a = max(self.n_answerable, 1)
        n_u = max(self.n_unanswerable, 1)
        return {
            "answerable": self.n_answerable,
            "unanswerable": self.n_unanswerable,
            "hit@5": round(self.hit5 / n_a, 3),
            "mrr": round(self.mrr / n_a, 3),
            "false_refusal_rate": round(self.false_refusals / n_a, 3),
            "refusal_accuracy": round(self.correct_refusals / n_u, 3),
            **{
                f"refusal_acc[{r}]": round(ok / max(t, 1), 3)
                for r, (ok, t) in sorted(self.refusal_by_reason.items())
            },
            "faithfulness": round(sum(self.faithful) / len(self.faithful), 3)
            if self.faithful
            else None,
            "citation_precision": round(sum(self.cite_prec) / len(self.cite_prec), 3)
            if self.cite_prec
            else None,
            "avg_latency_ms": round(self.total_latency_ms / (self.n_answerable + n_u), 1),
            "total_cost_usd": round(self.total_cost, 6),
        }
