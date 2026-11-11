"""Eval harness: retrieval quality, refusal behaviour and faithfulness across configurations.

    python -m eval.run                      # FakeProvider, offline, deterministic
    python -m eval.run --provider gemini    # needs GEMINI_API_KEY; uses an LLM judge

Compares BM25-only, vector-only and hybrid retrieval at two chunk sizes. Writes a markdown table
and a JSON dump to eval/results/. Numbers in the README come only from this script's output.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from eval.metrics import (
    Aggregate,
    citation_precision,
    heuristic_faithfulness,
    hit_at_k,
    reciprocal_rank,
    section_id,
)
from policypilot.config import Settings
from policypilot.engine import Engine
from policypilot.ingest.chunker import Chunker
from policypilot.llm.base import LLMProvider
from policypilot.llm.factory import build_provider
from policypilot.models import User
from policypilot.retrieval.base import Retriever
from policypilot.retrieval.bm25 import BM25Retriever
from policypilot.retrieval.hybrid import HybridRetriever
from policypilot.retrieval.vector import VectorRetriever

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "golden.json"
RESULTS = HERE / "results"

JUDGE_SYSTEM = (
    "You are a strict grader. Given CONTEXT and an ANSWER, reply with exactly YES if every factual "
    "claim in the ANSWER is supported by the CONTEXT, otherwise reply with exactly NO."
)


def llm_faithfulness(provider: LLMProvider, answer: str, cited_texts: list[str]) -> float:
    prompt = "CONTEXT:\n" + "\n---\n".join(cited_texts) + f"\n\nANSWER:\n{answer}\n\nVerdict:"
    verdict = provider.complete(JUDGE_SYSTEM, prompt).text.strip().upper()
    return 1.0 if verdict.startswith("YES") else 0.0


def build_retriever(kind: str, provider: LLMProvider, rrf_k: int) -> Retriever:
    if kind == "bm25":
        return BM25Retriever()
    if kind == "vector":
        return VectorRetriever(provider)
    if kind == "hybrid":
        return HybridRetriever(BM25Retriever(), VectorRetriever(provider), rrf_k=rrf_k)
    raise ValueError(kind)


def run_config(
    items: list[dict],
    provider: LLMProvider,
    retriever_kind: str,
    chunk_tokens: int,
    settings: Settings,
    use_llm_judge: bool,
) -> tuple[dict, list[dict]]:
    chunker = Chunker(chunk_tokens=chunk_tokens, overlap_tokens=min(50, chunk_tokens // 5))
    retriever = build_retriever(retriever_kind, provider, settings.rrf_k)
    engine = Engine(settings, provider=provider, retriever=retriever, chunker=chunker)
    # BM25-only has no fused score; its raw scores are on a different scale. Disable the
    # fused-score gate for the non-hybrid baselines so the comparison is about retrieval, and
    # leave refusal to the model + citation grounding layers.
    if retriever_kind != "hybrid":
        engine.settings = settings.model_copy(update={"refusal_threshold": 0.0})
    stats = engine.ingest()

    agg = Aggregate()
    rows: list[dict] = []
    for item in items:
        user = User(**item["user"])
        expected = set(item["expected"])
        t0 = time.perf_counter()
        result = engine.ask(item["question"], user)
        latency = (time.perf_counter() - t0) * 1000
        ranked = [t.chunk_id for t in result.retrieved]
        row = {
            "id": item["id"],
            "answerable": item["answerable"],
            "refused": result.refused,
            "top5": [section_id(c) for c in ranked[:5]],
            "citations": [c.chunk_id for c in result.citations],
            "answer": result.answer,
        }
        agg.total_cost += result.cost_usd
        agg.total_latency_ms += latency
        if item["answerable"]:
            agg.n_answerable += 1
            row["hit@5"] = hit_at_k(ranked, expected, 5)
            row["rr"] = reciprocal_rank(ranked, expected)
            agg.hit5 += row["hit@5"]
            agg.mrr += row["rr"]
            if result.refused:
                agg.false_refusals += 1
            else:
                cited_ids = [c.chunk_id for c in result.citations]
                cp = citation_precision(cited_ids, expected)
                if cp is not None:
                    agg.cite_prec.append(cp)
                cited_texts = _cited_texts(engine, result.citations)
                f = (
                    llm_faithfulness(provider, result.answer, cited_texts)
                    if use_llm_judge
                    else heuristic_faithfulness(result.answer, cited_texts)
                )
                row["faithful"] = f
                agg.faithful.append(f)
        else:
            agg.n_unanswerable += 1
            reason = item.get("refusal_reason", "out_of_scope")
            ok, total = agg.refusal_by_reason.get(reason, [0, 0])
            agg.refusal_by_reason[reason] = [ok + int(result.refused), total + 1]
            if result.refused:
                agg.correct_refusals += 1
            # An access-control case that is *not* refused is still fine only if nothing from
            # the forbidden region leaked; record that separately.
            if reason == "access":
                row["leaked_region"] = any(
                    c.region.value not in {r.value for r in user.visible_regions()}
                    for c in result.citations
                )
        rows.append(row)

    summary = {
        "retriever": retriever_kind,
        "chunk_tokens": chunk_tokens,
        "chunks": stats.chunks,
        **agg.summary(),
        "region_leaks": sum(1 for r in rows if r.get("leaked_region")),
    }
    return summary, rows


def _cited_texts(engine: Engine, citations) -> list[str]:
    """Pull the raw chunk texts for the cited sections from whichever retriever is active."""
    texts: list[str] = []
    bm25 = getattr(engine.retriever, "bm25", engine.retriever)
    chunks = getattr(bm25, "_chunks", None)
    if chunks is None:
        vec = getattr(engine.retriever, "vector", engine.retriever)
        # Vector-only: fetch payloads by scrolling; the corpus is tiny.
        points, _ = vec.client.scroll(vec.collection, limit=10_000, with_payload=True)
        chunks = [type("C", (), p.payload) for p in points]  # lightweight attribute access
    wanted = {section_id(c.chunk_id) for c in citations}
    for c in chunks:
        if section_id(c.chunk_id) in wanted:
            texts.append(c.text)
    return texts


def to_markdown(summaries: list[dict], provider_name: str, judge: str) -> str:
    cols = [
        "retriever",
        "chunk_tokens",
        "chunks",
        "hit@5",
        "mrr",
        "false_refusal_rate",
        "refusal_accuracy",
        "refusal_acc[access]",
        "refusal_acc[out_of_scope]",
        "region_leaks",
        "faithfulness",
        "citation_precision",
        "avg_latency_ms",
        "total_cost_usd",
    ]
    lines = [
        f"# Eval results — provider `{provider_name}`, judge: {judge}",
        "",
        f"Run at {datetime.now(UTC).isoformat(timespec='seconds')} over "
        f"{summaries[0]['answerable'] + summaries[0]['unanswerable']} golden questions "
        f"({summaries[0]['answerable']} answerable, {summaries[0]['unanswerable']} unanswerable).",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    for s in summaries:
        lines.append("| " + " | ".join(str(s.get(c, "")) for c in cols) + " |")
    lines += [
        "",
        "Metric notes: hit@5 / MRR are at *section* granularity (doc_id:section). "
        "`refusal_accuracy` is the share of unanswerable questions that got the refusal text; "
        "`region_leaks` counts answers that cited a document outside the user's region "
        "(must be 0).",
    ]
    if provider_name == "fake":
        lines += [
            "",
            "**Fake-provider caveat.** These numbers validate the pipeline end-to-end, not "
            "retrieval or answer quality. Fake embeddings are hashed lexical vectors (so "
            "`vector` ≈ a noisier BM25 and `hybrid` cannot beat BM25 here), the fake model "
            "answers whenever ≥40% of question tokens overlap a chunk (so refusal accuracy is "
            "a lower bound), and faithfulness is a numeric-grounding heuristic, not an LLM "
            "judge. Run with `--provider gemini` for real numbers.",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252, which cannot print the table's typographic characters.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="fake", choices=["fake", "gemini", "openai"])
    # 350 is the production size; 150 forces windowing of the longer sections. Larger sizes
    # (e.g. 700) produce the same chunks as 350 on this corpus because no section is that long.
    ap.add_argument("--chunk-sizes", default="150,350")
    ap.add_argument("--retrievers", default="bm25,vector,hybrid")
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args(argv)

    settings = Settings(llm_provider=args.provider)
    provider = build_provider(settings)
    use_llm_judge = provider.name != "fake"
    items = json.loads(GOLDEN.read_text(encoding="utf-8"))["items"]

    summaries: list[dict] = []
    all_rows: dict[str, list[dict]] = {}
    for chunk_tokens in (int(x) for x in args.chunk_sizes.split(",")):
        for kind in args.retrievers.split(","):
            print(f"running {kind:7s} chunk={chunk_tokens} ...", end=" ", flush=True)
            summary, rows = run_config(items, provider, kind, chunk_tokens, settings, use_llm_judge)
            summaries.append(summary)
            all_rows[f"{kind}-{chunk_tokens}"] = rows
            print(
                f"hit@5={summary['hit@5']} mrr={summary['mrr']} "
                f"refusal_acc={summary['refusal_accuracy']} leaks={summary['region_leaks']}"
            )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    judge = "LLM-as-judge" if use_llm_judge else "heuristic (numeric grounding)"
    md = to_markdown(summaries, provider.name, judge)
    (out / f"results_{provider.name}.md").write_text(md, encoding="utf-8", newline="\n")
    (out / f"results_{provider.name}.json").write_text(
        json.dumps({"summaries": summaries, "rows": all_rows}, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    print()
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
