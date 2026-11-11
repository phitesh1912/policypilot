# Eval results — provider `fake`, judge: heuristic (numeric grounding)

Run at 2026-11-11T09:23:10+00:00 over 50 golden questions (40 answerable, 10 unanswerable).

| retriever | chunk_tokens | chunks | hit@5 | mrr | false_refusal_rate | refusal_accuracy | refusal_acc[access] | refusal_acc[out_of_scope] | region_leaks | faithfulness | citation_precision | avg_latency_ms | total_cost_usd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 150 | 88 | 1.0 | 0.952 | 0.0 | 0.4 | 0.25 | 0.5 | 0 | 1.0 | 0.9 | 5.7 | 0.011571 |
| vector | 150 | 88 | 0.975 | 0.729 | 0.0 | 0.7 | 0.5 | 0.833 | 0 | 1.0 | 0.85 | 7.7 | 0.012464 |
| hybrid | 150 | 88 | 0.975 | 0.911 | 0.0 | 0.5 | 0.25 | 0.667 | 0 | 1.0 | 0.9 | 9.5 | 0.012287 |
| bm25 | 350 | 87 | 1.0 | 0.954 | 0.0 | 0.4 | 0.25 | 0.5 | 0 | 1.0 | 0.9 | 3.6 | 0.011695 |
| vector | 350 | 87 | 0.95 | 0.704 | 0.025 | 0.7 | 0.5 | 0.833 | 0 | 1.0 | 0.846 | 6.9 | 0.012276 |
| hybrid | 350 | 87 | 0.975 | 0.895 | 0.0 | 0.5 | 0.25 | 0.667 | 0 | 1.0 | 0.9 | 7.7 | 0.012406 |

Metric notes: hit@5 / MRR are at *section* granularity (doc_id:section). `refusal_accuracy` is the share of unanswerable questions that got the refusal text; `region_leaks` counts answers that cited a document outside the user's region (must be 0).

**Fake-provider caveat.** These numbers validate the pipeline end-to-end, not retrieval or answer quality. Fake embeddings are hashed lexical vectors (so `vector` ≈ a noisier BM25 and `hybrid` cannot beat BM25 here), the fake model answers whenever ≥40% of question tokens overlap a chunk (so refusal accuracy is a lower bound), and faithfulness is a numeric-grounding heuristic, not an LLM judge. Run with `--provider gemini` for real numbers.
