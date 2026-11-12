# Eval results — provider `groq+fastembed` (chat: openai/gpt-oss-20b, embeddings: BAAI/bge-small-en-v1.5), judge: LLM-as-judge (openai/gpt-oss-20b)

Run at 2026-11-12T04:31:50+00:00 over 50 golden questions (40 answerable, 10 unanswerable).

| retriever | chunk_tokens | chunks | mode | hit@5 | mrr | false_refusal_rate | refusal_accuracy | refusal_acc[access] | refusal_acc[out_of_scope] | region_leaks | faithfulness | citation_precision | avg_latency_ms | total_cost_usd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 150 | 88 | retrieval-only | 1.0 | 0.952 |  |  |  |  |  |  |  |  |  |
| vector | 150 | 88 | retrieval-only | 1.0 | 0.963 |  |  |  |  |  |  |  |  |  |
| hybrid | 150 | 88 | retrieval-only | 1.0 | 0.963 |  |  |  |  |  |  |  |  |  |
| bm25 | 350 | 87 | retrieval-only | 1.0 | 0.954 |  |  |  |  |  |  |  |  |  |
| vector | 350 | 87 | retrieval-only | 1.0 | 0.95 |  |  |  |  |  |  |  |  |  |
| hybrid | 350 | 87 | full | 1.0 | 0.963 | 0.0 | 1.0 | 1.0 | 1.0 | 0 | 0.95 | 0.988 | 1343.1 | 0.013746 |

Metric notes: hit@5 / MRR are at *section* granularity (doc_id:section). `refusal_accuracy` is the share of unanswerable questions that got the refusal text; `region_leaks` counts answers that cited a document outside the user's region (must be 0). Rows in `retrieval-only` mode score retrieval without calling the model, so their answer-level columns are blank.
