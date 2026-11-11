# PolicyPilot — Interview Prep

Everything a reviewer might ask about this project, with the honest answer. Numbers quoted here
are from `eval/results/results_fake.md` unless stated otherwise.

## Part 1 — Design decisions

### Why RAG at all, instead of fine-tuning or just asking the model?

HR policy changes (new leave year, new tax regime) and differs by region. Fine-tuning bakes stale
facts into weights and can't tell you *where* an answer came from. RAG keeps the source of truth in
documents that HR already maintains, lets us cite them, lets us re-ingest in seconds, and — the
part that matters most here — lets us apply **per-user access control at retrieval time**. You
can't filter a fine-tuned model by region.

### Why hybrid retrieval (BM25 + vectors) rather than one or the other?

They fail differently. BM25 is exact on rare tokens — `401(k)`, `FMLA`, `A1 to A3`, `Form 16` —
which embeddings blur. Embeddings catch paraphrase — "time off when my dad died" → bereavement
leave — which BM25 misses completely. HR questions contain both kinds of signal. Hybrid is
cheap insurance: BM25 is in-memory and microseconds.

Honest caveat from the eval: with the fake provider, BM25 alone has the best MRR (0.954) and
hybrid is second (0.895). That is because fake embeddings *are* hashed lexical vectors, so the
dense side adds noise, not paraphrase understanding. The comparison that matters is the Gemini
run, which hasn't been executed yet. I'd rather show that than make up a number.

### Why Reciprocal Rank Fusion, and why k=60?

BM25 scores (unbounded, corpus-dependent) and cosine similarities (−1..1) are not on the same
scale, so you can't add them. RRF only uses *ranks*: `score(d) = Σ 1/(k + rank_r(d))`. It needs no
normalisation and no tuning per corpus, and it rewards documents both retrievers agree on. k=60 is
the value from Cormack, Clarke & Buettcher (2009); larger k flattens the difference between rank 1
and rank 2, smaller k makes rank 1 dominate. I kept the paper's value and tested the formula
directly (`tests/test_hybrid.py::test_rrf_scores_match_formula`).

A side benefit I use: with k=60, rank 1 in *one* list scores 0.0164, rank 1 in *both* scores
0.0328. So a fused threshold of 0.02 reads as "both retrievers agree this is relevant" — that's the
retrieval gate.

### Why chunk by headings, and why 350 tokens with 50 overlap?

Policy documents are already structured: one heading = one rule. A chunk that is exactly one
section is self-contained, has a natural citation id (`doc:section-slug`), and doesn't straddle two
unrelated rules. Fixed-size chunking would split "Earned leave" mid-sentence and produce chunks
that start with "…up to a maximum of 45 days" with no subject.

350 tokens is a ceiling, not a target: most sections are 60–150 tokens and become one chunk; only
sections longer than the ceiling are windowed, with 50 tokens of overlap so no sentence is cut
without appearing whole in a neighbour. The eval showed 150 vs 350 differs by a single chunk
(88 vs 87) on this corpus — the honest conclusion is that chunk size barely matters *for this
corpus* because the sections are short. It would matter for long, flat documents.

Trade-offs: bigger chunks → more context per hit, fewer hits fit the budget, citations less
precise. Smaller → more precise citations, more risk the answer spans two chunks.

### Why prefix each chunk with its heading path?

The embedding and BM25 both "see" `Leave Policy — India > Sick leave (SL)` before the body. A
body like "12 days per leave year" is meaningless on its own; with the prefix, both retrievers
know what it's 12 days *of*. The prefix is excluded from the window budget so it never eats the
body.

### How does the refusal path work, and why three layers?

1. **Retrieval gate** — if the top fused score is below θ (0.02), the model isn't called. Cheap,
   deterministic, catches gibberish and off-domain questions.
2. **Prompt** — the system prompt says: answer only from context, tag every factual sentence,
   otherwise reply exactly *"I don't know — please contact HR."*
3. **Citation validation** — tags in the output are checked against the tags actually supplied.
   Unknown tags are stripped; an answer with *no* valid tag is replaced by the refusal.

Why three: the gate can't detect semantic unanswerability (bug #3 in BUILD_NOTES: "office wifi
password" shares the word "office" with the hybrid-work section and ranks first in both lists).
The prompt can be ignored by a weak model. The post-check is the one layer the model can't talk
its way past, but it only works if the first two give it something to check.

### Why is access control inside retrieval instead of a filter on the results?

If you take top-10 then filter, a US-region user could get 10 US chunks, an IN user filtering the
same top-10 could get *zero* results and a refusal — or worse, with a post-filter bug, a leak. By
filtering inside BM25 scoring and as a Qdrant `query_filter`, forbidden chunks never compete for a
slot. The model never sees them, so there's nothing to leak even under prompt injection. The eval
has a `region_leaks` column precisely to prove this stays zero (0 across 300 answers).

### Why does the semantic cache need a scope?

A semantic cache keyed only by query embedding is a cross-tenant leak: a US employee asks "how
much sick leave do I get", the answer (80 hours) is cached, an IN employee asks the same words and
gets the US answer. The cache key is `(embedding, frozenset(visible_regions))`. Same words,
different scope → miss. Tested in `test_cache.py` and `test_engine.py`.

### Why implement the LRU cache by hand?

The JD asks for DSA. A dict gives O(1) lookup by key; a doubly linked list gives O(1) move-to-front
and O(1) evict-from-tail. The *semantic* lookup is O(n·d) — a scan over n cached vectors — which is
fine for n ≤ 256 (microseconds vs an LLM round trip). If n grew large the scan would move into the
vector DB as a second collection. `functools.lru_cache` can't do similarity lookup or scoping.

### Why a token budget packer, and why greedy?

The context window is a hard cap and a cost. Rank already encodes relevance, so greedy first-fit
in rank order is the right objective: never drop the rank-1 chunk to fit two rank-8 chunks. First-fit
(skip what doesn't fit, keep going) beats first-fail (stop at the first that doesn't fit) because
a later, smaller chunk may still add value. It's O(n) and trivially testable.

### Why a provider interface with a fake?

Three reasons. CI has no API key. Tests must be deterministic. And the demo must work offline.
`FakeProvider` embeds by hashing stemmed tokens into fixed pseudo-random unit vectors and
"answers" by picking the context block with the most token overlap (≥2 tokens and ≥40% of the
question) and returning its first sentences with the block's citation tag. It exercises every
code path — cache, RRF, gate, grounding, streaming, refusal — without a network call.

### Why is BM25 rebuilt at startup?

It's in-memory and the corpus is 87 chunks; rebuilding takes milliseconds. Persisting it would add
a serialization format and a staleness problem for no gain at this size. The embedded Qdrant store
*is* persisted on disk; with a real provider, startup re-embeds the corpus (a few seconds, ~87
embedding calls). The obvious next step is a content hash per chunk to skip unchanged ones.

### What fails, and how would you improve it?

- **Multi-hop questions** ("can I encash the leave I carried forward?") need two sections; the
  packer includes both only if both rank in the top-k. Fix: query decomposition or a reranker.
- **Numbers inside tables** (the US PTO accrual table) — BM25 handles them, embeddings may not.
  Fix: table-aware chunking that repeats the header row per chunk.
- **Refusal threshold is a global constant.** Fix: calibrate θ per corpus from the eval, or
  replace with a small cross-encoder relevance score.
- **Stemmer is heuristic.** Fine for HR vocabulary, wrong for general text. Swap for Snowball if
  the corpus broadens.
- **Identity comes from the request body.** In production, from the JWT; the `User` model and
  access logic are unchanged.
- **No reranker.** A cross-encoder over the fused top-20 is the single biggest quality lever left.
- **Faithfulness judge** is heuristic offline, LLM-as-judge online; both can be fooled. A
  human-labelled subset is the real fix.

## Part 2 — 25 likely interview questions

**1. Walk me through what happens when I call `/ask`.**
Embed the question once → scoped semantic-cache lookup → BM25 and vector search in parallel, each
region-filtered → RRF fuse → gate on fused score → pack top chunks into the token budget → model
answers under a citation-forcing prompt → validate citations → cache → return answer, citations,
token counts, cost, retrieval trace, latency.

**2. What is BM25 and how did you implement it?**
Okapi BM25 scores a document for a query as Σ over query terms of IDF × tf·(k1+1) / (tf + k1·(1−b+b·|d|/avgdl)).
`k1=1.5` controls term-frequency saturation, `b=0.75` controls length normalisation. I store per-doc
term frequencies, document frequencies and average length; tokenization lowercases, normalises
`401(k)` → `401k`, drops a small stopword list, and applies a light stemmer. Unit tests check the
formula against hand computation, saturation, and length normalisation.

**3. What does IDF's "+1" do?**
The classic IDF `log((N−df+0.5)/(df+0.5))` goes negative for terms in more than half the documents.
The `+1` inside the log keeps it ≥ 0 so very common terms contribute nothing rather than penalising.

**4. Why did your tokenizer need a stemmer? What went wrong?**
"match" didn't match "matches"; "accrued" didn't match "accrue". My first stemmer was inconsistent
(`accrued → accru`, `accrue → accrue`), which a test caught. The fix was consistency over
correctness: apply the same crude rules to queries and documents.

**5. Explain cosine similarity and why vectors are normalised.**
Cosine = dot(a,b)/(|a||b|). If both are unit-length, cosine is just the dot product — cheaper, and
Qdrant's `COSINE` distance assumes it. Every provider normalises on the way out.

**6. What is an embedding? What are its limits?**
A dense vector where semantic similarity ≈ geometric closeness. Limits: exact tokens (IDs,
acronyms, numbers) are blurred; long chunks dilute; the model's training data decides what counts
as "similar". That's why BM25 is kept alongside.

**7. What's RRF and why not weighted score fusion?**
Rank-only fusion, `Σ 1/(k+rank)`. Weighted score fusion needs the scores on one scale, which
requires per-corpus normalisation that drifts as the corpus changes. RRF has one parameter and no
calibration.

**8. How do you stop the model hallucinating?**
Three layers: don't call it when retrieval is weak; instruct it to cite and refuse; verify the
citations it produced against the chunks it was given, and refuse if none are valid. Then
*measure* it — faithfulness and refusal accuracy are in the eval.

**9. How does the system handle "I'm from India, show me US payroll"?**
It can't retrieve US chunks for an IN user, so the model never sees them. The eval includes this
as an explicit access case and counts region leaks (0).

**10. What is prompt injection and what's your defence?**
Untrusted text (the question, or a document) containing instructions to the model. Defence in
depth: structural (the data the attacker wants is never retrieved), prompt (rule 4 says ignore
embedded instructions), and output validation (citations must match supplied tags). Tests cover
"ignore all previous instructions…" at the engine and API layers.

**11. Why is the cache keyed by embedding and not the string?**
"How many sick days do I get" and "sick leave entitlement?" should hit the same entry. Cosine ≥
0.95 captures near-paraphrases; the scope key prevents cross-region reuse.

**12. Complexity of your cache operations?**
`put`: O(1) amortised (dict insert + list push + possible tail evict). `get`: O(n·d) scan for the
best cosine, then O(1) move-to-front. n is capped at 256.

**13. What's a token? Why count them?**
The model's unit of text (~¾ of a word in English). The context window is a token cap, and
pricing is per token. `tiktoken` gives exact counts for OpenAI-style tokenizers; for Gemini it's a
close proxy — good enough for budgeting, and the real usage comes back in the response.

**14. How did you choose the context budget of 1800 tokens?**
Roughly 5–8 section chunks. Enough for a multi-part answer, small enough to keep latency and cost
low on a free-tier model. It's an env var; the UI shows the actual packed count per query.

**15. Why stream? How?**
Perceived latency. Server-Sent Events over a `StreamingResponse`: `meta` (retrieval trace, so the
UI can show what was found before the first token), `token`*, `done` (the fully grounded result —
the client replaces the streamed text with it, because citation validation runs on the full text).
The gate runs before any token, so a refusal never starts as a half-answer.

**16. How would you deploy and scale this?**
Single container (Dockerfile, non-root, tiktoken baked in) on Render. Scaling: stateless API
behind a load balancer; move Qdrant to Qdrant Cloud (one env var), move BM25 to a shared store
or accept per-replica rebuild, move the cache to Redis with the same scope key.

**17. How do you keep secrets out of the repo?**
`.env` is git-ignored; `render.yaml` marks `GEMINI_API_KEY` as `sync: false` and generates
`ADMIN_TOKEN`; tests never need a key because of the fake provider.

**18. What did the eval teach you?**
That chunk size 350 vs 700 was a no-op on this corpus (identical rows) — always check the variable
changes the artefact. That region isolation holds (0 leaks). That a rank-based gate can't judge
semantic unanswerability. And that fake-provider numbers must be labelled as pipeline validation,
not quality.

**19. What's MRR and why use it alongside hit@5?**
Mean reciprocal rank = average of 1/rank of the first correct result. hit@5 says "was it in the
window"; MRR says "how close to the top". A reranker would move MRR more than hit@5.

**20. What's the difference between `vector` and `hybrid` being close in hit@5 but far in MRR?**
Both find the right section somewhere in the top 5 most of the time, but BM25's exact-token
signal puts it at rank 1 more often. RRF recovers most of that (0.895 vs 0.704 for vector-only).

**21. Where would a reranker go?**
After RRF, over the fused top-20: a cross-encoder scores (question, chunk) pairs jointly, which is
far more accurate than bi-encoder cosine, and is affordable on 20 items.

**22. What would you do with PDFs that have no headings?**
The loader already takes PDFs with a metadata sidecar. Without headings, `split_sections` falls
back to one section (the title), which then gets windowed. Better: detect heading-like lines by
font size via `pypdf`'s layout mode, or fall back to sentence-boundary windows.

**23. How do you know your access control works?**
Unit tests at the retriever level (BM25 and vector), engine level (cache scoping, injection), API
level, and an eval column that would turn non-zero on regression. Plus the design: it's a
pre-filter, so the failure mode is "too few results", not "leak".

**24. What did you *not* let the AI assistant do?**
Write eval numbers into the README; "fix" lint by rewriting FastAPI dependency injection; claim
the hybrid retriever beats BM25 when the fake-provider run says otherwise. Each generated module
was accepted only once its tests passed and the failures were understood (BUILD_NOTES has five).

**25. If you had one more week?**
In order: run the Gemini eval and tune θ from it; add a cross-encoder reranker; content-hash
chunks to skip re-embedding; a human-labelled faithfulness subset; JWT-based identity.
