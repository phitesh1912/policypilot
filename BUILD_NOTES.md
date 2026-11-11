# Build notes — bugs caught during AI-assisted development

Running log of real defects found in AI-generated code while building PolicyPilot, and the test
that catches each one. This feeds the README's "AI-assisted development" section. Only real
findings go here.

## 1. BM25 tokenizer split "401(k)" into `401` + `k` (2026-11-11)

- **Where:** `src/policypilot/retrieval/bm25.py`, `tokenize()`.
- **Symptom:** `test_real_corpus_sanity` — query *"401k employer match"* ranked the Remote Work
  policy above the US Payroll FAQ. The payroll doc writes "401(k)", which the regex tokenizer broke
  into two tokens, so the query token `401k` never matched. Separately, "match" never matched
  "matches" because there was no stemming at all.
- **Fix:** normalise bracketed suffixes glued to a word (`401(k)` → `401k`) before tokenizing, and
  add a deliberately light suffix stemmer (`matches`→`match`, `days`→`day`, `policies`→`policy`)
  that leaves short tokens like `pto` untouched.
- **Tests:** `tests/test_bm25.py::test_tokenize_normalises_bracketed_suffixes`,
  `::test_stem_is_conservative`, `::test_real_corpus_sanity`.
- **Lesson:** lexical retrieval is only as good as its tokenizer; a corpus-level sanity test on
  real questions caught what unit tests on toy strings could not.

## 2. Stemmer was inconsistent between inflections (2026-11-11)

- **Where:** `stem()` in `bm25.py`, first version.
- **Symptom:** `accrued` → `accru` but `accrue` → `accrue`, and `employees` → `employe` via a
  generic `es` rule. A query in the base form could never match a document in the past tense.
- **Fix:** restrict the `es` rule to the Porter cases (`ches/shes/xes/sses/zes`) and drop a
  trailing `e` after suffix removal so all inflections meet at one stem. Consistency matters more
  than linguistic correctness for a ranking function, as long as queries and docs get the same
  treatment.
- **Tests:** `tests/test_bm25.py::test_stem_collapses_inflections_consistently`.

## 3. The retrieval gate cannot detect unanswerable questions on its own (2026-11-11)

- **Where:** `Engine._gate` + `FakeProvider._answer`.
- **Symptom:** `test_unanswerable_question_is_refused` — *"What is the office wifi password?"*
  passed the fused-score gate because "office" put the hybrid-work section at the top of both
  retrievers, and the fake model then "answered" with the office-attendance rules.
- **Why it matters:** RRF scores are rank-based; rank 1 always exists, so a vocabulary overlap
  with the corpus looks identical to a real hit. The gate catches gibberish and off-domain
  questions cheaply; semantic unanswerability has to be judged by the model under the system
  prompt, and then verified by citation grounding.
- **Fix:** the fake now requires ≥2 overlapping tokens and ≥40% coverage of the question before it
  answers, mirroring the real model's instruction to refuse. The refusal-accuracy metric in the
  eval harness is what measures this layer for real providers.
- **Tests:** `tests/test_fake_provider.py::test_complete_refuses_on_a_single_shared_word`,
  `tests/test_engine.py::test_unanswerable_question_is_refused`.

## 4. Chunk-size comparison was a no-op (2026-11-11)

- **Where:** `eval/run.py` default `--chunk-sizes 350,700`.
- **Symptom:** the first eval run produced byte-identical rows for 350 and 700 tokens, both with
  87 chunks. No section in the handbook exceeds 350 tokens, so the windowing path never ran and
  the "two chunk sizes" comparison measured nothing.
- **Fix:** compare 150 vs 350 so the smaller size actually splits long sections. Lesson: check
  that an experimental variable changes the artefact (chunk count) before trusting the metric.
- **Tests:** none needed; the results table now shows different chunk counts per size.

## 5. Golden set had 51 items, not 50 (2026-11-11)

- **Where:** `eval/golden.json`.
- **Symptom:** `test_golden_set_is_well_formed_and_targets_real_sections` failed on the count.
  The same test also validates that every expected `doc:section` id exists in the real corpus,
  which is what makes the retrieval metrics trustworthy.
- **Fix:** dropped one redundant IN payroll question.
