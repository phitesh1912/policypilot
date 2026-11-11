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
