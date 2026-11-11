import math

import pytest

from policypilot.models import Region, Role, User
from policypilot.retrieval.bm25 import BM25Index, BM25Retriever, stem, tokenize
from tests.conftest import make_doc


def test_tokenize_lowercases_drops_stopwords_and_stems():
    assert tokenize("How many Sick-Leave days do I get?") == ["many", "sick", "leav", "day", "get"]
    assert tokenize("Employees' PTO") == ["employe", "pto"]
    assert tokenize("employee") == ["employe"]


def test_tokenize_normalises_bracketed_suffixes():
    # Regression: "401(k)" used to split into "401" + "k", so the query "401k" never matched.
    assert tokenize("the 401(k) plan") == ["401k", "plan"]
    assert tokenize("401k") == ["401k"]


def test_stem_is_conservative():
    assert stem("matches") == "match"
    assert stem("policies") == "policy"
    assert stem("pto") == "pto"
    assert stem("days") == "day"
    assert stem("2026") == "2026"
    assert stem("hours") == "hour"


def test_stem_collapses_inflections_consistently():
    # Regression: an earlier version stemmed "accrued" -> "accru" but left "accrue" alone, so a
    # query using the base form could never match a document using the past tense.
    assert stem("accrue") == stem("accrued") == stem("accrues")
    assert stem("leave") == stem("leaves")
    assert stem("employee") == stem("employees")
    assert stem("match") == stem("matches") == stem("matching")


def test_idf_matches_formula():
    idx = BM25Index()
    idx.fit(["kiwi mango", "kiwi plum", "fig"])
    # 'kiwi' appears in 2 of 3 docs (idf takes an already-tokenized term, as score() passes it)
    assert idx.idf("kiwi") == pytest.approx(math.log((3 - 2 + 0.5) / (2 + 0.5) + 1))
    # unseen term gets the maximum idf
    assert idx.idf("zzz") == pytest.approx(math.log((3 + 0.5) / 0.5 + 1))
    # idf is never negative, even for a term in every doc
    idx.fit(["x", "x", "x"])
    assert idx.idf("x") >= 0


def test_score_matches_hand_computation():
    idx = BM25Index(k1=1.5, b=0.75)
    idx.fit(["cat cat dog", "dog bird"])  # lens 3,2 ; avgdl 2.5
    q = ["cat"]
    f, dl, avgdl = 2, 3, 2.5
    idf = math.log((2 - 1 + 0.5) / (1 + 0.5) + 1)
    expected = idf * f * (1.5 + 1) / (f + 1.5 * (1 - 0.75 + 0.75 * dl / avgdl))
    assert idx.score(q, 0) == pytest.approx(expected)
    assert idx.score(q, 1) == 0.0


def test_term_frequency_saturates():
    idx = BM25Index()
    idx.fit(["pay", "pay pay pay pay pay pay pay pay pay pay", "zzz"])
    s1, s10 = idx.scores("pay")[:2]
    assert s10 > s1
    # 10x the term frequency gives well under 10x the score
    assert s10 < 3 * s1


def test_length_normalisation_prefers_shorter_doc_with_same_tf():
    idx = BM25Index()
    idx.fit(["overtime rate", "overtime rate plus many many other unrelated words here"])
    short, long = idx.scores("overtime")
    assert short > long


def _corpus_retriever() -> BM25Retriever:
    from policypilot.ingest.chunker import Chunker

    docs = [
        make_doc("in-doc", "# L\n\n## Earned leave\n\nAccrue 1.5 days per month.", Region.IN),
        make_doc("us-doc", "# L\n\n## PTO\n\nAccrue 4.62 hours per pay period.", Region.US),
        make_doc("global", "# B\n\n## Wellness\n\nUSD 500 wellness allowance per year."),
    ]
    r = BM25Retriever()
    r.index(Chunker().chunk_documents(docs))
    return r


def test_retriever_ranks_relevant_chunk_first():
    r = _corpus_retriever()
    hits = r.search("wellness allowance", User(role=Role.EMPLOYEE, region=Region.IN), k=5)
    assert hits[0].chunk.doc_id == "global"
    assert hits[0].rank == 1
    assert hits[0].source == "bm25"


def test_retriever_enforces_region_access():
    r = _corpus_retriever()
    in_user = User(role=Role.EMPLOYEE, region=Region.IN)
    us_user = User(role=Role.EMPLOYEE, region=Region.US)
    hr_user = User(role=Role.HR, region=Region.IN)

    assert {h.chunk.doc_id for h in r.search("accrue", in_user)} == {"in-doc"}
    assert {h.chunk.doc_id for h in r.search("accrue", us_user)} == {"us-doc"}
    assert {h.chunk.doc_id for h in r.search("accrue", hr_user)} == {"in-doc", "us-doc"}


def test_retriever_drops_zero_score_and_respects_k():
    r = _corpus_retriever()
    assert r.search("quantum chromodynamics", User(), k=5) == []
    assert len(r.search("accrue leave pto wellness", User(role=Role.HR), k=1)) == 1


def test_real_corpus_sanity(chunks):
    r = BM25Retriever()
    r.index(chunks)
    in_user = User(role=Role.EMPLOYEE, region=Region.IN)
    top = r.search("how many days of casual leave do I get", in_user, k=3)
    assert top[0].chunk.doc_id == "leave-policy-in"
    assert top[0].chunk.section_slug == "casual-leave-cl"
    us_user = User(role=Role.EMPLOYEE, region=Region.US)
    top = r.search("401k employer match", us_user, k=3)
    assert top[0].chunk.doc_id == "payroll-faq-us"
    # An IN employee asking about US-only content must get nothing from US docs
    for hit in r.search("401k employer match", in_user, k=10):
        assert hit.chunk.region != Region.US
