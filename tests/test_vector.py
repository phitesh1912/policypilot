from policypilot.ingest.chunker import Chunker
from policypilot.llm.fake import FakeProvider
from policypilot.models import Region, Role, User
from policypilot.retrieval.vector import VectorRetriever, point_id
from tests.conftest import make_doc


def _retriever():
    docs = [
        make_doc("in-doc", "# L\n\n## Earned leave\n\nAccrue 1.5 days per month.", Region.IN),
        make_doc("us-doc", "# L\n\n## PTO\n\nAccrue 4.62 hours per pay period.", Region.US),
        make_doc("global", "# B\n\n## Wellness\n\nUSD 500 wellness allowance per year."),
    ]
    r = VectorRetriever(FakeProvider())
    r.index(Chunker().chunk_documents(docs))
    return r


def test_point_id_is_stable_uuid():
    assert point_id("a:b") == point_id("a:b")
    assert point_id("a:b") != point_id("a:c")
    assert len(point_id("x")) == 36


def test_index_and_count():
    r = _retriever()
    assert r.count() == 3


def test_reindex_replaces_old_points():
    r = _retriever()
    r.index(Chunker().chunk_documents([make_doc("only", "# T\n\n## S\n\nhello")]))
    assert r.count() == 1


def test_search_ranks_relevant_first_with_scores_and_ranks():
    r = _retriever()
    hits = r.search("wellness allowance", User(role=Role.EMPLOYEE, region=Region.IN), k=5)
    assert hits[0].chunk.doc_id == "global"
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))
    assert hits[0].source == "vector"


def test_search_enforces_region_prefilter():
    r = _retriever()
    in_user = User(role=Role.EMPLOYEE, region=Region.IN)
    us_user = User(role=Role.EMPLOYEE, region=Region.US)
    hr_user = User(role=Role.HR, region=Region.IN)
    assert {h.chunk.doc_id for h in r.search("accrue", in_user, k=10)} == {"in-doc", "global"}
    assert {h.chunk.doc_id for h in r.search("accrue", us_user, k=10)} == {"us-doc", "global"}
    assert {h.chunk.doc_id for h in r.search("accrue", hr_user, k=10)} == {
        "in-doc",
        "us-doc",
        "global",
    }


def test_search_on_empty_store_returns_nothing():
    r = VectorRetriever(FakeProvider(), collection="never-indexed")
    assert r.search("anything", User(), k=3) == []


def test_real_corpus_vector_sanity(chunks):
    r = VectorRetriever(FakeProvider())
    r.index(chunks)
    us_user = User(role=Role.EMPLOYEE, region=Region.US)
    hits = r.search("401k employer match vesting", us_user, k=5)
    assert any(h.chunk.doc_id == "payroll-faq-us" for h in hits)
    in_user = User(role=Role.EMPLOYEE, region=Region.IN)
    assert all(h.chunk.region != Region.US for h in r.search("401k employer match", in_user, k=20))
