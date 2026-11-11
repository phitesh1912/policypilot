import pytest

from policypilot.ingest.chunker import Chunker
from policypilot.llm.fake import FakeProvider
from policypilot.models import Region, RetrievedChunk, Role, User
from policypilot.retrieval.bm25 import BM25Retriever
from policypilot.retrieval.hybrid import HybridRetriever, rrf_fuse
from policypilot.retrieval.vector import VectorRetriever
from tests.conftest import make_doc


def _rc(chunk, rank, source, score=1.0):
    return RetrievedChunk(chunk=chunk, score=score, rank=rank, source=source)


@pytest.fixture
def three_chunks():
    docs = [make_doc(f"d{i}", f"# T\n\n## S\n\nbody {i}") for i in range(3)]
    return Chunker().chunk_documents(docs)


def test_rrf_scores_match_formula(three_chunks):
    a, b, c = three_chunks
    fused = rrf_fuse([[_rc(a, 1, "bm25"), _rc(b, 2, "bm25")], [_rc(b, 1, "vector")]], k=60)
    by_id = {f.chunk.chunk_id: f for f in fused}
    assert by_id[b.chunk_id].score == pytest.approx(1 / 62 + 1 / 61)
    assert by_id[a.chunk_id].score == pytest.approx(1 / 61)
    assert c.chunk_id not in by_id


def test_rrf_prefers_agreement_over_single_top_rank(three_chunks):
    a, b, _ = three_chunks
    # a is #1 for bm25 only; b is #2 for bm25 and #1 for vector -> b wins
    fused = rrf_fuse([[_rc(a, 1, "bm25"), _rc(b, 2, "bm25")], [_rc(b, 1, "vector")]])
    assert fused[0].chunk.chunk_id == b.chunk_id
    assert fused[0].rank == 1
    assert fused[0].source == "hybrid"
    assert fused[0].sources == {"bm25": 2, "vector": 1}


def test_rrf_is_deterministic_on_ties(three_chunks):
    a, b, _ = three_chunks
    fused = rrf_fuse([[_rc(a, 1, "bm25")], [_rc(b, 1, "vector")]])
    assert [f.chunk.chunk_id for f in fused] == sorted([a.chunk_id, b.chunk_id])


def test_rrf_ignores_raw_scores(three_chunks):
    a, b, _ = three_chunks
    fused = rrf_fuse([[_rc(a, 1, "bm25", score=1000.0), _rc(b, 2, "bm25", score=0.001)]])
    assert fused[0].score == pytest.approx(1 / 61)
    assert fused[1].score == pytest.approx(1 / 62)


def _hybrid(chunks):
    h = HybridRetriever(BM25Retriever(), VectorRetriever(FakeProvider()), rrf_k=60)
    h.index(chunks)
    return h


def test_hybrid_real_corpus_finds_answer_and_respects_access(chunks):
    h = _hybrid(chunks)
    in_user = User(role=Role.EMPLOYEE, region=Region.IN)
    hits = h.search("how many casual leave days do I get", in_user, k=5)
    assert hits[0].chunk.doc_id == "leave-policy-in"
    assert hits[0].chunk.section_slug == "casual-leave-cl"
    assert hits[0].source == "hybrid"
    assert set(hits[0].sources) == {"bm25", "vector"}
    assert all(h_.chunk.region != Region.US for h_ in h.search("401k match", in_user, k=20))


def test_hybrid_rank_limited_to_k(chunks):
    h = _hybrid(chunks)
    assert len(h.search("leave", User(role=Role.HR), k=3)) == 3


def test_hybrid_accepts_precomputed_query_vector(chunks):
    h = _hybrid(chunks)
    provider = h.vector.provider
    q = "wellness allowance"
    vec = provider.embed_one(q)
    before = len(provider.calls)
    a = h.search(q, User(), k=3, query_vector=vec)
    assert len([c for c in provider.calls[before:] if c[0] == "embed"]) == 0
    b = h.search(q, User(), k=3)
    assert [x.chunk.chunk_id for x in a] == [x.chunk.chunk_id for x in b]
