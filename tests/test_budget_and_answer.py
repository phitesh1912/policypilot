import pytest

from policypilot.answer import CITATION_RE, build_prompt, ground
from policypilot.budget import Pricing, pack
from policypilot.ingest.chunker import Chunker
from policypilot.llm.base import CONTEXT_HEADER, QUESTION_HEADER, REFUSAL_TEXT
from policypilot.models import Region, RetrievedChunk, TokenUsage
from tests.conftest import make_doc


def _hits(sizes):
    docs = [make_doc(f"d{i}", "# T\n\n## S\n\n" + "w " * n) for i, n in enumerate(sizes)]
    chunks = Chunker(chunk_tokens=2000, overlap_tokens=10).chunk_documents(docs)
    out = []
    for r, c in enumerate(chunks, start=1):
        # Force exact token counts so the test reasons about sizes, not tokenizer quirks.
        c = c.model_copy(update={"token_count": sizes[r - 1]})
        out.append(RetrievedChunk(chunk=c, score=1 / r, rank=r, source="hybrid"))
    return out


def test_pack_is_greedy_first_fit_in_rank_order():
    hits = _hits([300, 500, 200, 100])
    selected, used = pack(hits, budget_tokens=650)
    assert [h.chunk.doc_id for h in selected] == ["d0", "d2", "d3"]  # 500 skipped, later ones fit
    assert used == 600


def test_pack_never_exceeds_budget_and_keeps_rank_one_when_possible():
    hits = _hits([400, 400, 400])
    selected, used = pack(hits, 450)
    assert [h.chunk.doc_id for h in selected] == ["d0"]
    assert used == 400
    assert pack(hits, 10) == ([], 0)


def test_pricing():
    p = Pricing(input_per_1m=0.30, output_per_1m=2.50, embed_per_1m=0.15)
    assert p.cost(TokenUsage(prompt=1_000_000)) == pytest.approx(0.30)
    assert p.cost(TokenUsage(completion=400_000)) == pytest.approx(1.0)
    assert p.cost(TokenUsage(prompt=1000, completion=100, embedding=20)) == pytest.approx(
        0.0003 + 0.00025 + 0.000003
    )


def test_build_prompt_layout():
    hits = _hits([10, 10])
    prompt = build_prompt("What?", hits)
    assert prompt.count(CONTEXT_HEADER) == 2
    assert prompt.index(CONTEXT_HEADER) < prompt.index(QUESTION_HEADER)
    assert prompt.endswith("What?")
    assert "[d0:s]" in prompt


def test_citation_regex():
    assert CITATION_RE.findall("x [leave-policy-in:earned-leave-el]. y [payroll-faq-us:401k].") == [
        ("leave-policy-in", "earned-leave-el"),
        ("payroll-faq-us", "401k"),
    ]
    assert CITATION_RE.findall("[Not A Tag] [doc:Section]") == []


def test_ground_keeps_valid_citations_in_order_once():
    hits = _hits([10, 10])
    text = "Fact one [d0:s]. Fact two [d1:s]. Fact three [d0:s]."
    answer, cits, refused = ground(text, hits)
    assert not refused
    assert answer == text
    assert [c.tag for c in cits] == ["[d0:s]", "[d1:s]"]
    assert cits[0].doc_id == "d0" and cits[0].region == Region.GLOBAL


def test_ground_strips_unknown_tags_but_keeps_answer_if_some_valid():
    hits = _hits([10])
    answer, cits, refused = ground("Real [d0:s]. Made up [ghost-doc:nope].", hits)
    assert not refused
    assert "[ghost-doc:nope]" not in answer
    assert "[d0:s]" in answer
    assert [c.tag for c in cits] == ["[d0:s]"]


def test_ground_refuses_when_no_valid_citation():
    hits = _hits([10])
    answer, cits, refused = ground("Confident prose with no tags at all.", hits)
    assert refused and answer == REFUSAL_TEXT and cits == []
    answer, _, refused = ground("Prose citing a doc you never saw [ghost:x].", hits)
    assert refused and answer == REFUSAL_TEXT


def test_ground_detects_model_refusal_variants():
    hits = _hits([10])
    for text in (REFUSAL_TEXT, "I don't know.", "  i don't know — ask HR  "):
        answer, cits, refused = ground(text, hits)
        assert refused and answer == REFUSAL_TEXT and cits == []
