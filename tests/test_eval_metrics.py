import json
from pathlib import Path

from eval.metrics import (
    Aggregate,
    citation_precision,
    heuristic_faithfulness,
    hit_at_k,
    reciprocal_rank,
    section_id,
)

from policypilot.ingest.chunker import Chunker
from policypilot.ingest.loader import load_directory
from tests.conftest import POLICIES_DIR


def test_section_id_strips_window_suffix_only():
    assert section_id("doc:sec-2") == "doc:sec"
    assert section_id("doc:sec") == "doc:sec"
    assert section_id("payroll-faq-us:401-k-contributions") == "payroll-faq-us:401-k-contributions"


def test_hit_and_rr():
    ranked = ["a:x", "b:y-1", "c:z"]
    assert hit_at_k(ranked, {"b:y"}, 5) == 1.0
    assert hit_at_k(ranked, {"b:y"}, 1) == 0.0
    assert reciprocal_rank(ranked, {"b:y"}) == 0.5
    assert reciprocal_rank(ranked, {"nope:q"}) == 0.0


def test_citation_precision():
    assert citation_precision([], {"a:x"}) is None
    assert citation_precision(["a:x", "b:y"], {"a:x"}) == 0.5


def test_heuristic_faithfulness():
    ctx = ["Employees receive 12 days of paid sick leave. Carry forward up to 24 days."]
    assert heuristic_faithfulness("You get 12 days [l:s].", ctx) == 1.0
    assert heuristic_faithfulness("You get 15 days [l:s].", ctx) == 0.0
    assert heuristic_faithfulness("Sick leave is paid and carries forward.", ctx) == 1.0
    assert heuristic_faithfulness("Bring a unicorn certificate.", ctx) == 0.0


def test_aggregate_summary_handles_empty():
    s = Aggregate().summary()
    assert s["hit@5"] == 0.0 and s["faithfulness"] is None


def test_golden_set_is_well_formed_and_targets_real_sections():
    data = json.loads((Path(__file__).parents[1] / "eval" / "golden.json").read_text("utf-8"))
    items = data["items"]
    assert len(items) == 50
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids))
    chunks = Chunker().chunk_documents(load_directory(POLICIES_DIR))
    real_sections = {section_id(c.chunk_id) for c in chunks}
    for item in items:
        assert isinstance(item["answerable"], bool)
        if item["answerable"]:
            assert item["expected"], item["id"]
            for sec in item["expected"]:
                assert sec in real_sections, f"{item['id']} -> {sec} not in corpus"
        else:
            assert item["expected"] == []
            assert item["refusal_reason"] in {"out_of_scope", "access"}
    assert sum(1 for i in items if not i["answerable"]) >= 10
