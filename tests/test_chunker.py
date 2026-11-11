import pytest

from policypilot.ingest.chunker import Chunker, slugify, split_sections
from policypilot.models import Region
from tests.conftest import make_doc


def test_split_sections_tracks_heading_hierarchy():
    text = "# Title\n\nintro\n\n## A\n\na body\n\n### A1\n\na1 body\n\n## B\n\nb body\n"
    sections = split_sections(text, "Doc")
    paths = [s.heading_path for s in sections]
    assert paths == [
        ("Title",),
        ("Title", "A"),
        ("Title", "A", "A1"),
        ("Title", "B"),
    ]
    assert sections[2].body == "a1 body"


def test_split_sections_drops_empty_headings_and_keeps_preamble():
    text = "preamble text\n\n# Empty\n\n## Real\n\ncontent"
    sections = split_sections(text, "Doc")
    assert [s.heading_path for s in sections] == [("Doc",), ("Empty", "Real")]


def test_slugify():
    assert slugify("Earned leave (EL)") == "earned-leave-el"
    assert slugify("  Paid time off / PTO ") == "paid-time-off-pto"
    assert slugify("!!!") == "section"


def test_small_section_is_one_chunk_with_heading_prefix(chunker: Chunker):
    doc = make_doc("tiny", "# Leave\n\n## Sick leave\n\nYou get 12 days.")
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.chunk_id == "tiny:sick-leave"
    assert c.text.startswith("Leave > Sick leave\n\n")
    assert "12 days" in c.text
    assert c.citation == "[tiny:sick-leave]"
    assert c.token_count == chunker.count_tokens(c.text)


def test_long_section_is_windowed_with_overlap():
    chunker = Chunker(chunk_tokens=40, overlap_tokens=10)
    body = " ".join(f"word{i}" for i in range(200))
    doc = make_doc("long", f"# T\n\n## S\n\n{body}")
    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 1
    assert [c.chunk_id for c in chunks[:3]] == ["long:s", "long:s-1", "long:s-2"]
    # Every window body fits the budget (prefix excluded, as documented).
    for c in chunks:
        body_only = c.text.split("\n\n", 1)[1]
        assert chunker.count_tokens(body_only) <= 40
    # Consecutive windows overlap: the tail of one appears in the head of the next.
    first_body = chunks[0].text.split("\n\n", 1)[1]
    second_body = chunks[1].text.split("\n\n", 1)[1]
    tail = first_body.split()[-2:]
    assert all(w in second_body for w in tail)
    # Nothing is lost: the full body is reconstructable from the windows.
    assert all(f"word{i}" in " ".join(c.text for c in chunks) for i in range(200))


def test_windows_cover_exact_multiple_without_empty_trailing_window():
    chunker = Chunker(chunk_tokens=10, overlap_tokens=2)
    tokens = list(range(26))  # step 8 -> windows at 0, 8, 16 (covers to 26)
    windows = chunker._windows(tokens)
    assert windows[0] == list(range(0, 10))
    assert windows[-1][-1] == 25
    assert all(len(w) > 0 for w in windows)


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError):
        Chunker(chunk_tokens=50, overlap_tokens=50)


def test_positions_are_sequential_and_metadata_propagates(chunker: Chunker):
    doc = make_doc("meta", "# T\n\n## A\n\na\n\n## B\n\nb", region=Region.IN)
    chunks = chunker.chunk_document(doc)
    assert [c.position for c in chunks] == [0, 1]
    assert {c.region for c in chunks} == {Region.IN}
    assert {c.doc_id for c in chunks} == {"meta"}


def test_real_corpus_chunks_within_budget(chunks, chunker: Chunker):
    assert len(chunks) > 50
    for c in chunks:
        body_only = c.text.split("\n\n", 1)[1]
        assert chunker.count_tokens(body_only) <= chunker.chunk_tokens, c.chunk_id
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk ids must be unique"
