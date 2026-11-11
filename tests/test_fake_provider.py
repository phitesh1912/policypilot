import numpy as np

from policypilot.llm.base import CONTEXT_HEADER, QUESTION_HEADER, REFUSAL_TEXT
from policypilot.llm.fake import FakeProvider


def _cos(a, b):
    return float(np.dot(a, b))


def test_embeddings_are_deterministic_and_unit_length():
    p1, p2 = FakeProvider(), FakeProvider()
    a = p1.embed_one("casual leave days per year")
    b = p2.embed_one("casual leave days per year")
    assert a == b
    assert abs(np.linalg.norm(a) - 1.0) < 1e-6
    assert len(a) == p1.embed_dim


def test_similar_texts_are_closer_than_unrelated():
    p = FakeProvider()
    q = p.embed_one("how many sick leave days do I get")
    close = p.embed_one("Employees receive 12 days of paid sick leave per leave year.")
    far = p.embed_one("Employees may enrol in the stock purchase plan at a 15% discount.")
    assert _cos(q, close) > _cos(q, far)


def test_empty_text_embeds_to_zero_vector():
    p = FakeProvider()
    assert p.embed_one("the and of") == [0.0] * p.embed_dim


def _prompt(blocks, question):
    ctx = "".join(f"{CONTEXT_HEADER}{cite}\n{body}\n\n" for cite, body in blocks)
    return f"{ctx}{QUESTION_HEADER}\n{question}"


def test_complete_refuses_without_context():
    p = FakeProvider()
    c = p.complete("sys", _prompt([], "how many days?"))
    assert c.text == REFUSAL_TEXT
    assert c.completion_tokens > 0


def test_complete_refuses_when_nothing_overlaps():
    p = FakeProvider()
    blocks = [("[doc:a]", "Title > A\n\nThe stock plan opens in May.")]
    assert p.complete("sys", _prompt(blocks, "quantum physics?")).text == REFUSAL_TEXT


def test_complete_refuses_on_a_single_shared_word():
    # Regression: "office wifi password" overlapped the remote-work policy on "office" alone and
    # the fake happily "answered" with the hybrid-work rules.
    p = FakeProvider()
    blocks = [("[rw:hybrid]", "Remote > Hybrid\n\nWork from the office two days a week.")]
    assert p.complete("sys", _prompt(blocks, "what is the office wifi password?")).text == (
        REFUSAL_TEXT
    )


def test_complete_picks_best_block_and_cites_it():
    p = FakeProvider()
    blocks = [
        (
            "[leave:sick]",
            "Leave > Sick\n\nYou get 12 days of sick leave. A note is needed after 2.",
        ),
        (
            "[leave:casual]",
            "Leave > Casual\n\nYou get 7 days of casual leave. It cannot carry over.",
        ),
    ]
    c = p.complete("sys", _prompt(blocks, "how many casual leave days?"))
    assert c.text.endswith("[leave:casual]")
    assert c.text.startswith("You get 7 days of casual leave.")
    assert "sick" not in c.text


def test_stream_concatenates_to_complete():
    p = FakeProvider()
    blocks = [("[d:s]", "T > S\n\nAnswer here. More.")]
    prompt = _prompt(blocks, "answer?")
    assert "".join(p.stream("sys", prompt)).strip() == p.complete("sys", prompt).text
