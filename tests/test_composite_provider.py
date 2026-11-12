import pytest

from policypilot.config import Settings
from policypilot.llm.base import CONTEXT_HEADER, QUESTION_HEADER
from policypilot.llm.composite import CompositeProvider
from policypilot.llm.factory import build_provider
from policypilot.llm.fake import FakeProvider


class _StubEmbedder:
    name = "stub"
    embed_model = "stub-embed"
    embed_dim = 3

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[1.0, 0.0, 0.0] for _ in texts]


def test_composite_routes_chat_and_embeddings_separately():
    chat = FakeProvider()
    emb = _StubEmbedder()
    p = CompositeProvider(chat, emb)
    assert p.name == "fake+stub"
    assert p.chat_model == "fake-chat"
    assert p.embed_model == "stub-embed" and p.embed_dim == 3
    assert p.embed(["a", "b"]) == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert emb.calls == 1
    assert not any(c[0] == "embed" for c in chat.calls)
    prompt = (
        f"{CONTEXT_HEADER}[d:s]\nT > S\n\nThe carry forward cap is 45 days.\n\n"
        f"{QUESTION_HEADER}\nwhat is the carry forward cap in days?"
    )
    assert p.complete("sys", prompt).text.endswith("[d:s]")
    assert "".join(p.stream("sys", prompt)).strip() == p.complete("sys", prompt).text


def test_groq_requires_key_and_fake_needs_nothing():
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        build_provider(Settings(llm_provider="groq", groq_api_key=None))
    assert build_provider(Settings(llm_provider="fake")).name == "fake"
    with pytest.raises(ValueError):
        build_provider(Settings(llm_provider="nope"))
