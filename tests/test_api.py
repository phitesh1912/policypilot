import json

import pytest
from fastapi.testclient import TestClient

from policypilot.api import create_app
from policypilot.config import Settings
from policypilot.engine import Engine
from policypilot.llm.base import REFUSAL_TEXT
from policypilot.llm.fake import FakeProvider
from policypilot.retrieval.bm25 import BM25Retriever
from policypilot.retrieval.hybrid import HybridRetriever
from policypilot.retrieval.vector import VectorRetriever
from tests.conftest import POLICIES_DIR


def _client(admin_token: str | None = "secret") -> TestClient:
    settings = Settings(llm_provider="fake", policies_dir=POLICIES_DIR, admin_token=admin_token)
    provider = FakeProvider()
    engine = Engine(
        settings,
        provider=provider,
        retriever=HybridRetriever(BM25Retriever(), VectorRetriever(provider)),
    )
    return TestClient(create_app(settings, engine))


@pytest.fixture(scope="module")
def client() -> TestClient:
    with _client() as c:
        yield c


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["provider"] == "fake"
    assert body["retriever"] == "hybrid"
    assert body["indexed_chunks"] > 50


def test_ask_returns_grounded_answer(client: TestClient):
    r = client.post(
        "/ask",
        json={"question": "How many casual leave days do I get?", "user": {"region": "IN"}},
    )
    assert r.status_code == 200
    body = r.json()
    assert not body["refused"]
    assert "7 days" in body["answer"]
    assert body["citations"][0]["doc_id"] == "leave-policy-in"
    assert set(body["tokens"]) >= {"prompt", "completion", "embedding", "context"}
    assert body["cost_usd"] > 0


def test_ask_region_isolation_via_api(client: TestClient):
    q = "How much does Acme match on the 401(k)?"
    in_body = client.post("/ask", json={"question": q, "user": {"region": "IN"}}).json()
    us_body = client.post("/ask", json={"question": q, "user": {"region": "US"}}).json()
    assert all(c["region"] != "US" for c in in_body["citations"])
    assert us_body["citations"][0]["doc_id"] == "payroll-faq-us"


def test_ask_refuses_unanswerable(client: TestClient):
    body = client.post("/ask", json={"question": "office wifi password?"}).json()
    assert body["refused"] and body["answer"] == REFUSAL_TEXT


def test_ask_validation(client: TestClient):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "x" * 2001}).status_code == 422
    assert (
        client.post("/ask", json={"question": "hi", "user": {"region": "MARS"}}).status_code == 422
    )
    assert client.post("/ask", json={"question": "hi", "user": {"role": "god"}}).status_code == 422


def test_stream_emits_meta_tokens_done(client: TestClient):
    with client.stream(
        "GET", "/ask/stream", params={"question": "What is the wellness allowance?", "region": "IN"}
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        raw = "".join(r.iter_text())
    events = []
    for block in raw.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        events.append((lines["event"], json.loads(lines["data"])))
    kinds = [k for k, _ in events]
    assert kinds[0] == "meta" and kinds[-1] == "done" and "token" in kinds
    done = events[-1][1]["result"]
    assert "500" in done["answer"]
    assert done["citations"][0]["doc_id"] == "benefits-handbook"


def test_ingest_requires_admin_token(client: TestClient):
    assert client.post("/ingest").status_code == 401
    assert client.post("/ingest", headers={"Authorization": "Bearer wrong"}).status_code == 401
    r = client.post("/ingest", headers={"Authorization": "Bearer secret"})
    assert r.status_code == 200
    assert r.json()["documents"] == 7


def test_ingest_disabled_without_configured_token():
    with _client(admin_token=None) as c:
        assert c.post("/ingest", headers={"Authorization": "Bearer anything"}).status_code == 503
