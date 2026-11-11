"""FastAPI application: /ask, /ask/stream (SSE), /ingest, /health, plus the built React UI."""

from __future__ import annotations

import json
import secrets
from collections.abc import Iterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from policypilot import __version__
from policypilot.config import Settings, get_settings
from policypilot.engine import Engine
from policypilot.models import AskResult, Region, Role, User

MAX_QUESTION_CHARS = 2000


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    user: User = Field(default_factory=User)


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    seconds: float


class HealthResponse(BaseModel):
    status: str
    version: str
    provider: str
    chat_model: str
    embed_model: str
    retriever: str
    indexed_chunks: int
    cache: dict[str, int]


def create_app(settings: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine or Engine(settings)
        # BM25 lives in memory, so (re)build the index on every start. With the embedded Qdrant
        # store this also re-embeds the corpus; it is small (<100 chunks) so that is acceptable.
        app.state.engine.ingest()
        yield

    app = FastAPI(title="PolicyPilot", version=__version__, lifespan=lifespan)

    def get_engine(request: Request) -> Engine:
        return request.app.state.engine

    def require_admin(authorization: str | None = Header(default=None)) -> None:
        if not settings.admin_token:
            raise HTTPException(503, "ingest is disabled: set ADMIN_TOKEN to enable it")
        expected = f"Bearer {settings.admin_token}"
        if not authorization or not secrets.compare_digest(authorization, expected):
            raise HTTPException(401, "admin token required")

    @app.get("/health", response_model=HealthResponse)
    def health(eng: Engine = Depends(get_engine)) -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            provider=eng.provider.name,
            chat_model=eng.provider.chat_model,
            embed_model=eng.provider.embed_model,
            retriever=eng.retriever.name,
            indexed_chunks=eng.indexed_chunks,
            cache={"size": len(eng.cache), "hits": eng.cache.hits, "misses": eng.cache.misses},
        )

    @app.post("/ask", response_model=AskResult)
    def ask(req: AskRequest, eng: Engine = Depends(get_engine)) -> AskResult:
        return eng.ask(req.question, req.user)

    @app.get("/ask/stream")
    def ask_stream(
        question: str = Query(min_length=1, max_length=MAX_QUESTION_CHARS),
        role: Role = Role.EMPLOYEE,
        region: Region = Region.GLOBAL,
        eng: Engine = Depends(get_engine),
    ) -> StreamingResponse:
        user = User(role=role, region=region)

        def events() -> Iterator[str]:
            for ev in eng.ask_stream(question, user):
                payload = {k: v for k, v in ev.items() if k != "type"}
                yield f"event: {ev['type']}\ndata: {json.dumps(payload)}\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_admin)])
    def ingest(eng: Engine = Depends(get_engine)) -> IngestResponse:
        stats = eng.ingest()
        return IngestResponse(documents=stats.documents, chunks=stats.chunks, seconds=stats.seconds)

    if settings.static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(settings.static_dir), html=True), name="ui")

    return app


app = create_app()
