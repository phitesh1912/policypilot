"""Core data types shared across ingestion, retrieval and the API."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Region(StrEnum):
    IN = "IN"
    US = "US"
    GLOBAL = "GLOBAL"


class Role(StrEnum):
    EMPLOYEE = "employee"
    MANAGER = "manager"
    HR = "hr"
    ADMIN = "admin"


class User(BaseModel):
    """The caller's identity as supplied by the (trusted) API layer."""

    role: Role = Role.EMPLOYEE
    region: Region = Region.GLOBAL

    def visible_regions(self) -> set[Region]:
        """Regions whose documents this user may read.

        GLOBAL documents are visible to everyone. Region-specific documents are visible only to
        users in that region. HR and admin users see everything.
        """
        if self.role in (Role.HR, Role.ADMIN):
            return {Region.IN, Region.US, Region.GLOBAL}
        return {Region.GLOBAL, self.region}


class Document(BaseModel):
    """A source document after parsing, before chunking."""

    doc_id: str
    title: str
    region: Region
    department: str
    version: str
    text: str
    source_path: str


class Chunk(BaseModel):
    """A retrievable unit: one heading-aware slice of a document."""

    chunk_id: str
    doc_id: str
    title: str
    section: str
    section_slug: str
    region: Region
    department: str
    version: str
    text: str
    token_count: int
    position: int = Field(description="0-based index of the chunk within its document")

    @property
    def citation(self) -> str:
        """The inline citation tag the model emits, e.g. ``[leave-policy-in:earned-leave]``."""
        return f"[{self.doc_id}:{self.section_slug}]"


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    rank: int
    source: str = Field(description="'bm25', 'vector' or 'hybrid'")
    sources: dict[str, int] = Field(
        default_factory=dict, description="per-retriever rank, populated by the hybrid fuser"
    )


class Citation(BaseModel):
    tag: str
    doc_id: str
    title: str
    section: str
    chunk_id: str
    region: Region
    version: str


class TokenUsage(BaseModel):
    prompt: int = 0
    completion: int = 0
    embedding: int = 0
    context: int = Field(0, description="tokens of retrieved context packed into the prompt")

    @property
    def total(self) -> int:
        return self.prompt + self.completion + self.embedding


class RetrievalTrace(BaseModel):
    chunk_id: str
    doc_id: str
    section: str
    score: float
    sources: dict[str, int]
    in_context: bool


class AskResult(BaseModel):
    answer: str
    citations: list[Citation]
    refused: bool
    cached: bool = False
    tokens: TokenUsage
    cost_usd: float
    retrieved: list[RetrievalTrace] = Field(default_factory=list)
    latency_ms: float = 0.0
