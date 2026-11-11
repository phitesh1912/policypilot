from pathlib import Path

import pytest

from policypilot.ingest.chunker import Chunker
from policypilot.ingest.loader import load_directory
from policypilot.models import Chunk, Document, Region

POLICIES_DIR = Path(__file__).resolve().parents[1] / "data" / "policies"


@pytest.fixture(scope="session")
def documents() -> list[Document]:
    return load_directory(POLICIES_DIR)


@pytest.fixture(scope="session")
def chunker() -> Chunker:
    return Chunker(chunk_tokens=350, overlap_tokens=50)


@pytest.fixture(scope="session")
def chunks(documents: list[Document], chunker: Chunker) -> list[Chunk]:
    return chunker.chunk_documents(documents)


def make_doc(doc_id: str, text: str, region: Region = Region.GLOBAL) -> Document:
    return Document(
        doc_id=doc_id,
        title=doc_id.replace("-", " ").title(),
        region=region,
        department="HR",
        version="1",
        text=text,
        source_path=f"<memory:{doc_id}>",
    )
