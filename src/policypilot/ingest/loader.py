"""Load policy documents (markdown with YAML frontmatter, or PDF) into ``Document`` objects."""

from __future__ import annotations

from pathlib import Path

import frontmatter
from pypdf import PdfReader

from policypilot.models import Document, Region

REQUIRED_META = ("doc_id", "title", "region", "department", "version")


class DocumentLoadError(ValueError):
    pass


def _validate_meta(meta: dict, path: Path) -> dict:
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        raise DocumentLoadError(f"{path.name}: missing frontmatter keys {missing}")
    try:
        meta["region"] = Region(str(meta["region"]).upper())
    except ValueError as e:
        raise DocumentLoadError(f"{path.name}: invalid region {meta['region']!r}") from e
    meta["version"] = str(meta["version"])
    return meta


def load_markdown(path: Path) -> Document:
    post = frontmatter.load(path)
    meta = _validate_meta(dict(post.metadata), path)
    return Document(
        doc_id=meta["doc_id"],
        title=meta["title"],
        region=meta["region"],
        department=meta["department"],
        version=meta["version"],
        text=post.content.strip(),
        source_path=str(path),
    )


def load_pdf(path: Path, *, meta: dict | None = None) -> Document:
    """Load a PDF. PDFs carry no frontmatter, so metadata comes from a sidecar ``<name>.meta.json``
    or the ``meta`` argument."""
    import json

    if meta is None:
        sidecar = path.with_suffix(".meta.json")
        if not sidecar.exists():
            raise DocumentLoadError(f"{path.name}: PDF needs a sidecar {sidecar.name}")
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    meta = _validate_meta(dict(meta), path)
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return Document(
        doc_id=meta["doc_id"],
        title=meta["title"],
        region=meta["region"],
        department=meta["department"],
        version=meta["version"],
        text="\n\n".join(pages).strip(),
        source_path=str(path),
    )


def load_directory(directory: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() == ".md":
            docs.append(load_markdown(path))
        elif path.suffix.lower() == ".pdf":
            docs.append(load_pdf(path))
    ids = [d.doc_id for d in docs]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise DocumentLoadError(f"duplicate doc_id(s): {sorted(dupes)}")
    return docs
