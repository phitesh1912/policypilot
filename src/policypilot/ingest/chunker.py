"""Heading-aware chunker with token-accurate sizing.

Strategy:
1. Split the document into sections at markdown headings (``#``..``###``). Each section keeps its
   heading path (e.g. ``Leave Policy — India > Earned leave (EL)``) so a chunk is self-describing.
2. If a section fits within ``chunk_tokens`` it becomes one chunk. Otherwise it is split into
   token windows of ``chunk_tokens`` with ``overlap_tokens`` of overlap, so no fact is cut in half
   at a window boundary without also appearing whole in a neighbour.
3. Every chunk is prefixed with its heading path so the embedding and BM25 both "see" the context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

from policypilot.models import Chunk, Document

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.*?)\s*$", re.MULTILINE)
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-") or "section"


@dataclass(frozen=True)
class Section:
    heading_path: tuple[str, ...]
    body: str

    @property
    def title(self) -> str:
        return self.heading_path[-1] if self.heading_path else ""


def split_sections(text: str, title: str) -> list[Section]:
    """Split markdown into sections, tracking the heading hierarchy.

    Text before the first heading is attached to the document title. Empty sections (a heading
    followed immediately by another heading) are dropped.
    """
    matches = list(_HEADING_RE.finditer(text))
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (level, heading)

    def path() -> tuple[str, ...]:
        return tuple(h for _, h in stack)

    preamble = text[: matches[0].start()] if matches else text
    if preamble.strip():
        sections.append(Section((title,), preamble.strip()))

    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading = m.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading))
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end() : end].strip()
        if body:
            sections.append(Section(path(), body))
    return sections


class Chunker:
    def __init__(
        self,
        chunk_tokens: int = 350,
        overlap_tokens: int = 50,
        encoding_name: str = "cl100k_base",
    ) -> None:
        if overlap_tokens >= chunk_tokens:
            raise ValueError("overlap_tokens must be smaller than chunk_tokens")
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.enc = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        return len(self.enc.encode(text))

    def _windows(self, tokens: list[int]) -> list[list[int]]:
        if len(tokens) <= self.chunk_tokens:
            return [tokens]
        step = self.chunk_tokens - self.overlap_tokens
        windows = []
        start = 0
        while start < len(tokens):
            windows.append(tokens[start : start + self.chunk_tokens])
            if start + self.chunk_tokens >= len(tokens):
                break
            start += step
        return windows

    def chunk_document(self, doc: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        position = 0
        for section in split_sections(doc.text, doc.title):
            # The heading path is prefixed to the text so each chunk carries its own context.
            # Note: the prefix counts toward the chunk budget, so windows are computed on the
            # body and the prefix is added afterwards.
            prefix = " > ".join(section.heading_path)
            body_tokens = self.enc.encode(section.body)
            section_slug = slugify(section.title)
            for i, window in enumerate(self._windows(body_tokens)):
                body = self.enc.decode(window).strip()
                text = f"{prefix}\n\n{body}"
                suffix = f"-{i}" if i else ""
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}:{section_slug}{suffix}",
                        doc_id=doc.doc_id,
                        title=doc.title,
                        section=section.title,
                        section_slug=section_slug,
                        region=doc.region,
                        department=doc.department,
                        version=doc.version,
                        text=text,
                        token_count=self.count_tokens(text),
                        position=position,
                    )
                )
                position += 1
        return chunks

    def chunk_documents(self, docs: list[Document]) -> list[Chunk]:
        out: list[Chunk] = []
        for d in docs:
            out.extend(self.chunk_document(d))
        return out
