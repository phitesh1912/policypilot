"""BM25 (Okapi) implemented from scratch — no external ranking library.

Score for a query q over document d:

    sum over terms t in q of  IDF(t) * tf(t,d) * (k1 + 1)
                              -------------------------------------------
                              tf(t,d) + k1 * (1 - b + b * |d| / avgdl)

with IDF(t) = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 ), the "+1" variant that keeps IDF
non-negative for very common terms.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from policypilot.models import Chunk, RetrievedChunk, User
from policypilot.retrieval.base import Retriever, allowed

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# "401(k)" -> "401k", "comp(off)" -> "compoff": a bracketed suffix glued to a word is part of it.
_BRACKET_SUFFIX_RE = re.compile(r"(\w)\((\w{1,3})\)")

# Small, deliberately conservative stopword list: enough to stop "the"/"is" dominating short
# queries, not so aggressive that policy terms ("off", "pay") disappear.
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
        "my",
        "me",
        "our",
        "we",
        "do",
        "does",
        "can",
    ]
)


def stem(token: str) -> str:
    """Very light suffix stripping so 'matches'/'match', 'days'/'day', 'accrued'/'accrue' meet.

    Deliberately not a full Porter stemmer: HR vocabulary is small and over-stemming
    ("policies" -> "polici" is fine, but "pto" must stay "pto") costs more than it gains.
    """
    if len(token) <= 3 or token.isdigit():
        return token
    stemmed = token
    for suffix, repl in (
        ("ies", "y"),
        ("sses", "ss"),
        ("ches", "ch"),
        ("shes", "sh"),
        ("xes", "x"),
        ("zes", "z"),
        ("ing", ""),
        ("ed", ""),
        ("s", ""),
    ):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            stemmed = token[: -len(suffix)] + repl
            # 'ing'/'ed' removal can leave a doubled consonant ("matted" -> "matt"); trim it.
            if suffix in ("ing", "ed") and len(stemmed) > 3 and stemmed[-1] == stemmed[-2]:
                stemmed = stemmed[:-1]
            break
    # Drop a trailing 'e' so "accrue" / "accrued" / "accrues" all meet at "accru". Crude, but it
    # is applied identically to queries and documents, which is all a ranking function needs.
    if len(stemmed) > 4 and stemmed.endswith("e"):
        stemmed = stemmed[:-1]
    return stemmed


def tokenize(text: str) -> list[str]:
    text = _BRACKET_SUFFIX_RE.sub(r"\1\2", text.lower())
    return [stem(t) for t in _TOKEN_RE.findall(text) if t not in STOPWORDS]


class BM25Index:
    """Pure scoring structure, independent of access control. Kept separate so it can be unit
    tested against hand-computed values."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_tfs: list[Counter[str]] = []
        self.doc_lens: list[int] = []
        self.df: Counter[str] = Counter()
        self.avgdl: float = 0.0
        self.n: int = 0

    def fit(self, texts: list[str]) -> None:
        self.doc_tfs = []
        self.doc_lens = []
        self.df = Counter()
        for text in texts:
            toks = tokenize(text)
            tf = Counter(toks)
            self.doc_tfs.append(tf)
            self.doc_lens.append(len(toks))
            self.df.update(tf.keys())
        self.n = len(texts)
        self.avgdl = (sum(self.doc_lens) / self.n) if self.n else 0.0

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.n - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query_tokens: list[str], doc_index: int) -> float:
        tf = self.doc_tfs[doc_index]
        dl = self.doc_lens[doc_index]
        norm = self.k1 * (1.0 - self.b + self.b * dl / self.avgdl) if self.avgdl else self.k1
        total = 0.0
        for term in query_tokens:
            f = tf.get(term)
            if not f:
                continue
            total += self.idf(term) * f * (self.k1 + 1.0) / (f + norm)
        return total

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        return [self.score(q, i) for i in range(self.n)]


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._index = BM25Index(k1=k1, b=b)
        self._chunks: list[Chunk] = []

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        self._index.fit([c.text for c in self._chunks])

    def search(self, query: str, user: User, k: int = 10) -> list[RetrievedChunk]:
        scored = [
            (s, i)
            for i, s in enumerate(self._index.scores(query))
            if s > 0 and allowed(self._chunks[i], user)
        ]
        scored.sort(key=lambda p: (-p[0], p[1]))
        return [
            RetrievedChunk(chunk=self._chunks[i], score=s, rank=r, source=self.name)
            for r, (s, i) in enumerate(scored[:k], start=1)
        ]
