"""
BM25 keyword index over law chunks (design doc §3.1).

BM25 exists alongside the Pinecone vector index specifically because exact
matches — article numbers, decree numbers like "145/2020/NĐ-CP", defined
legal terms — are things sparse keyword search nails and dense embeddings
sometimes blur.

CHATBOT_MIGRATION_PLAN.md §D1 (applied here): the original tokenizer had no
accent-folding step, so a user typing without Vietnamese diacritics ("dieu
30 khoan 2 noi gi", common on mobile/older keyboards) got weak BM25 matches
against a corpus indexed with full diacritics. Per the citation-fast-path
skill's explicit warning, accent-folding is NOT applied to the primary,
diacritics-preserving index (folding could conflate distinct words that only
differ by accent, e.g. "hòa giải" vs "hoa giải", hurting precision on
ordinary semantic queries). Instead, a SECOND, parallel BM25 index is built
over accent-folded tokens and exposed as `query_folded()` — `hybrid_search.py`
wires this in as one extra, lower-weight RRF channel
(`config.RRF_WEIGHT_FOLDED`), so an accent-insensitive match can still
surface a result without diluting the precision of the primary channel.
"""
from __future__ import annotations

import pickle
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from backend import config
from backend.models import LawChunk, RetrievedChunk

# Simple Vietnamese-aware tokenizer: lowercase, split on non-word chars but
# keep intra-word diacritics and alphanumeric decree numbers like 145/2020/NĐ-CP
# from being shattered.
_TOKEN_RE = re.compile(r"[^\W_]+(?:[/\-][^\W_]+)*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def fold_accents(text: str) -> str:
    """D1: strip Vietnamese diacritics for accent-insensitive matching.
    Only used for the auxiliary folded BM25 channel and citation-reference
    parsing (backend/retrieval/citation_fastpath.py) — never for the primary
    semantic BM25 channel or for embeddings."""
    text = text.replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize("NFD", text)
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")


def tokenize_folded(text: str) -> list[str]:
    return tokenize(fold_accents(text))


@dataclass
class BM25IndexData:
    chunk_ids: list[str]
    law_ids: list[str]
    aids: list[int]
    article_nums: list[int | None]
    texts: list[str]
    statuses: list[str]


class BM25Index:
    def __init__(self):
        self._bm25 = None
        self._bm25_folded = None  # D1: accent-insensitive auxiliary index
        self._data: BM25IndexData | None = None

    def build(self, chunks: list[LawChunk], status_by_law: dict[str, str] | None = None) -> None:
        from rank_bm25 import BM25Okapi

        status_by_law = status_by_law or {}
        child_chunks = [c for c in chunks if c.level == "child"]
        corpus_tokens = [tokenize(c.text) for c in child_chunks]
        folded_tokens = [tokenize_folded(c.text) for c in child_chunks]

        self._bm25 = BM25Okapi(corpus_tokens)
        self._bm25_folded = BM25Okapi(folded_tokens)
        self._data = BM25IndexData(
            chunk_ids=[c.chunk_id for c in child_chunks],
            law_ids=[c.law_id for c in child_chunks],
            aids=[c.aid for c in child_chunks],
            article_nums=[c.article_num for c in child_chunks],
            texts=[c.text for c in child_chunks],
            statuses=[status_by_law.get(c.law_id, "unknown") for c in child_chunks],
        )

    def save(self, path: str | Path = config.BM25_INDEX_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "bm25_folded": self._bm25_folded, "data": self._data}, f)

    def load(self, path: str | Path = config.BM25_INDEX_PATH) -> None:
        path = Path(path)
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self._bm25 = payload["bm25"]
        self._bm25_folded = payload.get("bm25_folded")  # tolerate older pickles without it
        self._data = payload["data"]

    def _passes(self, idx: int, law_id: str | None, require_active: bool) -> bool:
        if law_id and self._data.law_ids[idx] != law_id:
            return False
        if require_active and self._data.statuses[idx] not in ("active", "unknown"):
            return False
        return True

    def _results_from_scores(self, scores, top_k: int, law_id: str | None, require_active: bool,
                              source: str) -> list[RetrievedChunk]:
        candidates = [(i, s) for i, s in enumerate(scores) if self._passes(i, law_id, require_active)]
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:top_k]
        return [
            RetrievedChunk(
                chunk_id=self._data.chunk_ids[i],
                law_id=self._data.law_ids[i],
                aid=self._data.aids[i],
                article_num=self._data.article_nums[i],
                text=self._data.texts[i],
                score=float(s),
                source=source,
            )
            for i, s in candidates
            if s > 0
        ]

    def query(
        self,
        text: str,
        top_k: int = config.BM25_TOP_K,
        law_id: str | None = None,
        require_active: bool = True,
    ) -> list[RetrievedChunk]:
        """Primary, diacritics-preserving channel — unchanged behavior."""
        if self._bm25 is None or self._data is None:
            raise RuntimeError("BM25Index not built/loaded. Call .build() or .load() first.")
        scores = self._bm25.get_scores(tokenize(text))
        return self._results_from_scores(scores, top_k, law_id, require_active, source="bm25")

    def query_folded(
        self,
        text: str,
        top_k: int = config.BM25_TOP_K,
        law_id: str | None = None,
        require_active: bool = True,
    ) -> list[RetrievedChunk]:
        """D1: accent-insensitive auxiliary channel. Returns [] gracefully
        (never raises) if the index was built before this field existed, so
        older BM25 pickles keep working without a mandatory rebuild."""
        if self._bm25_folded is None or self._data is None:
            return []
        scores = self._bm25_folded.get_scores(tokenize_folded(text))
        return self._results_from_scores(scores, top_k, law_id, require_active, source="bm25")


_singleton: BM25Index | None = None


def get_bm25_index() -> BM25Index:
    """Lazily load the singleton BM25 index from disk."""
    global _singleton
    if _singleton is None:
        _singleton = BM25Index()
        _singleton.load()
    return _singleton
