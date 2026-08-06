from __future__ import annotations

import pickle
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from backend import config
from backend.models import LawChunk, RetrievedChunk

_TOKEN_RE = re.compile(r"[^\W_]+(?:[/\-][^\W_]+)*", re.UNICODE)


def tokenize(text: str) -> list:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def fold_accents(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize("NFD", text)
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")


def tokenize_folded(text: str) -> list:
    return tokenize(fold_accents(text))


@dataclass
class BM25IndexData:
    chunk_ids: list
    law_ids: list
    aids: list
    article_nums: list
    texts: list
    statuses: list


class BM25Index:
    def __init__(self):
        self._bm25 = None
        self._bm25_folded = None
        self._data = None

    def build(self, chunks, status_by_law=None) -> None:
        from rank_bm25 import BM25Okapi
        status_by_law = status_by_law or {}
        child_chunks = [c for c in chunks if c.level == "child"]
        corpus_tokens = [tokenize(c.text) for c in child_chunks]
        folded_tokens = [tokenize_folded(c.text) for c in child_chunks]
        self._bm25 = BM25Okapi(corpus_tokens)
        self._bm25_folded = BM25Okapi(folded_tokens)
        self._data = BM25IndexData(
            chunk_ids=[c.chunk_id for c in child_chunks], law_ids=[c.law_id for c in child_chunks],
            aids=[c.aid for c in child_chunks], article_nums=[c.article_num for c in child_chunks],
            texts=[c.text for c in child_chunks],
            statuses=[status_by_law.get(c.law_id, "unknown") for c in child_chunks],
        )

    def save(self, path=config.BM25_INDEX_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "bm25_folded": self._bm25_folded, "data": self._data}, f)

    def load(self, path=config.BM25_INDEX_PATH) -> None:
        path = Path(path)
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self._bm25 = payload["bm25"]
        self._bm25_folded = payload.get("bm25_folded")
        self._data = payload["data"]

    def _passes(self, idx, law_id, require_active) -> bool:
        if law_id and self._data.law_ids[idx] != law_id:
            return False
        if require_active and self._data.statuses[idx] not in ("active", "unknown"):
            return False
        return True

    def _results_from_scores(self, scores, top_k, law_id, require_active, source) -> list:
        candidates = [(i, s) for i, s in enumerate(scores) if self._passes(i, law_id, require_active)]
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:top_k]
        return [
            RetrievedChunk(chunk_id=self._data.chunk_ids[i], law_id=self._data.law_ids[i], aid=self._data.aids[i],
                            article_num=self._data.article_nums[i], text=self._data.texts[i], score=float(s), source=source)
            for i, s in candidates if s > 0
        ]

    def query(self, text, top_k=config.BM25_TOP_K, law_id=None, require_active=True) -> list:
        if self._bm25 is None or self._data is None:
            raise RuntimeError("BM25Index not built/loaded. Call .build() or .load() first.")
        scores = self._bm25.get_scores(tokenize(text))
        return self._results_from_scores(scores, top_k, law_id, require_active, source="bm25")

    def query_folded(self, text, top_k=config.BM25_TOP_K, law_id=None, require_active=True) -> list:
        if self._bm25_folded is None or self._data is None:
            return []
        scores = self._bm25_folded.get_scores(tokenize_folded(text))
        return self._results_from_scores(scores, top_k, law_id, require_active, source="bm25")


_singleton = None


def get_bm25_index() -> BM25Index:
    global _singleton
    if _singleton is None:
        _singleton = BM25Index()
        _singleton.load()
    return _singleton
