"""
Hybrid search (design doc §3.1): fuse BM25 and Pinecone vector results with
Reciprocal Rank Fusion (RRF) rather than a weighted score sum, because BM25
scores and cosine similarities live on incomparable scales — RRF sidesteps
that by fusing on *rank* instead of raw score.

    RRF(d) = sum over retrievers r of  weight_r / (k + rank_r(d))

CHATBOT_MIGRATION_PLAN.md §2.1 (kept) + §D1 (added): retrieval-purpose-
agnostic, reused as-is for chat questions. NEW: a third BM25 channel
(`bm25.query_folded`, accent-insensitive — see indexing/bm25_index.py) is
now fused in at a lower weight (`config.RRF_WEIGHT_FOLDED`), so a question
typed without Vietnamese diacritics still gets a decent keyword match
without diluting precision on normal, accented queries (that risk is why
the folded channel is additive, not a replacement for the primary one).
"""
from __future__ import annotations

from typing import Optional

from backend import config
from backend.indexing import bm25_index, vector_store
from backend.models import RetrievedChunk
from backend.retrieval.ner import extract_entities, mask_person_org_entities
from backend.retrieval.querry_transform import decompose_query, rewrite_query


def _rrf_fuse_weighted(
    channels: list[tuple[list[RetrievedChunk], float]],
    k: int = config.RRF_K,
) -> list[RetrievedChunk]:
    """Fuse several (result_list, weight) channels by weighted reciprocal
    rank. weight=1.0 for every channel reproduces the original unweighted
    RRF used before query decomposition was introduced."""
    scores: dict[str, float] = {}
    best_chunk: dict[str, RetrievedChunk] = {}

    for results, weight in channels:
        for rank, chunk in enumerate(results, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + weight / (k + rank)
            best_chunk.setdefault(chunk.chunk_id, chunk)

    fused = [
        RetrievedChunk(
            chunk_id=cid,
            law_id=best_chunk[cid].law_id,
            aid=best_chunk[cid].aid,
            article_num=best_chunk[cid].article_num,
            text=best_chunk[cid].text,
            score=score,
            source="fused",
        )
        for cid, score in scores.items()
    ]
    fused.sort(key=lambda c: c.score, reverse=True)
    return fused


def hybrid_search(
    query: str,
    law_id: Optional[str] = None,
    require_active: bool = True,
    use_query_rewriting: bool = True,
    use_decomposition: bool = True,
    top_k: int = 30,
) -> list[RetrievedChunk]:
    """Run BM25 + vector search (optionally over multiple query rewrites and
    NER-decomposed legal-aspect sub-queries) plus one accent-folded BM25
    channel (D1), then fuse everything with weighted RRF.

    Metadata filtering (law_id / active-only) is pushed down into both the
    BM25 index and the Pinecone query themselves (design doc §3.2).
    """
    bm25 = bm25_index.get_bm25_index()

    entities = extract_entities(query)
    law_query = mask_person_org_entities(query, entities)

    channels: list[tuple[list[RetrievedChunk], float]] = []

    base_queries = rewrite_query(law_query) if use_query_rewriting else [law_query]
    for q in base_queries:
        channels.append((
            bm25.query(q, top_k=config.BM25_TOP_K, law_id=law_id, require_active=require_active),
            config.RRF_WEIGHT_STANDARD,
        ))
        channels.append((
            vector_store.query(q, top_k=config.VECTOR_TOP_K, law_id=law_id, require_active=require_active),
            config.RRF_WEIGHT_STANDARD,
        ))

    # D1: one accent-folded auxiliary channel per hybrid_search call (not
    # per rewrite variant, to avoid multiplying query fan-out for a
    # secondary signal) — run against the original (masked) query.
    channels.append((
        bm25.query_folded(law_query, top_k=config.BM25_TOP_K, law_id=law_id, require_active=require_active),
        config.RRF_WEIGHT_FOLDED,
    ))

    if use_decomposition and config.QUERY_DECOMPOSITION_ENABLED:
        for sub_q in decompose_query(
            query, masked_query=law_query, n_subqueries=config.QUERY_DECOMPOSITION_MAX_SUBQUERIES
        ):
            channels.append((
                vector_store.query(sub_q, top_k=config.VECTOR_TOP_K, law_id=law_id, require_active=require_active),
                config.RRF_WEIGHT_AGENT,
            ))

    fused = _rrf_fuse_weighted(channels)
    return fused[:top_k]
