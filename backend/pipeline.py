from __future__ import annotations

import logging
import pickle
from functools import lru_cache

from backend import config
from backend.ingestion.metadata import extract_cross_references
from backend.models import LawChunk, RetrievedChunk
from backend.retrieval.citation_fastpath import get_fastpath_index
from backend.retrieval.hybrid_search import hybrid_search
from backend.retrieval.querry_transform import decompose_query
from backend.retrieval.rerank import rerank

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_parent_lookup() -> dict:
    try:
        with open(config.PARENT_LOOKUP_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        log.warning("parent lookup not found at %s — rerank will use child-chunk text only.", config.PARENT_LOOKUP_PATH)
        return {}


def _enrich_with_parent_text(candidates: list) -> list:
    parent_lookup = _get_parent_lookup()
    if not parent_lookup:
        return candidates
    enriched = []
    for c in candidates:
        parent = parent_lookup.get(f"{c.law_id}_a{c.aid}")
        enriched.append(c.model_copy(update={"text": parent.text}) if parent else c)
    return enriched


def _resolve_cross_references(reranked: list, already_seen: set) -> list:
    idx = get_fastpath_index()
    article_lookup = idx.get("article_lookup", {})
    chunk_by_id = idx.get("chunk_by_id", {})
    resolved = []
    for c in reranked[: config.CROSSREF_TOP_N_SOURCE_CHUNKS]:
        for ref in extract_cross_references(c.text, default_law_id=c.law_id):
            chunk_ids = article_lookup.get((ref["law_id"], ref["aid"]))
            if not chunk_ids:
                continue
            for cid in chunk_ids:
                if cid in already_seen:
                    continue
                target = chunk_by_id.get(cid)
                if target is None:
                    continue
                already_seen.add(cid)
                resolved.append(RetrievedChunk(chunk_id=target.chunk_id, law_id=target.law_id, aid=target.aid,
                                                article_num=target.article_num, text=target.text, score=0.0, source="crossref"))
    return resolved


def collect_law_evidence(query_text: str) -> list:
    candidates = hybrid_search(query_text, top_k=config.RERANK_TOP_K)
    child_text_by_id = {c.chunk_id: c.text for c in candidates}

    reranked = rerank(query_text, _enrich_with_parent_text(candidates), top_k=config.FINAL_LAW_TOP_K)
    reranked = [c.model_copy(update={"text": child_text_by_id.get(c.chunk_id, c.text)}) for c in reranked]

    if config.RETRIEVAL_EVALUATOR_ENABLED and (not reranked or reranked[0].score < config.RETRIEVAL_EVALUATOR_SCORE_THRESHOLD):
        seen_ids = {c.chunk_id for c in candidates}
        extra_candidates = []

        sub_queries = decompose_query(query_text, n_subqueries=config.QUERY_DECOMPOSITION_MAX_SUBQUERIES)
        for sub_q in sub_queries:
            new_from_decomp = hybrid_search(sub_q, top_k=config.RERANK_TOP_K, use_query_rewriting=False, use_decomposition=False)
            for c in new_from_decomp:
                if c.chunk_id not in seen_ids:
                    seen_ids.add(c.chunk_id)
                    extra_candidates.append(c)

        if config.CROSSREF_ENABLED and config.CROSSREF_MAX_HOPS >= 1:
            extra_candidates.extend(_resolve_cross_references(reranked, seen_ids))

        if extra_candidates:
            merged = {c.chunk_id: c for c in candidates}
            for c in extra_candidates:
                merged.setdefault(c.chunk_id, c)
            merged_candidates = list(merged.values())
            child_text_by_id.update({c.chunk_id: c.text for c in extra_candidates})
            reranked = rerank(query_text, _enrich_with_parent_text(merged_candidates), top_k=config.FINAL_LAW_TOP_K)
            reranked = [c.model_copy(update={"text": child_text_by_id.get(c.chunk_id, c.text)}) for c in reranked]

    return reranked
