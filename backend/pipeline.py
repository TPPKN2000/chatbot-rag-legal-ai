"""
Law-evidence retrieval orchestration (design doc pipeline.py role).

CHATBOT_MIGRATION_PLAN.md §A1 (removed here): the ALQAC case-outcome branch
is gone — `collect_case_evidence()`, `_case_api_budget()`, the Case Content
API import, and `predict_outcome`/`build_case_digest`/`SubmissionRecord`
assembly all moved out. There is no more "case" with a fixed API budget or a
single verdict to assemble; `backend/chat_pipeline.py` is the new
orchestration entry point for a chat turn, and it calls
`collect_law_evidence()` below (kept, retrieval-purpose-agnostic) directly.

CHATBOT_MIGRATION_PLAN.md §2.3 item 6 / C2 (added here): the retrieval-
evaluator's extra round now also resolves single-hop cross-references
(`backend.ingestion.metadata.extract_cross_references`, previously dead
code — nothing called it) found in the top reranked chunks, so a question
like "Điều 12 có ngoại lệ nào không?" can pull in whichever OTHER article
Điều 12 itself points to, not just what's directly indexed against the
question text. Gated behind the same retrieval-evaluator trigger as the
decomposition round (conversational-retrieval skill: "không chạy mặc định
mọi câu hỏi để tránh tăng latency/token vô ích"), and capped at
`config.CROSSREF_MAX_HOPS` (1 — no recursive chasing of a chain of
references).
"""
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
def _get_parent_lookup() -> dict[str, LawChunk]:
    """Lazily load the parent (whole-Điều) chunk lookup persisted by
    scripts/build_index.py. Returns {} if the index hasn't been (re)built
    yet, so rerank degrades gracefully to child-only text instead of
    crashing."""
    try:
        with open(config.PARENT_LOOKUP_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        log.warning(
            "parent lookup not found at %s — rerank will use child-chunk text only. "
            "Re-run scripts/build_index.py to generate it.",
            config.PARENT_LOOKUP_PATH,
        )
        return {}


def _enrich_with_parent_text(candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Swap in whole-article (parent) text for reranking only (a
    cross-encoder scores a (query, chunk) pair more accurately with fuller
    context). The child chunk_id/aid are preserved so citations stay
    precise; text is swapped back to the child text after rerank."""
    parent_lookup = _get_parent_lookup()
    if not parent_lookup:
        return candidates
    enriched = []
    for c in candidates:
        parent = parent_lookup.get(f"{c.law_id}_a{c.aid}")
        enriched.append(c.model_copy(update={"text": parent.text}) if parent else c)
    return enriched


def _resolve_cross_references(reranked: list[RetrievedChunk], already_seen: set[str]) -> list[RetrievedChunk]:
    """C2: for the top `config.CROSSREF_TOP_N_SOURCE_CHUNKS` reranked
    chunks, extract cross-references and resolve any NEW one (not already in
    the candidate pool) to a retrievable chunk via the article-number lookup
    (the same index citation_fastpath.py uses — extract_cross_references
    returns REFERENCED ARTICLE NUMBERS, not internal aids, so this must go
    through the (law_id, article_num) index, not a direct aid lookup — see
    metadata.py's docstring on this exact gotcha)."""
    idx = get_fastpath_index()
    article_lookup = idx.get("article_lookup", {})
    chunk_by_id = idx.get("chunk_by_id", {})

    resolved: list[RetrievedChunk] = []
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
                resolved.append(
                    RetrievedChunk(
                        chunk_id=target.chunk_id,
                        law_id=target.law_id,
                        aid=target.aid,
                        article_num=target.article_num,
                        text=target.text,
                        score=0.0,
                        source="crossref",
                    )
                )
    return resolved


def collect_law_evidence(query_text: str) -> list[RetrievedChunk]:
    """Hybrid search (design doc §3) -> cross-encoder rerank on parent-article
    context -> child text restored for citation precision -> optional
    retrieval-evaluator re-round (decomposition + C2 cross-reference hop).
    """
    candidates = hybrid_search(query_text, top_k=config.RERANK_TOP_K)
    child_text_by_id = {c.chunk_id: c.text for c in candidates}

    reranked = rerank(query_text, _enrich_with_parent_text(candidates), top_k=config.FINAL_LAW_TOP_K)
    reranked = [
        c.model_copy(update={"text": child_text_by_id.get(c.chunk_id, c.text)}) for c in reranked
    ]

    if (
        config.RETRIEVAL_EVALUATOR_ENABLED
        and (not reranked or reranked[0].score < config.RETRIEVAL_EVALUATOR_SCORE_THRESHOLD)
    ):
        seen_ids = {c.chunk_id for c in candidates}
        extra_candidates: list[RetrievedChunk] = []

        sub_queries = decompose_query(query_text, n_subqueries=config.QUERY_DECOMPOSITION_MAX_SUBQUERIES)
        for sub_q in sub_queries:
            # ACTION_PLAN.md §C1 (SPEED, carried over): rewriting disabled
            # here too — each decomposed sub-query is already short/specific,
            # re-running it through rewrite_query() would multiply fan-out
            # by up to QUERY_DECOMPOSITION_MAX_SUBQUERIES for no benefit.
            new_from_decomp = hybrid_search(
                sub_q, top_k=config.RERANK_TOP_K, use_query_rewriting=False, use_decomposition=False,
            )
            for c in new_from_decomp:
                if c.chunk_id not in seen_ids:
                    seen_ids.add(c.chunk_id)
                    extra_candidates.append(c)

        # C2: single-hop cross-reference resolution, seeded from round-1's
        # top reranked chunks (capped by config.CROSSREF_MAX_HOPS — 1, no
        # recursive chasing of a reference chain).
        if config.CROSSREF_ENABLED and config.CROSSREF_MAX_HOPS >= 1:
            extra_candidates.extend(_resolve_cross_references(reranked, seen_ids))

        if extra_candidates:
            merged = {c.chunk_id: c for c in candidates}
            for c in extra_candidates:
                merged.setdefault(c.chunk_id, c)
            merged_candidates = list(merged.values())
            child_text_by_id.update({c.chunk_id: c.text for c in extra_candidates})

            reranked = rerank(query_text, _enrich_with_parent_text(merged_candidates), top_k=config.FINAL_LAW_TOP_K)
            reranked = [
                c.model_copy(update={"text": child_text_by_id.get(c.chunk_id, c.text)}) for c in reranked
            ]

    return reranked
