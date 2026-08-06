from __future__ import annotations
import logging
from typing import Optional
from backend import config
from backend.indexing import bm25_index, vector_store
from backend.models import RetrievedChunk
from backend.retrieval.ner import extract_entities, mask_person_org_entities
from backend.retrieval.querry_transform import decompose_query, rewrite_query

log = logging.getLogger(__name__)

_vector_unavailable_warned = False


def _safe_vector_query(text: str, **kwargs) -> list:
    """Bọc vector_store.query() để một lỗi Pinecone (mất kết nối, chưa cấu
    hình PINECONE_API_KEY, index chưa tồn tại, thiếu package `pinecone`...)
    làm hybrid_search DEGRADE về BM25-only thay vì crash toàn bộ chat turn.

    Phát hiện qua smoke-test thực tế (coding_plan.md A1): trước bản vá này,
    lỗi Pinecone propagate thẳng lên handle_chat_turn() không bị bắt ở đâu
    — KHÔNG nhất quán với triết lý "một bước phụ lỗi không được chặn cả
    pipeline" đã áp dụng cho condense_question/decompose_query/
    rewrite_query/build_conversation_digest. Mất kênh vector làm giảm
    recall (còn BM25 + BM25-folded), không phải lý do từ chối cả câu hỏi.

    Chỉ log warning MỘT LẦN mỗi process để tránh ngập log khi Pinecone down
    kéo dài trong thực tế.
    """
    global _vector_unavailable_warned
    try:
        return vector_store.query(text, **kwargs)
    except Exception as e:
        if not _vector_unavailable_warned:
            log.warning(
                "vector_store.query() thất bại (%s: %s) — degrade về BM25-only. Kiểm tra "
                "PINECONE_API_KEY/kết nối mạng nếu đây không phải môi trường dev/sandbox.",
                type(e).__name__, e,
            )
            _vector_unavailable_warned = True
        return []


def _rrf_fuse_weighted(channels: list, k: int = config.RRF_K) -> list:
    scores: dict = {}
    best_chunk: dict = {}
    for results, weight in channels:
        for rank, chunk in enumerate(results, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + weight / (k + rank)
            best_chunk.setdefault(chunk.chunk_id, chunk)
    fused = [
        RetrievedChunk(chunk_id=cid, law_id=best_chunk[cid].law_id, aid=best_chunk[cid].aid,
                        article_num=best_chunk[cid].article_num, text=best_chunk[cid].text, score=score, source="fused")
        for cid, score in scores.items()
    ]
    fused.sort(key=lambda c: c.score, reverse=True)
    return fused


def hybrid_search(query: str, law_id: Optional[str] = None, require_active: bool = True,
                   use_query_rewriting: bool = True, use_decomposition: bool = True, top_k: int = 30) -> list:
    bm25 = bm25_index.get_bm25_index()
    entities = extract_entities(query)
    law_query = mask_person_org_entities(query, entities)
    channels = []
    base_queries = rewrite_query(law_query) if use_query_rewriting else [law_query]
    for q in base_queries:
        channels.append((bm25.query(q, top_k=config.BM25_TOP_K, law_id=law_id, require_active=require_active), config.RRF_WEIGHT_STANDARD))
        channels.append((_safe_vector_query(q, top_k=config.VECTOR_TOP_K, law_id=law_id, require_active=require_active), config.RRF_WEIGHT_STANDARD))
    channels.append((bm25.query_folded(law_query, top_k=config.BM25_TOP_K, law_id=law_id, require_active=require_active), config.RRF_WEIGHT_FOLDED))
    if use_decomposition and config.QUERY_DECOMPOSITION_ENABLED:
        for sub_q in decompose_query(query, masked_query=law_query, n_subqueries=config.QUERY_DECOMPOSITION_MAX_SUBQUERIES):
            channels.append((_safe_vector_query(sub_q, top_k=config.VECTOR_TOP_K, law_id=law_id, require_active=require_active), config.RRF_WEIGHT_AGENT))
    fused = _rrf_fuse_weighted(channels)
    return fused[:top_k]
