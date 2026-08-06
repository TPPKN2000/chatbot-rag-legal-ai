from __future__ import annotations
from functools import lru_cache
from backend import config
from backend.models import RetrievedChunk


@lru_cache(maxsize=1)
def _get_reranker():
    from sentence_transformers import CrossEncoder
    device = config.DEVICE
    try:
        import torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
    except ImportError:
        device = "cpu"
    return CrossEncoder(config.RERANKER_MODEL_NAME, device=device)


def rerank(query: str, candidates: list, top_k: int = config.FINAL_LAW_TOP_K) -> list:
    if not candidates:
        return []
    model = _get_reranker()
    pairs = [(query, c.text) for c in candidates]
    scores = model.predict(pairs)
    reranked = [
        RetrievedChunk(chunk_id=c.chunk_id, law_id=c.law_id, aid=c.aid, article_num=c.article_num,
                        text=c.text, score=float(s), source="reranked")
        for c, s in zip(candidates, scores)
    ]
    reranked.sort(key=lambda c: c.score, reverse=True)
    return reranked[:top_k]
