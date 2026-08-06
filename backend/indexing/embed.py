from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np

from backend import config


@lru_cache(maxsize=1)
def _get_embedder():
    from sentence_transformers import SentenceTransformer
    device = config.DEVICE
    try:
        import torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
    except ImportError:
        device = "cpu"
    return SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=device)


def embed_texts(texts: Sequence[str], batch_size: int = 32, normalize: bool = True, is_query: bool = False) -> np.ndarray:
    """B2 (coding_plan.md): `is_query=True` prefixes each text with
    `config.EMBEDDING_QUERY_INSTRUCTION` trước khi encode.

    Lý do cần tham số này: một số model instruction-aware (vd.
    Qwen3-Embedding) BẤT ĐỐI XỨNG giữa query/document — CHỈ query cần
    prefix, document encode thô. Đảo ngược 2 nhánh này sẽ âm thầm làm giảm
    chất lượng retrieval mà KHÔNG raise lỗi gì — nguy hiểm hơn một crash.

    Với model hiện tại (AITeamVN/Vietnamese_Embedding, KHÔNG instruction-aware),
    `EMBEDDING_QUERY_INSTRUCTION` không có tác dụng nếu để rỗng — tham số này
    an toàn để giữ mặc định `is_query=False` ở mọi call site cũ (document
    encoding trong vector_store.upsert_chunks không đổi hành vi).

    `truncate_dim=config.EMBEDDING_DIM` chỉ được truyền khi thực sự cần MRL
    (model hỗ trợ Matryoshka Representation Learning, vd. Qwen3-Embedding) —
    sentence-transformers bỏ qua tham số này êm nếu model không hỗ trợ, NHƯNG
    yêu cầu sentence-transformers>=3.1. Vẫn giữ optional (try/except) để
    không phá vỡ môi trường đang dùng bản cũ hơn.
    """
    if not texts:
        return np.zeros((0, config.EMBEDDING_DIM), dtype="float32")
    if is_query and config.EMBEDDING_QUERY_INSTRUCTION:
        texts = [f"Instruct: {config.EMBEDDING_QUERY_INSTRUCTION}\nQuery: {t}" for t in texts]
    model = _get_embedder()
    encode_kwargs = dict(batch_size=batch_size, normalize_embeddings=normalize, convert_to_numpy=True,
                          show_progress_bar=False)
    if config.EMBEDDING_USE_MRL_TRUNCATE:
        try:
            vectors = model.encode(list(texts), truncate_dim=config.EMBEDDING_DIM, **encode_kwargs)
        except TypeError:
            # sentence-transformers < 3.1 không có truncate_dim — degrade về
            # encode thường thay vì crash (EMBEDDING_DIM khi đó phải khớp
            # sẵn dimension gốc của model, người vận hành tự đảm bảo).
            vectors = model.encode(list(texts), **encode_kwargs)
    else:
        vectors = model.encode(list(texts), **encode_kwargs)
    return vectors.astype("float32")


def embed_query(text: str) -> list:
    """Luôn qua nhánh is_query=True — instruction-prefix chỉ áp dụng ở đây,
    KHÔNG áp dụng trong upsert_chunks() (document encoding)."""
    return embed_texts([text], is_query=True)[0].tolist()
