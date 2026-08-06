from __future__ import annotations

from functools import lru_cache
from typing import Optional

from backend import config
from backend.indexing.embed import embed_query, embed_texts
from backend.models import LawChunk, RetrievedChunk


@lru_cache(maxsize=1)
def _get_client():
    from pinecone import Pinecone
    if not config.PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is not set. Add it to your .env file (see README §Configuration).")
    return Pinecone(api_key=config.PINECONE_API_KEY)


def ensure_index() -> None:
    from pinecone import ServerlessSpec
    pc = _get_client()
    existing = {i["name"] for i in pc.list_indexes()}
    if config.PINECONE_INDEX_NAME not in existing:
        pc.create_index(name=config.PINECONE_INDEX_NAME, dimension=config.EMBEDDING_DIM, metric="cosine",
                         spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION))


@lru_cache(maxsize=1)
def _get_index():
    ensure_index()
    return _get_client().Index(config.PINECONE_INDEX_NAME)


def reset_index_cache() -> None:
    _get_index.cache_clear()
    _get_client.cache_clear()


def _chunk_metadata(chunk: LawChunk, extra: Optional[dict] = None) -> dict:
    meta = {
        "law_id": chunk.law_id, "aid": chunk.aid,
        "article_num": chunk.article_num if chunk.article_num is not None else -1,
        "level": chunk.level, "parent_id": chunk.parent_id or "",
        "breadcrumb": chunk.breadcrumb, "text": chunk.text,
    }
    if extra:
        meta.update(extra)
    return meta


def upsert_chunks(chunks, status_by_law=None, batch_size: int = 100) -> int:
    index = _get_index()
    child_chunks = [c for c in chunks if c.level == "child"]
    status_by_law = status_by_law or {}
    count = 0
    for i in range(0, len(child_chunks), batch_size):
        batch = child_chunks[i:i + batch_size]
        vectors = embed_texts([c.text for c in batch])
        upserts = []
        for chunk, vec in zip(batch, vectors):
            extra = {"status": status_by_law.get(chunk.law_id, "unknown")}
            upserts.append({"id": chunk.chunk_id, "values": vec.tolist(), "metadata": _chunk_metadata(chunk, extra)})
        index.upsert(vectors=upserts, namespace=config.PINECONE_NAMESPACE)
        count += len(upserts)
    return count


def query(text: str, top_k: int = config.VECTOR_TOP_K, law_id=None, require_active: bool = True) -> list:
    index = _get_index()
    vec = embed_query(text)
    flt: dict = {}
    if law_id:
        flt["law_id"] = {"$eq": law_id}
    if require_active:
        flt["status"] = {"$in": ["active", "unknown"]}
    res = index.query(vector=vec, top_k=top_k, namespace=config.PINECONE_NAMESPACE, include_metadata=True, filter=flt or None)
    out = []
    for match in res.get("matches", []):
        md = match.get("metadata", {})
        raw_article_num = md.get("article_num", -1)
        out.append(RetrievedChunk(
            chunk_id=match["id"], law_id=md.get("law_id", ""), aid=int(md.get("aid", -1)),
            article_num=int(raw_article_num) if raw_article_num not in (None, -1) else None,
            text=md.get("text", ""), score=float(match.get("score", 0.0)), source="vector",
        ))
    return out


def delete_namespace() -> None:
    _get_index().delete(delete_all=True, namespace=config.PINECONE_NAMESPACE)
