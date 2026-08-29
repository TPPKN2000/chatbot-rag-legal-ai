from __future__ import annotations

import importlib.util
import re
import unicodedata

from backend import config
from backend.ingestion.parser import RawArticle, split_article_into_khoan_diem
from backend.models import LawChunk

_RE_DIEU_NUM = re.compile(r"Điều\s+(\d+)", re.IGNORECASE)


def _fallback_count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _load_token_encoder():
    if importlib.util.find_spec("tiktoken") is None:
        return None
    import tiktoken
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


_ENC = _load_token_encoder()


def _count_tokens(text: str) -> int:
    if _ENC is None:
        return _fallback_count_tokens(text)
    return len(_ENC.encode(text))


_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.;])\s+")


def _soft_split_oversized(text: str, max_chars: int = config.CHILD_MAX_CHARS) -> list:
    if len(text) <= max_chars:
        return [text]
    sentences = [s for s in _SENTENCE_BOUNDARY_RE.split(text) if s]
    if not sentences:
        return [text]
    parts, buf = [], ""
    for s in sentences:
        if buf and len(buf) + 1 + len(s) > max_chars:
            parts.append(buf.strip())
            buf = s
        else:
            buf = f"{buf} {s}".strip() if buf else s
    if buf:
        parts.append(buf.strip())
    return parts or [text]


def _ascii_id(value: str) -> str:
    value = value.replace("Đ", "D").replace("đ", "d")
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_value = re.sub(r"[^A-Za-z0-9_.-]+", "-", ascii_value)
    return ascii_value.strip("-") or "chunk"


def _parse_real_article_num(article: RawArticle):
    m = _RE_DIEU_NUM.search(article.title or "")
    if m:
        return int(m.group(1))
    m = _RE_DIEU_NUM.search(article.body or "")
    if m:
        return int(m.group(1))
    return None


def _breadcrumb(law_id, chuong, muc, aid, khoan_no=None, diem_no=None, article_num=None) -> str:
    parts = [f"Luật {law_id}"]
    if chuong:
        parts.append(chuong)
    if muc:
        parts.append(muc)
    parts.append(f"Điều {article_num if article_num is not None else aid}")
    if khoan_no:
        parts.append(f"Khoản {khoan_no}")
    if diem_no:
        parts.append(f"Điểm {diem_no}")
    return " > ".join(parts)


def chunk_article(article: RawArticle) -> list:
    chunks = []
    article_num = _parse_real_article_num(article)
    law_id_for_chunk_id = _ascii_id(article.law_id)
    parent_id = f"{law_id_for_chunk_id}_a{article.aid}"
    parent_text = (article.title + "\n" + article.body).strip()
    chunks.append(LawChunk(
        chunk_id=parent_id, law_id=article.law_id, aid=article.aid, article_num=article_num,
        breadcrumb=_breadcrumb(article.law_id, article.chuong, article.muc, article.aid, article_num=article_num),
        level="parent", parent_id=None, text=parent_text, token_count=_count_tokens(parent_text),
    ))

    splits = split_article_into_khoan_diem(article.body)
    for i, split in enumerate(splits):
        if not split.text:
            continue
        suffix = f"_k{split.khoan_no or i}"
        if split.diem_no:
            suffix += f"_d{split.diem_no}"
        sub_parts = _soft_split_oversized(split.text)
        for j, part_text in enumerate(sub_parts):
            part_suffix = suffix if len(sub_parts) == 1 else f"{suffix}_p{j}"
            child_id = f"{parent_id}{part_suffix}"
            chunks.append(LawChunk(
                chunk_id=child_id, law_id=article.law_id, aid=article.aid, article_num=article_num,
                breadcrumb=_breadcrumb(article.law_id, article.chuong, article.muc, article.aid,
                                        split.khoan_no, split.diem_no, article_num=article_num),
                level="child", parent_id=parent_id, khoan_no=split.khoan_no, diem_no=split.diem_no,
                text=part_text, token_count=_count_tokens(part_text),
            ))
    return chunks


def _ascii_id_component(value: str) -> str:
    """ASCII-hoá một thành phần ngắn (chữ cái điểm/khoản, vd. 'đ' -> 'd') để
    dùng trong chunk_id/Pinecone vector id. KHÔNG dùng cho breadcrumb hiển
    thị — chỉ cho chuỗi ID. Nguyên nhân gốc: RE_DIEM trong parser.py khớp cả
    'đ' (điểm sau 'd' trong liệt kê a) b) ... d) đ) e)...), ký tự này không
    phải ASCII khiến Pinecone từ chối upsert."""
    if not value:
        return value
    folded = value.replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize("NFKD", folded)
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")


def chunk_articles(articles: list) -> list:
    splits = split_article_into_khoan_diem(article.body)
    for i, split in enumerate(splits):
        if not split.text:
            continue
        suffix = f"_k{split.khoan_no or i}"
        if split.diem_no:
            suffix += f"_d{split.diem_no}"


def build_parent_lookup(chunks: list) -> dict:
    return {c.chunk_id: c for c in chunks if c.level == "parent"}


def build_article_num_lookup(chunks: list) -> dict:
    lookup: dict = {}
    for c in chunks:
        if c.level != "parent" or c.article_num is None:
            continue
        lookup.setdefault((c.law_id, c.article_num), []).append(c.chunk_id)
    return lookup


def build_khoan_lookup(chunks: list) -> dict:
    lookup: dict = {}
    for c in chunks:
        if c.level != "child" or c.article_num is None or c.khoan_no is None:
            continue
        lookup.setdefault((c.law_id, c.article_num, c.khoan_no), []).append(c.chunk_id)
    return lookup
