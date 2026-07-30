"""
Hierarchical chunking for the law corpus (design doc §2.2).

Parent chunk  = the whole Điều (article) — used to re-attach full context
                to the LLM once a child chunk has been retrieved.
Child chunk   = an individual Khoản/Điểm inside the article — the unit that
                actually gets embedded and searched.

Chunking is entirely rule-based on the Chương > Mục > Điều > Khoản > Điểm
structure (never a fixed token-window split) — a legal clause's meaning
frequently hinges on a preceding qualifier like "trừ trường hợp" that a hard
token cut could separate from its clause.

CHATBOT_MIGRATION_PLAN.md §2.1 / §D2 (applied here):

`article.aid` is the corpus's internal/global article id — for the ALQAC
corpus this was confirmed (system_adjustments_v4.md §2.3, ACTION_PLAN.md §A4)
to NOT equal the printed "Điều N" number in the vast majority of articles
(observed aids like 50882, 53082 vs. small "Điều N" numbers). The OLD
`_breadcrumb()` baked `f"Điều {aid}"` straight from that internal id, which
is fine for round-tripping citations back to the same chunk (a stable key),
but is actively MISLEADING if ever shown to a person asking a legal
question — a user citing "Điều 50882" would rightly be confused, unlike the
ALQAC competition where breadcrumbs were never rendered to a human.

Fix: each chunk now also carries `article_num` — the REAL printed article
number, parsed from the article's own title/body text the same way
ACTION_PLAN.md §A4's `build_aid_to_article_num_map()` did for evaluation.
`aid` remains the stable internal identifier used everywhere citations are
verified against (`allowed_citation_keys()` in prompt_builder.py, the
hallucination guard in generate.py) — only *display* text should ever
prefer `article_num`. See `backend/generation/prompt_builder.py` for where
this is consumed.
"""
from __future__ import annotations

import importlib.util
import re
import unicodedata

from backend import config
from backend.ingestion.parser import RawArticle, split_article_into_khoan_diem
from backend.models import LawChunk

# Same pattern used by ACTION_PLAN.md §A4 / test_all_backend.py to recover
# the printed article number from free text.
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


def _soft_split_oversized(text: str, max_chars: int = config.CHILD_MAX_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    sentences = [s for s in _SENTENCE_BOUNDARY_RE.split(text) if s]
    if not sentences:
        return [text]

    parts: list[str] = []
    buf = ""
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
    """Return an ASCII-only identifier component safe for Pinecone vector IDs."""
    value = value.replace("Đ", "D").replace("đ", "d")
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_value = re.sub(r"[^A-Za-z0-9_.-]+", "-", ascii_value)
    return ascii_value.strip("-") or "chunk"


def _parse_real_article_num(article: RawArticle) -> int | None:
    """Best-effort recovery of the printed 'Điều N' number for DISPLAY
    purposes only (D2). Never used for citation identity — `aid` stays the
    identity key. Falls back to None (caller then falls back to `aid`) if
    neither title nor body carries a parseable 'Điều N'."""
    m = _RE_DIEU_NUM.search(article.title or "")
    if m:
        return int(m.group(1))
    m = _RE_DIEU_NUM.search(article.body or "")
    if m:
        return int(m.group(1))
    return None


def _breadcrumb(law_id: str, chuong: str | None, muc: str | None, aid: int,
                 khoan_no: str | None = None, diem_no: str | None = None,
                 article_num: int | None = None) -> str:
    parts = [f"Luật {law_id}"]
    if chuong:
        parts.append(chuong)
    if muc:
        parts.append(muc)
    # D2: prefer the REAL printed article number for the human-readable
    # breadcrumb; fall back to the internal aid only when we couldn't parse
    # one (keeps old behavior for corpora where aid genuinely *is* the
    # printed number).
    parts.append(f"Điều {article_num if article_num is not None else aid}")
    if khoan_no:
        parts.append(f"Khoản {khoan_no}")
    if diem_no:
        parts.append(f"Điểm {diem_no}")
    return " > ".join(parts)


def chunk_article(article: RawArticle) -> list[LawChunk]:
    """Produce one parent chunk + N child chunks for a single article."""
    chunks: list[LawChunk] = []

    article_num = _parse_real_article_num(article)

    law_id_for_chunk_id = _ascii_id(article.law_id)
    parent_id = f"{law_id_for_chunk_id}_a{article.aid}"
    parent_text = (article.title + "\n" + article.body).strip()
    chunks.append(
        LawChunk(
            chunk_id=parent_id,
            law_id=article.law_id,
            aid=article.aid,
            article_num=article_num,
            breadcrumb=_breadcrumb(article.law_id, article.chuong, article.muc, article.aid,
                                    article_num=article_num),
            level="parent",
            parent_id=None,
            text=parent_text,
            token_count=_count_tokens(parent_text),
        )
    )

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
            chunks.append(
                LawChunk(
                    chunk_id=child_id,
                    law_id=article.law_id,
                    aid=article.aid,
                    article_num=article_num,
                    breadcrumb=_breadcrumb(
                        article.law_id, article.chuong, article.muc, article.aid,
                        split.khoan_no, split.diem_no, article_num=article_num,
                    ),
                    level="child",
                    parent_id=parent_id,
                    khoan_no=split.khoan_no,
                    diem_no=split.diem_no,
                    text=part_text,
                    token_count=_count_tokens(part_text),
                )
            )
    return chunks


def chunk_articles(articles: list[RawArticle]) -> list[LawChunk]:
    all_chunks: list[LawChunk] = []
    for art in articles:
        all_chunks.extend(chunk_article(art))
    return all_chunks


def build_parent_lookup(chunks: list[LawChunk]) -> dict[str, LawChunk]:
    """Map parent_id -> parent LawChunk, so retrieval can re-attach full
    article context to a matched child chunk at generation time."""
    return {c.chunk_id: c for c in chunks if c.level == "parent"}


def build_article_num_lookup(chunks: list[LawChunk]) -> dict[tuple[str, int], list[str]]:
    """CHATBOT_MIGRATION_PLAN.md §A3 (citation-fast-path skill): map
    (law_id, REAL printed article_num) -> [parent_chunk_id, ...] so a user
    query like "Điều 12 khoản 2" can be resolved directly, without a
    hybrid_search round-trip. A list (not a single id) is returned because
    an ambiguous article number shared by >1 law_id is exactly the signal
    the fast-path must use to bail out (see citation_fastpath.py).

    Only articles with a successfully-parsed `article_num` participate —
    articles where the real number couldn't be recovered are simply absent
    from this index (fast-path correctly falls through to full retrieval
    for those instead of guessing).
    """
    lookup: dict[tuple[str, int], list[str]] = {}
    for c in chunks:
        if c.level != "parent" or c.article_num is None:
            continue
        lookup.setdefault((c.law_id, c.article_num), []).append(c.chunk_id)
    return lookup


def build_khoan_lookup(chunks: list[LawChunk]) -> dict[tuple[str, int, str], list[str]]:
    """citation-fast-path skill: map (law_id, article_num, khoan_no) -> child
    chunk_id(s) covering that Khoản (possibly several if it was soft-split,
    see `_soft_split_oversized`), so a question like "Điều 12 khoản 2 nói
    gì?" can be answered with exactly that Khoản's verbatim text instead of
    the whole article. Điểm-level narrowing (when the user also names a
    specific điểm) is done at answer-build time by filtering
    `LawChunk.diem_no` among the chunks this returns, not by a separate
    lookup table — the same Khoản rarely soft-splits AND has Điểm children
    at once for the fast-path's target use case, so this stays a single,
    simple index.
    """
    lookup: dict[tuple[str, int, str], list[str]] = {}
    for c in chunks:
        if c.level != "child" or c.article_num is None or c.khoan_no is None:
            continue
        lookup.setdefault((c.law_id, c.article_num, c.khoan_no), []).append(c.chunk_id)
    return lookup
