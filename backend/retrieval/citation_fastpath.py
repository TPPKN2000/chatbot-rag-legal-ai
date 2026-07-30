"""
Citation fast-path (CHATBOT_MIGRATION_PLAN.md §A3 / citation-fast-path skill).

When a question is a pure lookup of one specific Điều/Khoản/Điểm ("Điều 12
khoản 2 nói gì?", "dieu 5 quy dinh gi" without diacritics), running the full
hybrid_search -> rerank -> LLM-generate pipeline is:
  1. Unnecessarily slow (multiple BM25/vector/LLM calls for a lookup).
  2. A paraphrase-risk: the LLM could reword the provision and silently
     invert its meaning (dropping a "trừ trường hợp..." qualifier) — the
     same verbatim-text principle already enforced in generation/compress.py
     and prompt_builder.py applies with even more force here, since there's
     no reason to risk it at all for a pure lookup.
  3. Redundant: if (law_id, article_num[, khoan, diem]) is uniquely
     resolvable, the answer IS a lookup, not an inference.

Ported/adapted from the ideas documented in
skills/legacy-prototype-salvage/SKILL.md (AI/data_utils.py::
extract_legal_reference / is_generic_reference_question,
AI/legal_spans.py::extract_structured_answer, AI/retrieval.py's
accent-folding + MIN_RETRIEVAL_CONFIDENCE abstention principle) — those
modules aren't present in this checkout to import directly, so the regex
and heuristics are re-implemented here against `backend/`'s own data model
(LawChunk / chunker.py's article_num + khoan_no/diem_no fields), not copied
verbatim.

Activation requires ALL of:
  - The (contextualized) question parses to a specific article_num via
    `extract_legal_reference()`.
  - It reads as a GENERIC reference question (`is_generic_reference_question`)
    — no additional situational/conditional content ("nếu... thì sao").
  - Exactly ONE (law_id, article_num) combination resolves in the corpus.
    Ambiguous (>1 law sharing that article number, and no law hinted) ->
    fast-path declines, caller falls through to full retrieval.

If any condition fails, `try_lookup()` returns None and the caller
(`backend/chat_pipeline.py`) must proceed with the normal pipeline.
"""
from __future__ import annotations

import logging
import pickle
import re
from functools import lru_cache

from backend import config
from backend.indexing.bm25_index import fold_accents
from backend.models import ChatAnswer, CitedProvision, LawChunk

log = logging.getLogger(__name__)

# --- Reference parsing (accent-insensitive: matched against folded text) ---
_RE_DIEU = re.compile(r"\bdieu\s+(\d+)\b", re.IGNORECASE)
_RE_KHOAN = re.compile(r"\bkhoan\s+(\d+)\b", re.IGNORECASE)
_RE_DIEM = re.compile(r"\bdiem\s+([a-z])\b", re.IGNORECASE)

# Generic "tell me the content" tails vs. situational/conditional cues that
# signal the question needs actual reasoning, not a lookup.
_GENERIC_TAIL_RE = re.compile(
    r"\b(noi gi|la gi|quy dinh gi|quy dinh nhu the nao|noi dung (la gi|the nao)|cho biet|"
    r"quy dinh ra sao|the nao)\b",
    re.IGNORECASE,
)
_CONDITIONAL_CUES = (
    "neu", "trong truong hop", "khi nao", "ap dung cho", "ap dung nhu the nao",
    "doi voi truong hop", "thi sao", "trong tinh huong", "lam sao neu", "vi sao",
    "tai sao", "so sanh", "khac gi", "khac nhau",
)


class LegalReference:
    __slots__ = ("article_num", "khoan_no", "diem_no", "law_id_hint")

    def __init__(self, article_num: int, khoan_no: str | None = None,
                 diem_no: str | None = None, law_id_hint: str | None = None):
        self.article_num = article_num
        self.khoan_no = khoan_no
        self.diem_no = diem_no
        self.law_id_hint = law_id_hint


def extract_legal_reference(text: str) -> LegalReference | None:
    """Parse "Điều X [khoản Y] [điểm Z]" out of free text, regardless of word
    order or diacritics. Returns None if no article number is found at all
    — a bare "khoản 2" with no "Điều" is not resolvable on its own."""
    folded = fold_accents(text)

    m_dieu = _RE_DIEU.search(folded)
    if not m_dieu:
        return None

    article_num = int(m_dieu.group(1))
    m_khoan = _RE_KHOAN.search(folded)
    m_diem = _RE_DIEM.search(folded)

    return LegalReference(
        article_num=article_num,
        khoan_no=m_khoan.group(1) if m_khoan else None,
        diem_no=m_diem.group(1) if m_diem else None,
        # NOTE: no law_id_hint extraction attempted — corpus law_ids (e.g.
        # "47/2010/QH12") are rarely spoken/typed by a user verbatim. When
        # this stays None and the article number is ambiguous across laws,
        # `try_lookup()` correctly declines rather than guessing.
        law_id_hint=None,
    )


def is_generic_reference_question(text: str, ref: LegalReference) -> bool:
    """True iff the question is asking for the ENTIRE content of the
    referenced unit ("Điều 12 khoản 2 nói gì?"), not for how it applies to a
    specific situation ("Điều 12 áp dụng thế nào cho ly hôn đơn phương?").
    Only generic reference questions are eligible for the fast-path — an
    applied question needs actual reasoning over facts, which is exactly
    what the fast-path must NOT attempt (it never calls an LLM)."""
    folded = fold_accents(text).lower()

    # Strip the reference tokens themselves and generic "what does it say"
    # tails before checking what's left.
    stripped = _RE_DIEU.sub(" ", folded)
    stripped = _RE_KHOAN.sub(" ", stripped)
    stripped = _RE_DIEM.sub(" ", stripped)
    stripped = _GENERIC_TAIL_RE.sub(" ", stripped)

    if any(cue in stripped for cue in _CONDITIONAL_CUES):
        return False

    # Anything substantial left over (beyond stray punctuation/short filler
    # words) signals extra situational content the fast-path can't handle.
    leftover_words = [w for w in re.findall(r"[a-z]+", stripped) if len(w) > 2]
    return len(leftover_words) == 0


# ---------------------------------------------------------------------------
# Index loading (built by scripts/build_index.py alongside the parent lookup)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_fastpath_index() -> dict:
    """{"article_lookup": {(law_id, article_num): [chunk_id,...]},
        "khoan_lookup": {(law_id, article_num, khoan_no): [chunk_id,...]},
        "chunk_by_id": {chunk_id: LawChunk}}
    Returns an empty-shell dict (fast-path always declines) if the index
    hasn't been built yet, rather than raising — a missing fast-path index
    must degrade to "always use full retrieval", never crash the chatbot.
    """
    try:
        with open(config.ARTICLE_NUM_LOOKUP_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        log.warning(
            "citation fast-path index not found at %s — fast-path is effectively "
            "disabled until scripts/build_index.py is (re)run.",
            config.ARTICLE_NUM_LOOKUP_PATH,
        )
        return {"article_lookup": {}, "khoan_lookup": {}, "chunk_by_id": {}}


def lookup_exact_chunk(ref: LegalReference) -> LawChunk | list[LawChunk] | None:
    """Resolve `ref` to exactly one target:
      - a single parent LawChunk (whole article) if no khoản was named,
      - a list of child LawChunks covering that Khoản if one was named
        (a list because an oversized Khoản may have been soft-split),
      - None if unresolvable OR ambiguous (>1 law_id matches and no hint).
    """
    idx = get_fastpath_index()

    if ref.khoan_no is not None:
        matches = {
            law_id: ids
            for (law_id, art_num, khoan_no), ids in idx["khoan_lookup"].items()
            if art_num == ref.article_num and khoan_no == ref.khoan_no
            and (ref.law_id_hint is None or law_id == ref.law_id_hint)
        }
        if len(matches) != 1:
            return None  # not found, or ambiguous across laws
        (chunk_ids,) = matches.values()
        chunks = [idx["chunk_by_id"][cid] for cid in chunk_ids if cid in idx["chunk_by_id"]]
        if ref.diem_no is not None:
            narrowed = [c for c in chunks if (c.diem_no or "").lower() == ref.diem_no.lower()]
            if narrowed:
                chunks = narrowed
        return chunks or None

    matches = {
        law_id: ids
        for (law_id, art_num), ids in idx["article_lookup"].items()
        if art_num == ref.article_num and (ref.law_id_hint is None or law_id == ref.law_id_hint)
    }
    if len(matches) != 1:
        return None  # not found, or ambiguous — caller falls through to full retrieval
    (chunk_ids,) = matches.values()
    if len(chunk_ids) != 1:
        return None
    return idx["chunk_by_id"].get(chunk_ids[0])


def build_fastpath_answer(target: LawChunk | list[LawChunk]) -> ChatAnswer:
    """Build the verbatim answer. NEVER paraphrases — uses `chunk.text`
    exactly as stored (same "never compress/reword verbatim law text"
    principle as generation/compress.py), with only a short lead-in
    sentence added ("Theo {breadcrumb}:")."""
    chunks = target if isinstance(target, list) else [target]
    chunks = sorted(chunks, key=lambda c: (c.diem_no or "", c.khoan_no or ""))

    lead = chunks[0].breadcrumb
    body = "\n".join(c.text.strip() for c in chunks)
    answer_text = f"Theo {lead}:\n\n{body}"

    citations = [
        CitedProvision(law_id=c.law_id, aid=c.aid, article_num=c.article_num)
        for c in {c.chunk_id: c for c in chunks}.values()
    ]
    return ChatAnswer(answer=answer_text, citations=citations, is_fastpath=True)


def try_lookup(question: str) -> ChatAnswer | None:
    """Top-level entry point for `chat_pipeline.py`. Returns a ready-to-send
    `ChatAnswer` if the fast-path applies, else None (caller must fall
    through to hybrid_search -> rerank -> generate)."""
    ref = extract_legal_reference(question)
    if ref is None:
        return None
    if not is_generic_reference_question(question, ref):
        return None
    target = lookup_exact_chunk(ref)
    if target is None:
        return None
    return build_fastpath_answer(target)
