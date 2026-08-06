from __future__ import annotations

import logging
import pickle
import re
from functools import lru_cache

from backend import config
from backend.indexing.bm25_index import fold_accents
from backend.models import ChatAnswer, CitedProvision, LawChunk

log = logging.getLogger(__name__)

_RE_DIEU = re.compile(r"\bdieu\s+(\d+)\b", re.IGNORECASE)
_RE_KHOAN = re.compile(r"\bkhoan\s+(\d+)\b", re.IGNORECASE)
_RE_DIEM = re.compile(r"\bdiem\s+([a-z])\b", re.IGNORECASE)

_GENERIC_TAIL_RE = re.compile(
    r"\b(noi gi|la gi|quy dinh gi|quy dinh nhu the nao|noi dung (la gi|the nao)|cho biet|"
    r"quy dinh ra sao|the nao)\b", re.IGNORECASE,
)
_CONDITIONAL_CUES = (
    "neu", "trong truong hop", "khi nao", "ap dung cho", "ap dung nhu the nao",
    "doi voi truong hop", "thi sao", "trong tinh huong", "lam sao neu", "vi sao",
    "tai sao", "so sanh", "khac gi", "khac nhau",
)


class LegalReference:
    __slots__ = ("article_num", "khoan_no", "diem_no", "law_id_hint")

    def __init__(self, article_num, khoan_no=None, diem_no=None, law_id_hint=None):
        self.article_num = article_num
        self.khoan_no = khoan_no
        self.diem_no = diem_no
        self.law_id_hint = law_id_hint


def extract_legal_reference(text: str):
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
        law_id_hint=None,
    )


def is_generic_reference_question(text: str, ref: LegalReference) -> bool:
    folded = fold_accents(text).lower()
    stripped = _RE_DIEU.sub(" ", folded)
    stripped = _RE_KHOAN.sub(" ", stripped)
    stripped = _RE_DIEM.sub(" ", stripped)
    stripped = _GENERIC_TAIL_RE.sub(" ", stripped)
    if any(cue in stripped for cue in _CONDITIONAL_CUES):
        return False
    leftover_words = [w for w in re.findall(r"[a-z]+", stripped) if len(w) > 2]
    return len(leftover_words) == 0


@lru_cache(maxsize=1)
def get_fastpath_index() -> dict:
    try:
        with open(config.ARTICLE_NUM_LOOKUP_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        log.warning("citation fast-path index not found at %s — fast-path is effectively disabled until "
                    "scripts/build_index.py is (re)run.", config.ARTICLE_NUM_LOOKUP_PATH)
        return {"article_lookup": {}, "khoan_lookup": {}, "chunk_by_id": {}}


def lookup_exact_chunk(ref: LegalReference):
    idx = get_fastpath_index()
    if ref.khoan_no is not None:
        matches = {
            law_id: ids for (law_id, art_num, khoan_no), ids in idx["khoan_lookup"].items()
            if art_num == ref.article_num and khoan_no == ref.khoan_no
            and (ref.law_id_hint is None or law_id == ref.law_id_hint)
        }
        if len(matches) != 1:
            return None
        (chunk_ids,) = matches.values()
        chunks = [idx["chunk_by_id"][cid] for cid in chunk_ids if cid in idx["chunk_by_id"]]
        if ref.diem_no is not None:
            narrowed = [c for c in chunks if (c.diem_no or "").lower() == ref.diem_no.lower()]
            if narrowed:
                chunks = narrowed
        return chunks or None

    matches = {
        law_id: ids for (law_id, art_num), ids in idx["article_lookup"].items()
        if art_num == ref.article_num and (ref.law_id_hint is None or law_id == ref.law_id_hint)
    }
    if len(matches) != 1:
        return None
    (chunk_ids,) = matches.values()
    if len(chunk_ids) != 1:
        return None
    return idx["chunk_by_id"].get(chunk_ids[0])


def build_fastpath_answer(target) -> ChatAnswer:
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


def try_lookup(question: str):
    ref = extract_legal_reference(question)
    if ref is None:
        return None
    if not is_generic_reference_question(question, ref):
        return None
    target = lookup_exact_chunk(ref)
    if target is None:
        return None
    return build_fastpath_answer(target)
