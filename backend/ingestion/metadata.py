from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

from backend.ingestion.parser import RawLawDocument
from backend.models import LawMetadata

RE_CROSS_REF = re.compile(
    r"(?:Điều\s+(\d+)(?:\s*,\s*khoản\s+(\d+))?)\s*(?:của\s+)?([\w\d]+/\d{4}/[A-ZĐ\-]+)?",
    re.IGNORECASE,
)


def _parse_date(value):
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def infer_status(doc: RawLawDocument, as_of=None) -> str:
    as_of = as_of or date.today()
    expiry = _parse_date(doc.expiry_date)
    effective = _parse_date(doc.effective_date)
    if expiry and expiry <= as_of:
        return "expired"
    if effective and effective > as_of:
        return "unknown"
    return "active"


def build_metadata(doc: RawLawDocument, superseded_by=None, supersedes=None) -> LawMetadata:
    return LawMetadata(
        law_id=doc.law_id, doc_type=doc.doc_type, issuing_body=doc.issuing_body,
        issue_date=doc.issue_date, effective_date=doc.effective_date, expiry_date=doc.expiry_date,
        status=infer_status(doc), superseded_by=superseded_by, supersedes=supersedes,
    )


def extract_cross_references(article_text: str, default_law_id: str) -> list:
    refs = []
    for m in RE_CROSS_REF.finditer(article_text):
        aid_str, _khoan, law_num = m.groups()
        if not aid_str:
            continue
        refs.append({"law_id": law_num or default_law_id, "aid": int(aid_str)})
    return refs


def metadata_filter(metadatas: dict, law_id=None, require_active=True, doc_type=None) -> set:
    allowed = set()
    for lid, meta in metadatas.items():
        if law_id and lid != law_id:
            continue
        if doc_type and meta.doc_type != doc_type:
            continue
        if require_active and meta.status not in ("active", "unknown"):
            continue
        allowed.add(lid)
    return allowed
