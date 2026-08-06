from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

RE_DIEU = re.compile(r"^\s*Điều\s+(\d+)\s*[.:]?\s*(.*)$", re.MULTILINE)
RE_KHOAN = re.compile(r"^\s*(\d+)\s*\.\s+")
RE_DIEM = re.compile(r"^\s*([a-zđ])\s*\)\s+", re.IGNORECASE)


@dataclass
class RawArticle:
    law_id: str
    aid: int
    title: str = ""
    chuong: Optional[str] = None
    muc: Optional[str] = None
    body: str = ""


@dataclass
class RawLawDocument:
    law_id: str
    doc_type: Optional[str] = None
    issuing_body: Optional[str] = None
    issue_date: Optional[str] = None
    effective_date: Optional[str] = None
    expiry_date: Optional[str] = None
    articles: list = field(default_factory=list)


def load_law_corpus(path):
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("laws") or raw.get("data") or [raw]

    docs = []
    for law in raw:
        law_id = str(law.get("law_id") or law.get("id"))
        doc = RawLawDocument(
            law_id=law_id, doc_type=law.get("doc_type") or law.get("type"),
            issuing_body=law.get("issuing_body"), issue_date=law.get("issue_date"),
            effective_date=law.get("effective_date"), expiry_date=law.get("expiry_date"),
        )
        for art in law.get("articles") or law.get("content") or []:
            aid = art.get("aid")
            if aid is None:
                aid = art.get("id")
            text = art.get("text") or art.get("content") or art.get("content_Article") or ""
            doc.articles.append(RawArticle(
                law_id=law_id, aid=int(aid), title=art.get("title", ""),
                chuong=art.get("chuong") or art.get("chapter"), muc=art.get("muc") or art.get("section"),
                body=text.strip(),
            ))
        docs.append(doc)
    return docs


@dataclass
class KhoanSplit:
    khoan_no: Optional[str]
    diem_no: Optional[str]
    text: str


def split_article_into_khoan_diem(body: str) -> list:
    lines = [l for l in body.splitlines() if l.strip()]
    if not lines:
        return []
    units = []
    current_khoan = None
    current_lines = []
    current_diem = None

    def flush():
        if current_lines:
            units.append(KhoanSplit(khoan_no=current_khoan, diem_no=current_diem, text=" ".join(current_lines).strip()))

    for line in lines:
        khoan_match = RE_KHOAN.match(line)
        diem_match = RE_DIEM.match(line)
        if khoan_match:
            flush()
            current_khoan = khoan_match.group(1)
            current_diem = None
            current_lines = [line[khoan_match.end():].strip()]
        elif diem_match:
            flush()
            current_diem = diem_match.group(1)
            current_lines = [line[diem_match.end():].strip()]
        else:
            current_lines.append(line.strip())
    flush()
    if not units:
        units = [KhoanSplit(khoan_no=None, diem_no=None, text=body.strip())]
    return units
