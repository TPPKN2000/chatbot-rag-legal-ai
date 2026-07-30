"""
Scope/intent guardrail + disclaimer (CHATBOT_MIGRATION_PLAN.md §2.3 item 3 /
§4 C1). Originally proposed in docs/system_design_v0.md §7.4 but never
implemented for the ALQAC pipeline (single-turn, no free-form user input to
guard against) — now needed for a chatbot that accepts arbitrary questions.

Two separate concerns, kept as two separate cheap checks rather than one
LLM call per turn (consistent with the project's existing "reuse a cheap
signal instead of an LLM judge" pattern — see
`pipeline.collect_law_evidence`'s retrieval-evaluator, which reuses the
rerank score instead of adding an LLM-judge call):

1. Trivial out-of-scope detection (`is_trivially_out_of_scope`) — a small,
   curated pattern list for UNAMBIGUOUSLY non-legal input (greetings, chit-
   chat, "what's the weather"). Deliberately narrow/high-precision: a false
   positive here means point-blank refusing a legitimate legal question, so
   this must never guess. Genuinely ambiguous non-legal questions are left
   to the existing retrieval-confidence refusal gate in chat_pipeline.py —
   if nothing relevant is in the law corpus, that gate already declines.
   This function only short-circuits the OBVIOUS cases before spending a
   retrieval round on them at all.

2. Personalized-advice disclaimer (`should_attach_disclaimer`) —
   prompt_builder.CHAT_SYSTEM_PROMPT rule #4 already asks the model to add
   this itself, but (same reasoning as the hard refusal gate in generate.py)
   a small model can't be trusted to follow a soft prompt rule 100% of the
   time. This is a hard, code-level backstop: if the question matches
   personalized-advice phrasing, the disclaimer is appended unconditionally,
   regardless of what the model did or didn't say.

`NEGATIVE_SAFETY_QUERIES` below is shared with `test/test_chatbot.py`'s
abstention-correctness eval (eval-harness-chatbot skill) so the guardrail
and its test stay in sync — extending one list benefits both.
"""
from __future__ import annotations

import re

from backend import config
from backend.indexing.bm25_index import fold_accents
from backend.models import ChatAnswer

# Shared with test/test_chatbot.py's abstention-correctness eval (mirrors
# AI/evaluate.py::NEGATIVE_SAFETY_QUERIES per the legacy-prototype-salvage
# skill, extended for this corpus's specific out-of-scope failure modes).
NEGATIVE_SAFETY_QUERIES: list[str] = [
    "thời tiết ngày mai thế nào",
    "hôm nay có bóng đá không",
    "kể cho tôi một câu chuyện cười",
    "1 + 1 bằng mấy",
    "bạn là ai",
    "xin chào",
    "hello",
]

_TRIVIAL_OUT_OF_SCOPE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^\s*(xin\s*)?ch[aà]o\b",
        r"^\s*hello\b",
        r"^\s*hi\b",
        r"th[oờ]i ti[eế]t",
        r"b[oó]ng đ[aá]",
        r"c[aâ]u chuy[eệ]n c[uườ]{0,3}i",
        r"b[aạ]n l[aà] ai\b",
        r"\d\s*\+\s*\d\s*b[aằ]ng",
    )
]

# Phrasing that signals the user wants a personalized recommendation/
# decision, not a lookup of what the law says — prompt_builder rule #4's
# hard-coded backstop.
_ADVICE_SEEKING_CUES = [
    "toi nen", "co nen khong", "co nen", "giup toi quyet dinh", "theo ban thi toi nen",
    "toi phai lam gi", "truong hop cua toi thi", "neu la ban thi",
]


def is_trivially_out_of_scope(question: str) -> bool:
    """High-precision only: True means "confidently not a legal question",
    never a maybe. See module docstring for why this stays narrow."""
    if not config.GUARDRAIL_ENABLED:
        return False
    q = question.strip()
    if not q:
        return False
    return any(p.search(q) for p in _TRIVIAL_OUT_OF_SCOPE_PATTERNS)


def should_attach_disclaimer(question: str) -> bool:
    if not config.GUARDRAIL_ENABLED:
        return False
    folded = fold_accents(question).lower()
    return any(cue in folded for cue in _ADVICE_SEEKING_CUES)


def out_of_scope_answer() -> ChatAnswer:
    return ChatAnswer(
        answer=config.OUT_OF_SCOPE_MESSAGE,
        is_abstention=True,
        abstention_reason="trivially out of scope (guardrail)",
    )


def apply_disclaimer_if_needed(question: str, answer: ChatAnswer) -> ChatAnswer:
    """Hard-code the disclaimer onto the answer when the question pattern-
    matches personalized-advice-seeking — regardless of whether the model
    already added something similar itself (a duplicate disclaimer is a far
    smaller problem than a missing one on a legal-advice surface)."""
    if answer.is_abstention:
        return answer
    if not should_attach_disclaimer(question):
        return answer
    if config.LEGAL_DISCLAIMER in answer.answer:
        return answer
    answer.answer = answer.answer.rstrip() + "\n\n" + config.LEGAL_DISCLAIMER
    return answer
