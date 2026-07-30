"""
Chat turn orchestration (CHATBOT_MIGRATION_PLAN.md §A2).

This is the chatbot's equivalent of the old `backend/pipeline.py::process_case()`
— but for a multi-turn, session-based question instead of a single ALQAC
case. One call = one chat turn.

Order of operations (each step documented inline with which skill/plan
section it implements):
  0. Guardrail: trivially-out-of-scope short-circuit (guardrail.py, C1).
  1. Question contextualization: condense the latest question against
     session history into a standalone question (conversational-retrieval
     skill, Bước 1) — uses a conversation digest instead of raw history once
     the session is long (Bước 2).
  2. Citation fast-path: bypass retrieval+LLM entirely for a pure
     "Điều X [khoản Y] nói gì?" lookup (citation-fast-path skill, A3).
  3. Full retrieval: backend.pipeline.collect_law_evidence() (hybrid search
     -> rerank -> retrieval-evaluator re-round, now including the C2
     cross-reference hop).
  4. Hard refusal gate: if retrieval confidence is still weak after the
     evaluator's extra round, refuse in code — never let the LLM guess from
     weak context (grounded-chat-generation skill, "Nhánh từ chối").
  5. Generation + hallucination guard (generation/generate.py).
  6. Guardrail disclaimer backstop (C1) + session history update.

Streaming and non-streaming entry points are both provided; both share
steps 0-4 (`_prepare_turn`) so the refusal/fast-path logic can never drift
between the two paths.
"""
from __future__ import annotations

import logging

from backend import config, guardrail
from backend.generation.conversation_digest import build_conversation_digest
from backend.generation.generate import generate_chat_answer, generate_chat_answer_stream
from backend.models import ChatAnswer, RetrievedChunk
from backend.pipeline import collect_law_evidence
from backend.retrieval import citation_fastpath
from backend.retrieval.querry_transform import condense_question
from backend.session_store import store as session_store

log = logging.getLogger(__name__)


def _record_turn(session_id: str, question: str, answer_text: str) -> None:
    session_store.append(session_id, "user", question)
    session_store.append(session_id, "assistant", answer_text)


def _history_for_condense(session_id: str) -> list:
    """conversational-retrieval skill, gotcha: don't confuse condense-
    question's input (fed to the LLM to rewrite the CURRENT question) with
    the conversation digest (a separate summary, fed into the final
    generation prompt). Below CONVERSATION_DIGEST_TRIGGER_TURNS this uses
    the raw turn list directly; above it, feeding condense_question() the
    full raw transcript would defeat the point of digesting — so a single
    synthetic "assistant" turn carrying the digest text is used instead,
    which condense_question's prompt format handles the same as any other
    turn.
    """
    from backend.models import ChatTurn

    history = session_store.history(session_id)
    if len(history) <= config.CONVERSATION_DIGEST_TRIGGER_TURNS:
        return history
    digest = build_conversation_digest(history)
    return [ChatTurn(role="assistant", content=f"(Tóm tắt hội thoại trước đó: {digest})")]


class _PreparedTurn:
    """Internal result of steps 0-4, shared by the streaming and
    non-streaming entry points."""

    __slots__ = ("early_answer", "contextualized_question", "law_chunks", "conversation_digest")

    def __init__(self, early_answer: ChatAnswer | None = None, contextualized_question: str = "",
                 law_chunks: list[RetrievedChunk] | None = None, conversation_digest: str = ""):
        self.early_answer = early_answer
        self.contextualized_question = contextualized_question
        self.law_chunks = law_chunks or []
        self.conversation_digest = conversation_digest


def _prepare_turn(session_id: str, question: str) -> _PreparedTurn:
    # --- 0. Guardrail: trivially out of scope -----------------------------
    if guardrail.is_trivially_out_of_scope(question):
        return _PreparedTurn(early_answer=guardrail.out_of_scope_answer())

    # --- 1. Question contextualization -------------------------------------
    history = _history_for_condense(session_id)
    contextualized_question = condense_question(history, question)

    # --- 2. Citation fast-path ----------------------------------------------
    fastpath_hit = citation_fastpath.try_lookup(contextualized_question)
    if fastpath_hit is not None:
        return _PreparedTurn(early_answer=fastpath_hit, contextualized_question=contextualized_question)

    # --- 3. Full retrieval ---------------------------------------------------
    law_chunks = collect_law_evidence(contextualized_question)

    # --- 4. Hard refusal gate -------------------------------------------------
    # grounded-chat-generation skill: this is a CODE-LEVEL gate, not a prompt
    # instruction — a small model can't be trusted to reliably self-refuse.
    if not law_chunks or law_chunks[0].score < config.RETRIEVAL_EVALUATOR_SCORE_THRESHOLD:
        log.info(
            "session=%s: refusing to answer — best retrieval score %.3f below threshold %.3f",
            session_id, law_chunks[0].score if law_chunks else -1.0,
            config.RETRIEVAL_EVALUATOR_SCORE_THRESHOLD,
        )
        return _PreparedTurn(
            early_answer=ChatAnswer(
                answer=config.ABSTENTION_MESSAGE,
                is_abstention=True,
                abstention_reason="retrieval confidence below threshold after evaluator round",
            ),
            contextualized_question=contextualized_question,
        )

    conversation_digest = build_conversation_digest(session_store.history(session_id))
    return _PreparedTurn(
        contextualized_question=contextualized_question,
        law_chunks=law_chunks,
        conversation_digest=conversation_digest,
    )


def handle_chat_turn(session_id: str, question: str) -> ChatAnswer:
    """Blocking entry point: one full chat turn in, one ChatAnswer out."""
    prepared = _prepare_turn(session_id, question)
    if prepared.early_answer is not None:
        answer = guardrail.apply_disclaimer_if_needed(question, prepared.early_answer)
        _record_turn(session_id, question, answer.answer)
        return answer

    answer = generate_chat_answer(
        prepared.contextualized_question, prepared.law_chunks, prepared.conversation_digest,
    )
    answer = guardrail.apply_disclaimer_if_needed(question, answer)
    _record_turn(session_id, question, answer.answer)
    return answer


def handle_chat_turn_stream(session_id: str, question: str):
    """Streaming entry point (B3). Yields text chunks; records the full turn
    into session history once the generator is exhausted by the caller.

    NOTE: the fast-path and refusal branches yield their ENTIRE fixed answer
    as a single chunk — see citation-fast-path skill: "Streaming KHÔNG áp
    dụng cho citation fast-path... không qua LLM nên không cần/không thể
    stream token." The same applies to refusal messages (also not
    LLM-generated).
    """
    prepared = _prepare_turn(session_id, question)
    if prepared.early_answer is not None:
        answer = guardrail.apply_disclaimer_if_needed(question, prepared.early_answer)
        yield answer.answer
        _record_turn(session_id, question, answer.answer)
        return

    chunks: list[str] = []
    for piece in generate_chat_answer_stream(
        prepared.contextualized_question, prepared.law_chunks, prepared.conversation_digest,
    ):
        chunks.append(piece)
        yield piece

    full_text = "".join(chunks).strip()
    # Disclaimer backstop is applied AFTER the stream for the same reason
    # citations can't be retroactively fixed mid-stream (see generate.py's
    # module docstring) — appended as one more chunk if needed.
    if guardrail.should_attach_disclaimer(question) and config.LEGAL_DISCLAIMER not in full_text:
        disclaimer_chunk = "\n\n" + config.LEGAL_DISCLAIMER
        yield disclaimer_chunk
        full_text += disclaimer_chunk

    _record_turn(session_id, question, full_text)
