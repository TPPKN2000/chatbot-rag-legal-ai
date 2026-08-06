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
    from backend.models import ChatTurn
    history = session_store.history(session_id)
    if len(history) <= config.CONVERSATION_DIGEST_TRIGGER_TURNS:
        return history
    digest = build_conversation_digest(history)
    return [ChatTurn(role="assistant", content=f"(Tóm tắt hội thoại trước đó: {digest})")]


class _PreparedTurn:
    __slots__ = ("early_answer", "contextualized_question", "law_chunks", "conversation_digest")

    def __init__(self, early_answer=None, contextualized_question: str = "", law_chunks=None, conversation_digest: str = ""):
        self.early_answer = early_answer
        self.contextualized_question = contextualized_question
        self.law_chunks = law_chunks or []
        self.conversation_digest = conversation_digest


def _prepare_turn(session_id: str, question: str) -> _PreparedTurn:
    if guardrail.is_trivially_out_of_scope(question):
        return _PreparedTurn(early_answer=guardrail.out_of_scope_answer())

    history = _history_for_condense(session_id)
    contextualized_question = condense_question(history, question)

    fastpath_hit = citation_fastpath.try_lookup(contextualized_question)
    if fastpath_hit is not None:
        return _PreparedTurn(early_answer=fastpath_hit, contextualized_question=contextualized_question)

    law_chunks = collect_law_evidence(contextualized_question)

    if not law_chunks or law_chunks[0].score < config.RETRIEVAL_EVALUATOR_SCORE_THRESHOLD:
        log.info("session=%s: refusing to answer — best retrieval score %.3f below threshold %.3f",
                  session_id, law_chunks[0].score if law_chunks else -1.0, config.RETRIEVAL_EVALUATOR_SCORE_THRESHOLD)
        return _PreparedTurn(
            early_answer=ChatAnswer(answer=config.ABSTENTION_MESSAGE, is_abstention=True,
                                     abstention_reason="retrieval confidence below threshold after evaluator round"),
            contextualized_question=contextualized_question,
        )

    conversation_digest = build_conversation_digest(session_store.history(session_id))
    return _PreparedTurn(contextualized_question=contextualized_question, law_chunks=law_chunks, conversation_digest=conversation_digest)


def handle_chat_turn(session_id: str, question: str) -> ChatAnswer:
    prepared = _prepare_turn(session_id, question)
    if prepared.early_answer is not None:
        answer = guardrail.apply_disclaimer_if_needed(question, prepared.early_answer)
        _record_turn(session_id, question, answer.answer)
        return answer
    answer = generate_chat_answer(prepared.contextualized_question, prepared.law_chunks, prepared.conversation_digest)
    answer = guardrail.apply_disclaimer_if_needed(question, answer)
    _record_turn(session_id, question, answer.answer)
    return answer


def handle_chat_turn_stream(session_id: str, question: str):
    prepared = _prepare_turn(session_id, question)
    if prepared.early_answer is not None:
        answer = guardrail.apply_disclaimer_if_needed(question, prepared.early_answer)
        yield answer.answer
        _record_turn(session_id, question, answer.answer)
        return

    chunks = []
    for piece in generate_chat_answer_stream(prepared.contextualized_question, prepared.law_chunks, prepared.conversation_digest):
        chunks.append(piece)
        yield piece

    full_text = "".join(chunks).strip()
    if guardrail.should_attach_disclaimer(question) and config.LEGAL_DISCLAIMER not in full_text:
        disclaimer_chunk = "\n\n" + config.LEGAL_DISCLAIMER
        yield disclaimer_chunk
        full_text += disclaimer_chunk

    _record_turn(session_id, question, full_text)
