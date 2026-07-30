"""
Final chat generation step + grounding verification pass
(grounded-chat-generation skill).

CHATBOT_MIGRATION_PLAN.md §2.2/§B2 (replaces the ALQAC JSON-verdict
`predict_outcome()`): there is no more 4-label JSON schema to parse. The
model returns free-form Vietnamese prose with inline citations formatted
"[Điều X, law_id]" (see prompt_builder.CHAT_SYSTEM_PROMPT rule #2). The
hallucination guard therefore moves from "parse JSON, filter a
law_citations array" to "regex-scan the prose for citation brackets, verify
each against the closed set the model was shown, and strip/flag any that
aren't" — same MECHANISM (whitelist verification against
`allowed_citation_keys`), different INPUT shape.

Streaming/verification tradeoff (documented, not hidden): true token-by-
token streaming can't retroactively fix an already-displayed hallucinated
citation. `generate_chat_answer_stream()` therefore buffers the full raw
generation server-side, verifies it, and re-emits the VERIFIED text in
word-sized chunks — the person still sees a live-typing effect, but never
sees an unverified citation on screen. This is a deliberate correctness-
over-latency choice for a legal-advice surface; see module note below if a
future iteration wants true incremental streaming with retroactive
correction UI support instead.

The hard "refuse to answer" gate driven by retrieval confidence
(`config.RETRIEVAL_EVALUATOR_SCORE_THRESHOLD`) is enforced by the CALLER
(`backend/chat_pipeline.py`), not here — see that module and the
grounded-chat-generation skill's "Nhánh từ chối" section for why this must
be a hard, code-level gate rather than left to the model's own judgement.
"""
from __future__ import annotations

import logging
import re

from backend import config
from backend.generation.prompt_builder import allowed_citation_map, build_chat_prompt
from backend.models import ChatAnswer, CitedProvision, RetrievedChunk, generate_text, generate_text_stream

log = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[Điều\s+(\d+)\s*,\s*([^\]]+)\]")

_UNVERIFIED_NOTE = "\n\nLưu ý: một số nội dung trên chưa được xác minh với corpus hiện có."


def verify_and_strip_hallucinated_citations(
    answer_text: str,
    allowed: dict[tuple[str, int]] | dict,
) -> tuple[str, list[CitedProvision], int]:
    """Scan `answer_text` for "[Điều X, law_id]" citations, verify each
    against `allowed` (from `prompt_builder.allowed_citation_map`).

    A citation NOT in `allowed` is replaced in-place with
    "[cần xác minh thêm]" (not silently deleted — a silent deletion could
    leave the surrounding sentence reading as if it were still grounded,
    which is worse than visibly flagging it). Returns (filtered_text,
    surviving_citations, dropped_count).
    """
    dropped = 0
    citations: list[CitedProvision] = []
    seen_keys: set[tuple[str, int]] = set()

    def _check(match: re.Match) -> str:
        nonlocal dropped
        try:
            num = int(match.group(1))
        except ValueError:
            dropped += 1
            return "[cần xác minh thêm]"
        law_id = match.group(2).strip()
        key = (law_id, num)
        chunk = allowed.get(key)
        if chunk is None:
            dropped += 1
            return "[cần xác minh thêm]"
        if key not in seen_keys:
            seen_keys.add(key)
            citations.append(CitedProvision(law_id=chunk.law_id, aid=chunk.aid, article_num=chunk.article_num))
        return match.group(0)

    filtered = _CITATION_RE.sub(_check, answer_text)
    return filtered, citations, dropped


def _finalize(raw_answer: str, law_chunks: list[RetrievedChunk]) -> ChatAnswer:
    allowed = allowed_citation_map(law_chunks)
    filtered_text, citations, dropped = verify_and_strip_hallucinated_citations(raw_answer, allowed)

    if dropped and not citations:
        # No grounded citation survived at all — flag it explicitly rather
        # than let the prose read as if it were still fully sourced.
        filtered_text = filtered_text.rstrip() + _UNVERIFIED_NOTE

    return ChatAnswer(
        answer=filtered_text.strip(),
        citations=citations,
        dropped_hallucinated_citations=dropped,
    )


def generate_chat_answer(
    contextualized_question: str,
    law_chunks: list[RetrievedChunk],
    conversation_digest: str,
    max_new_tokens: int = config.CHAT_ANSWER_MAX_NEW_TOKENS,
    temperature: float = 0.3,
) -> ChatAnswer:
    """Blocking chat answer generation + grounding verification.

    Unlike ALQAC's predict_outcome(), a generation/parsing failure here does
    NOT need a "safe default label" — there's no label to default to. It
    degrades to an honest error message instead, still flagged via
    `is_abstention` so the caller/UI can treat it consistently with a
    retrieval-triggered refusal.
    """
    system_prompt, user_prompt = build_chat_prompt(contextualized_question, law_chunks, conversation_digest)
    try:
        raw = generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    except Exception as e:
        log.warning("generate_chat_answer: generation failed: %s", e)
        return ChatAnswer(
            answer="Xin lỗi, đã có lỗi khi tạo câu trả lời. Bạn vui lòng thử lại.",
            is_abstention=True,
            abstention_reason=f"generation failed: {e}",
        )

    return _finalize(raw, law_chunks)


def generate_chat_answer_stream(
    contextualized_question: str,
    law_chunks: list[RetrievedChunk],
    conversation_digest: str,
    max_new_tokens: int = config.CHAT_ANSWER_MAX_NEW_TOKENS,
    temperature: float = 0.3,
):
    """Streaming variant (B3). See module docstring for the buffer-then-
    verify-then-replay tradeoff. Yields word-sized text chunks of the
    VERIFIED answer, then returns (via StopIteration) — callers that need
    the structured `ChatAnswer` (citations list, dropped count) should use
    `generate_chat_answer()` instead, or collect this generator's chunks and
    call `verify_and_strip_hallucinated_citations` themselves if they need
    both a live stream AND the structured result from one call.
    """
    system_prompt, user_prompt = build_chat_prompt(contextualized_question, law_chunks, conversation_digest)
    try:
        raw_chunks = list(
            generate_text_stream(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        )
    except Exception as e:
        log.warning("generate_chat_answer_stream: generation failed: %s", e)
        yield "Xin lỗi, đã có lỗi khi tạo câu trả lời. Bạn vui lòng thử lại."
        return

    raw = "".join(raw_chunks)
    result = _finalize(raw, law_chunks)
    for word in result.answer.split(" "):
        yield word + " "
