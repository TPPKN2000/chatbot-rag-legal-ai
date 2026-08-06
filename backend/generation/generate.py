from __future__ import annotations

import logging
import re

from backend import config
from backend.generation.prompt_builder import allowed_citation_map, build_chat_prompt
from backend.models import ChatAnswer, CitedProvision, RetrievedChunk, generate_text, generate_text_stream

log = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[Điều\s+(\d+)\s*,\s*([^\]]+)\]")
_UNVERIFIED_NOTE = "\n\nLưu ý: một số nội dung trên chưa được xác minh với corpus hiện có."


def verify_and_strip_hallucinated_citations(answer_text: str, allowed):
    dropped = 0
    citations = []
    seen_keys = set()

    def _check(match):
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


def _finalize(raw_answer: str, law_chunks) -> ChatAnswer:
    allowed = allowed_citation_map(law_chunks)
    filtered_text, citations, dropped = verify_and_strip_hallucinated_citations(raw_answer, allowed)
    if dropped and not citations:
        filtered_text = filtered_text.rstrip() + _UNVERIFIED_NOTE
    return ChatAnswer(answer=filtered_text.strip(), citations=citations, dropped_hallucinated_citations=dropped)


def generate_chat_answer(contextualized_question, law_chunks, conversation_digest,
                          max_new_tokens=config.CHAT_ANSWER_MAX_NEW_TOKENS, temperature=0.3) -> ChatAnswer:
    system_prompt, user_prompt = build_chat_prompt(contextualized_question, law_chunks, conversation_digest)
    try:
        raw = generate_text(system_prompt=system_prompt, user_prompt=user_prompt,
                             max_new_tokens=max_new_tokens, temperature=temperature)
    except Exception as e:
        log.warning("generate_chat_answer: generation failed: %s", e)
        return ChatAnswer(answer="Xin lỗi, đã có lỗi khi tạo câu trả lời. Bạn vui lòng thử lại.",
                           is_abstention=True, abstention_reason=f"generation failed: {e}")
    return _finalize(raw, law_chunks)


def generate_chat_answer_stream(contextualized_question, law_chunks, conversation_digest,
                                 max_new_tokens=config.CHAT_ANSWER_MAX_NEW_TOKENS, temperature=0.3):
    system_prompt, user_prompt = build_chat_prompt(contextualized_question, law_chunks, conversation_digest)
    try:
        raw_chunks = list(generate_text_stream(system_prompt=system_prompt, user_prompt=user_prompt,
                                                max_new_tokens=max_new_tokens, temperature=temperature))
    except Exception as e:
        log.warning("generate_chat_answer_stream: generation failed: %s", e)
        yield "Xin lỗi, đã có lỗi khi tạo câu trả lời. Bạn vui lòng thử lại."
        return
    raw = "".join(raw_chunks)
    result = _finalize(raw, law_chunks)
    for word in result.answer.split(" "):
        yield word + " "
