"""
Prompt compression (design doc §4.2).

STATUS: CURRENTLY UNUSED, KEPT INTENTIONALLY (unchanged from the ALQAC-era
decision — CHATBOT_MIGRATION_PLAN.md doesn't call for revisiting this).
Nothing in the chatbot's `chat_pipeline.py` calls
`compress_case_evidence()`/`compress_auxiliary_text()` either — the
conversation-digest role is filled by `generation/conversation_digest.py`
(an LLM-summarization approach), same as `case_digest.py` did for ALQAC.

CRITICAL legal-domain rule (unchanged, still applies to the chatbot): NEVER
compress the verbatim text of a law provision. Losing a connective like
"trừ trường hợp" ("except in the case of") or "ngoại trừ" ("excluding") can
invert the meaning of a clause. Only *auxiliary* context is eligible for
compression — and in the chatbot, that auxiliary role is filled by
conversation history, not case evidence.

Implementation uses LLMLingua when available; falls back to a cheap
sentence-count truncation (never token-level truncation) if `llmlingua`
isn't installed.
"""
from __future__ import annotations

from functools import lru_cache

from backend import config


@lru_cache(maxsize=1)
def _get_compressor():
    from llmlingua import PromptCompressor

    return PromptCompressor()


def _sentence_fallback_compress(text: str, target_ratio: float) -> str:
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    if not sentences:
        return text
    keep_n = max(1, round(len(sentences) * target_ratio))
    return ". ".join(sentences[:keep_n]) + "."


def compress_auxiliary_text(
    text: str,
    target_ratio: float = config.COMPRESSION_TARGET_RATIO,
) -> str:
    """Compress a block of AUXILIARY context only. Callers must never pass
    verbatim law-provision text here."""
    if not config.COMPRESSION_ENABLED or not text.strip():
        return text
    try:
        compressor = _get_compressor()
        result = compressor.compress_prompt(text, rate=target_ratio)
        return result.get("compressed_prompt", text)
    except Exception:
        return _sentence_fallback_compress(text, target_ratio)
