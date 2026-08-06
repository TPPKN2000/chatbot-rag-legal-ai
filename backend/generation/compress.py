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


def compress_auxiliary_text(text: str, target_ratio: float = config.COMPRESSION_TARGET_RATIO) -> str:
    if not config.COMPRESSION_ENABLED or not text.strip():
        return text
    try:
        compressor = _get_compressor()
        result = compressor.compress_prompt(text, rate=target_ratio)
        return result.get("compressed_prompt", text)
    except Exception:
        return _sentence_fallback_compress(text, target_ratio)
