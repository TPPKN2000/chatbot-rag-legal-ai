"""
Single shared module for:
  1. Pydantic schemas used across ingestion, indexing, retrieval and
     generation (keeping these in one place avoids circular imports between
     pipeline stages).
  2. The central LLM loading/generation interface (`generate_text` +
     `generate_text_stream`), used by query rewriting, question
     contextualization, conversation digesting, and final chat generation.

CHATBOT_MIGRATION_PLAN.md §2.2 (removed here): the ALQAC 4-label schema
(`Prediction` literal, `SubmissionRecord`, `LawEvidenceItem`,
`CaseQuery`/`CaseEvidenceHit` tied to the Case Content API) is gone — there
is no more "case" with a single fixed verdict label to score. In its place:
`ChatTurn`/`ChatSession` for multi-turn state, and `CitedProvision` /
`ChatAnswer` for the free-form, citation-annotated prose answer (see
`backend/generation/prompt_builder.py` and `generate.py`).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import BaseModel, Field

from backend import config


# ---------------------------------------------------------------------------
# Ingestion / law corpus structures
# ---------------------------------------------------------------------------
class LawChunk(BaseModel):
    """A single retrievable unit of the law corpus.

    Parent chunks correspond to a full "Điều" (article); child chunks
    correspond to individual "Khoản"/"Điểm" nested inside it.
    """

    chunk_id: str
    law_id: str
    aid: int = Field(..., description="Stable internal article id within the corpus — the identity key "
                                        "used for citation verification (allowed_citation_keys).")
    article_num: Optional[int] = Field(
        default=None,
        description="D2 (CHATBOT_MIGRATION_PLAN.md): best-effort REAL printed 'Điều N' number, parsed "
                     "from the article's own title/body. Prefer this for anything shown to a user; `aid` "
                     "is not guaranteed to match the printed number (see chunker.py docstring). None if "
                     "unparseable — callers should fall back to `aid` for display in that case.",
    )
    breadcrumb: str = Field(..., description='e.g. "Luật ... > Chương II > Mục 1 > Điều 12 > Khoản 3"')
    level: Literal["parent", "child"] = "child"
    parent_id: Optional[str] = None
    khoan_no: Optional[str] = Field(
        default=None, description="A3 citation-fast-path: structured Khoản number, None for parent chunks."
    )
    diem_no: Optional[str] = Field(
        default=None, description="A3 citation-fast-path: structured Điểm letter, None if this unit is a "
                                    "whole Khoản (no further Điểm split) or a parent chunk."
    )
    text: str
    token_count: int = 0


class LawMetadata(BaseModel):
    law_id: str
    doc_type: Optional[str] = None
    issuing_body: Optional[str] = None
    issue_date: Optional[str] = None
    effective_date: Optional[str] = None
    expiry_date: Optional[str] = None
    status: Literal["active", "expired", "amended", "unknown"] = "unknown"
    superseded_by: Optional[str] = None
    supersedes: Optional[str] = None


# ---------------------------------------------------------------------------
# Retrieval results
# ---------------------------------------------------------------------------
class RetrievedChunk(BaseModel):
    chunk_id: str
    law_id: str
    aid: int
    article_num: Optional[int] = None
    text: str
    score: float
    source: Literal["bm25", "vector", "fused", "reranked", "fastpath", "crossref"] = "fused"


# ---------------------------------------------------------------------------
# Conversation state (replaces CaseQuery — no more single-shot "case")
# ---------------------------------------------------------------------------
class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    session_id: str
    question: str


# ---------------------------------------------------------------------------
# Chat answer + citation instrumentation
# ---------------------------------------------------------------------------
class CitedProvision(BaseModel):
    """One (law_id, aid) pair that survived the hallucination guard for a
    given chat answer — kept for logging/eval (eval-harness-chatbot skill's
    groundedness check) and so a UI can hyperlink citations without
    re-parsing the prose answer, even though the model's raw output is
    free-form text, not JSON."""
    law_id: str
    aid: int
    article_num: Optional[int] = None


class ChatAnswer(BaseModel):
    answer: str
    citations: list[CitedProvision] = Field(default_factory=list)
    dropped_hallucinated_citations: int = 0
    is_fastpath: bool = False
    is_abstention: bool = False
    abstention_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# LLM loading & generation
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_generation_model():
    """Load the causal LM + tokenizer once and cache them for the process
    lifetime.

    - No `device_map`: unnecessary for a sub-1B model, and a source of
      version-dependent accelerate/transformers errors. Load normally and
      `.to(device)`.
    - `dtype` vs `torch_dtype`: transformers is mid-deprecation between the
      two kwarg names across versions; try the new name first, fall back.
    - Explicit `attn_implementation="sdpa"`: avoids an import-time crash if
      transformers auto-selects flash-attention-2 but `flash-attn` isn't
      installed.
    - Loud failure if `tokenizer.chat_template` is missing (usually means a
      base, non-chat checkpoint was configured).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = config.DEVICE
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(config.GENERATION_MODEL_NAME, trust_remote_code=True)
    if tokenizer.chat_template is None:
        raise RuntimeError(
            f"{config.GENERATION_MODEL_NAME} has no chat_template — check that "
            "GENERATION_MODEL_NAME points at an '-Instruct'/'-Chat' checkpoint, "
            "not a base model."
        )

    dtype_val = torch.float16 if device.startswith("cuda") else torch.float32
    load_kwargs = dict(trust_remote_code=True, attn_implementation=config.GENERATION_ATTN_IMPL)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.GENERATION_MODEL_NAME, dtype=dtype_val, **load_kwargs
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            config.GENERATION_MODEL_NAME, torch_dtype=dtype_val, **load_kwargs
        )

    model = model.to(device)
    model.eval()
    return tokenizer, model, device


def _encode_chat(system_prompt: str, user_prompt: str):
    """Shared chat-template encoding step used by both the blocking and
    streaming generation paths — kept in one place so the BatchEncoding vs.
    torch.Tensor normalization fix (see module history) is never duplicated
    or accidentally dropped by one of the two callers."""
    tokenizer, model, device = _get_generation_model()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    template_kwargs = dict(add_generation_prompt=True, return_tensors="pt")
    try:
        encoded = tokenizer.apply_chat_template(
            messages, enable_thinking=config.GENERATION_ENABLE_THINKING, **template_kwargs
        )
    except TypeError:
        encoded = tokenizer.apply_chat_template(messages, **template_kwargs)

    from transformers.tokenization_utils_base import BatchEncoding

    encoded = encoded.to(device) if hasattr(encoded, "to") else encoded

    if isinstance(encoded, (BatchEncoding, dict)):
        model_inputs = dict(encoded)
        input_length = model_inputs["input_ids"].shape[-1]
        generate_args = ()
        generate_kwargs = model_inputs
    else:
        input_length = encoded.shape[-1]
        generate_args = (encoded,)
        generate_kwargs = {}

    return tokenizer, model, device, input_length, generate_args, generate_kwargs


def generate_text(
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = config.GENERATION_MAX_NEW_TOKENS_DEFAULT,
    temperature: float = 0.3,
    top_p: float = 0.9,
) -> str:
    """Single-turn, blocking chat-style generation used by query rewriting,
    question contextualization (condense_question), the conversation
    digest, and non-streaming chat answers.

    Raises on failure (deliberately) — callers that must degrade gracefully
    already wrap this in try/except; callers that cannot degrade should let
    it propagate.
    """
    import torch

    tokenizer, model, device, input_length, generate_args, generate_kwargs = _encode_chat(
        system_prompt, user_prompt
    )

    do_sample = temperature > 0
    with torch.no_grad():
        output = model.generate(
            *generate_args,
            **generate_kwargs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    new_tokens = output[0][input_length:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text


def generate_text_stream(
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = config.GENERATION_MAX_NEW_TOKENS_DEFAULT,
    temperature: float = 0.3,
    top_p: float = 0.9,
):
    """B3 (grounded-chat-generation skill): token-streaming variant of
    `generate_text`, for the final chat answer only (query rewriting /
    condensing / digesting stay on the blocking path — nothing there is
    user-facing token-by-token).

    Reuses `_encode_chat()` so the BatchEncoding/dict-vs-Tensor
    normalization is shared with the blocking path rather than
    re-implemented — the skill's documented gotcha is exactly a version
    where that normalization got dropped for the streaming branch.

    NOT used by the citation fast-path (backend/retrieval/citation_fastpath.py)
    — that path never calls the LLM at all, so there's nothing to stream.

    Yields decoded text chunks as they're produced. Raises if generation
    setup fails (mirrors generate_text's fail-loud contract) — the caller
    (chat_pipeline.py) is responsible for the hard-refusal branch, which
    must short-circuit BEFORE this is ever called.
    """
    from threading import Thread

    from transformers import TextIteratorStreamer

    tokenizer, model, device, _input_length, generate_args, generate_kwargs = _encode_chat(
        system_prompt, user_prompt
    )

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    do_sample = temperature > 0

    thread_kwargs = dict(generate_kwargs)
    thread = Thread(
        target=model.generate,
        args=generate_args,
        kwargs=dict(
            **thread_kwargs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        ),
    )
    thread.start()
    for token_text in streamer:
        if token_text:
            yield token_text
    thread.join()
