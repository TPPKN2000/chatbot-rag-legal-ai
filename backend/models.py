from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import BaseModel, Field

from backend import config


class LawChunk(BaseModel):
    chunk_id: str
    law_id: str
    aid: int = Field(..., description="Stable internal article id within the corpus.")
    article_num: Optional[int] = Field(default=None, description="D2: real printed 'Điều N' number.")
    breadcrumb: str
    level: Literal["parent", "child"] = "child"
    parent_id: Optional[str] = None
    khoan_no: Optional[str] = None
    diem_no: Optional[str] = None
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


class RetrievedChunk(BaseModel):
    chunk_id: str
    law_id: str
    aid: int
    article_num: Optional[int] = None
    text: str
    score: float
    source: Literal["bm25", "vector", "fused", "reranked", "fastpath", "crossref"] = "fused"


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    session_id: str
    question: str


class CitedProvision(BaseModel):
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


@lru_cache(maxsize=1)
def _get_generation_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = config.DEVICE
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(config.GENERATION_MODEL_NAME, trust_remote_code=True)
    if tokenizer.chat_template is None:
        raise RuntimeError(f"{config.GENERATION_MODEL_NAME} has no chat_template.")

    dtype_val = torch.float16 if device.startswith("cuda") else torch.float32
    load_kwargs = dict(trust_remote_code=True, attn_implementation=config.GENERATION_ATTN_IMPL)
    try:
        model = AutoModelForCausalLM.from_pretrained(config.GENERATION_MODEL_NAME, dtype=dtype_val, **load_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(config.GENERATION_MODEL_NAME, torch_dtype=dtype_val, **load_kwargs)

    model = model.to(device)
    model.eval()
    return tokenizer, model, device


def _encode_chat(system_prompt: str, user_prompt: str):
    tokenizer, model, device = _get_generation_model()
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    template_kwargs = dict(add_generation_prompt=True, return_tensors="pt")
    try:
        encoded = tokenizer.apply_chat_template(messages, enable_thinking=config.GENERATION_ENABLE_THINKING, **template_kwargs)
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


def generate_text(system_prompt: str, user_prompt: str, max_new_tokens: int = config.GENERATION_MAX_NEW_TOKENS_DEFAULT,
                   temperature: float = 0.3, top_p: float = 0.9) -> str:
    import torch
    tokenizer, model, device, input_length, generate_args, generate_kwargs = _encode_chat(system_prompt, user_prompt)
    do_sample = temperature > 0
    with torch.no_grad():
        output = model.generate(*generate_args, **generate_kwargs, max_new_tokens=max_new_tokens, do_sample=do_sample,
                                 temperature=temperature if do_sample else None, top_p=top_p if do_sample else None,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    new_tokens = output[0][input_length:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text


def generate_text_stream(system_prompt: str, user_prompt: str, max_new_tokens: int = config.GENERATION_MAX_NEW_TOKENS_DEFAULT,
                          temperature: float = 0.3, top_p: float = 0.9):
    from threading import Thread
    from transformers import TextIteratorStreamer

    tokenizer, model, device, _input_length, generate_args, generate_kwargs = _encode_chat(system_prompt, user_prompt)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    do_sample = temperature > 0
    thread_kwargs = dict(generate_kwargs)
    thread = Thread(target=model.generate, args=generate_args, kwargs=dict(
        **thread_kwargs, streamer=streamer, max_new_tokens=max_new_tokens, do_sample=do_sample,
        temperature=temperature if do_sample else None, top_p=top_p if do_sample else None,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id))
    thread.start()
    for token_text in streamer:
        if token_text:
            yield token_text
    thread.join()
