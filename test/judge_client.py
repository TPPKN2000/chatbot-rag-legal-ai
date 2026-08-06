"""
Judge client cho test/eval_faithfulness.py (coding_plan.md D3).

Nguyên tắc bắt buộc (D3): model chấm điểm (judge) KHÔNG được là chính model
sinh câu trả lời (Qwen3.5-0.8B local, backend/models.py::generate_text). Một
model nhỏ khó tự phát hiện lỗi của chính nó — FaithJudge/TREC 2025 RAG Track
đều dùng judge tách biệt, thường mạnh hơn model được đánh giá.

File này gọi Anthropic API trực tiếp qua `requests` (KHÔNG qua LangChain —
giữ tinh thần no-framework của core pipeline; test/ là ngoài luồng sản phẩm
nên nếu có vi phạm cũng chấp nhận được, nhưng ở đây không cần vì requests
là đủ).

Nếu không có ANTHROPIC_API_KEY, `is_judge_available()` trả False — MỌI lời
gọi judge trong eval_faithfulness.py phải SKIP (không raise ra ngoài, không
âm thầm dùng model sinh nội bộ thay thế — đó chính là điều D3 cấm).
"""
from __future__ import annotations

import os

import requests

_ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
# Model MẶC ĐỊNH cho judge — cố ý KHÔNG trùng model sinh (Qwen3.5-0.8B local).
# Override qua env JUDGE_MODEL nếu muốn đổi (vd. sang model mạnh hơn khi có
# Nhóm B NVIDIA NIM/Llama-3.1-8B — xem D3 "Ưu tiên 2" trong coding_plan.md).
_JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-4-6")


def is_judge_available() -> bool:
    """True nếu có ANTHROPIC_API_KEY trong env. Không gọi network — chỉ
    kiểm tra config, để test_chatbot.py có thể quyết định skip trục
    faithfulness/helpfulness sớm mà không tốn 1 lệnh gọi API."""
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def judge_generate(system_prompt: str, user_prompt: str, max_tokens: int = 300, temperature: float = 0.0) -> str:
    """Lệnh gọi chặn (blocking) tới Anthropic API. Raises khi lỗi — caller
    PHẢI gọi is_judge_available() trước và tự bắt exception (một lỗi judge
    không được làm sập cả eval run, cùng triết lý fallback nhất quán đã
    dùng xuyên suốt querry_transform.py/case_digest.py)."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set — call is_judge_available() first.")

    resp = requests.post(
        _ANTHROPIC_API_URL,
        headers={"x-api-key": api_key, "anthropic-version": _ANTHROPIC_VERSION, "content-type": "application/json"},
        json={
            "model": _JUDGE_MODEL,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    text_blocks = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
    return "".join(text_blocks).strip()
