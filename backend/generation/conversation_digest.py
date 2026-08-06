from __future__ import annotations
from backend import config
from backend.models import ChatTurn, generate_text

_DIGEST_SYSTEM_PROMPT = (
    "Bạn là trợ lý tóm tắt hội thoại. Từ lịch sử hội thoại dưới đây, hãy viết "
    "một đoạn TÓM TẮT NGẮN (tối đa 150 từ) nêu: chủ đề pháp lý đang được hỏi, "
    "các thông tin/tình tiết quan trọng người dùng đã cung cấp, và các điều "
    "luật đã được trích dẫn trước đó (nếu có). KHÔNG suy đoán, KHÔNG thêm "
    "thông tin ngoài những gì đã có trong hội thoại."
)
_EMPTY_DIGEST = "(Chưa có lịch sử hội thoại đáng kể.)"


def _format_turn(turn) -> str:
    role = turn.role if isinstance(turn, ChatTurn) else turn.get("role", "user")
    content = turn.content if isinstance(turn, ChatTurn) else turn.get("content", "")
    speaker = "Người dùng" if role == "user" else "Trợ lý"
    return f"{speaker}: {content}"


def build_conversation_digest(history) -> str:
    if not history:
        return _EMPTY_DIGEST
    joined = "\n".join(_format_turn(t) for t in history)
    if not joined.strip():
        return _EMPTY_DIGEST
    try:
        digest = generate_text(system_prompt=_DIGEST_SYSTEM_PROMPT, user_prompt=joined,
                                max_new_tokens=config.CONVERSATION_DIGEST_MAX_NEW_TOKENS, temperature=0.2).strip()
        return digest or _EMPTY_DIGEST
    except Exception:
        return joined[:800]
