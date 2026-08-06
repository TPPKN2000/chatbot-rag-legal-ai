from __future__ import annotations
from backend import config
from backend.models import ChatTurn, generate_text

_REWRITE_SYSTEM_PROMPT = (
    "Bạn là trợ lý pháp lý. Nhiệm vụ: viết lại câu hỏi/tình huống sau đây thành "
    "3 đến 5 câu hỏi tương đương, dùng thuật ngữ pháp lý chính xác thay cho "
    "ngôn ngữ đời thường. Mỗi câu hỏi trên một dòng, không đánh số, không giải thích thêm."
)
_DECOMPOSE_SYSTEM_PROMPT = (
    "Bạn là trợ lý pháp lý. Cho tình huống dưới đây, hãy liệt kê 3-4 khía cạnh pháp lý "
    "riêng biệt cần tra cứu. Mỗi khía cạnh viết thành MỘT câu hỏi ngắn, không đánh số, "
    "không giải thích thêm, KHÔNG được tự bịa nội dung điều luật cụ thể."
)
_CONDENSE_SYSTEM_PROMPT = (
    "Bạn nhận được lịch sử hội thoại và câu hỏi mới nhất của người dùng. "
    "Hãy viết lại câu hỏi mới nhất thành MỘT câu hỏi độc lập, đầy đủ ngữ nghĩa, "
    "không cần đọc lịch sử vẫn hiểu được. KHÔNG trả lời câu hỏi, KHÔNG thêm "
    "thông tin luật pháp nào, chỉ diễn đạt lại đúng ý người dùng."
)


def rewrite_query(query: str, n_variants: int = config.QUERY_REWRITE_MAX_VARIANTS, max_new_tokens: int = 256) -> list:
    try:
        raw = generate_text(system_prompt=_REWRITE_SYSTEM_PROMPT, user_prompt=query, max_new_tokens=max_new_tokens, temperature=0.7)
    except Exception:
        return [query]
    variants = [line.strip("-• \t") for line in raw.splitlines() if line.strip()]
    variants = [v for v in variants if len(v) > 5][:n_variants]
    return [query] + variants if variants else [query]


def decompose_query(query: str, masked_query=None, n_subqueries: int = 4, max_new_tokens: int = 200) -> list:
    try:
        raw = generate_text(system_prompt=_DECOMPOSE_SYSTEM_PROMPT, user_prompt=masked_query or query,
                             max_new_tokens=max_new_tokens, temperature=0.5)
    except Exception:
        return []
    lines = [l.strip("-•\t ") for l in raw.splitlines() if l.strip()]
    return [l for l in lines if len(l) > 5][:n_subqueries]


def _format_history(history) -> str:
    lines = []
    for turn in history:
        role = turn.role if isinstance(turn, ChatTurn) else turn.get("role", "user")
        content = turn.content if isinstance(turn, ChatTurn) else turn.get("content", "")
        speaker = "Người dùng" if role == "user" else "Trợ lý"
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


def condense_question(history, latest_question: str, max_new_tokens: int = config.CONDENSE_QUESTION_MAX_NEW_TOKENS) -> str:
    if not history:
        return latest_question
    try:
        result = generate_text(system_prompt=_CONDENSE_SYSTEM_PROMPT,
                                user_prompt=_format_history(history) + f"\n\nCâu hỏi mới: {latest_question}",
                                max_new_tokens=max_new_tokens, temperature=0.2).strip()
        return result or latest_question
    except Exception:
        return latest_question
