from __future__ import annotations
from backend.models import ChatTurn, RetrievedChunk

CHAT_SYSTEM_PROMPT = """Bạn là trợ lý tra cứu pháp luật Việt Nam.

QUY TẮC BẮT BUỘC:
1. Chỉ trả lời dựa trên các điều luật và tóm tắt hội thoại được cung cấp bên dưới.
   KHÔNG bịa thêm điều khoản, số hiệu văn bản không có trong danh sách.
2. Mọi luận điểm PHẢI trích dẫn cụ thể, viết dạng "[Điều {số điều}, {law_id}]"
   ngay sau luận điểm đó trong câu trả lời — số điều PHẢI lấy đúng từ danh sách
   "CÁC ĐIỀU LUẬT LIÊN QUAN" bên dưới, không tự đổi số.
3. Nếu ngữ cảnh được cung cấp KHÔNG đủ để trả lời có căn cứ, hãy nói rõ điều đó và
   đề nghị người dùng cung cấp thêm chi tiết — TUYỆT ĐỐI KHÔNG đoán khi thiếu căn cứ.
4. Đây là công cụ hỗ trợ tra cứu, KHÔNG thay thế tư vấn pháp lý chính thức từ luật sư.
   Với câu hỏi mang tính tư vấn cá nhân hoá (ví dụ "tôi nên làm gì trong tình huống X"),
   nhắc người dùng cân nhắc gặp luật sư cho quyết định cuối cùng.
5. Trả lời bằng văn xuôi tự nhiên, tiếng Việt, KHÔNG dùng JSON, KHÔNG dùng markdown
   code fence.
"""


def _display_num(c: RetrievedChunk) -> int:
    return c.article_num if c.article_num is not None else c.aid


def _format_law_section(chunks: list) -> str:
    if not chunks:
        return "(Không có điều luật liên quan nào được truy hồi.)"
    lines = []
    for c in chunks:
        lines.append(f"- [{c.law_id} | Điều {_display_num(c)}]\n{c.text.strip()}")
    return "\n\n".join(lines)


def build_chat_prompt(contextualized_question: str, law_chunks: list, conversation_digest: str):
    law_section = _format_law_section(law_chunks)
    user_prompt = f"""LỊCH SỬ HỘI THOẠI (tóm tắt): {conversation_digest.strip()}

CÂU HỎI HIỆN TẠI: {contextualized_question.strip()}

CÁC ĐIỀU LUẬT LIÊN QUAN ĐÃ TRUY HỒI (nguyên văn — chỉ được trích dẫn trong danh sách này):
{law_section}

Hãy trả lời câu hỏi, trích dẫn đúng các điều luật trên theo định dạng "[Điều X, law_id]"."""
    return CHAT_SYSTEM_PROMPT, user_prompt


def allowed_citation_keys(law_chunks: list) -> set:
    return {(c.law_id, _display_num(c)) for c in law_chunks}


def allowed_citation_map(law_chunks: list) -> dict:
    return {(c.law_id, _display_num(c)): c for c in law_chunks}
