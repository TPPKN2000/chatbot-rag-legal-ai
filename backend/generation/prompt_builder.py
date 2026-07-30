"""
Chat prompt assembly (grounded-chat-generation skill).

CHATBOT_MIGRATION_PLAN.md §2.2/§B2 (replaces the ALQAC JSON-verdict prompt):
the old `SYSTEM_PROMPT` forced the model to always emit one of 4 fixed
labels via a JSON schema (`accepted_ratio_estimate` + categorical
`prediction`), even under low confidence (only the confidence NUMBER was
allowed to drop, per the old rule #3). A chatbot must do the opposite:
declining to answer under insufficient grounding is the CORRECT behavior,
not a fallback failure mode — see `generate.py`'s hard refusal gate, which
enforces this at the code level rather than trusting the model's own
judgement about when to refuse.

Grounding & verbatim-law-text principles carry over UNCHANGED from the
ALQAC-era prompt:
  - Only cite provisions in the retrieved list, never invent one.
  - Law-provision text is inserted VERBATIM (never compressed/paraphrased —
    see generation/compress.py) so no connective word can be silently
    dropped.

D2 (CHATBOT_MIGRATION_PLAN.md): citations shown to the model — and expected
back from it — use the chunk's REAL printed article number
(`RetrievedChunk.article_num`, falling back to the internal `aid` only when
unparseable), never the raw internal `aid` directly. `allowed_citation_keys`
and `allowed_citation_map` both key off this same "display number" so the
hallucination guard in generate.py verifies against exactly what the model
was shown, not a hidden internal id it never saw.
"""
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
    """D2: prefer the real printed article number; fall back to the
    internal aid only when it couldn't be parsed."""
    return c.article_num if c.article_num is not None else c.aid


def _format_law_section(chunks: list[RetrievedChunk]) -> str:
    """Verbatim law text — NEVER pass through compress_auxiliary_text."""
    if not chunks:
        return "(Không có điều luật liên quan nào được truy hồi.)"
    lines = []
    for c in chunks:
        lines.append(f"- [{c.law_id} | Điều {_display_num(c)}]\n{c.text.strip()}")
    return "\n\n".join(lines)


def build_chat_prompt(
    contextualized_question: str,
    law_chunks: list[RetrievedChunk],
    conversation_digest: str,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) ready for
    `backend.models.generate_text`/`generate_text_stream`.

    `conversation_digest` is the pre-condensed conversation-history summary
    from `generation/conversation_digest.build_conversation_digest` (mirrors
    how ALQAC's `build_prediction_prompt` took a pre-built `case_digest`
    rather than raw evidence).
    """
    law_section = _format_law_section(law_chunks)

    user_prompt = f"""LỊCH SỬ HỘI THOẠI (tóm tắt): {conversation_digest.strip()}

CÂU HỎI HIỆN TẠI: {contextualized_question.strip()}

CÁC ĐIỀU LUẬT LIÊN QUAN ĐÃ TRUY HỒI (nguyên văn — chỉ được trích dẫn trong danh sách này):
{law_section}

Hãy trả lời câu hỏi, trích dẫn đúng các điều luật trên theo định dạng "[Điều X, law_id]"."""

    return CHAT_SYSTEM_PROMPT, user_prompt


def allowed_citation_keys(law_chunks: list[RetrievedChunk]) -> set[tuple[str, int]]:
    """The closed set of (law_id, DISPLAY article number) pairs the model
    was actually shown — used by generate.py's verification pass to strip
    any citation the model invents despite rule #2, or gets right in
    content but wrong in identity (citing a number not in the shown list)."""
    return {(c.law_id, _display_num(c)) for c in law_chunks}


def allowed_citation_map(law_chunks: list[RetrievedChunk]) -> dict[tuple[str, int], RetrievedChunk]:
    """Same key space as `allowed_citation_keys`, but keeps the full chunk
    (including the internal `aid`) so a verified citation can still be
    logged/returned with its stable identity, not just its display number."""
    return {(c.law_id, _display_num(c)): c for c in law_chunks}
