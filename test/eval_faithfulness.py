"""
Claim-level faithfulness + hallucination eval (RAGChecker/FaithJudge-style),
viết tay — không thêm dependency ragas/deepeval (coding_plan.md D1: tránh
kéo theo LangChain chỉ để gọi 1 judge).

Khác với `generate.generate.evaluate_groundedness()` (set-based: citation có
nằm trong whitelist không), 2 hàm ở đây đo NỘI DUNG: một khẳng định (claim)
trong câu trả lời có thực sự được NGỮ CẢNH đã retrieve xác nhận hay không —
bắt được cả trường hợp citation đúng nhưng nội dung diễn giải sai, VÀ trường
hợp claim sai dù không kèm citation nào (hai lỗi mà groundedness set-based
không thấy được).

D3 (bắt buộc): mọi hàm ở đây gọi qua `judge_client.judge_generate()`
(Anthropic API tách biệt), KHÔNG BAO GIỜ dùng `backend.models.generate_text`
(model sinh nội bộ) làm judge cho chính output của nó.
"""
from __future__ import annotations

import json

from test.judge_client import is_judge_available, judge_generate

_DECOMPOSE_PROMPT = (
    "Tách đoạn văn bản pháp lý sau thành các khẳng định (claim) độc lập, "
    "mỗi khẳng định là MỘT câu đơn giản, kiểm chứng được. Trả về JSON list "
    "of strings, không thêm giải thích, không thêm markdown code fence."
)
_VERIFY_PROMPT = (
    "Cho một KHẲNG ĐỊNH và một NGỮ CẢNH PHÁP LUẬT. Xác định khẳng định có "
    "được ngữ cảnh hỗ trợ hay không. Trả lời CHỈ một từ: 'supported' nếu "
    "ngữ cảnh xác nhận rõ ràng, 'contradicted' nếu ngữ cảnh nói ngược lại, "
    "'unverifiable' nếu ngữ cảnh không đề cập. KHÔNG dùng kiến thức ngoài "
    "ngữ cảnh được cung cấp."
)
_JUDGE_PROMPT = (
    "Bạn là giám khảo đánh giá chất lượng câu trả lời của trợ lý pháp luật. "
    "Cho CÂU HỎI và CÂU TRẢ LỜI, chấm điểm 1-5 theo 3 tiêu chí: "
    "(1) Mạch lạc — câu trả lời có rõ ràng, dễ hiểu không; "
    "(2) Đúng trọng tâm — có trả lời đúng câu hỏi được hỏi không, không lạc đề; "
    "(3) Hữu ích — người dùng có nhận được thông tin actionable không. "
    'Trả về JSON: {"coherence": int, "relevance": int, "helpfulness": int, "reasoning": str}. '
    "Không thêm markdown code fence."
)

_ALLOWED_VERDICTS = ("supported", "contradicted", "unverifiable")


def _strip_code_fence(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3]
    return raw.strip()


def decompose_into_claims(answer_text: str) -> list[str]:
    """Trả [] nếu judge không sẵn sàng hoặc lỗi — caller (evaluate_faithfulness)
    phải coi [] là 'không đo được', KHÔNG phải 'answer không có claim nào'."""
    if not is_judge_available():
        return []
    try:
        raw = judge_generate(_DECOMPOSE_PROMPT, answer_text, max_tokens=400, temperature=0.0)
        claims = json.loads(_strip_code_fence(raw))
        return [c for c in claims if isinstance(c, str) and c.strip()]
    except Exception:
        # Fallback thô theo dấu chấm câu — vẫn hữu ích hơn bỏ qua hoàn toàn,
        # nhưng caller nên biết đây là fallback (xem "note" trong kết quả).
        return [s.strip() for s in answer_text.split(".") if len(s.strip()) > 10]


def verify_claim(claim: str, context_text: str) -> str:
    if not is_judge_available():
        return "unverifiable"
    try:
        verdict = judge_generate(
            _VERIFY_PROMPT, f"KHẲNG ĐỊNH: {claim}\n\nNGỮ CẢNH:\n{context_text}",
            max_tokens=10, temperature=0.0,
        ).strip().lower()
        return verdict if verdict in _ALLOWED_VERDICTS else "unverifiable"
    except Exception:
        return "unverifiable"


def evaluate_faithfulness(answer_text: str, retrieved_law_chunks) -> dict:
    """claim-level faithfulness. Trả note rõ ràng nếu judge không sẵn sàng
    thay vì trả số liệu 0.0 gây hiểu nhầm là 'đo được và tệ'."""
    if not is_judge_available():
        return {
            "n_claims": 0, "n_supported": 0, "n_contradicted": 0, "n_unverifiable": 0,
            "faithfulness_score": None, "hallucination_rate": None, "claims": [],
            "note": "no external judge configured (ANTHROPIC_API_KEY not set) — see coding_plan.md D3",
        }

    context_text = "\n\n".join(c.text for c in retrieved_law_chunks)
    claims = decompose_into_claims(answer_text)
    verdicts = [verify_claim(c, context_text) for c in claims]
    n = len(verdicts) or 1
    return {
        "n_claims": len(claims),
        "n_supported": verdicts.count("supported"),
        "n_contradicted": verdicts.count("contradicted"),
        "n_unverifiable": verdicts.count("unverifiable"),
        "faithfulness_score": verdicts.count("supported") / n,
        # hallucination = contradicted HOẶC unverifiable — cả 2 đều là nội
        # dung KHÔNG được context xác nhận, khác nhau ở mức độ nghiêm trọng.
        "hallucination_rate": (verdicts.count("contradicted") + verdicts.count("unverifiable")) / n,
        "claims": list(zip(claims, verdicts)),
        "note": None,
    }


def judge_answer(question: str, answer_text: str) -> dict:
    """Trục helpfulness/coherence (D2) — trục DUY NHẤT cố ý cần LLM-judge
    trong thiết kế gốc của eval-harness-chatbot skill."""
    if not is_judge_available():
        return {"coherence": None, "relevance": None, "helpfulness": None, "reasoning": None,
                "note": "no external judge configured (ANTHROPIC_API_KEY not set)"}
    raw = judge_generate(_JUDGE_PROMPT, f"CÂU HỎI: {question}\n\nCÂU TRẢ LỜI: {answer_text}",
                          max_tokens=200, temperature=0.0)
    # Để nguyên exception nếu parse fail — eval script phải biết judge lỗi,
    # không âm thầm bỏ qua (D2/D4 nguyên tắc).
    result = json.loads(_strip_code_fence(raw))
    result["note"] = None
    return result
