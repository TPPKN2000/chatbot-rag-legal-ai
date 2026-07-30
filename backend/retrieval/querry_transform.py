"""
Pre-retrieval query transformation.

- rewrite_query(): turns colloquial language into legal-register variants
  (paraphrases) so BM25/vector search hit the terms actually used in
  statutes.
- decompose_query(): asks the LLM to list distinct legal *aspects* that need
  looking up as short standalone questions — never hypothetical statute
  text (HyDE was removed upstream of this migration; see the ALQAC-era
  system_adjustments_v3.md §3 for the original rationale, which still
  applies unchanged).
- condense_question(): NEW (CHATBOT_MIGRATION_PLAN.md §2.3 / conversational-
  retrieval skill, Bước 1). ALQAC was single-turn (`case_query` was already
  a complete, self-contained situation description); a chatbot's follow-up
  turns are routinely elliptical ("Vậy nếu bên vay không trả thì sao?" only
  makes sense given the previous turn). This asks the LLM to rewrite the
  latest question into a standalone one BEFORE it ever reaches
  `hybrid_search`/`collect_law_evidence` — otherwise retrieval silently
  loses the conversational context.

All three transforms share the same fallback philosophy: a transform-step
failure must never block retrieval or the whole pipeline. Each degrades to
"pass the input through unchanged" on any exception.
"""
from __future__ import annotations

from backend import config
from backend.models import ChatTurn, generate_text

_REWRITE_SYSTEM_PROMPT = (
    "Bạn là trợ lý pháp lý. Nhiệm vụ: viết lại câu hỏi/tình huống sau đây thành "
    "3 đến 5 câu hỏi tương đương, dùng thuật ngữ pháp lý chính xác thay cho "
    "ngôn ngữ đời thường (ví dụ: 'đánh nhau' -> 'hành vi cố ý gây thương tích', "
    "'lấy trộm' -> 'hành vi trộm cắp tài sản'). Mỗi câu hỏi trên một dòng, "
    "không đánh số, không giải thích thêm."
)

_DECOMPOSE_SYSTEM_PROMPT = (
    "Bạn là trợ lý pháp lý. Cho tình huống dưới đây, hãy liệt kê 3-4 khía cạnh pháp lý "
    "riêng biệt cần tra cứu (ví dụ: quan hệ pháp luật tranh chấp, điều kiện có hiệu lực "
    "của hợp đồng/giao dịch, thời hiệu khởi kiện, nghĩa vụ chứng minh). "
    "Mỗi khía cạnh viết thành MỘT câu hỏi ngắn, không đánh số, không giải thích thêm, "
    "KHÔNG được tự bịa nội dung điều luật cụ thể — chỉ nêu khía cạnh cần tra."
)

# conversational-retrieval skill: explicitly forbid answering or inventing
# legal content here — this step ONLY rephrases what the user already said.
_CONDENSE_SYSTEM_PROMPT = (
    "Bạn nhận được lịch sử hội thoại và câu hỏi mới nhất của người dùng. "
    "Hãy viết lại câu hỏi mới nhất thành MỘT câu hỏi độc lập, đầy đủ ngữ nghĩa, "
    "không cần đọc lịch sử vẫn hiểu được. KHÔNG trả lời câu hỏi, KHÔNG thêm "
    "thông tin luật pháp nào, chỉ diễn đạt lại đúng ý người dùng."
)


def rewrite_query(
    query: str,
    n_variants: int = config.QUERY_REWRITE_MAX_VARIANTS,
    max_new_tokens: int = 256,
) -> list[str]:
    """Return `query` plus up to `n_variants` legal-register paraphrases.

    Falls back to just [query] if generation fails for any reason."""
    try:
        raw = generate_text(
            system_prompt=_REWRITE_SYSTEM_PROMPT,
            user_prompt=query,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
        )
    except Exception:
        return [query]

    variants = [line.strip("-• \t") for line in raw.splitlines() if line.strip()]
    variants = [v for v in variants if len(v) > 5][:n_variants]
    return [query] + variants if variants else [query]


def decompose_query(
    query: str,
    masked_query: str | None = None,
    n_subqueries: int = 4,
    max_new_tokens: int = 200,
) -> list[str]:
    """Decompose the question into distinct legal aspects to look up, as
    short standalone questions. Never generates hypothetical statute text.

    Returns [] on failure — callers should treat that as "skip the
    decomposition route", not as an error.
    """
    try:
        raw = generate_text(
            system_prompt=_DECOMPOSE_SYSTEM_PROMPT,
            user_prompt=masked_query or query,
            max_new_tokens=max_new_tokens,
            temperature=0.5,
        )
    except Exception:
        return []
    lines = [l.strip("-•\t ") for l in raw.splitlines() if l.strip()]
    return [l for l in lines if len(l) > 5][:n_subqueries]


def _format_history(history: list[ChatTurn] | list[dict]) -> str:
    lines = []
    for turn in history:
        role = turn.role if isinstance(turn, ChatTurn) else turn.get("role", "user")
        content = turn.content if isinstance(turn, ChatTurn) else turn.get("content", "")
        speaker = "Người dùng" if role == "user" else "Trợ lý"
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


def condense_question(
    history: list[ChatTurn] | list[dict],
    latest_question: str,
    max_new_tokens: int = config.CONDENSE_QUESTION_MAX_NEW_TOKENS,
) -> str:
    """conversational-retrieval skill, Bước 1: rewrite `latest_question` into
    a standalone question given `history`, so retrieval never has to see an
    elliptical follow-up ("Vậy nếu... thì sao?") in isolation.

    Returns `latest_question` unchanged (no-op) when there is no history yet
    (first turn of a session) — nothing to condense against. Also falls back
    to the raw question on any generation failure: a broken condense step
    must never block the whole turn, matching the fallback philosophy
    already used by rewrite_query/decompose_query.
    """
    if not history:
        return latest_question
    try:
        result = generate_text(
            system_prompt=_CONDENSE_SYSTEM_PROMPT,
            user_prompt=_format_history(history) + f"\n\nCâu hỏi mới: {latest_question}",
            max_new_tokens=max_new_tokens,
            temperature=0.2,
        ).strip()
        return result or latest_question
    except Exception:
        return latest_question
