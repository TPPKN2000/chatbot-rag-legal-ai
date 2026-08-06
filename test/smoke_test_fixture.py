"""
A1 (coding_plan.md): smoke-test end-to-end trên data/fixtures/corpus_mini.json,
KHÔNG cần GPU/Pinecone/HuggingFace network thật.

Mock 3 dependency nặng, mỗi cái có lý do cụ thể để không coi là "gian lận" test:
  - vector_store.query -> [] : không có PINECONE_API_KEY thật trong sandbox.
    BM25-only retrieval vẫn là một đường chạy thật, hợp lệ (hybrid_search đã
    thiết kế RRF để chịu được một channel rỗng).
  - rerank() -> passthrough theo thứ tự BM25 score : không có mạng tới
    HuggingFace để tải CrossEncoder thật. Đây KHÔNG kiểm tra chất lượng
    rerank, CHỈ kiểm tra pipeline không crash khi rerank trả về dữ liệu.
  - generate_text/generate_text_stream -> raise ngay lập tức : mô phỏng đúng
    "generation model không sẵn sàng" mà generate.py/querry_transform.py đã
    THIẾT KẾ SẴN fallback cho (xem docstring generate_chat_answer, gotcha
    trong grounded-chat-generation skill) — test này xác nhận đúng nhánh
    fallback đó chạy, không phải bỏ qua nó.

Acceptance criteria (coding_plan.md A1):
  1. "Điều 1 nói gì?" (khớp fixture) -> is_fastpath=True, đúng nguyên văn Điều 1.
  2. Câu hỏi ngoài phạm vi fixture -> is_abstention=True, KHÔNG crash, KHÔNG bịa.
  3. Câu hỏi trivially out-of-scope (guardrail) -> is_abstention=True, 0 lệnh
     gọi retrieval/generation nào (short-circuit sớm nhất).
"""
from __future__ import annotations

import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import backend.indexing.vector_store as vector_store
import backend.retrieval.rerank as rerank_module
import backend.models as models_module


def _mock_vector_query(text, top_k=None, law_id=None, require_active=True):
    return []


def _mock_rerank(query, candidates, top_k=8):
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)[:top_k]
    return [c.model_copy(update={"source": "reranked"}) for c in ranked]


def _mock_generate_text(*args, **kwargs):
    raise RuntimeError("generation model not available in this sandbox (no GPU/HF network)")


def _mock_generate_text_stream(*args, **kwargs):
    raise RuntimeError("generation model not available in this sandbox (no GPU/HF network)")
    yield  # pragma: no cover - makes this a generator function


def apply_mocks():
    vector_store.query = _mock_vector_query
    rerank_module.rerank = _mock_rerank
    models_module.generate_text = _mock_generate_text
    models_module.generate_text_stream = _mock_generate_text_stream
    # pipeline.py and querry_transform.py imported generate_text by name
    # (`from backend.models import ... generate_text`) — patch those bound
    # references too, not just the origin module.
    import backend.retrieval.querry_transform as qt
    qt.generate_text = _mock_generate_text
    import backend.generation.conversation_digest as cd
    cd.generate_text = _mock_generate_text
    import backend.generation.generate as gen
    gen.generate_text = _mock_generate_text
    gen.generate_text_stream = _mock_generate_text_stream


def run():
    apply_mocks()
    from backend.chat_pipeline import handle_chat_turn

    failures = []

    # --- Case 1: citation fast-path (khớp fixture) ---
    answer1 = handle_chat_turn("smoke-1", "Điều 1 nói gì?")
    ok1 = answer1.is_fastpath and "Bộ luật này quy định địa vị pháp lý" in answer1.answer
    print(f"[Case 1] fastpath={answer1.is_fastpath} abstention={answer1.is_abstention}")
    print(f"          answer[:80]={answer1.answer[:80]!r}")
    if not ok1:
        failures.append("Case 1: kỳ vọng is_fastpath=True với nguyên văn Điều 1")

    # --- Case 2: ngoài phạm vi fixture (BM25 sẽ không tìm thấy gì liên
    # quan -> retrieval-evaluator gate phải từ chối, KHÔNG crash) ---
    answer2 = handle_chat_turn(
        "smoke-2",
        "quy định về xử phạt vi phạm giao thông đường bộ khi vượt đèn đỏ là gì",
    )
    ok2 = answer2.is_abstention
    print(f"[Case 2] fastpath={answer2.is_fastpath} abstention={answer2.is_abstention} "
          f"reason={answer2.abstention_reason!r}")
    if not ok2:
        failures.append("Case 2: kỳ vọng is_abstention=True cho câu hỏi ngoài phạm vi fixture")

    # --- Case 3: trivially out-of-scope (guardrail short-circuit) ---
    answer3 = handle_chat_turn("smoke-3", "xin chào")
    ok3 = answer3.is_abstention and answer3.abstention_reason == "trivially out of scope (guardrail)"
    print(f"[Case 3] abstention={answer3.is_abstention} reason={answer3.abstention_reason!r}")
    if not ok3:
        failures.append("Case 3: kỳ vọng guardrail short-circuit cho câu chào hỏi")

    # --- Case 4: fast-path phải KHÔNG kích hoạt cho câu có điều kiện áp dụng ---
    answer4 = handle_chat_turn("smoke-4", "Điều 470 áp dụng thế nào nếu bên vay phá sản giữa kỳ hạn?")
    ok4 = not answer4.is_fastpath
    print(f"[Case 4] fastpath={answer4.is_fastpath} (kỳ vọng False)")
    if not ok4:
        failures.append("Case 4: fast-path KHÔNG được kích hoạt khi câu hỏi có điều kiện áp dụng")

    print()
    if failures:
        print(f"SMOKE TEST FAILED ({len(failures)} lỗi):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("SMOKE TEST PASSED — 4/4 case đúng kỳ vọng, pipeline không crash.")


if __name__ == "__main__":
    run()
