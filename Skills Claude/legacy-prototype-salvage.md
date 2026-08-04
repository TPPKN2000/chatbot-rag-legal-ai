---
name: legacy-prototype-salvage
description: Tài liệu tra cứu — dùng khi cần biết một ý tưởng/kỹ thuật cụ thể có thể lấy được từ đâu trong 2 codebase nháp đã tồn tại sẵn trong repo (AI/ và các script rời trong backend/) trước khi viết code mới từ đầu cho dự án LegalRAG chatbot. Không phải hướng dẫn triển khai — là bảng tra cứu nguồn gốc ý tưởng.
---

# Legacy Prototype Salvage — bảng tra cứu

Repo có 2 pipeline nháp khác hoàn toàn với `backend/` package chính (ALQAC). Trước khi viết mới bất kỳ thành phần nào cho chatbot, kiểm tra bảng dưới xem đã có sẵn ý tưởng/code tham khảo chưa.

## Nháp #1: `AI/` — extractive QA, SQLite FTS5, không vector DB
Kiến trúc: `data.json` → `repair_data.py` (sửa lỗi PDF line-wrap) → `prepare_data.py` (chia train/val/test theo article, augment câu hỏi) → `retrieval.py` (SQLite + FTS5, exact-citation branch + BM25-fallback) → `legal_spans.py` (cắt đoạn theo cấu trúc) → `train.py`/`model.py` (BiGRU reader tự train) → `predict.py` (kết hợp các đường).

| Cần gì | Lấy từ đâu trong `AI/` | Ghi chú |
|---|---|---|
| Parse "Điều X khoản Y điểm Z" từ câu hỏi tự nhiên, mọi thứ tự từ, có/không dấu | `data_utils.py::extract_legal_reference()` | Regex đã test kỹ (`AI/tests/test_pipeline.py::test_reference_parser_handles_accents_and_word_order`) |
| Phân loại câu hỏi đang hỏi cả điều/chỉ 1 khoản/chỉ 1 điểm | `data_utils.py::infer_question_kind()` | Dùng cho citation fast-path quyết định trả nguyên Điều hay chỉ 1 Khoản |
| Phát hiện câu hỏi "hỏi chung chung toàn bộ nội dung" vs "hỏi có điều kiện áp dụng cụ thể" | `data_utils.py::is_generic_reference_question()` | Quyết định có được bypass LLM (fast-path) hay không |
| Cắt chính xác 1 đoạn Khoản/Điểm từ văn bản Điều đầy đủ | `legal_spans.py::extract_structured_answer()` | Dựa ranh giới cấu trúc bằng regex, KHÔNG dùng model — an toàn 100%, không hallucination |
| Tokenizer bỏ dấu tiếng Việt cho tra cứu | `retrieval.py` — FTS5 `tokenize='unicode61 remove_diacritics 2'` | `bm25_index.py` hiện tại CHƯA có bước này |
| Ngưỡng từ chối khi độ tin cậy thấp | `retrieval.py::MIN_RETRIEVAL_CONFIDENCE = 0.85` | Nguyên tắc, không phải số cụ thể — cần tune lại cho corpus luật dân sự |
| Bộ câu hỏi test khả năng từ chối đúng lúc (an toàn) | `evaluate.py::NEGATIVE_SAFETY_QUERIES` + `evaluate_abstention_safety()` | Dùng làm mẫu cho eval harness mới |

**KHÔNG nên lấy:** model BiGRU tự train (`model.py::SimpleQAModel`) — chỉ đạt ~60% exact match ở chế độ full-context thật (`AI/evaluation_report.json::full_context_reader.exact_match = 0.6056`), thấp hơn nhiều so với việc dùng LLM sinh có grounding đã có trong `backend/`. Retrieval-index-based benchmark của `AI/` đạt 100% chỉ vì index chứa chính xác nội dung đang được test (xem cảnh báo trong `AI/README.md::"Is the current run finished?"` — "a human-authored external benchmark is still needed to estimate generalization").

## Nháp #2: script rời trong `backend/` — ChromaDB + bge-m3 + Ollama
File liên quan: `backend/embed.py`, `retriever.py`, `retriever_api.py`, `testvecto.py`, `testretriever.py`, `prompt.py`, `api.py`, `LLM.py`, và `server.js` (gọi trực tiếp `retriever_api.py`, KHÔNG đi qua package `backend/` chính).

| Cần gì | Lấy từ đâu | Ghi chú |
|---|---|---|
| Hợp đồng I/O đơn giản Node ↔ Python (JSON qua stdin/stdout) | `retriever_api.py` + `server.js::runQaModel()` | Đã chạy được thật; dùng làm baseline khi chưa có HTTP streaming server, nhưng nên thay bằng API HTTP thực sự cho production (subprocess-per-request không scale) |
| Schema field trích dẫn phẳng, dễ hiển thị UI | `embed.py` — metadata `{law_name, chapter, article, clause, point, title, level}` | Đối chiếu: `vector_store.py::_chunk_metadata()` hiện gộp hết vào 1 field `breadcrumb` string — nên tách lại field riêng để UI không cần tự parse |
| Serving LLM đơn giản không cần lo `device_map`/`dtype` | `LLM.py` (gọi qua `ollama.chat(model="qwen2.5:7b", ...)`) | Cân nhắc làm serving backend thay thế cho dev/local; giữ HF transformers in-process (`models.py`) cho production GPU thật |
| Câu prompt từ chối tối giản khi thiếu context | `prompt.py` — "Nếu Context không đủ thì nói: Không tìm thấy thông tin." | Chỉ lấy Ý TƯỞNG câu từ chối; KHÔNG copy nguyên prompt vì thiếu citation format và whitelist chống hallucination đã có trong `generate.py` |

**KHÔNG nên lấy:** `api.py` (route stub không có error handling/rate limit), toàn bộ pipeline chunking của nháp #2 (không thấy code chunking — có vẻ `legal_nodes.json` được build sẵn ở ngoài repo, không rule-based theo Điều/Khoản/Điểm như `chunker.py` của `backend/` chính) — dùng chunking rule-based hiện có, KHÔNG quay lại cách này.

## Nguyên tắc chung khi salvage
1. Lấy **ý tưởng & pattern**, không copy nguyên khối code — 2 nháp này chưa qua đánh giá kỹ (không có test suite tương đương `test/test_all_backend.py`, không có hallucination guard, không có rerank).
2. Mọi ý tưởng lấy về phải được cấy vào đúng vị trí tương ứng trong package `backend/` chính (xem `CHATBOT_MIGRATION_PLAN.md` mục 3.3), không tạo thêm một luồng code song song thứ 3.
3. Nếu một ý tưởng ở 2 nháp mâu thuẫn với nguyên tắc thiết kế đã có trong `backend/` chính (vd. nháp #2 dùng embedding đa ngôn ngữ `bge-m3` thay vì `AITeamVN/Vietnamese_Embedding` đã chọn), KHÔNG tự ý đổi mà không có A/B test — xem `docs/system_design_v0.md §5` (đã tự ghi chú nên A/B test `AITeamVN` vs các lựa chọn khác).
