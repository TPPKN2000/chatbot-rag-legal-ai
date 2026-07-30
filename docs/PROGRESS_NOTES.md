# PROGRESS_NOTES.md — Chatbot Migration (phiên thực hiện hiện tại)

Mục đích: cho phép một phiên làm việc SAU (hoặc chính Claude ở conversation
khác) tiếp tục đúng chỗ đã dừng, không cần đọc lại toàn bộ lịch sử chat.
Đối chiếu trực tiếp với bảng lộ trình ở `CHATBOT_MIGRATION_PLAN.md §4`.

Toàn bộ code được viết ra trong phiên này nằm ở `/home/claude/legalrag/`
(sandbox của Claude) — đây là bản dựng lại từ nội dung các file được dán
trong hội thoại (repo thật của người dùng KHÔNG được mount ở đây, chỉ có
`CHATBOT_MIGRATION_PLAN.md` trong `/mnt/project/`). Người dùng cần copy các
file này vào đúng vị trí trong repo thật của họ và đối chiếu kỹ trước khi
merge — đặc biệt các phần đụng tới `data/corpus_law_pub.json` thật (chưa có
sẵn ở đây để test end-to-end).

## Bảng trạng thái theo lộ trình (`CHATBOT_MIGRATION_PLAN.md §4`)

| # | Việc | Trạng thái | File | Ghi chú |
|---|---|---|---|---|
| A1 | Gỡ nhánh case-outcome khỏi pipeline.py, xoá case_api_client.py/submission.py khỏi luồng chính | ✅ Xong | `backend/pipeline.py` (viết lại), `case_api_client.py`/`submission.py` không còn trong cây thư mục mới | |
| A2 | Viết chat_pipeline.py | ✅ Xong | `backend/chat_pipeline.py` | `handle_chat_turn()` (blocking) + `handle_chat_turn_stream()` (B3) |
| A3 | Citation fast-path | ✅ Xong | `backend/retrieval/citation_fastpath.py` + `chunker.build_article_num_lookup/build_khoan_lookup` + `scripts/build_index.py` (persist) | Xem mục "Quyết định/đánh đổi" bên dưới — KHÔNG copy nguyên `AI/data_utils.py` (không có trong context), viết lại tương đương |
| B1 | Session/history store | ✅ Xong | `backend/session_store.py` | In-memory, đúng như plan yêu cầu "tối giản giai đoạn đầu" |
| B2 | Prompt tự do + citation inline + nhánh từ chối | ✅ Xong | `backend/generation/prompt_builder.py`, `generate.py`, `chat_pipeline.py` (hard refusal gate ở tầng code) | |
| B3 | Streaming | ✅ Xong (với đánh đổi đã ghi rõ) | `backend/models.py::generate_text_stream`, `generate.py::generate_chat_answer_stream` | Xem "Quyết định/đánh đổi" — buffer-rồi-verify-rồi-replay, KHÔNG stream token thô trực tiếp |
| C1 | Scope/intent guardrail + disclaimer | ✅ Xong (heuristic, không dùng LLM-judge) | `backend/guardrail.py` | Cố ý giữ hẹp/high-precision cho phần out-of-scope — xem ghi chú trong file |
| C2 | Multi-hop cross-reference | ✅ Xong | `backend/pipeline.py::_resolve_cross_references`, `backend/ingestion/metadata.py` (không đổi, chỉ được gọi) | Gate chung với vòng decomposition, 1 hop, dùng chung fastpath index |
| C3 | Eval harness mới | ✅ Khung xong, DỮ LIỆU chưa đầy đủ | `test/test_chatbot.py` | Groundedness + abstention + fastpath-false-positive chạy được ngay; recall@k và fastpath-positive-case cần `data/chatbot_eval_set.json` tự soạn (CHƯA có) |
| D1 | Accent-insensitive tokenizer cho BM25 | ✅ Xong | `backend/indexing/bm25_index.py` (`fold_accents`, `query_folded`, kênh RRF phụ trọng số thấp) | |
| D2 | Tách field trích dẫn khỏi breadcrumb | ✅ Xong (mở rộng hơn plan một chút — xem bên dưới) | `backend/ingestion/chunker.py` (`article_num`), `backend/indexing/vector_store.py` (`article_num` metadata field), `models.py` (`RetrievedChunk.article_num`) | |

**Tất cả 12 mục trong bảng lộ trình gốc đã có code.** Việc còn lại là
review/test bằng dữ liệu thật (xem mục "Chưa làm / cần xác nhận" bên dưới),
KHÔNG phải viết thêm code cho các mục đã liệt kê ở trên.

## Quyết định / đánh đổi đã tự chọn (KHÔNG có trong plan gốc, cần người dùng xác nhận)

1. **`aid` vs `article_num` — vấn đề kế thừa từ ALQAC, ảnh hưởng TOÀN BỘ
   citation hiển thị cho người dùng.** `system_adjustments_v4.md`/
   `ACTION_PLAN.md` đã xác nhận: corpus ALQAC có `aid` là ID nội bộ
   (vd. 50882), KHÔNG phải số "Điều N" thật. Bug này vốn chỉ ảnh hưởng phép
   đo (Micro Law F1) trong pipeline ALQAC vì không có ai đọc citation trực
   tiếp. Nhưng trong CHATBOT, citation được hiển thị thẳng cho người dùng —
   nếu không sửa, chatbot sẽ trả lời kiểu "[Điều 50882, ...]" hoàn toàn vô
   nghĩa với người dùng thật.
   → Đã CHỦ ĐỘNG thêm field `article_num` (số Điều in thật, parse từ
   `title`/`body` — cùng kỹ thuật `ACTION_PLAN.md §A4` đã dùng cho việc đo
   lường) vào `LawChunk`, `RetrievedChunk`, metadata Pinecone. Toàn bộ
   citation hiển thị cho người dùng (`prompt_builder.py`, `generate.py`,
   `citation_fastpath.py`) ưu tiên `article_num`, fallback về `aid` khi
   không parse được. `aid` vẫn là khoá định danh nội bộ dùng cho
   `allowed_citation_keys`/whitelist — KHÔNG đổi.
   **Cần xác nhận:** nếu corpus THẬT dùng cho chatbot khác corpus ALQAC và
   `aid` ở đó vốn đã = số Điều thật, cơ chế fallback này vẫn hoạt động đúng
   (article_num sẽ trùng aid), không cần sửa gì thêm — nhưng nên chạy thử
   `build_index.py` và xem log dòng "citation fast-path index: N/M articles
   have a parseable article_num" để xác nhận tỉ lệ parse thực tế.

2. **Streaming (B3) không stream token thô trực tiếp.** Skill
   `grounded-chat-generation` đưa ra `generate_text_stream()` stream thẳng
   token — nhưng nếu làm vậy, hallucination guard (xoá citation sai) không
   thể áp dụng NGƯỢC lên token đã hiển thị trên màn hình người dùng rồi.
   Đã chọn: `generate_chat_answer_stream()` buffer TOÀN BỘ output ở server,
   verify, rồi phát lại theo từng từ (word-by-word) của bản ĐÃ LỌC — người
   dùng vẫn thấy hiệu ứng gõ chữ, nhưng không bao giờ thấy citation chưa
   xác minh. Đánh đổi: mất độ trễ "token đầu tiên xuất hiện ngay lập tức"
   thật sự (phải đợi generate xong hết mới bắt đầu phát). Với model
   Qwen3.5-0.8B, độ trễ tổng thể vẫn nhỏ nên đây là lựa chọn ưu tiên đúng
   (an toàn > cảm giác mượt) cho một sản phẩm tư vấn pháp lý, nhưng NÊN bàn
   lại nếu độ trễ thực tế trên GPU thật lớn hơn dự kiến.

3. **Guardrail (C1) cố ý hẹp, không dùng LLM để phân loại scope.**
   `is_trivially_out_of_scope()` chỉ bắt các câu hỏi RÕ RÀNG ngoài phạm vi
   (chào hỏi, thời tiết, bóng đá...) bằng regex/keyword — KHÔNG dùng LLM
   phân loại ý định vì: (a) tốn thêm 1 lệnh gọi LLM mỗi lượt chat, (b) rủi
   ro false-positive cao hơn (từ chối nhầm câu hỏi pháp lý hợp lệ). Các câu
   hỏi ngoài phạm vi "mơ hồ" hơn được kỳ vọng tự nhiên bị chặn bởi cổng từ
   chối dựa trên điểm rerank đã có sẵn (retrieval sẽ không tìm thấy gì liên
   quan trong corpus luật). **Cần xác nhận bằng test thật:** chạy
   `test/test_chatbot.py`'s abstention axis với bộ câu hỏi ngoài phạm vi đa
   dạng hơn (mở rộng `NEGATIVE_SAFETY_QUERIES` trong `guardrail.py`) để xem
   cổng retrieval-score có đủ tin cậy không, hay cần thêm lớp guardrail
   thứ 2 (LLM-based) sau này.

4. **Cross-reference resolution (C2) dùng chung fastpath index thay vì xây
   riêng.** `conversational-retrieval` skill mô tả một hàm `resolve_chunk()`
   độc lập; đã thực hiện bằng cách tái dùng `citation_fastpath.get_fastpath_index()`
   (article_lookup + chunk_by_id) — đúng nguyên tắc "không tạo luồng code
   song song thứ 3" của `legacy-prototype-salvage` skill. Hệ quả: C2 phụ
   thuộc vào cùng file `config.ARTICLE_NUM_LOOKUP_PATH` — nếu file này thiếu
   (chưa chạy `build_index.py`), CẢ fastpath LẪN cross-reference hop đều
   im lặng không hoạt động (degrade an toàn, có log warning, không crash).

5. **`docs/` cũ (case_content_api_doc.md, evaluation.md,
   submission_example.json, test_design.md) KHÔNG được copy sang** — đúng
   theo migration plan §2.2 (những khái niệm này không còn áp dụng: không
   còn Case Content API, không còn công thức chấm điểm ALQAC, không còn
   file nộp bài). `docs/system_design_v0.md` (thiết kế gốc trước ALQAC-era
   adjustments) và các file `system_adjustments_v*.md`/`progress_notes_v*.md`
   /`ACTION_PLAN.md` KHÔNG được copy sang repo mới — chúng là lịch sử phát
   triển của pipeline ALQAC, không có trong context để copy verbatim, và
   không còn mô tả code hiện tại. Nếu người dùng muốn giữ lại làm tài liệu
   lịch sử, nên copy thủ công từ repo thật vào một thư mục `docs/archive/`.

## Chưa làm / cần xác nhận (việc còn lại thật sự, không phải đã có code)

1. **Chưa test end-to-end trên GPU/corpus thật.** Toàn bộ code trong phiên
   này mới qua `python3 -m py_compile` (không lỗi cú pháp) + đọc lại logic
   thủ công. CHƯA chạy được vì: (a) không có `data/corpus_law_pub.json`
   thật trong sandbox, (b) không có GPU/model weights để load Qwen3.5-0.8B/
   reranker/embedder/NER thật, (c) không có Pinecone API key thật.
   **Việc cần làm tiếp:** copy code vào repo thật, chạy
   `python -m scripts.build_index --corpus <corpus thật>`, kiểm tra log
   "citation fast-path index: N/M articles..." (mục Quyết định #1), rồi
   `python -m test.test_chatbot --skip-generation` trước (không cần model
   sinh), sau đó bỏ `--skip-generation` khi model đã sẵn sàng.

2. **Chưa có HTTP server (`/chat` route).** `chat_pipeline.py` mới là API
   Python nội bộ (`handle_chat_turn`, `handle_chat_turn_stream`) — chưa có
   FastAPI/Flask wrapper. Migration plan không yêu cầu mục này rõ ràng
   (không có trong bảng §4), nhưng cần thiết để thực sự "dùng được" như một
   chatbot — nên là việc tiếp theo hợp lý nhất nếu tiếp tục phiên sau.

3. **`data/chatbot_eval_set.json` chưa tồn tại** (xem C3 trong bảng trên) —
   cần người có domain knowledge (hoặc Claude, nếu được cấp corpus thật)
   tự soạn 30-50 câu `{"question": ..., "gold_aids": [...]}` theo đúng
   corpus thật sẽ dùng. `_FASTPATH_POSITIVE_CASES` trong `test_chatbot.py`
   cũng đang để trống vì cùng lý do (cần biết số Điều thật trong corpus để
   viết case dương chính xác).

4. **Chưa viết `data/` sample/fixture nào** để chạy thử nhanh không cần
   corpus đầy đủ (vd. 1 luật nhỏ, vài Điều, để smoke-test toàn bộ pipeline
   cục bộ). Nên cân nhắc làm việc này trước khi có corpus thật, để phát
   hiện lỗi runtime sớm hơn (vd. lỗi Pydantic, lỗi import) mà không cần chờ
   hạ tầng đầy đủ.

5. **`backend/guardrail.py`'s `NEGATIVE_SAFETY_QUERIES`** hiện chỉ có 7 câu
   mẫu (tối thiểu để minh hoạ) — plan/skill gợi ý mở rộng dần theo thời
   gian, chưa phải bộ đầy đủ.

6. **Chưa quyết định lưu trữ session lâu dài** (migration plan §6 đã tự ghi
   nhận đây là rủi ro treo, không phải việc bỏ sót của phiên này) —
   `session_store.py` viết theo interface tối giản (3 method) để dễ thay
   backend sau, nhưng vẫn là in-memory thuần, mất dữ liệu khi process
   restart.

## Nếu tiếp tục ở phiên sau, thứ tự nên làm

1. Copy toàn bộ `/home/claude/legalrag/` vào repo thật, đối chiếu từng file
   với bản gốc trong repo thật (không phải bản trong context hội thoại này)
   để chắc chắn không có phần nào của repo thật bị bỏ sót khi migrate
   (phiên này chỉ thấy được các file được dán vào hội thoại, có thể repo
   thật còn file khác không xuất hiện trong context).
2. Chạy `scripts/build_index.py` trên corpus thật, kiểm tra log coverage.
3. Viết `data/chatbot_eval_set.json` nhỏ (mục "Chưa làm" #3).
4. Chạy `test/test_chatbot.py --skip-generation`, sửa lỗi runtime nếu có.
5. Chạy full `test/test_chatbot.py` với model thật, review chất lượng câu
   trả lời + tỉ lệ groundedness/abstention.
6. Viết HTTP server wrapper nếu cần dùng thực tế (mục "Chưa làm" #2).
