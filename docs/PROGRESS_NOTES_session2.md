# PROGRESS_NOTES — Session 2 (thực thi `coding_plan.md`)

Tiếp nối `docs/PROGRESS_NOTES.md` (session 1 — dựng migration ALQAC → chatbot)
và việc tối ưu 6 skill (Nhóm C, đã làm ở phiên trước đó của session này).
Session này thực thi **Nhóm A, B (code), D** của `coding_plan.md`. Nhóm E
không cần code (chỉ đọc/quyết định) — xem ghi chú cuối file.

Toàn bộ code nằm ở `/home/claude/legalrag/` (sandbox Claude, dựng lại từ nội
dung đã dán trong hội thoại — repo thật của người dùng KHÔNG được mount ở
đây). Người dùng cần copy vào repo thật và đối chiếu kỹ.

## Nhóm A — Đóng khoảng trống migration

| # | Việc | Trạng thái | Ghi chú |
|---|---|---|---|
| A1 | Fixture nhỏ + smoke-test end-to-end | ✅ Xong, **chạy thật thành công** | `data/fixtures/corpus_mini.json` (4 Điều, có Khoản/Điểm), `test/smoke_test_fixture.py` mock 3 dependency nặng (Pinecone, CrossEncoder, LLM sinh) có kiểm soát. `python -m scripts.build_index --corpus data/fixtures/corpus_mini.json` chạy được, log "4/4 articles have a parseable article_num". 4/4 case smoke test đúng kỳ vọng (fastpath đúng nguyên văn, abstention đúng lúc, guardrail short-circuit đúng, fastpath không kích hoạt nhầm khi có điều kiện áp dụng). |
| A2 | HTTP server (`backend/server.py`, FastAPI) | ✅ Xong, **kiểm thử thật bằng TestClient** | 3 route: `/chat`, `/chat/stream`, `/healthz`. Đã test qua `fastapi.testclient.TestClient` (không phải chỉ compile) — cả 3 route trả đúng. Có TODO an toàn về session-ownership/auth trước khi public-facing (chưa làm — ngoài phạm vi A2 gốc). |
| A4 | Mở rộng `NEGATIVE_SAFETY_QUERIES` | ✅ Xong | 7 → 27 câu, phủ 6 nhóm: chit-chat, lĩnh vực khác, điều luật bịa, prompt injection cơ bản, input vô nghĩa (nhóm "tư vấn cá nhân hoá" để riêng làm tài liệu tham chiếu — KHÔNG đưa vào eval abstention mặc định vì đó là câu hỏi HỢP LỆ, chỉ cần disclaimer chứ không từ chối). |
| A3 | `data/chatbot_eval_set.json` | ⚠️ Xong bản **fixture-derived**, CHƯA phải bộ 30-50 câu production | 12 câu soạn từ `corpus_mini.json` (5 retrieval + 3 fastpath dương + 2 fastpath âm + 2 unsupported, theo gợi ý cân bằng grounded/unsupported của Nhóm E4). Khi có corpus thật, thay file này bằng bộ 40 Điều tự sinh theo đúng quy trình A3 mô tả — `test_chatbot.py` không cần sửa gì thêm, tự động dùng lại. |

**Phát hiện + vá thêm ngoài kế hoạch gốc:** chạy A1 thật (không chỉ
`py_compile`) lộ ra `backend/retrieval/hybrid_search.py` KHÔNG bắt lỗi khi
`vector_store.query()` thất bại (Pinecone down/chưa cấu hình) — lỗi
propagate thẳng lên `handle_chat_turn()`, crash cả turn của người dùng.
KHÔNG nhất quán với triết lý fallback đã áp dụng nhất quán ở
`condense_question`/`decompose_query`/`rewrite_query`/`build_conversation_digest`.
Đã vá bằng `_safe_vector_query()` — degrade về BM25-only + log warning 1
lần/process thay vì crash. Xác nhận bằng chạy thật (không mock): trước vá
crash với `ModuleNotFoundError`, sau vá trả về `is_abstention=True` đúng.

**Để ngỏ, KHÔNG tự ý sửa:** `backend/retrieval/rerank.py::rerank()` cũng
không có fallback khi CrossEncoder không tải được — nhưng đây là quyết định
kiến trúc lớn hơn (thiếu reranker ảnh hưởng chất lượng nhiều hơn thiếu 1
kênh vector song song với BM25 sẵn có). Để người dùng quyết định có muốn
thêm "degrade về thứ tự RRF-fused thô khi rerank lỗi" hay không, thay vì tự
ý thêm.

## Nhóm B — Chuẩn bị code cho embedding 4B (KHÔNG đổi default, KHÔNG reindex thật)

`backend/indexing/embed.py`: thêm `is_query` (instruction-prefix bất đối
xứng query/doc) + hỗ trợ MRL `truncate_dim` (degrade êm nếu
sentence-transformers < 3.1). Đã kiểm thử thật bằng model giả lập (3 case:
backward-compat mặc định, cấu hình kiểu Qwen3-Embedding qua config, document
encoding không bị ảnh hưởng — cả 3 đều PASS).

`backend/config.py`: thêm `EMBEDDING_QUERY_INSTRUCTION` (mặc định rỗng) +
`EMBEDDING_USE_MRL_TRUNCATE` (mặc định `false`).

**Quyết định có chủ đích:** KHÔNG đổi default `EMBEDDING_MODEL_NAME`/
`EMBEDDING_DIM`/`RERANKER_MODEL_NAME` sang Qwen3-family dù mẫu patch trong
`coding_plan.md` B2 có set default mới — vì B4 của chính tài liệu đó ghi rõ
"KHÔNG được merge vào production dù VRAM/latency ổn" nếu chưa pass A/B test
Recall@5 trên corpus thật. Đổi default trong `config.py` = merge vào
production ngay. Cách dùng đúng: override qua env (`EMBEDDING_MODEL_NAME`,
`EMBEDDING_DIM=1536`, `EMBEDDING_QUERY_INSTRUCTION=...`,
`EMBEDDING_USE_MRL_TRUNCATE=true`, `RERANKER_MODEL_NAME`, `INDEX_NAME=...-v2`)
— chi tiết đã ghi trong comment `config.py`.

**Chưa làm (không thể làm trong sandbox này):** B3 (reindex thật trên
Pinecone) và B4 (A/B test recall@5 thật) — cần GPU thật + corpus thật +
Pinecone API key thật + `data/chatbot_eval_set.json` bản đầy đủ (A3 thật,
không phải bản fixture). Sandbox này không có mạng tới `huggingface.co`
hoặc `api.pinecone.io` (chỉ PyPI/npm/GitHub) nên không thể tải model/kết
nối Pinecone thật để tự chạy B3/B4.

## Nhóm D — Faithfulness/hallucination + judge tách biệt

`test/judge_client.py` (D3): gọi Anthropic API trực tiếp qua `requests`
(không LangChain). `is_judge_available()` kiểm tra `ANTHROPIC_API_KEY` mà
không tốn network call. Xác nhận degrade đúng: không có key → trả `False`,
mọi hàm gọi judge trả `note` rõ ràng thay vì số liệu gây hiểu nhầm.

`test/eval_faithfulness.py` (D1+D2): `decompose_into_claims`, `verify_claim`,
`evaluate_faithfulness` (claim-level, 3 verdict: supported/contradicted/
unverifiable), `judge_answer` (helpfulness/coherence 1-5).

**Sửa 1 mâu thuẫn nội tại của `coding_plan.md`:** mẫu code D1 gốc gọi
`backend.models.generate_text` (model sinh nội bộ Qwen3.5-0.8B) làm judge —
nhưng D3 trong CÙNG tài liệu lại cấm chính điều này ("KHÔNG chấp nhận: model
tự chấm điểm chính nó"). Đã triển khai theo D3 (đúng), KHÔNG theo mẫu code D1
(sai) — `eval_faithfulness.py` gọi qua `judge_client.judge_generate()`.

`test/test_chatbot.py` (D4): thêm bước 5 (faithfulness/helpfulness), gate
bởi `is_judge_available()`, chỉ chạy khi `not args.skip_generation`. Đã kiểm
thử thật với judge giả lập (mock `judge_client.judge_generate`) — luồng chạy
hết không crash, log đúng format `"Faithfulness: X.XX | Hallucination rate:
X.XX | Helpfulness (avg 1-5): X.X"` như acceptance criteria D4 yêu cầu.

Đồng thời hoàn thiện `_FASTPATH_POSITIVE_CASES` (trước đây để trống trong
skill gốc) — giờ nạp từ `chatbot_eval_set.json` (`type=="fastpath"`), cả
dương lẫn âm, kiểm tra cả false-positive VÀ false-negative/sai nội dung.

## Nhóm E — không cần code

E1-E6 là các mục "đọc tài liệu tham khảo + quyết định", không có acceptance
criteria dạng code. Đối chiếu nhanh: E1 (cross-reference 1-hop) đã có sẵn
qua C2 (`config.CROSSREF_MAX_HOPS`, đổi được qua env nếu cần >1 hop sau
này); E6 (giữ hybrid BM25+dense) đã đúng kiến trúc hiện tại, không cần đổi.
E2/E3/E5 (fine-tune/benchmark VN-MTEB) phụ thuộc Nhóm B thật sự chạy trên
corpus thật — chưa tới lượt.

## Việc còn lại thật sự (không phải đã có code)

1. Chạy B3 (reindex) + B4 (A/B test) trên GPU/Pinecone/corpus thật — sandbox
   này không thể làm được (không có mạng HuggingFace/Pinecone).
2. A3 bản đầy đủ (30-50 câu trên corpus thật, không phải 12 câu fixture).
3. Quyết định có thêm fallback cho `rerank()` khi CrossEncoder lỗi hay không
   (để ngỏ, xem phần Nhóm A ở trên).
4. Auth/session-ownership cho `backend/server.py` trước khi public-facing
   (TODO đã ghi rõ trong code).
5. D3 "Ưu tiên 2" (dùng model 8B nội bộ làm judge thay vì Anthropic API) nếu
   Nhóm B (NVIDIA NIM) của repo ALQAC gốc được migrate — vẫn đang pending
   từ trước, chưa đổi gì thêm ở đây.
