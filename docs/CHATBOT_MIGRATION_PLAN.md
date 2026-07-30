# LegalRAG → Legal Chatbot — Kế hoạch chuyển đổi

> Phạm vi tài liệu: chuyển `backend/` (pipeline dự đoán kết quả vụ án cho ALQAC2026) thành một **chatbot RAG tra cứu/tư vấn pháp luật đa lượt**. Tài liệu này chỉ mô tả *thay đổi kiến trúc*; các quyết định thiết kế gốc (rule-based chunking, RRF, no-framework) không đổi trừ khi nêu rõ.
>
> Repo hiện có **3 lớp pipeline khác nhau**, không phải 1:
> 1. `backend/` (package Python có cấu trúc: `config.py`, `pipeline.py`, `ingestion/`, `indexing/`, `retrieval/`, `generation/`) — pipeline ALQAC chính, Pinecone + Qwen3.5-0.8B. **Đây là nền để build chatbot mới.**
> 2. `AI/` — một pipeline QA trích xuất hoàn toàn khác: SQLite FTS5 + regex citation parser + BiGRU reader tự train, không dùng vector DB/LLM sinh. Coi là **repo nháp #1**.
> 3. Các script rời nằm trực tiếp trong `backend/` (`embed.py`, `retriever.py`, `retriever_api.py`, `testvecto.py`, `testretriever.py`, `prompt.py`, `api.py`, `LLM.py`) — một prototype chatbot RAG khác, dùng ChromaDB + `BAAI/bge-m3` + Ollama, được `server.js` gọi trực tiếp (không đi qua package `backend/` ở mục 1!). Coi là **repo nháp #2**.
>
> Mục 3 của tài liệu này khai thác nháp #1 và #2 để lấy ý tưởng cho kiến trúc mới; mục 4–6 là kế hoạch chuyển đổi cụ thể trên nền pipeline chính (mục 1).

---

## 1. Đổi khung bài toán

| | Pipeline cũ (ALQAC) | Chatbot mới |
|---|---|---|
| Input | `{case_id, case_query}`, 1 lần/case | Câu hỏi tự do, nhiều lượt, có lịch sử hội thoại |
| Output | 1 trong 4 nhãn cố định + evidence list (JSON, để chấm điểm) | Câu trả lời văn xuôi, trích dẫn Điều/Khoản inline, streaming |
| Có "đáp án đúng" cố định? | Có (Outcome Accuracy) | Không — chỉ có "có căn cứ hay không" |
| Case Content API | Bắt buộc gọi để lấy case_evidence | Không tồn tại — không có "case" cụ thể |
| Ngân sách API (`E_i`) | Ràng buộc cứng, ảnh hưởng điểm | Không áp dụng |
| Vòng đời | Single-turn, stateless theo case_id | Multi-turn, có session/state |

---

## 2. Giữ nguyên / gỡ bỏ / thêm mới

### 2.1 Giữ nguyên gần như 100%
- `backend/ingestion/parser.py`, `chunker.py` (rule-based Chương>Mục>Điều>Khoản>Điểm + soft-split theo câu), `metadata.py` (lọc hiệu lực văn bản).
- `backend/indexing/*` (BM25 + Pinecone, đã fix cache `_get_index`).
- `backend/retrieval/hybrid_search.py`, `rerank.py`, `ner.py`, `querry_transform.py` — retrieval-purpose-agnostic, dùng được nguyên vẹn cho câu hỏi tự do. Càng quan trọng hơn vì câu hỏi chat thường ngắn/mơ hồ hơn `case_query`.
- Retrieval-evaluator loop trong `pipeline.collect_law_evidence()` (tái dùng điểm rerank làm ngưỡng, không cần LLM-judge riêng).
- Nguyên tắc grounding: `allowed_citation_keys()` + hallucination guard trong `generate.py` — đổi *nơi áp dụng*, không đổi *cơ chế*.

### 2.2 Gỡ bỏ
- `backend/case_api_client.py`, `collect_case_evidence()`, `_case_api_budget()` — không còn "case" để gọi API lấy bằng chứng.
- 4-nhãn schema + `_label_from_ratio()` / `USE_RATIO_DERIVED_LABEL` trong `generate.py`/`prompt_builder.py`.
- `backend/submission.py` + `_validate_submission()` (khớp `docs/submission_example.json`) — không còn nộp bài thi.
- `test/test_all_backend.py` (Outcome Accuracy / Micro Law F1 / ước lượng `E_i`) — không còn gold `verdict_label`.

### 2.3 Thêm mới
1. **Query contextualization** — condense (lịch sử hội thoại + câu hỏi mới) → câu hỏi độc lập trước khi vào `hybrid_search`. Tái dùng model 0.8B theo đúng phong cách `decompose_query` (không sinh nội dung luật giả định).
2. **Conversation memory** theo session_id, không theo case_id — rolling digest khi hội thoại dài (kế thừa vai trò của `case_digest.py`).
3. **Scope/intent guardrail** — kiểm tra câu hỏi có thuộc phạm vi pháp luật + gắn disclaimer "công cụ hỗ trợ tra cứu, không thay thế tư vấn pháp lý chính thức" (đã được đặt ra ở `docs/system_design_v0.md` §7.4 nhưng chưa từng cài đặt).
4. **Free-form answer generation** với citation inline, có nhánh từ chối lịch sự khi thiếu căn cứ (thay vì bị ép chọn 1/4 nhãn).
5. **Streaming** — đổi `generate_text()` từ blocking `model.generate()` sang `TextIteratorStreamer`.
6. **Multi-hop qua cross-reference** — kích hoạt lại `extract_cross_references()` trong `metadata.py` (hiện là dead code), dùng cho câu hỏi kiểu "Điều 12 có ngoại lệ nào không".
7. **Citation fast-path** — xem mục 3.1 (lấy từ nháp #1).
8. **Eval harness mới** — faithfulness / groundedness / retrieval recall@k, thay cho Outcome Accuracy.

---

## 3. Khai thác 2 repo nháp

### 3.1 Từ `AI/` (extractive QA, SQLite FTS5, không dùng vector DB)

| Thành phần | Ý tưởng lấy được | Áp dụng vào đâu |
|---|---|---|
| `AI/data_utils.py::extract_legal_reference()` | Regex parse "Điều X khoản Y điểm Z" bất kể thứ tự từ, chấp nhận cả không dấu (`dieu 30 khoan 2 diem a`) | **Citation fast-path**: nếu câu hỏi user chỉ hỏi đúng 1 điều khoản cụ thể (không cần suy luận), trả lời tức thì bằng exact-match, bỏ qua LLM hoàn toàn → 0% hallucination, latency ~0 |
| `AI/legal_spans.py::extract_structured_answer()` | Cắt chính xác đoạn Điều/Khoản/Điểm từ context đã có, dựa ranh giới cấu trúc (không phải semantic) | Kết hợp với chunk `parent`/`child` đã có trong `chunker.py` để trả nguyên văn khoản được hỏi, không qua LLM diễn giải lại → tránh lỗi paraphrase sai nghĩa |
| `AI/retrieval.py` — tokenizer `unicode61 remove_diacritics 2` trong FTS5 | Xử lý người dùng gõ không dấu | `bm25_index.py::tokenize()` hiện **không** chuẩn hoá bỏ dấu — nên thêm biến thể token không dấu làm kênh phụ trong `hybrid_search` |
| `AI/retrieval.py::MIN_RETRIEVAL_CONFIDENCE = 0.85` + từ chối thẳng khi dưới ngưỡng | Nguyên tắc "không đoán khi không chắc" | Áp cho `RETRIEVAL_EVALUATOR_SCORE_THRESHOLD` hiện có — nếu sau cả 2 vòng vẫn dưới ngưỡng, trả lời "không tìm thấy căn cứ" thay vì ép LLM trả lời |
| `AI/evaluate.py::NEGATIVE_SAFETY_QUERIES` (abstention safety probes) | Bộ câu hỏi ngoài phạm vi để test từ chối đúng cách | Đưa vào eval harness mới (mục 2.3.8) |

**Cảnh báo khi lấy từ `AI/`:** kiến trúc này dùng span-extraction trên context đã biết trước (không phải retrieval-rồi-generate thực sự), và model BiGRU tự train chỉ ~60% exact match ở chế độ full-context. Chỉ nên lấy **ý tưởng xử lý câu hỏi/citation**, không nên tái sử dụng model reader này.

### 3.2 Từ các script rời trong `backend/` (ChromaDB + bge-m3 + Ollama)

| Thành phần | Ý tưởng lấy được | Áp dụng vào đâu |
|---|---|---|
| `backend/retriever_api.py` | Hợp đồng I/O đơn giản: đọc JSON qua stdin, trả JSON qua stdout — đã được `server.js` gọi thành công | Mẫu tham khảo cho **giao diện chat backend ↔ Node**, dù nên thay bằng HTTP streaming thực sự thay vì spawn subprocess mỗi request |
| `backend/embed.py` — schema metadata Chroma (`law_name, chapter, article, clause, point, title, level`) | Schema phẳng, dễ hiển thị trích dẫn trên UI | Đối chiếu với `LawChunk`/`_chunk_metadata()` trong `backend/indexing/vector_store.py` — field `breadcrumb` hiện có đã gộp thông tin này thành 1 string; cân nhắc **tách lại thành field riêng** trong metadata Pinecone để UI hiển thị dễ hơn (không cần parse breadcrumb) |
| `backend/LLM.py` (gọi qua Ollama, model `qwen2.5:7b`) | Serving qua Ollama đơn giản hơn HF transformers in-process | Cân nhắc làm **serving backend thay thế** cho dev/local (không cần lo `device_map`/`dtype` như trong `models.py`), giữ HF transformers cho production/GPU thật |
| `backend/prompt.py` | Prompt tối giản: "chỉ trả lời dựa trên Context, nếu không đủ thì nói 'Không tìm thấy thông tin'" | Đúng tinh thần cần cho chatbot, nhưng **thiếu**: không có citation format, không có structure Điều/Khoản, không chống hallucination bằng whitelist như `generate.py` hiện có → **không thay thế** `prompt_builder.py`, chỉ lấy câu "từ chối lịch sự" làm baseline |
| `backend/api.py` | Bộ khung route tối giản `/chat` | Chỉ là stub tham khảo, thiếu retry/error handling/rate limit — **không nên port trực tiếp** |

**Cảnh báo khi lấy từ nháp #2:** đây là prototype chưa hoàn thiện — không có chunking rule-based (dựa vào `legal_nodes.json` được build sẵn ở đâu đó ngoài repo), không có reranking, không có hallucination guard, không có conversation memory. Giá trị chính là ở **pattern kiến trúc & schema**, không phải logic xử lý.

### 3.3 Kết luận khai thác
Không nên "hợp nhất" 2 nháp vào backend chính. Cách đúng: giữ `backend/` (package đã đánh giá kỹ, có test) làm nền, **cấy 4 ý tưởng cụ thể** vào:
1. Citation fast-path (từ `AI/`) → skill `citation-fast-path`.
2. Accent-insensitive tokenizer bổ sung cho BM25 (từ `AI/`).
3. Tách metadata trích dẫn thành field riêng thay vì gộp trong `breadcrumb` (từ nháp #2).
4. Ngưỡng từ chối cứng khi retrieval yếu (từ `AI/`).

---

## 4. Lộ trình triển khai (theo thứ tự ROI)

| # | Việc | File chính bị ảnh hưởng | Effort |
|---|---|---|---|
| A1 | Gỡ nhánh case-outcome khỏi `pipeline.py`, xoá `case_api_client.py`/`submission.py` khỏi luồng chính | `backend/pipeline.py` | Thấp |
| A2 | Viết `chat_pipeline.py`: condense-question → `collect_law_evidence` (tái dùng) → `generate_chat_answer()` | mới: `backend/chat_pipeline.py` | Trung bình |
| A3 | Citation fast-path (mục 3.1) — bypass LLM khi câu hỏi là citation thuần | mới: `backend/retrieval/citation_fastpath.py` | Thấp |
| B1 | Session/history store tối giản (in-memory trước) | mới: `backend/session_store.py` | Thấp |
| B2 | Prompt tự do + citation inline + nhánh từ chối | `backend/generation/prompt_builder.py`, `generate.py` | Trung bình |
| B3 | Streaming trong `generate_text()` | `backend/models.py` | Trung bình |
| C1 | Scope/intent guardrail + disclaimer | mới: `backend/guardrail.py` | Thấp |
| C2 | Multi-hop qua cross-reference (kích hoạt `extract_cross_references`) | `backend/ingestion/metadata.py`, `pipeline.py` | Trung bình |
| C3 | Eval harness mới (faithfulness/groundedness/recall@k) | mới: `test/test_chatbot.py` | Trung bình–Cao |
| D1 | Accent-insensitive tokenizer bổ sung cho BM25 | `backend/indexing/bm25_index.py` | Thấp |
| D2 | Tách field trích dẫn khỏi `breadcrumb` trong metadata Pinecone | `backend/indexing/vector_store.py`, `models.py` | Thấp |

Nhóm A nên làm trước (dọn nền + có luồng chat chạy được tối thiểu), nhóm B làm chatbot dùng được thật, nhóm C/D là tinh chỉnh chất lượng và có thể làm song song.

---

## 5. Skill đi kèm

Các skill dưới đây mô tả chi tiết từng mảnh việc ở mục 4, dùng cho coding agent (Claude Code) khi thực thi từng bước:

- `skills/citation-fast-path/SKILL.md` — A3, D1
- `skills/conversational-retrieval/SKILL.md` — A2, B1, C2
- `skills/grounded-chat-generation/SKILL.md` — B2, B3
- `skills/legacy-prototype-salvage/SKILL.md` — tham chiếu chi tiết mục 3, dùng khi cần tra lại "cái gì lấy được từ đâu"
- `skills/eval-harness-chatbot/SKILL.md` — C3

## 6. Việc CHƯA làm / rủi ro còn treo
- Chưa xác nhận corpus mở rộng ngoài phạm vi dân sự (lao động, hành chính...) — cần input JSON mới, code `parser.py`/`chunker.py` đã tổng quát nên không cần sửa.
- Chưa có quyết định lưu trữ session lâu dài (DB nào) — mục B1 tạm dùng in-memory.
- Việc bật streaming (B3) cần kiểm tra `TextIteratorStreamer` tương thích với cách `apply_chat_template` đang xử lý `BatchEncoding`/dict trong `models.py` hiện tại — cần test thực tế trên GPU thật trước khi merge.
