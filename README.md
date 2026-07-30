# **LegalRAG Chatbot**

Chatbot RAG tra cứu pháp luật dân sự Việt Nam, đa lượt, trích dẫn Điều/Khoản
inline, có nhánh từ chối khi thiếu căn cứ. Chuyển đổi từ pipeline dự đoán
kết quả vụ án ALQAC2026 theo `CHATBOT_MIGRATION_PLAN.md` — xem file đó cho
lý do/khảo sát đầy đủ trước khi đổi bất kỳ phần nào dưới đây.

> **Trạng thái migration:** xem `PROGRESS_NOTES.md` cho bảng trạng thái chi
> tiết từng mục (Nhóm A/B/C/D) — bao gồm cả những việc CHƯA làm và các
> quyết định/đánh đổi đã chọn khi mục đó không có hướng dẫn rõ trong plan.

## 1. Kiến trúc một lượt chat (`backend/chat_pipeline.py`)

```
ChatRequest(session_id, question)
            │
            ▼
   0. guardrail.is_trivially_out_of_scope()?  ──yes──▶ trả OUT_OF_SCOPE_MESSAGE
            │ no
            ▼
   1. condense_question(history hoặc conversation_digest, question)
      -> câu hỏi độc lập, không phụ thuộc ngữ cảnh trước
            │
            ▼
   2. citation_fastpath.try_lookup(contextualized_question)
      -> khớp "Điều X [khoản Y] nói gì?" thuần lookup?  ──yes──▶ trả nguyên văn, 0 lệnh gọi LLM
            │ no
            ▼
   3. pipeline.collect_law_evidence(contextualized_question)
      hybrid_search (BM25 + BM25-folded + Pinecone + query decomposition,
      weighted RRF) -> rerank (full-article context) -> retrieval-evaluator
      re-round (decomposition + C2 cross-reference hop) nếu điểm thấp
            │
            ▼
   4. best score < RETRIEVAL_EVALUATOR_SCORE_THRESHOLD?  ──yes──▶ trả ABSTENTION_MESSAGE
            │ no
            ▼
   5. build_conversation_digest(history) + generate_chat_answer()
      -> văn xuôi tiếng Việt, trích dẫn "[Điều X, law_id]" inline
      -> verify_and_strip_hallucinated_citations() lọc citation ngoài whitelist
            │
            ▼
   6. guardrail.apply_disclaimer_if_needed() + session_store ghi lại lượt
            │
            ▼
        ChatAnswer(answer, citations[], is_abstention, ...)
```

Bản streaming (`handle_chat_turn_stream`) chạy chung bước 0-4
(`_prepare_turn`), chỉ khác bước 5: generate token-by-token nhưng vẫn xác
minh citation TRƯỚC khi phát ra ("buffer rồi verify rồi replay theo từ" —
xem docstring `generate.py` cho lý do không stream token thô trực tiếp).

## 2. Cấu trúc thư mục

```
backend/
├── config.py                    # tham số qua .env — không còn Case API budget/4-nhãn
├── models.py                     # LawChunk/ChatAnswer/... + generate_text() + generate_text_stream() (B3)
├── chat_pipeline.py                # A2 — orchestration 1 lượt chat (entry point chính)
├── pipeline.py                      # collect_law_evidence() — hybrid search -> rerank -> evaluator + C2 cross-ref
├── session_store.py                  # B1 — session/history in-memory
├── guardrail.py                       # C1 — out-of-scope short-circuit + disclaimer backstop
├── generation/
│   ├── conversation_digest.py           # kế thừa vai trò case_digest.py — tóm tắt lịch sử hội thoại
│   ├── prompt_builder.py                 # CHAT_SYSTEM_PROMPT + build_chat_prompt() (B2)
│   ├── generate.py                        # generate_chat_answer() + hallucination guard trên văn xuôi
│   └── compress.py                         # KHÔNG dùng (giữ nguyên từ ALQAC, xem docstring)
├── retrieval/
│   ├── ner.py                               # mask tên riêng — không đổi
│   ├── querry_transform.py                   # rewrite_query/decompose_query (không đổi) + condense_question() (mới)
│   ├── hybrid_search.py                       # BM25 + BM25-folded (D1) + Pinecone, weighted RRF
│   ├── rerank.py                               # cross-encoder — không đổi
│   └── citation_fastpath.py                    # A3 — bypass LLM cho câu hỏi tra cứu thuần
├── indexing/
│   ├── embed.py                                 # không đổi
│   ├── bm25_index.py                             # D1 — thêm kênh BM25 bỏ dấu phụ trợ
│   └── vector_store.py                            # D2 — article_num tách field riêng khỏi breadcrumb
└── ingestion/
    ├── parser.py                                  # không đổi
    ├── chunker.py                                  # thêm article_num/khoan_no/diem_no (D2, A3)
    └── metadata.py                                  # extract_cross_references() nay ĐƯỢC gọi (C2)

scripts/
└── build_index.py    # + persist fastpath index (article_lookup/khoan_lookup/chunk_by_id)

test/
└── test_chatbot.py   # C3 — groundedness / abstention / fastpath / recall@k (KHÔNG dùng gold verdict_label nữa)
```

**Đã gỡ bỏ hoàn toàn** (không còn trong repo, theo `CHATBOT_MIGRATION_PLAN.md §2.2`):
`backend/case_api_client.py`, `backend/submission.py`, `test/test_all_backend.py`,
schema 4-nhãn (`Prediction`, `SubmissionRecord`, `LawEvidenceItem`,
`accepted_ratio_estimate`), `docs/case_content_api_doc.md`,
`docs/evaluation.md`, `docs/submission_example.json`, `docs/test_design.md`.

## 3. Cài đặt & chạy

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp backend/.env.example .env   # điền PINECONE_API_KEY tối thiểu
```

```bash
# Build index (BM25 + BM25-folded + Pinecone + parent lookup + fastpath index)
python -m scripts.build_index --corpus data/corpus_law_pub.json --rebuild-pinecone
```

```python
# Dùng thử 1 lượt chat trong Python
from backend.chat_pipeline import handle_chat_turn
answer = handle_chat_turn(session_id="demo", question="Điều 12 nói gì?")
print(answer.answer)
```

```bash
# Eval harness (groundedness / abstention / fastpath / recall@k)
python -m test.test_chatbot
python -m test.test_chatbot --skip-generation   # smoke test không cần load model sinh
```

## 4. Việc CHƯA làm / rủi ro còn treo

Xem `PROGRESS_NOTES.md` mục "Chưa làm / cần xác nhận" — tóm tắt nhanh:
- Chưa có HTTP server thực sự (FastAPI/Flask) bọc `chat_pipeline.py` —
  hiện chỉ có API Python nội bộ, chưa có route `/chat`.
- Chưa test end-to-end trên corpus/GPU thật (mọi thứ mới qua compile-check
  + đọc lại logic, giống tình trạng "chưa validate runtime" đã ghi nhận ở
  các phiên ALQAC trước).
- `test/test_chatbot.py`'s retrieval-recall@k và fastpath-positive-case
  cần một `data/chatbot_eval_set.json` tự soạn theo corpus thật — chưa có
  sẵn vì không biết trước nội dung corpus cụ thể.
- `docs/system_design_v0.md` (thiết kế gốc) chưa được viết lại/hợp nhất
  vào tài liệu chatbot — vẫn còn mô tả HyDE (đã bỏ từ thời ALQAC) và
  không nhắc gì tới multi-turn.

## 5. Giấy phép

MIT License.
