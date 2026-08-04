---
name: citation-fast-path
description: Dùng khi cần trả lời tức thì, không hallucination, cho các câu hỏi chỉ hỏi đúng nội dung một Điều/Khoản/Điểm cụ thể (vd. "Điều 12 khoản 2 nói gì?"), bằng cách bypass LLM sinh và trả nguyên văn từ corpus/index đã có. Áp dụng cho backend/retrieval/ trong dự án LegalRAG chatbot. Nguồn ý tưởng: AI/data_utils.py::extract_legal_reference, AI/legal_spans.py, AI/retrieval.py.
---

# Citation Fast-Path

## Vì sao cần
Khi câu hỏi chỉ là tra cứu nguyên văn một điều khoản đã biết rõ (article/clause/point), việc đưa qua toàn bộ pipeline hybrid_search → rerank → LLM sinh là:
1. Chậm không cần thiết (nhiều lệnh gọi BM25/vector/LLM).
2. Có rủi ro paraphrase sai — LLM có thể diễn giải lại thay vì trích nguyên văn, đổi nghĩa "trừ trường hợp..."
3. Không cần thiết vì nếu (law_id, article_num) xác định được, câu trả lời chính là một lookup, không phải một suy luận.

`AI/retrieval.py::_retrieve_exact_reference()` đã chứng minh cách này đạt ~100% exact match trên tập test khi câu hỏi đúng dạng citation.

## Khi nào kích hoạt fast-path
Chỉ kích hoạt khi **tất cả** đúng:
- Câu hỏi (sau khi contextualize với lịch sử hội thoại) parse được `article` cụ thể qua regex kiểu `extract_legal_reference()`.
- Câu hỏi là "generic reference question" — tức người dùng hỏi *toàn bộ nội dung* của đơn vị đó, không hỏi thêm điều kiện/tình huống áp dụng (dùng lại logic `is_generic_reference_question()` trong `AI/data_utils.py`).
- Xác định được duy nhất 1 chunk khớp `(law_id, article_num[, khoan, diem])` trong index — nếu mơ hồ (nhiều luật có cùng số Điều), KHÔNG dùng fast-path, rơi về pipeline retrieval đầy đủ.

Nếu không đủ điều kiện trên, không được cưỡng ép — chuyển ngay sang `hybrid_search()` bình thường.

## Cách triển khai trong `backend/`

1. Tạo `backend/retrieval/citation_fastpath.py`:
   - Copy & thích nghi `extract_legal_reference()` và `is_generic_reference_question()` từ `AI/data_utils.py` (regex đã xử lý cả không dấu và các trật tự từ khác nhau — không cần viết lại).
   - Viết `lookup_exact_chunk(law_id_hint, article_num, khoan=None, diem=None) -> LawChunk | None`, tra trực tiếp trong parent/child lookup đã có (`backend/pipeline.py::_get_parent_lookup()` hoặc thêm 1 index phụ `(article_num) -> [chunk_id,...]` build cùng lúc với `build_parent_lookup()` trong `chunker.py`).
   - Nếu `law_id_hint` không xác định (người dùng không nói rõ luật nào) và tra ra >1 kết quả → coi là "ambiguous", KHÔNG fast-path.

2. Trong `chat_pipeline.py`, gọi fast-path **trước** `collect_law_evidence()`:
   ```python
   fastpath_hit = citation_fastpath.try_lookup(contextualized_question)
   if fastpath_hit is not None:
       return build_fastpath_answer(fastpath_hit)  # trả nguyên văn + citation, không gọi LLM
   # ngược lại: rơi vào pipeline retrieval + generate bình thường
   ```

3. `build_fastpath_answer()` KHÔNG được paraphrase — dùng chính field `text` của chunk (giữ nguyên tinh thần "không cắt/nén văn bản luật" đã có trong `generation/compress.py`). Chỉ thêm 1 câu dẫn ngắn kiểu "Theo {breadcrumb}:".

## Bổ sung: accent-insensitive matching cho BM25 (D1 trong migration plan)
`AI/retrieval.py` dùng FTS5 với `tokenize='unicode61 remove_diacritics 2'` — xử lý được câu hỏi gõ không dấu ("dieu 30 khoan 2"). `backend/indexing/bm25_index.py::tokenize()` hiện KHÔNG có bước này.

Đề xuất: thêm một biến thể token bỏ dấu làm **kênh RRF phụ**, không thay thế kênh có dấu (tránh giảm precision cho văn bản có nghĩa khác nhau chỉ vì dấu câu — vd "hòa giải" vs "hoa giải"):
```python
def _fold_accents(text: str) -> str:
    import unicodedata
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")
```
Dùng hàm này CHỈ cho việc match citation reference (article/khoan number), KHÔNG áp dụng cho semantic BM25 matching toàn văn, để không làm loãng độ chính xác vốn có.

## Ngưỡng từ chối khi retrieval yếu
`AI/retrieval.py::MIN_RETRIEVAL_CONFIDENCE = 0.85` — nguyên tắc: nếu điểm confidence dưới ngưỡng, trả `None` (không đoán). Áp dụng tương tự vào `RETRIEVAL_EVALUATOR_SCORE_THRESHOLD` đã có trong `backend/config.py`:
- Nếu sau cả vòng retrieval-evaluator (2 vòng) vẫn dưới ngưỡng → trả lời "Không tìm thấy căn cứ pháp lý phù hợp trong corpus hiện có" thay vì để LLM tự bịa câu trả lời từ context yếu.
- Đây LÀ một thay đổi hành vi so với pipeline ALQAC cũ (vốn bắt buộc phải luôn ra 1 trong 4 nhãn dù confidence thấp) — trong chatbot, từ chối trả lời là lựa chọn hợp lệ và nên được khuyến khích.

## Kiểm thử
- Test case dương: "Điều 12 khoản 2 nói gì?", "dieu 5 quy dinh gi" (không dấu) → phải trả đúng nguyên văn, 0 lệnh gọi LLM.
- Test case cần rơi về pipeline đầy đủ: "Điều 12 áp dụng thế nào cho trường hợp ly hôn đơn phương?" (có thêm điều kiện áp dụng cụ thể) → `is_generic_reference_question()` phải trả `False`.
- Test case ambiguous: hỏi "Điều 5" khi có ≥2 luật cùng chứa Điều 5 mà không nói rõ luật nào → phải KHÔNG fast-path.
