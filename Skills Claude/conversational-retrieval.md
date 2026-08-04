---
name: conversational-retrieval
description: Dùng khi chuyển pipeline retrieval single-turn (case_query cố định) của LegalRAG sang chế độ hỏi-đáp nhiều lượt — cần condense câu hỏi theo lịch sử hội thoại, quản lý session, và (mở rộng) đi theo tham chiếu chéo giữa các điều luật. Áp dụng cho backend/chat_pipeline.py (mới) và backend/pipeline.py trong dự án LegalRAG chatbot.
---

# Conversational Retrieval

## Vấn đề cần giải quyết
`backend/pipeline.py::collect_law_evidence(query_text)` hiện nhận thẳng `case_query` — một đoạn mô tả tình huống đầy đủ, không phụ thuộc ngữ cảnh trước đó (vì ALQAC là single-turn). Trong chatbot, câu hỏi tiếp theo thường là tỉnh lược ngữ cảnh:

```
User: Hợp đồng vay tiền không có giấy tờ thì có được công nhận không?
Bot: [trả lời, trích Điều X]
User: Vậy nếu bên vay không trả thì sao?   <-- "vậy", "bên vay" phụ thuộc câu trước
```

Nếu đưa thẳng câu 2 vào `hybrid_search`, retrieval sẽ mất ngữ cảnh "hợp đồng vay tiền không giấy tờ".

## Bước 1 — Query contextualization (condense-question)
Thêm một lệnh gọi LLM nhỏ TRƯỚC `collect_law_evidence()`, theo đúng tinh thần `decompose_query()` đã có trong `backend/retrieval/querry_transform.py` (không sinh nội dung luật giả định, chỉ diễn giải lại ý người dùng):

```python
_CONDENSE_SYSTEM_PROMPT = (
    "Bạn nhận được lịch sử hội thoại và câu hỏi mới nhất của người dùng. "
    "Hãy viết lại câu hỏi mới nhất thành MỘT câu hỏi độc lập, đầy đủ ngữ nghĩa, "
    "không cần đọc lịch sử vẫn hiểu được. KHÔNG trả lời câu hỏi, KHÔNG thêm "
    "thông tin luật pháp nào, chỉ diễn đạt lại đúng ý người dùng."
)

def condense_question(history: list[dict], latest_question: str) -> str:
    if not history:
        return latest_question
    try:
        return generate_text(
            system_prompt=_CONDENSE_SYSTEM_PROMPT,
            user_prompt=_format_history(history) + f"\n\nCâu hỏi mới: {latest_question}",
            max_new_tokens=150,
            temperature=0.2,
        ).strip() or latest_question
    except Exception:
        return latest_question  # fallback an toàn: dùng câu hỏi thô, không chặn cả pipeline
```

Điểm quan trọng: fallback về câu hỏi gốc khi lỗi — giống triết lý fallback đã dùng nhất quán trong `querry_transform.py` và `case_digest.py` (một bước phụ lỗi không được sập cả pipeline).

## Bước 2 — Session/history store
Khác biệt cốt lõi với ALQAC: state không còn theo `case_id` (1 lần dùng) mà theo `session_id` (sống suốt phiên chat).

Tối giản (giai đoạn đầu, in-memory):
```python
# backend/session_store.py
from collections import defaultdict, deque

class SessionStore:
    def __init__(self, max_turns: int = 20):
        self._sessions: dict[str, deque] = defaultdict(lambda: deque(maxlen=max_turns))

    def append(self, session_id: str, role: str, content: str) -> None:
        self._sessions[session_id].append({"role": role, "content": content})

    def history(self, session_id: str) -> list[dict]:
        return list(self._sessions[session_id])

store = SessionStore()
```

Khi hội thoại dài (vd. > 10 lượt), thay vì đưa toàn bộ history thô vào `condense_question`, dùng lại đúng pattern của `backend/generation/case_digest.py` (đổi vai trò: tóm tắt hồ sơ vụ án → tóm tắt lịch sử hội thoại):
```python
def build_conversation_digest(history: list[dict]) -> str:
    # Cùng nguyên tắc build_case_digest(): không suy đoán thêm, chỉ nén.
    ...
```
Không cần viết mới từ đầu — copy cấu trúc try/except + fallback hard-truncate của `case_digest.py`.

## Bước 3 (mở rộng) — Multi-hop qua cross-reference
`backend/ingestion/metadata.py::extract_cross_references()` đã tồn tại nhưng là **dead code** — không có nơi nào trong `pipeline.py`/`hybrid_search.py` gọi nó.

Trường hợp cần: câu hỏi kiểu "Điều 12 có ngoại lệ nào không?" — câu trả lời đúng có thể nằm ở một Điều KHÁC được tham chiếu bên trong Điều 12, không phải trong chính văn bản Điều 12.

Cách kích hoạt lại, tái dùng retrieval-evaluator loop đã có trong `pipeline.collect_law_evidence()`:
1. Sau khi có top-k chunk từ rerank vòng 1, chạy `extract_cross_references(chunk.text, chunk.law_id)` trên mỗi chunk.
2. Với mỗi `{law_id, aid}` tham chiếu được tìm thấy mà CHƯA có trong candidate set, tra thẳng chunk tương ứng (không cần retrieval lại — đã biết chính xác `(law_id, aid)`) và gộp vào candidate set trước khi rerank lần cuối.
3. Giới hạn độ sâu = 1 hop (không đệ quy vô hạn theo chuỗi tham chiếu) để tránh nổ token và trôi dạt chủ đề.

```python
# trong collect_law_evidence(), sau bước rerank đầu tiên
cross_refs = []
for c in reranked[:3]:  # chỉ theo tham chiếu của top-3 chunk tin cậy nhất
    cross_refs.extend(extract_cross_references(c.text, default_law_id=c.law_id))
new_candidates = [resolve_chunk(ref) for ref in cross_refs if not_already_seen(ref)]
if new_candidates:
    reranked = rerank(query_text, candidates + new_candidates, top_k=config.FINAL_LAW_TOP_K)
```

## Gotcha
- Đừng nhầm "condense-question" (bước 1) với "case_digest" (bước 2) — bước 1 chạy TRƯỚC retrieval để sửa câu hỏi; bước 2 chạy khi lịch sử quá dài, độc lập với retrieval.
- `condense_question()` phải luôn trả về string non-empty; nếu LLM trả rỗng, dùng lại `latest_question` gốc — không được để retrieval nhận chuỗi rỗng.
- Multi-hop cross-reference chỉ nên bật khi retrieval-evaluator đã xác định điểm rerank vòng 1 dưới ngưỡng (giống cách vòng decomposition thứ 2 hiện đang được gate) — không chạy mặc định mọi câu hỏi để tránh tăng latency/token vô ích.
