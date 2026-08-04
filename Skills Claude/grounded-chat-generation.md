---
name: grounded-chat-generation
description: Dùng khi thay schema sinh JSON 4-nhãn (A_WIN/PARTIAL_A_WIN/PARTIAL_B_WIN/B_WIN) của pipeline ALQAC bằng câu trả lời chatbot dạng văn xuôi, có trích dẫn Điều/Khoản inline, streaming, và nhánh từ chối lịch sự khi thiếu căn cứ. Áp dụng cho backend/generation/prompt_builder.py và generate.py trong dự án LegalRAG chatbot.
---

# Grounded Chat Generation

## Điểm khác biệt cốt lõi so với `generate.py`/`prompt_builder.py` hiện tại
Pipeline ALQAC ép model LUÔN phải chọn 1 trong 4 nhãn dù confidence thấp (rule #3 trong `SYSTEM_PROMPT` cũ chỉ yêu cầu hạ confidence, không cho phép từ chối). Chatbot phải làm NGƯỢC LẠI: từ chối trả lời khi thiếu căn cứ là hành vi ĐÚNG, không phải fallback lỗi.

## Prompt mới (thay `prompt_builder.SYSTEM_PROMPT`)
Giữ nguyên các quy tắc grounding đã có (chỉ trích dẫn trong danh sách được cấp, không bịa điều khoản), bỏ toàn bộ phần liên quan `accepted_ratio_estimate`/4-nhãn:

```python
CHAT_SYSTEM_PROMPT = """Bạn là trợ lý tra cứu pháp luật Việt Nam.

QUY TẮC BẮT BUỘC:
1. Chỉ trả lời dựa trên các điều luật và tóm tắt hội thoại được cung cấp bên dưới.
   KHÔNG bịa thêm điều khoản, số hiệu văn bản không có trong danh sách.
2. Mọi luận điểm PHẢI trích dẫn (law_id, aid) cụ thể, viết dạng "[Điều {aid}, {law_id}]"
   ngay sau luận điểm đó trong câu trả lời.
3. Nếu ngữ cảnh được cung cấp KHÔNG đủ để trả lời có căn cứ, hãy nói rõ điều đó và
   đề nghị người dùng cung cấp thêm chi tiết — TUYỆT ĐỐI KHÔNG đoán khi thiếu căn cứ.
4. Đây là công cụ hỗ trợ tra cứu, KHÔNG thay thế tư vấn pháp lý chính thức từ luật sư.
   Với câu hỏi mang tính tư vấn cá nhân hoá (vd. "tôi nên làm gì trong tình huống X"),
   nhắc người dùng cân nhắc gặp luật sư cho quyết định cuối cùng.
5. Trả lời bằng văn xuôi tự nhiên, tiếng Việt, không dùng JSON, không markdown code fence.
"""
```

## `build_prediction_prompt` → `build_chat_prompt`
```python
def build_chat_prompt(
    contextualized_question: str,
    law_chunks: list[RetrievedChunk],
    conversation_digest: str,
) -> tuple[str, str]:
    law_section = _format_law_section(law_chunks)  # TÁI DÙNG nguyên hàm cũ — verbatim, không nén
    user_prompt = f"""LỊCH SỬ HỘI THOẠI (tóm tắt): {conversation_digest.strip()}

CÂU HỎI HIỆN TẠI: {contextualized_question.strip()}

CÁC ĐIỀU LUẬT LIÊN QUAN ĐÃ TRUY HỒI (nguyên văn):
{law_section}

Hãy trả lời câu hỏi, trích dẫn đúng các điều luật trên."""
    return CHAT_SYSTEM_PROMPT, user_prompt
```
`_format_law_section()` và `allowed_citation_keys()` trong `prompt_builder.py` **giữ nguyên 100%** — logic verbatim-law-text và whitelist citation là retrieval-purpose-agnostic.

## Hallucination guard — đổi input, giữ cơ chế
`generate.py::predict_outcome()` hiện parse JSON rồi lọc `law_citations` theo `allowed_citation_keys`. Với câu trả lời tự do, không còn JSON field `law_citations` — cần đổi sang **hậu kiểm bằng regex trên câu trả lời**:

```python
_CITATION_RE = re.compile(r"\[Điều\s+(\d+)\s*,\s*([^\]]+)\]")

def verify_and_strip_hallucinated_citations(
    answer_text: str,
    allowed: set[tuple[str, int]],
) -> tuple[str, int]:
    """Trả về (answer đã lọc, số citation bị gỡ). Citation không nằm trong `allowed`
    bị xoá khỏi text (không phải xoá cả câu — chỉ xoá phần ngoặc vuông trích dẫn),
    và câu văn liền kề được gắn cờ '[cần xác minh thêm]' thay vì im lặng bỏ qua,
    vì xoá lặng lẽ có thể khiến câu văn trông như vẫn có căn cứ."""
    dropped = 0
    def _check(match):
        nonlocal dropped
        aid, law_id = int(match.group(1)), match.group(2).strip()
        if (law_id, aid) in allowed:
            return match.group(0)
        dropped += 1
        return "[cần xác minh thêm]"
    return _CITATION_RE.sub(_check, answer_text), dropped
```
Nguyên tắc `_UNGROUNDED_CONFIDENCE_CEILING` cũ không còn áp dụng trực tiếp (không có field `confidence` số) nhưng tinh thần giữ lại: nếu `dropped > 0` và không còn citation hợp lệ nào trong câu trả lời, cân nhắc thêm 1 câu disclaimer tự động ở cuối: "Lưu ý: một số nội dung trên chưa được xác minh với corpus hiện có."

## Streaming
`backend/models.py::generate_text()` hiện dùng `model.generate()` blocking, trả cả câu trả lời cùng lúc. Đổi sang streaming:

```python
from transformers import TextIteratorStreamer
from threading import Thread

def generate_text_stream(system_prompt: str, user_prompt: str, **gen_kwargs):
    tokenizer, model, device = _get_generation_model()
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
    encoded = tokenizer.apply_chat_template(
        messages, enable_thinking=config.GENERATION_ENABLE_THINKING,
        add_generation_prompt=True, return_tensors="pt",
    ).to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    thread = Thread(target=model.generate, kwargs=dict(
        input_ids=encoded, streamer=streamer,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        **gen_kwargs,
    ))
    thread.start()
    for token_text in streamer:
        yield token_text
    thread.join()
```

**Gotcha quan trọng:** `models.py::generate_text()` hiện tại có logic xử lý `apply_chat_template` trả về cả `torch.Tensor` LẪN `BatchEncoding`/dict tuỳ version transformers (xem comment trong code hiện tại). Khi chuyển sang streaming, PHẢI giữ lại đúng nhánh normalize input này trước khi truyền vào `model.generate(..., streamer=...)`, nếu không sẽ tái diễn đúng lỗi `BatchEncoding has no .shape` đã từng được fix.

Streaming KHÔNG áp dụng cho citation fast-path (`skills/citation-fast-path`) — đường đó trả lời tức thì, không qua LLM nên không cần/không thể stream token.

## Nhánh từ chối (khi retrieval yếu)
Không đưa quyết định từ chối cho LLM tự quyết hoàn toàn (rule #3 chỉ là hướng dẫn mềm, model nhỏ có thể không tuân thủ nhất quán). Nên có một **rule cứng ở tầng code**, trước khi gọi LLM: nếu điểm rerank cao nhất < `RETRIEVAL_EVALUATOR_SCORE_THRESHOLD` sau cả 2 vòng, bỏ qua bước generate hoàn toàn và trả một câu từ chối cố định soạn sẵn (không phải do LLM sinh) — đảm bảo 100% nhất quán, không phụ thuộc việc model có tuân thủ rule #3 hay không.
