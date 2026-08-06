"""
HTTP server wrapper cho chat_pipeline.py (coding_plan.md A2).

CHATBOT_MIGRATION_PLAN.md/PROGRESS_NOTES.md đã tự xác nhận đây là mảnh còn
thiếu để chatbot "dùng được thật": chat_pipeline.py trước đây chỉ là API
Python nội bộ (handle_chat_turn/handle_chat_turn_stream), chưa có route
/chat nào expose ra ngoài.

Chạy: uvicorn backend.server:app --host 0.0.0.0 --port 8000

LƯU Ý AN TOÀN (đọc trước khi deploy public-facing):
Bản này KHÔNG có auth/session-ownership — bất kỳ client nào cũng có thể gửi
`session_id` tuỳ ý và đọc/ghi vào session của người khác nếu đoán được ID
(session_store.py hiện là in-memory, không gắn với người dùng đã xác thực).
Chấp nhận được cho bản đầu tiên (in-memory, single-tenant nội bộ / demo),
nhưng PHẢI thêm auth (vd. xác thực JWT/API key + validate session_id thuộc
đúng người gọi) trước khi public-facing thật — xem
CHATBOT_MIGRATION_PLAN.md §6 (lưu trữ session lâu dài) nếu cần đổi sang
backend có gắn user_id.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.chat_pipeline import handle_chat_turn, handle_chat_turn_stream

log = logging.getLogger(__name__)

app = FastAPI(title="LegalRAG Chatbot API")


class ChatIn(BaseModel):
    session_id: str
    question: str


@app.post("/chat")
def chat(payload: ChatIn):
    try:
        answer = handle_chat_turn(payload.session_id, payload.question)
    except Exception as e:
        log.exception("handle_chat_turn failed for session=%s", payload.session_id)
        raise HTTPException(status_code=500, detail=str(e))
    return answer.model_dump()


@app.post("/chat/stream")
def chat_stream(payload: ChatIn):
    def gen():
        try:
            for chunk in handle_chat_turn_stream(payload.session_id, payload.question):
                yield chunk
        except Exception as e:
            log.exception("handle_chat_turn_stream failed for session=%s", payload.session_id)
            # Đã bắt đầu stream nên không thể raise HTTPException giữa
            # chừng — phát 1 chunk lỗi rõ ràng thay vì im lặng cắt kết nối.
            yield f"\n\n[Lỗi hệ thống: {e}]"

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
