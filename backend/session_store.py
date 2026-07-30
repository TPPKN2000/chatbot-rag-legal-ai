"""
Session/history store (CHATBOT_MIGRATION_PLAN.md §B1 / conversational-
retrieval skill, Bước 2).

Core difference from ALQAC: state is no longer keyed by `case_id` (used
once, then discarded) but by `session_id` (lives for the duration of a chat
session, across many turns).

Deliberately minimal for the first cut, as the migration plan calls for
("Tối giản (giai đoạn đầu, in-memory)"): a process-local, in-memory store.
`CHATBOT_MIGRATION_PLAN.md §6` explicitly flags long-term session storage
(which DB) as an open decision — this module is written so swapping the
backing store later (Redis, a DB table) only requires reimplementing
`SessionStore`'s three methods, not touching any caller.
"""
from __future__ import annotations

import threading
from collections import defaultdict, deque

from backend import config
from backend.models import ChatTurn


class SessionStore:
    def __init__(self, max_turns: int = config.SESSION_MAX_TURNS):
        self._max_turns = max_turns
        self._sessions: dict[str, deque] = defaultdict(lambda: deque(maxlen=self._max_turns))
        self._lock = threading.Lock()

    def append(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            self._sessions[session_id].append(ChatTurn(role=role, content=content))

    def history(self, session_id: str) -> list[ChatTurn]:
        with self._lock:
            return list(self._sessions[session_id])

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def turn_count(self, session_id: str) -> int:
        with self._lock:
            return len(self._sessions[session_id])


# Process-wide singleton — mirrors the case_api_client singleton pattern
# used elsewhere in backend/ for shared, process-lifetime state.
store = SessionStore()
