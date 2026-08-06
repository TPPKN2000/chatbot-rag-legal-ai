from __future__ import annotations

import threading
from collections import defaultdict, deque

from backend import config
from backend.models import ChatTurn


class SessionStore:
    def __init__(self, max_turns: int = config.SESSION_MAX_TURNS):
        self._max_turns = max_turns
        self._sessions: dict = defaultdict(lambda: deque(maxlen=self._max_turns))
        self._lock = threading.Lock()

    def append(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            self._sessions[session_id].append(ChatTurn(role=role, content=content))

    def history(self, session_id: str) -> list:
        with self._lock:
            return list(self._sessions[session_id])

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def turn_count(self, session_id: str) -> int:
        with self._lock:
            return len(self._sessions[session_id])


store = SessionStore()
