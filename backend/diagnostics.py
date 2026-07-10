"""Debug-visibility tap on the existing pipeline logging -- purely additive,
touches zero pipeline logic in agent/*.py. Every log line throughout
agent/main.py, agent/vision.py, agent/styling.py, agent/download.py, and
agent/fireworks_client.py already follows a "[task_id] message" convention;
this just captures those lines per task_id so the API can surface *why* a
clip fell back (real Fireworks error, empty Stage-A description, template
fallback, etc.) without needing to read Render's server logs directly.
"""
from __future__ import annotations

import logging
import re
import threading

_TASK_ID_RE = re.compile(r"^\[([^\]]+)\]")


class TaskLogCapture(logging.Handler):
    def __init__(self, level: int = logging.INFO):
        super().__init__(level=level)
        self._lock = threading.Lock()
        self._buffers: dict[str, list[str]] = {}

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            return
        match = _TASK_ID_RE.match(msg)
        if not match:
            return
        task_id = match.group(1)
        with self._lock:
            self._buffers.setdefault(task_id, []).append(f"{record.levelname}: {msg}")

    def pop(self, task_id: str) -> list[str]:
        with self._lock:
            return self._buffers.pop(task_id, [])
