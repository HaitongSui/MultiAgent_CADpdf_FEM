"""Shared blackboard: the single communication medium between agents.

Agents never call each other directly.  They read the artefacts they need
from the blackboard, post the artefacts they produce, and may post
*messages* (findings, warnings, requests to other agents).  The orchestrator
watches the blackboard to decide who runs next.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

log = logging.getLogger(__name__)


@dataclass
class Message:
    sender: str
    level: str                     # INFO / WARNING / ERROR / REQUEST
    text: str
    target: Optional[str] = None   # agent name for REQUEST messages
    payload: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        to = f" -> {self.target}" if self.target else ""
        return f"[{self.level:<7}] {self.sender}{to}: {self.text}"


class Blackboard:
    def __init__(self, **initial: Any) -> None:
        self._data: dict[str, Any] = dict(initial)
        self._producer: dict[str, str] = {k: "user" for k in initial}
        self._version: dict[str, int] = {k: 1 for k in initial}
        self.messages: list[Message] = []

    # ---- artefacts -------------------------------------------------------
    def post(self, key: str, value: Any, sender: str) -> None:
        self._data[key] = value
        self._producer[key] = sender
        self._version[key] = self._version.get(key, 0) + 1
        log.debug("%s posted %s", sender, key)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def has(self, key: str) -> bool:
        return key in self._data

    def remove(self, key: str) -> None:
        self._data.pop(key, None)
        self._producer.pop(key, None)

    def keys(self) -> list[str]:
        return list(self._data)

    def version(self, key: str) -> int:
        return self._version.get(key, 0)

    def producer(self, key: str) -> Optional[str]:
        return self._producer.get(key)

    # ---- messages --------------------------------------------------------
    def say(self, sender: str, text: str, level: str = "INFO", target: Optional[str] = None,
            **payload: Any) -> Message:
        msg = Message(sender, level, text, target, payload)
        self.messages.append(msg)
        getattr(log, {"ERROR": "error", "WARNING": "warning"}.get(level, "info"))(str(msg))
        return msg

    def warnings(self) -> list[Message]:
        return [m for m in self.messages if m.level in ("WARNING", "ERROR")]

    def pop_requests(self) -> list[Message]:
        reqs = [m for m in self.messages if m.level == "REQUEST" and not m.payload.get("_done")]
        for r in reqs:
            r.payload["_done"] = True
        return reqs
