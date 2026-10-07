"""Base class for all agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from .blackboard import Blackboard


class Agent(ABC):
    """An agent consumes blackboard keys (``requires``) and produces others
    (``provides``).  ``optional`` keys are consumed if present.

    ``params`` holds tunable settings that other agents can change through
    REQUEST messages (e.g. the validator asking the mesher for a finer mesh).
    """

    name: ClassVar[str] = "agent"
    requires: ClassVar[tuple[str, ...]] = ()
    optional: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = dict(params)

    def ready(self, bb: Blackboard) -> bool:
        return all(bb.has(k) for k in self.requires)

    def handle_request(self, bb: Blackboard, payload: dict) -> bool:
        """Apply a REQUEST from another agent.  Return True if a re-run is needed."""
        changed = False
        for k, v in payload.items():
            if k.startswith("_"):
                continue
            if self.params.get(k) != v:
                self.params[k] = v
                changed = True
        return changed

    @abstractmethod
    def run(self, bb: Blackboard) -> None:
        """Do the work: read from and post to the blackboard."""

    # helpers ---------------------------------------------------------------
    def info(self, bb: Blackboard, text: str, **kw: Any) -> None:
        bb.say(self.name, text, "INFO", **kw)

    def warn(self, bb: Blackboard, text: str, **kw: Any) -> None:
        bb.say(self.name, text, "WARNING", **kw)

    def error(self, bb: Blackboard, text: str, **kw: Any) -> None:
        bb.say(self.name, text, "ERROR", **kw)

    def request(self, bb: Blackboard, target: str, text: str, **payload: Any) -> None:
        bb.say(self.name, text, "REQUEST", target=target, **payload)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"
