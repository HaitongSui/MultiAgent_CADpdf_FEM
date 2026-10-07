"""Data-driven orchestrator.

Scheduling rule: an agent runs when all of its ``requires`` keys are on the
blackboard and either it has never run, one of its inputs changed version
since its last run, or another agent sent it a REQUEST that changed its
parameters.  This gives a pipeline in the normal case and feedback loops
(validator -> mesher -> ... -> validator) when agents ask for rework.
"""

from __future__ import annotations

import logging
import time
from typing import Iterable, Optional

from .agent import Agent
from .blackboard import Blackboard

log = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    pass


class Orchestrator:
    def __init__(self, agents: Iterable[Agent], max_passes: int = 80) -> None:
        self.agents: list[Agent] = list(agents)
        names = [a.name for a in self.agents]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate agent names: {names}")
        self.max_passes = max_passes
        self._seen: dict[str, dict[str, int]] = {}
        self._dirty: set[str] = set()
        self.trace: list[tuple[str, float]] = []

    def agent(self, name: str) -> Optional[Agent]:
        return next((a for a in self.agents if a.name == name), None)

    def _inputs(self, a: Agent, bb: Blackboard) -> dict[str, int]:
        return {k: bb.version(k) for k in (*a.requires, *a.optional)}

    def _should_run(self, a: Agent, bb: Blackboard) -> bool:
        if not a.ready(bb):
            return False
        if a.name in self._dirty:
            return True
        return self._seen.get(a.name) != self._inputs(a, bb)

    def _dispatch_requests(self, bb: Blackboard) -> None:
        for req in bb.pop_requests():
            target = self.agent(req.target or "")
            if target is None:
                bb.say("orchestrator", f"request to unknown agent '{req.target}' ignored", "WARNING")
                continue
            if target.handle_request(bb, req.payload):
                self._dirty.add(target.name)

    def run(self, bb: Blackboard) -> Blackboard:
        for n_pass in range(1, self.max_passes + 1):
            ran = False
            for a in self.agents:
                if not self._should_run(a, bb):
                    continue
                self._dirty.discard(a.name)
                t0 = time.perf_counter()
                try:
                    a.run(bb)
                except Exception as exc:  # surface any agent failure with context
                    bb.say(a.name, f"failed: {exc}", "ERROR")
                    raise PipelineError(f"agent '{a.name}' failed: {exc}") from exc
                self._seen[a.name] = self._inputs(a, bb)
                self.trace.append((a.name, time.perf_counter() - t0))
                ran = True
                self._dispatch_requests(bb)
                # restart the sweep so upstream reruns propagate in order
                break
            if not ran:
                missing = {a.name: [k for k in a.requires if not bb.has(k)]
                           for a in self.agents if not a.ready(bb)}
                if missing:
                    bb.say("orchestrator", f"agents never became ready: {missing}", "WARNING")
                log.info("orchestration finished after %d steps", len(self.trace))
                return bb
        raise PipelineError(f"no convergence after {self.max_passes} scheduling steps")
