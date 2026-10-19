"""Persistent memory for recursive self-improvement.

Layout (one directory, plain JSON/Markdown so it can be reviewed in git)::

    memory/
      knowledge.json    verified claims per task (facts, sections, cables, index)
      lessons.md        reading lessons distilled from verified episodes
      episodes/NNN_<task>.json   every attempt: claims, checks, verdict, cost

Only verified claims and their lessons are committed; failed attempts are
kept as episodes so the curriculum can see them.  A frozen memory refuses
writes (test-time reuse).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from .claims import Claims


class MemoryFrozen(RuntimeError):
    pass


class Memory:
    def __init__(self, root: str | Path, frozen: bool = False) -> None:
        self.root = Path(root)
        self.frozen = frozen
        (self.root / "episodes").mkdir(parents=True, exist_ok=True)
        self._kpath = self.root / "knowledge.json"
        self._lpath = self.root / "lessons.md"
        self.knowledge: dict[str, dict] = (json.loads(self._kpath.read_text(encoding="utf-8"))
                                           if self._kpath.exists() else {})
        self.lessons: list[str] = self._read_lessons()

    # ---- reading ---------------------------------------------------------
    def _read_lessons(self) -> list[str]:
        if not self._lpath.exists():
            return []
        return [ln[2:].strip() for ln in self._lpath.read_text(encoding="utf-8").splitlines()
                if ln.startswith("- ")]

    def claims(self, task: str) -> Optional[Claims]:
        d = self.knowledge.get(task)
        return Claims.model_validate(d["claims"]) if d else None

    def has(self, task: str) -> bool:
        return task in self.knowledge

    def episodes(self, task: Optional[str] = None) -> list[dict]:
        eps = [json.loads(p.read_text(encoding="utf-8"))
               for p in sorted((self.root / "episodes").glob("*.json"))]
        return [e for e in eps if task is None or e["task"] == task]

    @property
    def version(self) -> str:
        h = hashlib.sha256()
        for p in (self._kpath, self._lpath):
            if p.exists():
                h.update(p.read_bytes())
        return h.hexdigest()[:12]

    # ---- writing -----------------------------------------------------------
    def _guard(self) -> None:
        if self.frozen:
            raise MemoryFrozen("memory is frozen (test-time reuse)")

    def record_episode(self, episode: dict) -> Path:
        self._guard()
        n = len(list((self.root / "episodes").glob("*.json"))) + 1
        path = self.root / "episodes" / f"{n:03d}_{episode['task']}.json"
        path.write_text(json.dumps(episode, indent=1, ensure_ascii=False), encoding="utf-8")
        return path

    def evict(self, task: str, failed: list[dict], by: str = "memory audit") -> Path:
        """Drop a committed claim that no longer passes the (updated) checks, with the
        lessons that came with it; the audit is recorded as a failed episode so the
        curriculum re-practises the task with the failures as a hint."""
        self._guard()
        entry = self.knowledge.pop(task)
        self._kpath.write_text(json.dumps(self.knowledge, indent=1, ensure_ascii=False), encoding="utf-8")
        gone = {ls.strip() for ls in entry["claims"].get("lessons", [])}
        kept = {ls.strip() for t, e in self.knowledge.items() for ls in e["claims"].get("lessons", [])}
        self.lessons = [ls for ls in self.lessons if ls not in gone or ls in kept]
        self._write_lessons()
        return self.record_episode({"task": task, "audit": True, "ok": False, "reader": by,
                                    "page": None, "checks": failed, "llm": {}, "spent_usd": {},
                                    "claims": entry["claims"]})

    def _write_lessons(self) -> None:
        body = "# Reading lessons (verified)\n\n" + "\n".join(f"- {ls}" for ls in self.lessons) + "\n"
        self._lpath.write_text(body, encoding="utf-8")

    def commit(self, claims: Claims, checks: list[dict], episode_file: str) -> list[str]:
        """Store verified claims; append new lessons.  Returns the lessons added."""
        self._guard()
        self.knowledge[claims.task] = {"claims": claims.model_dump(), "checks": checks,
                                       "episode": episode_file}
        self._kpath.write_text(json.dumps(self.knowledge, indent=1, ensure_ascii=False), encoding="utf-8")
        added = [ls for ls in claims.lessons if ls.strip() and ls.strip() not in self.lessons]
        if added:
            self.lessons += [ls.strip() for ls in added]
            self._write_lessons()
        return added
