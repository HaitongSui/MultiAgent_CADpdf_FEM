"""LLM backends for the RSI agents (all return parsed JSON).

* ClaudeCodeBackend  - runs the Claude Code CLI headless (``claude -p``), which
                       uses the environment's existing Claude login; each call is
                       a fresh, independent context that reads the image files
                       with its Read tool.  No API key needed.
* AnthropicBackend   - Messages API with base64 images (needs ANTHROPIC_API_KEY).
* auto_backend()     - API key -> API; else `claude` on PATH -> CLI; else None.

Every backend tracks spend so the loop can enforce a budget.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional


class BackendError(RuntimeError):
    pass


def extract_json(text: str):
    """First JSON object/array in a model reply (tolerates code fences)."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise BackendError(f"no JSON in reply: {text[:200]!r}")
    dec = json.JSONDecoder()
    obj, _ = dec.raw_decode(text[start:])
    return obj


class Backend:
    name = "backend"

    def __init__(self) -> None:
        self.spent_usd = 0.0
        self.calls = 0

    def ask(self, prompt: str, images: list[Path]) -> dict:   # pragma: no cover - interface
        raise NotImplementedError


class ClaudeCodeBackend(Backend):
    name = "claude-code-cli"

    def __init__(self, model: Optional[str] = None, timeout: int = 900, max_turns: int = 12) -> None:
        super().__init__()
        self.model, self.timeout, self.max_turns = model, timeout, max_turns
        self.exe = shutil.which("claude")
        if not self.exe:
            raise BackendError("Claude Code CLI ('claude') not found on PATH")

    def ask(self, prompt: str, images: list[Path]) -> dict:
        listing = "\n".join(f"- {p}" for p in images)
        full = (f"{prompt}\n\nImage files (open each with the Read tool before answering):\n{listing}\n\n"
                "Answer with one JSON object only, no prose.")
        cmd = [self.exe, "-p", full, "--allowedTools", "Read", "--max-turns", str(self.max_turns),
               "--output-format", "json"]
        if self.model:
            cmd += ["--model", self.model]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout,
                                  cwd=str(Path(images[0]).parent) if images else None)
        except subprocess.TimeoutExpired as exc:
            raise BackendError(f"claude CLI timed out after {self.timeout}s") from exc
        self.calls += 1
        try:
            out = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise BackendError(f"claude CLI returned non-JSON: {proc.stdout[-300:]} {proc.stderr[-300:]}") from exc
        self.spent_usd += float(out.get("total_cost_usd") or 0.0)
        if out.get("is_error"):
            raise BackendError(f"claude CLI error: {out.get('result')}")
        return extract_json(out.get("result") or "")


class AnthropicBackend(Backend):
    name = "anthropic-api"
    # approximate list prices (USD per token) used only for the budget guard
    PRICE_IN, PRICE_OUT = 4e-6, 20e-6

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "high", client=None) -> None:
        super().__init__()
        import anthropic

        self.client = client or anthropic.Anthropic()
        self.model, self.effort = model, effort

    def ask(self, prompt: str, images: list[Path]) -> dict:
        content = []
        for p in images:
            content.append({"type": "text", "text": Path(p).name})
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/png",
                "data": base64.b64encode(Path(p).read_bytes()).decode()}})
        content.append({"type": "text", "text": prompt + "\n\nAnswer with one JSON object only."})
        resp = self.client.beta.messages.create(
            model=self.model, max_tokens=16000, thinking={"type": "adaptive"},
            output_config={"effort": self.effort}, betas=["server-side-fallback-2026-07-01"],
            fallbacks="default", messages=[{"role": "user", "content": content}])
        self.calls += 1
        u = resp.usage
        self.spent_usd += u.input_tokens * self.PRICE_IN + u.output_tokens * self.PRICE_OUT
        if resp.stop_reason == "refusal":
            raise BackendError("model declined the request")
        text = "".join(b.text for b in resp.content if b.type == "text")
        return extract_json(text)


def auto_backend(prefer: str = "auto") -> Optional[Backend]:
    if prefer in ("auto", "api") and os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicBackend()
    if prefer in ("auto", "cli") and shutil.which("claude"):
        return ClaudeCodeBackend()
    return None
