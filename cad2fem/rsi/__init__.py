"""Recursive self-improvement (RSI) for reading drawing sets.

Curriculum, Actor, Verifier and Memory agents practise on the sheets, keep
only verified knowledge and lessons, and reuse the frozen memory to build the
model (in the spirit of RSIAgent: broad-then-deep exploration, grounded
verification, persistent memory, fixed model weights).
"""

from .loop import RSIConfig, run_rsi

__all__ = ["RSIConfig", "run_rsi"]
