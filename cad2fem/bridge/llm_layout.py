"""Optional Claude-backed reading of general-arrangement notes.

Same contract as the part-level LLMAnnotationAgent: structured output into a
strict schema, and graceful fallback (nothing posted) on any failure.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from ..core import Agent, Blackboard
from .layout_rules import material
from .models import BridgeLayout


class LLMBearing(BaseModel):
    support: str = Field(description="A0 (first abutment), P1, P2 ... (piers), An (last abutment)")
    kind: Literal["FIXED", "SLIDING_X", "SLIDING_XY", "MONOLITHIC"] = Field(
        description="FIXED=fixed bearing, SLIDING_X=longitudinally movable/guided, "
                    "SLIDING_XY=free in plan, MONOLITHIC=deck rigidly connected to pier")


class LLMPier(BaseModel):
    pier: str = Field(description="P1, P2, ...")
    height_m: float


class LLMLayout(BaseModel):
    title: Optional[str]
    spans_m: list[float] = Field(description="Span lengths in metres, in order. Empty if not stated.")
    piers: list[LLMPier]
    bearings: list[LLMBearing]
    deck_material: Optional[str] = Field(description="Grade, e.g. C50 or Q345")
    pier_material: Optional[str]
    secondary_dead_kN_per_m: Optional[float]
    lane_q_kN_per_m: Optional[float]
    lane_P_kN: Optional[float]
    lanes: Optional[int]
    element_size_m: Optional[float]
    remarks: list[str]


SYSTEM = (
    "You are the layout-reading agent of a pipeline that builds a global spine (beam) finite "
    "element model of a girder bridge from its drawing set. You receive the text found on the "
    "sheets (title blocks, notes, dimensions). Extract the bridge layout. Only report values the "
    "drawing states or clearly implies; leave a field null or a list empty rather than guess. "
    "Dimension numbers on the elevation are in the drawing's stated unit; convert spans and "
    "heights to metres. List assumptions in remarks."
)


def to_layout(s: LLMLayout) -> BridgeLayout:
    lay = BridgeLayout(source="llm", notes=list(s.remarks))
    if s.title:
        lay.title = s.title
        lay.explicit.add("title")
    if s.spans_m:
        lay.spans = list(s.spans_m)
        lay.explicit.add("spans")
    if s.piers:
        lay.pier_heights = {p.pier.upper(): p.height_m for p in s.piers}
        lay.explicit.add("pier_heights")
    if s.bearings:
        lay.bearings = {b.support.upper(): b.kind for b in s.bearings}
        lay.explicit.add("bearings")
    for attr, value in (("deck_material", s.deck_material), ("pier_material", s.pier_material)):
        mat = material(value) if value else None
        if mat:
            setattr(lay, attr, mat)
            lay.explicit.add(attr)
    if s.secondary_dead_kN_per_m:
        lay.secondary_dead = s.secondary_dead_kN_per_m
        lay.explicit.add("secondary_dead")
    if s.lane_q_kN_per_m or s.lane_P_kN:
        lay.lane_q = s.lane_q_kN_per_m or 0.0
        lay.lane_P = s.lane_P_kN or 0.0
        lay.lanes = s.lanes or 1
        lay.explicit.add("lane")
    if s.element_size_m:
        lay.element_size = s.element_size_m
        lay.explicit.add("element_size")
    return lay


class LLMLayoutAgent(Agent):
    name = "llm_layout"
    requires = ("sheets", "sheet_kinds")
    provides = ("layout_llm",)

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", client=None) -> None:
        super().__init__(model=model, effort=effort)
        self._client = client

    def run(self, bb: Blackboard) -> None:
        text = "\n\n".join(f"<sheet kind='{k}' source='{s.source}'>\n" + "\n".join(s.lines) + "\n</sheet>"
                           for s, k in zip(bb["sheets"], bb["sheet_kinds"]))
        try:
            import anthropic

            client = self._client or anthropic.Anthropic()
            response = client.beta.messages.parse(
                model=self.params["model"],
                max_tokens=16000,
                system=SYSTEM,
                thinking={"type": "adaptive"},
                output_config={"effort": self.params["effort"]},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=[{"role": "user", "content": text}],
                output_format=LLMLayout,
            )
        except Exception as exc:  # optional agent: never take the pipeline down
            self.warn(bb, f"LLM layout extraction unavailable ({type(exc).__name__}: {exc})")
            return
        if response.stop_reason == "refusal" or response.parsed_output is None:
            self.warn(bb, f"no LLM layout (stop_reason={response.stop_reason}); using rules")
            return
        lay = to_layout(response.parsed_output)
        for r in response.parsed_output.remarks:
            self.info(bb, f"LLM remark: {r}")
        bb.post("layout_llm", lay, self.name)
