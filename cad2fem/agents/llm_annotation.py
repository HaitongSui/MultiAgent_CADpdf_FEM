"""LLMAnnotationAgent: optional Claude-backed reading of drawing notes.

Free-form notes ("the plate is welded along its left side and pulled with
50 MPa on the opposite end") are hard to cover with regexes.  This agent asks
Claude to map the drawing text onto a strict schema.  It is optional: if the
``anthropic`` package or credentials are missing, or the call fails, it posts
nothing and the SpecAgent falls back to the rule-based result.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from ..core import Agent, Blackboard
from ..models import AnalysisSpec, BoundaryCondition, Load, Material

LOCATION_HELP = ("Boundary selector. One of LEFT, RIGHT, TOP, BOTTOM (extreme edges of the part "
                 "bounding box), HOLE<n> (n-th hole, numbered left-to-right then bottom-to-top, "
                 "starting at 1), HOLES (all holes), X=<value> or Y=<value> (straight edge at that "
                 "model coordinate in mm).")


class LLMBoundaryCondition(BaseModel):
    location: str = Field(description=LOCATION_HELP)
    dofs: list[int] = Field(description="Constrained degrees of freedom: 1 = x, 2 = y.")
    kind: str = Field(description="FIXED, PINNED, ROLLER, SYMMETRY ...")


class LLMLoad(BaseModel):
    location: str = Field(description=LOCATION_HELP)
    kind: Literal["PRESSURE", "FORCE"]
    magnitude: float = Field(description="PRESSURE in MPa, positive pushes INTO the body (tension "
                                         "is negative). FORCE in N, total resultant, always >= 0.")
    direction_x: float = Field(description="FORCE only: unit direction x component, else 0.")
    direction_y: float = Field(description="FORCE only: unit direction y component, else 0.")


class LLMSpec(BaseModel):
    title: Optional[str]
    units: Optional[str] = Field(description="Drawing length unit: mm, cm, m or in.")
    scale: Optional[float] = Field(description="Scale denominator N for 'SCALE 1:N'.")
    thickness_mm: Optional[float]
    analysis: Literal["PLANE_STRESS", "PLANE_STRAIN"]
    material_name: Optional[str]
    youngs_modulus_mpa: Optional[float]
    poisson_ratio: Optional[float]
    mesh_size_mm: Optional[float]
    element_order: Optional[int] = Field(description="1 linear, 2 quadratic, null if not stated.")
    boundary_conditions: list[LLMBoundaryCondition]
    loads: list[LLMLoad]
    remarks: list[str] = Field(description="Ambiguities or assumptions you had to make.")


SYSTEM = (
    "You are the annotation-reading agent in a pipeline that converts 2D CAD drawings into "
    "plane-stress/plane-strain finite element models (units mm, N, MPa). You receive every text "
    "line found on the drawing sheet: title block, dimensions and notes. Extract the analysis "
    "setup. Only report values the drawing actually states or clearly implies; use null for "
    "anything absent rather than guessing. Dimension numbers belong to geometry, not to the "
    "analysis setup - ignore them. List any assumption in remarks."
)


def to_analysis_spec(s: LLMSpec) -> AnalysisSpec:
    from .. import materials

    spec = AnalysisSpec(source="llm", analysis=s.analysis, notes=list(s.remarks))
    values = {"title": s.title, "units": s.units and s.units.lower(), "scale": s.scale,
              "thickness": s.thickness_mm, "mesh_size": s.mesh_size_mm,
              "element_order": s.element_order if s.element_order in (1, 2) else None}
    for key, value in values.items():
        if value:
            setattr(spec, key, value)
            spec.explicit.add(key)
    if s.analysis == "PLANE_STRAIN":
        spec.explicit.add("analysis")
    if s.material_name or s.youngs_modulus_mpa:
        mat = materials.lookup(s.material_name or "") or Material(name=s.material_name or "MAT")
        if s.youngs_modulus_mpa:
            mat.E = s.youngs_modulus_mpa
        if s.poisson_ratio:
            mat.nu = s.poisson_ratio
        spec.material = mat
        spec.explicit.add("material")
    spec.bcs = [BoundaryCondition(b.location.upper().replace(" ", ""),
                                  tuple(sorted({d for d in b.dofs if d in (1, 2)})) or (1, 2),
                                  kind=b.kind.upper()) for b in s.boundary_conditions]
    spec.loads = [Load(ld.location.upper().replace(" ", ""), ld.kind, ld.magnitude,
                       (ld.direction_x, ld.direction_y)) for ld in s.loads]
    return spec


class LLMAnnotationAgent(Agent):
    name = "llm_annotation"
    requires = ("drawing",)
    provides = ("spec_llm",)

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", client=None) -> None:
        super().__init__(model=model, effort=effort)
        self._client = client

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def run(self, bb: Blackboard) -> None:
        drawing = bb["drawing"]
        text = "\n".join(ln.text for ln in drawing.lines)
        if not text.strip():
            self.warn(bb, "no text on drawing; skipping LLM extraction")
            return
        try:
            import anthropic
        except ImportError:
            self.warn(bb, "anthropic package not installed; skipping LLM extraction")
            return
        try:
            response = self._get_client().beta.messages.parse(
                model=self.params["model"],
                max_tokens=16000,
                system=SYSTEM,
                thinking={"type": "adaptive"},
                output_config={"effort": self.params["effort"]},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=[{"role": "user",
                           "content": f"<drawing_text>\n{text}\n</drawing_text>"}],
                output_format=LLMSpec,
            )
        except anthropic.AuthenticationError:
            self.warn(bb, "no valid Anthropic credentials; using rule-based annotations only")
            return
        except anthropic.RateLimitError as exc:
            self.warn(bb, f"rate limited ({exc}); using rule-based annotations only")
            return
        except anthropic.APIStatusError as exc:
            self.warn(bb, f"API error {exc.status_code}: {exc.message}; using rule-based annotations")
            return
        except anthropic.APIConnectionError as exc:
            self.warn(bb, f"cannot reach Anthropic API ({exc}); using rule-based annotations")
            return
        except Exception as exc:  # optional agent: never take the pipeline down
            self.warn(bb, f"LLM extraction unavailable ({type(exc).__name__}: {exc}); "
                          "using rule-based annotations")
            return

        if response.stop_reason == "refusal":
            self.warn(bb, "model declined the request; using rule-based annotations")
            return
        parsed = response.parsed_output
        if parsed is None:
            self.warn(bb, f"no structured output (stop_reason={response.stop_reason})")
            return
        spec = to_analysis_spec(parsed)
        for r in parsed.remarks:
            self.info(bb, f"LLM remark: {r}")
        self.info(bb, f"extracted {len(spec.bcs)} BCs, {len(spec.loads)} loads via {response.model}")
        bb.post("spec_llm", spec, self.name)
