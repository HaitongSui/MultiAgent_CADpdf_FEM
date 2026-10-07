"""SpecAgent: reconciles rule-based, LLM and user-supplied analysis settings."""

from __future__ import annotations

import copy

from ..core import Agent, Blackboard
from ..models import AnalysisSpec

FIELDS = ("title", "units", "scale", "thickness", "analysis", "element_order", "mesh_size",
          "material")


class SpecAgent(Agent):
    """Priority: user overrides > explicit rule matches > explicit LLM values
    > defaults.  For BCs/loads the rule list wins when non-empty unless
    ``prefer='llm'``; disagreements are reported."""

    name = "spec"
    requires = ("spec_rules",)
    optional = ("spec_llm", "overrides")
    provides = ("spec",)

    def __init__(self, prefer: str = "rules") -> None:
        super().__init__(prefer=prefer)

    def run(self, bb: Blackboard) -> None:
        rules: AnalysisSpec = bb["spec_rules"]
        llm: AnalysisSpec | None = bb.get("spec_llm")
        spec = copy.deepcopy(rules)
        prefer_llm = self.params["prefer"] == "llm"

        if llm is not None:
            spec.source = "rules+llm"
            for f in FIELDS:
                if f in llm.explicit and (f not in rules.explicit or prefer_llm):
                    if f in rules.explicit and getattr(rules, f) != getattr(llm, f):
                        self.warn(bb, f"{f}: rules={getattr(rules, f)!r} vs llm={getattr(llm, f)!r}"
                                      " -> using LLM")
                    setattr(spec, f, copy.deepcopy(getattr(llm, f)))
                    spec.explicit.add(f)
                elif f in llm.explicit and getattr(rules, f) != getattr(llm, f):
                    self.warn(bb, f"{f}: rules={getattr(rules, f)!r} vs llm={getattr(llm, f)!r}"
                                  " -> keeping rule value")
            for attr in ("bcs", "loads"):
                r, m = getattr(rules, attr), getattr(llm, attr)
                if m and (not r or prefer_llm):
                    setattr(spec, attr, copy.deepcopy(m))
                    self.info(bb, f"{attr} taken from LLM ({len(m)})")
                elif m and r and _signature(r) != _signature(m):
                    self.warn(bb, f"{attr} differ between rules {_signature(r)} and LLM "
                                  f"{_signature(m)}; keeping rules (use prefer='llm' to switch)")
            spec.notes = rules.notes + [f"LLM: {n}" for n in llm.notes]

        for k, v in (bb.get("overrides") or {}).items():
            if v is None:
                continue
            if not hasattr(spec, k):
                self.warn(bb, f"unknown override '{k}' ignored")
                continue
            setattr(spec, k, v)
            spec.explicit.add(k)

        if not spec.bcs:
            self.warn(bb, "no boundary conditions found in drawing notes")
        if not spec.loads:
            self.warn(bb, "no loads found in drawing notes")
        if "thickness" not in spec.explicit:
            self.warn(bb, f"thickness not stated; using {spec.thickness:g} mm")
        if "material" not in spec.explicit:
            self.warn(bb, f"material not stated; using {spec.material.name}")
        bb.post("spec", spec, self.name)


def _signature(items) -> list[tuple]:
    out = []
    for it in items:
        if hasattr(it, "dofs"):
            out.append((it.location, tuple(it.dofs)))
        else:
            out.append((it.location, it.kind, round(it.magnitude, 6)))
    return sorted(out)
