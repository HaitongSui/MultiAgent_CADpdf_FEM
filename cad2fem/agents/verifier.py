"""VerifierAgent: solves the model with the built-in CST solver as a smoke test."""

from __future__ import annotations

import numpy as np

from ..core import Agent, Blackboard
from ..fem.solver import solve


class VerifierAgent(Agent):
    name = "verifier"
    requires = ("fe_model", "validation")
    provides = ("verification",)

    def run(self, bb: Blackboard) -> None:
        model = bb["fe_model"]
        if not model.constraints:
            self.warn(bb, "skipping solve: no constraints")
            return
        try:
            res = solve(model)
        except np.linalg.LinAlgError as exc:
            self.error(bb, f"solve failed: {exc}")
            return
        applied = res.applied.sum(axis=0)
        reaction = res.reactions.sum(axis=0)
        scale = max(np.abs(res.applied).sum(), 1e-300)
        eq_err = float(np.linalg.norm(applied + reaction) / scale)
        x0, y0, x1, y1 = model.geometry.bbox
        umax = float(np.linalg.norm(res.u, axis=1).max())
        out = {
            "max_displacement_mm": umax,
            "max_von_mises_mpa": float(res.von_mises.max()),
            "mean_von_mises_mpa": float(np.average(res.von_mises, weights=model.mesh.element_areas())),
            "applied_resultant_n": applied.tolist(),
            "reaction_resultant_n": reaction.tolist(),
            "equilibrium_error": eq_err,
            "result": res,
        }
        if eq_err > 1e-6:
            self.warn(bb, f"equilibrium error {eq_err:.2e}")
        if umax > 0.1 * max(x1 - x0, y1 - y0):
            self.warn(bb, f"very large displacement ({umax:.3g} mm): check constraints/units")
        self.info(bb, f"CST check: |u|max={umax:.4g} mm, von Mises max={out['max_von_mises_mpa']:.4g} "
                      f"MPa, reactions=({reaction[0]:.4g}, {reaction[1]:.4g}) N")
        bb.post("verification", out, self.name)
