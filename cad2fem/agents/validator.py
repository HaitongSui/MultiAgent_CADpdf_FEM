"""ValidatorAgent: checks the assembled model and asks for rework.

Checks
------
* every element positively oriented, minimum-angle statistics
* mesh boundary conforms to the geometry (no missing / bridging triangles)
* holes resolved with enough elements around their circumference
* constraints remove all rigid-body modes (rank test)
* loads present and non-zero

Mesh problems trigger a REQUEST to the MeshAgent (smaller ``refine``) up to
``max_rework`` times; the model is approved (``validation`` posted) once it
passes or the rework budget is exhausted.
"""

from __future__ import annotations

import numpy as np

from ..core import Agent, Blackboard
from ..fem.geom import distance_to_segments, loop_segments
from ..models import FEModel


def min_angles(nodes: np.ndarray, tri: np.ndarray) -> np.ndarray:
    p = nodes[tri[:, :3]]
    angs = []
    for i in range(3):
        a, b, c = p[:, i], p[:, (i + 1) % 3], p[:, (i + 2) % 3]
        u, v = b - a, c - a
        cosv = (u * v).sum(1) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-300)
        angs.append(np.degrees(np.arccos(np.clip(cosv, -1, 1))))
    return np.min(angs, axis=0)


def rigid_body_rank(model: FEModel) -> int:
    rows = []
    nodes = model.mesh.nodes
    for name, dofs, _, _ in model.constraints:
        for n in model.mesh.node_sets[name]:
            x, y = nodes[n]
            for d in dofs:
                # [tx, ty, rz] mode values at this dof
                rows.append([1.0, 0.0, -y] if d == 1 else [0.0, 1.0, x])
    if not rows:
        return 0
    R = np.asarray(rows)
    scale = np.abs(R).max(axis=0)
    return int(np.linalg.matrix_rank(R / np.maximum(scale, 1e-300), tol=1e-8))


class ValidatorAgent(Agent):
    name = "validator"
    requires = ("fe_model",)
    optional = ("mesh_size_used",)
    provides = ("validation",)

    def __init__(self, min_angle_warn: float = 15.0, bad_fraction: float = 0.02,
                 min_hole_segments: int = 12, max_rework: int = 2, refine_step: float = 0.7) -> None:
        super().__init__(min_angle_warn=min_angle_warn, bad_fraction=bad_fraction,
                         min_hole_segments=min_hole_segments, max_rework=max_rework,
                         refine_step=refine_step, rework_done=0)

    def run(self, bb: Blackboard) -> None:
        model: FEModel = bb["fe_model"]
        mesh, g = model.mesh, model.geometry
        h = bb.get("mesh_size_used") or 1.0
        issues: list[str] = []
        mesh_issues: list[str] = []

        areas = mesh.element_areas()
        if (areas <= 0).any():
            mesh_issues.append(f"{int((areas <= 0).sum())} inverted/degenerate elements")
        ang = min_angles(mesh.nodes, mesh.elements)
        bad = float((ang < self.params["min_angle_warn"]).mean())
        if bad > self.params["bad_fraction"]:
            mesh_issues.append(f"{bad:.1%} of elements have a min angle < "
                               f"{self.params['min_angle_warn']} deg")

        # boundary conformity
        a, b = loop_segments([lp.points for lp in g.loops])
        bedges = mesh.boundary_edges()
        mids = np.array([mesh.nodes[list(mesh.face_nodes(e, f))].mean(axis=0) for e, f in bedges])
        off = distance_to_segments(mids, a, b) if len(mids) else np.zeros(0)
        n_off = int((off > 0.3 * h).sum())
        if n_off:
            mesh_issues.append(f"{n_off} mesh boundary edges do not lie on the part outline")

        for k, hole in enumerate(g.holes, 1):
            n_seg = sum(1 for lid in mesh.boundary_loop.values() if lid == k)
            n_seg //= 2 if mesh.order == 2 else 1
            if n_seg < self.params["min_hole_segments"]:
                mesh_issues.append(f"HOLE{k} resolved by only {n_seg} element edges")

        area_err = abs(mesh.element_areas().sum() - g.area) / g.area
        if area_err > 0.02:
            mesh_issues.append(f"mesh area differs from geometry by {area_err:.1%}")

        # physics set-up
        if not model.constraints:
            issues.append("no constraints: model is unrestrained")
        else:
            rank = rigid_body_rank(model)
            if rank < 3:
                issues.append(f"constraints suppress only {rank}/3 rigid-body modes")
        if not model.nodal_forces:
            issues.append("no loads applied")
        else:
            tot = model.total_nodal_forces().sum(axis=0)
            if np.allclose(tot, 0) and not model.pressures:
                issues.append("applied loads sum to zero")

        issues += mesh_issues
        for msg in issues:
            self.warn(bb, msg)

        if mesh_issues and self.params["rework_done"] < self.params["max_rework"]:
            self.params["rework_done"] += 1
            refine = self.params["refine_step"] ** self.params["rework_done"]
            bb.remove("validation")
            self.request(bb, "mesh", f"mesh quality issues -> refine to x{refine:.3g} "
                                     f"(attempt {self.params['rework_done']})", refine=refine)
            return

        stats = {"nodes": mesh.n_nodes, "elements": mesh.n_elements,
                 "min_angle_deg": float(ang.min()), "mean_min_angle_deg": float(ang.mean()),
                 "area_error": float(area_err), "rework": self.params["rework_done"]}
        ok = not issues
        self.info(bb, f"{'PASSED' if ok else 'PASSED WITH WARNINGS'}: "
                      f"min angle {stats['min_angle_deg']:.1f} deg, area err {area_err:.2%}")
        bb.post("validation", {"ok": ok, "issues": issues, "stats": stats}, self.name)
