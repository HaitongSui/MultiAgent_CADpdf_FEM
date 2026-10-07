"""MeshAgent: discretises the part into triangles."""

from __future__ import annotations

from ..core import Agent, Blackboard
from ..fem.mesher import default_mesh_size, to_quadratic, triangulate


class MeshAgent(Agent):
    """Params: ``mesh_size`` (mm, overrides spec), ``refine`` (multiplier on the
    size; the validator lowers it when it requests a finer mesh)."""

    name = "mesh"
    requires = ("geometry", "spec")
    provides = ("mesh",)

    def __init__(self, mesh_size: float | None = None, refine: float = 1.0,
                 smooth_iters: int = 6, min_hole_segments: int = 16) -> None:
        super().__init__(mesh_size=mesh_size, refine=refine, smooth_iters=smooth_iters,
                         min_hole_segments=min_hole_segments)

    def run(self, bb: Blackboard) -> None:
        g = bb["geometry"]
        spec = bb["spec"]
        h0 = self.params["mesh_size"] or spec.mesh_size or default_mesh_size(g)
        h = h0 * self.params["refine"]
        mesh = triangulate(g, h, self.params["smooth_iters"], self.params["min_hole_segments"])
        if spec.element_order == 2:
            mesh = to_quadratic(mesh, g)
        self.info(bb, f"h={h:.4g} mm -> {mesh.n_nodes} nodes, {mesh.n_elements} "
                      f"{'Tri6' if mesh.order == 2 else 'Tri3'} elements")
        bb.post("mesh_size_used", h, self.name)
        bb.post("mesh", mesh, self.name)
