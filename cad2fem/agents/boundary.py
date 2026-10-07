"""BoundaryAgent: maps textual BC/load locations onto mesh entities."""

from __future__ import annotations

import re

import numpy as np

from ..core import Agent, Blackboard
from ..models import FEModel, Mesh


def select_edges(mesh: Mesh, location: str, tol: float) -> list[tuple[int, int]]:
    """Boundary element faces matching a selector (LEFT, HOLE2, X=10 ...)."""
    loc = location.upper().replace(" ", "")
    bl = mesh.boundary_loop
    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    outer = [n for n, k in bl.items() if k == 0]
    if not outer:
        return []
    ox, oy = x[outer], y[outer]

    def coord_test(axis: np.ndarray, value: float):
        return lambda n: abs(axis[n] - value) <= tol

    m = re.fullmatch(r"HOLE(\d+)", loc)
    if m:
        k = int(m.group(1))
        test = lambda n: bl.get(n) == k  # noqa: E731
    elif loc == "HOLES":
        test = lambda n: bl.get(n, 0) > 0  # noqa: E731
    elif loc == "LEFT":
        test = coord_test(x, ox.min())
    elif loc == "RIGHT":
        test = coord_test(x, ox.max())
    elif loc == "BOTTOM":
        test = coord_test(y, oy.min())
    elif loc == "TOP":
        test = coord_test(y, oy.max())
    elif m := re.fullmatch(r"([XY])=([-+]?[\d.]+(?:[eE][-+]?\d+)?)", loc):
        test = coord_test(x if m.group(1) == "X" else y, float(m.group(2)))
    else:
        raise ValueError(f"unknown location selector '{location}'")

    edges = []
    for e, f in mesh.boundary_edges():
        a, b = mesh.face_nodes(e, f)
        if a in bl and b in bl and test(a) and test(b):
            if loc.startswith("HOLE") and not (bl[a] == bl[b]):
                continue
            edges.append((e, f))
    return edges


def edge_nodes(mesh: Mesh, e: int, f: int) -> list[int]:
    a, b = mesh.face_nodes(e, f)
    if mesh.order == 2:
        return [a, int(mesh.elements[e][3 + f - 1]), b]
    return [a, b]


def equivalent_forces(mesh: Mesh, edges, traction_fn) -> tuple[np.ndarray, np.ndarray]:
    """Consistent nodal forces for an edge traction.

    ``traction_fn(p_a, p_b)`` returns the force per unit length (vector) on
    that edge (already multiplied by thickness).  Returns (node ids, forces).
    """
    acc: dict[int, np.ndarray] = {}
    for e, f in edges:
        nodes = edge_nodes(mesh, e, f)
        pa, pb = mesh.nodes[nodes[0]], mesh.nodes[nodes[-1]]
        L = float(np.linalg.norm(pb - pa))
        q = np.asarray(traction_fn(pa, pb), float) * L
        weights = (0.5, 0.5) if len(nodes) == 2 else (1 / 6, 2 / 3, 1 / 6)
        for n, w in zip(nodes, weights):
            acc[n] = acc.get(n, np.zeros(2)) + w * q
    ids = np.array(sorted(acc), dtype=int)
    return ids, np.array([acc[i] for i in ids]).reshape(-1, 2)


class BoundaryAgent(Agent):
    name = "boundary"
    requires = ("mesh", "spec", "geometry")
    provides = ("fe_model",)

    def run(self, bb: Blackboard) -> None:
        mesh: Mesh = bb["mesh"]
        spec = bb["spec"]
        g = bb["geometry"]
        x0, y0, x1, y1 = g.bbox
        tol = 1e-4 * max(x1 - x0, y1 - y0)
        mesh.node_sets = {"NALL": np.arange(mesh.n_nodes)}
        mesh.surfaces = {}
        model = FEModel(mesh=mesh, spec=spec, geometry=g)
        t = spec.thickness

        for i, bc in enumerate(spec.bcs, 1):
            edges = select_edges(mesh, bc.location, tol)
            if not edges:
                self.error(bb, f"BC '{bc.kind} {bc.location}' matches no boundary edge")
                continue
            name = f"BC{i}_{_safe(bc.location)}"
            mesh.node_sets[name] = np.unique([n for e, f in edges for n in edge_nodes(mesh, e, f)])
            model.constraints.append((name, tuple(bc.dofs), bc.value, f"{bc.kind} {bc.location}"))
            self.info(bb, f"{name}: {len(mesh.node_sets[name])} nodes, dofs {bc.dofs}")

        for i, ld in enumerate(spec.loads, 1):
            edges = select_edges(mesh, ld.location, tol)
            if not edges:
                self.error(bb, f"load '{ld.kind} {ld.location}' matches no boundary edge")
                continue
            name = f"LOAD{i}_{_safe(ld.location)}"
            if ld.kind == "PRESSURE":
                mesh.surfaces[name] = edges
                model.pressures.append((name, ld.magnitude))

                def traction(pa, pb, p=ld.magnitude):
                    d = pb - pa
                    n_out = np.array([d[1], -d[0]]) / max(np.linalg.norm(d), 1e-300)  # CCW elems
                    return -p * n_out * t
            else:
                length = sum(np.linalg.norm(np.subtract(*mesh.nodes[list(mesh.face_nodes(e, f))]))
                             for e, f in edges)
                direction = np.asarray(ld.direction, float)
                direction = direction / max(np.linalg.norm(direction), 1e-300)

                # total force spread uniformly along the selection (N per mm)
                def traction(pa, pb, q=ld.magnitude / length * direction):
                    return q
            ids, forces = equivalent_forces(mesh, edges, traction)
            mesh.node_sets[name] = ids
            model.nodal_forces[name] = forces
            tot = forces.sum(axis=0)
            self.info(bb, f"{name}: {ld.kind} {ld.magnitude:g} on {len(edges)} edges, "
                          f"resultant ({tot[0]:.4g}, {tot[1]:.4g}) N")
        bb.post("fe_model", model, self.name)


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").upper()
