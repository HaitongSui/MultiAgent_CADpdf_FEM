"""Minimal linear-elastic CST (Tri3) solver used to sanity-check models.

Quadratic meshes are checked on their corner-node (Tri3) sub-mesh.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from ..models import FEModel


@dataclass
class SolveResult:
    u: np.ndarray               # (n, 2) displacements
    stress: np.ndarray          # (m, 3) sx, sy, txy per element
    von_mises: np.ndarray       # (m,)
    reactions: np.ndarray       # (n, 2)
    applied: np.ndarray         # (n, 2)


def d_matrix(E: float, nu: float, analysis: str) -> np.ndarray:
    if analysis == "PLANE_STRAIN":
        c = E / ((1 + nu) * (1 - 2 * nu))
        return c * np.array([[1 - nu, nu, 0], [nu, 1 - nu, 0], [0, 0, (1 - 2 * nu) / 2]])
    c = E / (1 - nu ** 2)
    return c * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])


def _b_matrices(xy: np.ndarray):
    """xy: (m, 3, 2) -> B (m, 3, 6), area (m,)"""
    x, y = xy[..., 0], xy[..., 1]
    b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], axis=1)
    c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], axis=1)
    area = 0.5 * (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0])
    B = np.zeros((len(xy), 3, 6))
    B[:, 0, 0::2] = b
    B[:, 1, 1::2] = c
    B[:, 2, 0::2] = c
    B[:, 2, 1::2] = b
    return B / (2 * area)[:, None, None], area


def solve(model: FEModel) -> SolveResult:
    mesh, spec = model.mesh, model.spec
    tri = mesh.corner_elements()
    n = mesh.n_nodes
    D = d_matrix(spec.material.E, spec.material.nu, spec.analysis)
    B, area = _b_matrices(mesh.nodes[tri])
    t = spec.thickness
    Ke = np.einsum("mji,jk,mkl->mil", B, D, B) * (area * t)[:, None, None]
    dofs = np.repeat(tri * 2, 2, axis=1) + np.tile([0, 1], 3)
    rows = np.repeat(dofs, 6, axis=1).ravel()
    cols = np.tile(dofs, (1, 6)).ravel()
    K = coo_matrix((Ke.ravel(), (rows, cols)), shape=(2 * n, 2 * n)).tocsr()

    # lumped loads: mid-side forces of quadratic meshes go to the edge ends
    F = model.total_nodal_forces()
    if mesh.order == 2:
        F = _lump_to_corners(model, F)
    f = F.ravel()

    fixed = np.zeros(2 * n, bool)
    u = np.zeros(2 * n)
    for name, dof_list, value, _ in model.constraints:
        for d in dof_list:
            idx = mesh.node_sets[name] * 2 + (d - 1)
            fixed[idx] = True
            u[idx] = value
    used = np.zeros(2 * n, bool)
    used[dofs.ravel()] = True
    free = used & ~fixed
    rhs = f[free] - K[free][:, fixed] @ u[fixed]
    Kff = K[free][:, free]
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        try:
            u[free] = spsolve(Kff.tocsc(), rhs)
        except (MatrixRankWarning, RuntimeError) as exc:
            raise np.linalg.LinAlgError(f"singular stiffness matrix ({exc})") from exc
    if not np.all(np.isfinite(u)):
        raise np.linalg.LinAlgError("singular stiffness matrix (insufficient constraints?)")
    r = K @ u - f
    reactions = np.where(fixed, r, 0.0).reshape(-1, 2)
    ue = u[dofs]
    stress = np.einsum("ij,mjk,mk->mi", D, B, ue)
    sx, sy, txy = stress.T
    if spec.analysis == "PLANE_STRAIN":
        sz = spec.material.nu * (sx + sy)
        vm = np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * txy ** 2)
    else:
        vm = np.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3 * txy ** 2)
    return SolveResult(u.reshape(-1, 2), stress, vm, reactions, F)


def _lump_to_corners(model: FEModel, F: np.ndarray) -> np.ndarray:
    mesh = model.mesh
    F = F.copy()
    for tri in mesh.elements:
        for k, (a, b) in enumerate(((0, 1), (1, 2), (2, 0))):
            m = tri[3 + k]
            if np.any(F[m]):
                F[tri[a]] += F[m] / 2
                F[tri[b]] += F[m] / 2
                F[m] = 0.0
    return F
