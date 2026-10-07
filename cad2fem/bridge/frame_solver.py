"""Linear static 3D frame solver (Euler-Bernoulli beams, springs, rigid links).

Used by the BridgeVerifierAgent as an independent check of the assembled
model, and to produce internal-force diagrams for the report.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, identity, lil_matrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from .models import Beam, FrameModel


def rotation(p1: np.ndarray, p2: np.ndarray, vec_y) -> tuple[np.ndarray, float]:
    """3x3 matrix with rows = local x, y, z unit vectors; and member length."""
    d = p2 - p1
    L = float(np.linalg.norm(d))
    ex = d / L
    vy = np.asarray(vec_y, float)
    ey = vy - (vy @ ex) * ex
    ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    return np.vstack([ex, ey, ez]), L


def local_stiffness(E, G, A, Iyy, Izz, J, L) -> np.ndarray:
    k = np.zeros((12, 12))
    EA, GJ = E * A / L, G * J / L
    k[0, 0] = k[6, 6] = EA
    k[0, 6] = k[6, 0] = -EA
    k[3, 3] = k[9, 9] = GJ
    k[3, 9] = k[9, 3] = -GJ
    # bending in local x-y plane (v, theta_z) -> Izz
    a, b, c, d = 12 * E * Izz / L ** 3, 6 * E * Izz / L ** 2, 4 * E * Izz / L, 2 * E * Izz / L
    for (i, j), val in {(1, 1): a, (7, 7): a, (1, 7): -a, (1, 5): b, (1, 11): b, (5, 7): -b,
                        (7, 11): -b, (5, 5): c, (11, 11): c, (5, 11): d}.items():
        k[i, j] = k[j, i] = val
    # bending in local x-z plane (w, theta_y) -> Iyy
    a, b, c, d = 12 * E * Iyy / L ** 3, 6 * E * Iyy / L ** 2, 4 * E * Iyy / L, 2 * E * Iyy / L
    for (i, j), val in {(2, 2): a, (8, 8): a, (2, 8): -a, (2, 4): -b, (2, 10): -b, (4, 8): b,
                        (8, 10): b, (4, 4): c, (10, 10): c, (4, 10): d}.items():
        k[i, j] = k[j, i] = val
    return k


def uniform_load_vector(R: np.ndarray, L: float, q_global) -> np.ndarray:
    """Consistent global nodal loads (12,) for a uniform load per length given
    in global axes."""
    qx, qy, qz = R @ np.asarray(q_global, float)
    f = np.zeros(12)
    f[0] = f[6] = qx * L / 2
    f[1] = f[7] = qy * L / 2
    f[5], f[11] = qy * L ** 2 / 12, -qy * L ** 2 / 12
    f[2] = f[8] = qz * L / 2
    f[4], f[10] = -qz * L ** 2 / 12, qz * L ** 2 / 12
    T = np.kron(np.eye(4), R)
    return T.T @ f


def beam_matrices(model: FrameModel, b: Beam):
    R, L = rotation(model.nodes[b.n1], model.nodes[b.n2], b.vec_y)
    s = model.sections[b.section]
    m = model.materials[b.material]
    k = local_stiffness(m.E, m.G, s.A, s.Iyy, s.Izz, s.J, L)
    T = np.kron(np.eye(4), R)
    return k, T, L


@dataclass
class FrameResult:
    u: np.ndarray            # (n, 6)
    reactions: np.ndarray    # (n, 6) at supported nodes
    end_forces: np.ndarray   # (n_beams, 12) local end forces (element on nodes)
    applied: np.ndarray      # (n, 6)


def assemble(model: FrameModel):
    n = model.n_nodes
    rows, cols, vals = [], [], []
    for b in model.beams:
        k, T, _ = beam_matrices(model, b)
        kg = T.T @ k @ T
        dofs = np.r_[6 * b.n1 + np.arange(6), 6 * b.n2 + np.arange(6)]
        rows.append(np.repeat(dofs, 12))
        cols.append(np.tile(dofs, 12))
        vals.append(kg.ravel())
    for s in model.springs:
        i, j = 6 * s.n1 + s.dof - 1, 6 * s.n2 + s.dof - 1
        rows.append([i, j, i, j])
        cols.append([i, j, j, i])
        vals.append([s.k, s.k, -s.k, -s.k])
    if rows:
        K = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(6 * n, 6 * n)).tocsr()
    else:
        K = coo_matrix((6 * n, 6 * n)).tocsr()
    return K


def constraint_map(model: FrameModel):
    """Matrix C (6n x n_red) with u_full = C u_red for rigid links
    (u_slave = u_master + theta_master x r, theta_slave = theta_master)."""
    n = model.n_nodes
    slaves = {s: m for m, s in model.rigid}
    # resolve chains master->slave->slave
    def root(i):
        seen = set()
        while i in slaves and i not in seen:
            seen.add(i)
            i = slaves[i]
        return i
    keep = [d for i in range(n) if i not in slaves for d in range(6 * i, 6 * i + 6)]
    col = {d: k for k, d in enumerate(keep)}
    C = lil_matrix((6 * n, len(keep)))
    for d in keep:
        C[d, col[d]] = 1.0
    for s in slaves:
        m = root(s)
        r = model.nodes[s] - model.nodes[m]
        for a in range(3):
            C[6 * s + a, col[6 * m + a]] = 1.0
            C[6 * s + 3 + a, col[6 * m + 3 + a]] = 1.0
        # theta x r
        rx, ry, rz = r
        C[6 * s + 0, col[6 * m + 4]] += rz
        C[6 * s + 0, col[6 * m + 5]] += -ry
        C[6 * s + 1, col[6 * m + 3]] += -rz
        C[6 * s + 1, col[6 * m + 5]] += rx
        C[6 * s + 2, col[6 * m + 3]] += ry
        C[6 * s + 2, col[6 * m + 4]] += -rx
    return C.tocsr(), keep


def solve(model: FrameModel, case: str) -> FrameResult:
    n = model.n_nodes
    K = assemble(model)
    F = model.load_cases[case].ravel()
    C, keep = constraint_map(model) if model.rigid else (identity(6 * n, format="csr"),
                                                         list(range(6 * n)))
    fixed_full = np.zeros(6 * n, bool)
    for node, dofs in model.supports.items():
        for d in dofs:
            fixed_full[6 * node + d - 1] = True
    col = {d: k for k, d in enumerate(keep)}
    fixed_red = np.zeros(len(keep), bool)
    for d in np.where(fixed_full)[0]:
        if d not in col:
            raise ValueError(f"support on a rigid-link slave dof {d}")
        fixed_red[col[d]] = True
    Kr = (C.T @ K @ C).tocsr()
    Fr = C.T @ F
    free = ~fixed_red
    Kff = Kr[free][:, free].tocsc()
    ur = np.zeros(len(keep))
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        try:
            ur[free] = spsolve(Kff, Fr[free])
        except (MatrixRankWarning, RuntimeError) as exc:
            raise np.linalg.LinAlgError(
                f"singular stiffness - mechanism, check bearings/supports ({exc})") from exc
    if not np.all(np.isfinite(ur)):
        raise np.linalg.LinAlgError("singular stiffness (mechanism - check bearings/supports)")
    # mechanism check: residual of the reduced system
    res = Kff @ ur[free] - Fr[free]
    if np.linalg.norm(res) > 1e-6 * max(np.linalg.norm(Fr), 1.0):
        raise np.linalg.LinAlgError("ill-conditioned / singular stiffness (mechanism?)")
    u = C @ ur
    r = K @ u - F
    reactions = np.where(fixed_full, r, 0.0).reshape(n, 6)

    ends = np.zeros((len(model.beams), 12))
    for e, b in enumerate(model.beams):
        k, T, _ = beam_matrices(model, b)
        dofs = np.r_[6 * b.n1 + np.arange(6), 6 * b.n2 + np.arange(6)]
        ends[e] = k @ (T @ u[dofs])
    # remove the equivalent-load part so end forces are true member forces
    for e, fl in model.element_loads.get(case, {}).items():
        _, T, _ = beam_matrices(model, model.beams[e])
        ends[e] -= T @ fl
    return FrameResult(u.reshape(n, 6), reactions, ends, F.reshape(n, 6))
