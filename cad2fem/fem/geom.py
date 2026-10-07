"""Planar geometry helpers: segment noding, face extraction, containment."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from ..models import polygon_area


# --------------------------------------------------------------------------- #
# containment / distance
# --------------------------------------------------------------------------- #
def points_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised even-odd ray casting.  ``poly`` is (n, 2), not closed."""
    pts = np.atleast_2d(pts)
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x1, y1 = poly[:, 0][None, :], poly[:, 1][None, :]
    x2, y2 = np.roll(poly[:, 0], -1)[None, :], np.roll(poly[:, 1], -1)[None, :]
    cond = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    return (cond & (x < xin)).sum(axis=1) % 2 == 1


def distance_to_segments(pts: np.ndarray, a: np.ndarray, b: np.ndarray,
                         chunk: int = 4096) -> np.ndarray:
    """Minimum distance from each point to a set of segments a[i]-b[i]."""
    out = np.empty(len(pts))
    ab = b - a
    ab2 = np.maximum((ab ** 2).sum(axis=1), 1e-300)
    for s in range(0, len(pts), chunk):
        p = pts[s:s + chunk, None, :]
        t = np.clip(((p - a[None]) * ab[None]).sum(axis=2) / ab2[None], 0.0, 1.0)
        proj = a[None] + t[..., None] * ab[None]
        out[s:s + chunk] = np.sqrt(((p - proj) ** 2).sum(axis=2)).min(axis=1)
    return out


def loop_segments(loops: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    a = np.vstack(loops)
    b = np.vstack([np.roll(lp, -1, axis=0) for lp in loops])
    return a, b


# --------------------------------------------------------------------------- #
# noding: split segments at mutual intersections
# --------------------------------------------------------------------------- #
def split_segments(segs: np.ndarray, tol: float) -> np.ndarray:
    """Split segments (n, 2, 2) at every intersection / T-junction."""
    n = len(segs)
    if n == 0:
        return segs
    p, r = segs[:, 0], segs[:, 1] - segs[:, 0]
    lo, hi = segs.min(axis=1) - tol, segs.max(axis=1) + tol
    splits: list[list[float]] = [[0.0, 1.0] for _ in range(n)]
    lengths = np.linalg.norm(r, axis=1)
    for i in range(n - 1):
        j = np.arange(i + 1, n)
        j = j[np.all(lo[j] <= hi[i], axis=1) & np.all(hi[j] >= lo[i], axis=1)]
        if j.size == 0:
            continue
        s = r[j]
        denom = r[i, 0] * s[:, 1] - r[i, 1] * s[:, 0]
        qp = p[j] - p[i]
        ok = np.abs(denom) > 1e-12 * lengths[i] * lengths[j]
        if not ok.any():
            continue
        j, s, qp, denom = j[ok], s[ok], qp[ok], denom[ok]
        t = (qp[:, 0] * s[:, 1] - qp[:, 1] * s[:, 0]) / denom
        u = (qp[:, 0] * r[i, 1] - qp[:, 1] * r[i, 0]) / denom
        ti = tol / max(lengths[i], 1e-12)
        tj = tol / np.maximum(lengths[j], 1e-12)
        hit = (t >= -ti) & (t <= 1 + ti) & (u >= -tj) & (u <= 1 + tj)
        for jj, tt, uu in zip(j[hit], t[hit], u[hit]):
            if ti < tt < 1 - ti:
                splits[i].append(float(tt))
            if tol / lengths[jj] < uu < 1 - tol / lengths[jj]:
                splits[jj].append(float(uu))
    out = []
    for k in range(n):
        ts = sorted(set(splits[k]))
        for t0, t1 in zip(ts[:-1], ts[1:]):
            out.append([p[k] + t0 * r[k], p[k] + t1 * r[k]])
    return np.asarray(out).reshape(-1, 2, 2)


# --------------------------------------------------------------------------- #
# planar graph and faces
# --------------------------------------------------------------------------- #
class PlanarGraph:
    def __init__(self, segs: np.ndarray, tol: float) -> None:
        pts = segs.reshape(-1, 2)
        tree = cKDTree(pts)
        parent = np.arange(len(pts))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a, b in tree.query_pairs(tol):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        roots = np.array([find(i) for i in range(len(pts))])
        uniq, inv = np.unique(roots, return_inverse=True)
        self.vertices = np.array([pts[roots == u].mean(axis=0) for u in uniq])
        idx = inv.reshape(-1, 2)
        edges = {(min(a, b), max(a, b)) for a, b in idx if a != b}
        self.edges = sorted(edges)
        self.adj: dict[int, list[int]] = defaultdict(list)
        for a, b in self.edges:
            self.adj[a].append(b)
            self.adj[b].append(a)
        for v, nb in self.adj.items():
            d = self.vertices[nb] - self.vertices[v]
            ang = np.arctan2(d[:, 1], d[:, 0])
            self.adj[v] = [nb[k] for k in np.argsort(ang)]

    def components(self) -> list[set[int]]:
        seen: set[int] = set()
        comps = []
        for v in self.adj:
            if v in seen:
                continue
            stack, comp = [v], set()
            while stack:
                u = stack.pop()
                if u in comp:
                    continue
                comp.add(u)
                stack.extend(self.adj[u])
            seen |= comp
            comps.append(comp)
        return comps

    def faces(self, comp: set[int]) -> list[list[int]]:
        """All faces of a connected component (bounded faces are CCW,
        the unbounded face is CW)."""
        used: set[tuple[int, int]] = set()
        faces = []
        for u in comp:
            for v in self.adj[u]:
                if (u, v) in used:
                    continue
                face = []
                a, b = u, v
                while (a, b) not in used:
                    used.add((a, b))
                    face.append(a)
                    nb = self.adj[b]
                    k = nb.index(a)
                    a, b = b, nb[(k - 1) % len(nb)]
                faces.append(face)
        return faces

    def polygon(self, face: list[int]) -> np.ndarray:
        return self.vertices[remove_spikes(face)]


def remove_spikes(face: list[int]) -> list[int]:
    """Remove dangling back-and-forth excursions (A,B,A) from a face cycle."""
    f = list(face)
    changed = True
    while changed and len(f) > 2:
        changed = False
        n = len(f)
        for i in range(n):
            if f[(i - 1) % n] == f[(i + 1) % n]:
                for k in sorted({i, (i + 1) % n}, reverse=True):
                    del f[k]
                changed = True
                break
    return f


def simplify_collinear(poly: np.ndarray, angle_tol_deg: float = 0.5) -> np.ndarray:
    """Drop vertices where the polygon goes straight on."""
    if len(poly) < 4:
        return poly
    keep = []
    n = len(poly)
    for i in range(n):
        a, b, c = poly[i - 1], poly[i], poly[(i + 1) % n]
        u, v = b - a, c - b
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu < 1e-12 or nv < 1e-12:
            continue
        cross = (u[0] * v[1] - u[1] * v[0]) / (nu * nv)
        dot = (u @ v) / (nu * nv)
        if abs(math.degrees(math.atan2(cross, dot))) > angle_tol_deg:
            keep.append(i)
    return poly[keep] if len(keep) >= 3 else poly


def ccw(poly: np.ndarray) -> np.ndarray:
    return poly if polygon_area(poly) > 0 else poly[::-1].copy()
