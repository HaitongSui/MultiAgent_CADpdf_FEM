"""Dependency-light 2D triangular mesher for polygons with holes.

Boundary loops are resampled at the target size (corners preserved), the
interior is filled with an equilateral lattice, the point cloud is Delaunay
triangulated (scipy), triangles outside the domain are dropped and interior
nodes are smoothed.  Good enough for plates/brackets; for production use
plug in gmsh via the same Mesh dataclass.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import Delaunay

from ..models import Geometry, Loop, Mesh
from .geom import distance_to_segments, loop_segments, points_in_polygon


def circle_segments(lp: Loop, h: float, min_segments: int = 16) -> int:
    return max(min_segments, int(math.ceil(2 * math.pi * lp.radius / h)))


def resample_loop(lp: Loop, h: float, corner_deg: float = 25.0, min_segments: int = 16) -> np.ndarray:
    p = lp.points
    n = len(p)
    if lp.is_circle and lp.center is not None and lp.radius:
        m = circle_segments(lp, h, min_segments)
        a0 = math.atan2(p[0, 1] - lp.center[1], p[0, 0] - lp.center[0])
        a = a0 + np.linspace(0, 2 * math.pi, m, endpoint=False)
        return np.c_[lp.center[0] + lp.radius * np.cos(a), lp.center[1] + lp.radius * np.sin(a)]

    # corners = vertices with a significant turn
    prev, nxt = np.roll(p, 1, axis=0), np.roll(p, -1, axis=0)
    u, v = p - prev, nxt - p
    turn = np.degrees(np.abs(np.arctan2(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0],
                                        (u * v).sum(axis=1))))
    corners = list(np.where(turn > corner_deg)[0])
    if not corners:
        corners = [0]
    out = []
    for ci, c in enumerate(corners):
        c2 = corners[(ci + 1) % len(corners)]
        idx = list(range(c, c2 + 1)) if c2 > c else list(range(c, n)) + list(range(0, c2 + 1))
        chain = p[idx]
        seg = np.linalg.norm(np.diff(chain, axis=0), axis=1)
        s = np.r_[0.0, np.cumsum(seg)]
        L = s[-1]
        k = max(1, int(round(L / h)))
        t = np.linspace(0.0, L, k + 1)[:-1]
        out.append(np.c_[np.interp(t, s, chain[:, 0]), np.interp(t, s, chain[:, 1])])
    return np.vstack(out)


def default_mesh_size(g: Geometry) -> float:
    x0, y0, x1, y1 = g.bbox
    h = min(x1 - x0, y1 - y0) / 12.0
    for hole in g.holes:
        per = float(np.linalg.norm(np.roll(hole.points, -1, axis=0) - hole.points, axis=1).sum())
        h = min(h, per / 16.0)
    return h


def hole_rings(lp: Loop, h: float, min_segments: int, growth: float = 1.25):
    """Concentric point rings around a small circular hole, growing from the
    hole's edge spacing to ``h``.  Returns (points, local spacing, radius of
    the graded zone)."""
    m = circle_segments(lp, h, min_segments)
    r, c = lp.radius, np.asarray(lp.center)
    s = 2 * math.pi * r / m
    rk, pts, spacing = r, [], []
    k = 0
    while s < 0.95 * h:
        rk += s * math.sqrt(3) / 2
        s = min(s * growth, h)
        k += 1
        mk = max(m, int(round(2 * math.pi * rk / s)))
        a = (k % 2) * math.pi / mk + np.linspace(0, 2 * math.pi, mk, endpoint=False)
        pts.append(np.c_[c[0] + rk * np.cos(a), c[1] + rk * np.sin(a)])
        spacing.append(np.full(mk, 2 * math.pi * rk / mk))
    if not pts:
        return np.zeros((0, 2)), np.zeros(0), 0.0
    return np.vstack(pts), np.concatenate(spacing), rk


def triangulate(g: Geometry, h: float, smooth_iters: int = 6, min_hole_segments: int = 16) -> Mesh:
    bnd_loops = [resample_loop(lp, h, min_segments=min_hole_segments) for lp in g.loops]
    loop_id = np.concatenate([np.full(len(b), i) for i, b in enumerate(bnd_loops)])
    bnd = np.vstack(bnd_loops)

    # equilateral lattice
    x0, y0, x1, y1 = g.bbox
    dy = h * math.sqrt(3) / 2
    ys = np.arange(y0 + dy / 2, y1, dy)
    pts = []
    for j, y in enumerate(ys):
        off = (h / 2) * (j % 2)
        xs = np.arange(x0 + off + h / 4, x1, h)
        pts.append(np.c_[xs, np.full_like(xs, y)])
    interior = np.vstack(pts) if pts else np.zeros((0, 2))
    spacing = np.full(len(interior), h)

    # graded rings around small circular holes; the lattice is cleared there
    for lp in g.holes:
        if not (lp.is_circle and lp.radius):
            continue
        rp, rs, rz = hole_rings(lp, h, min_hole_segments)
        if not len(rp):
            continue
        keep = np.linalg.norm(interior - np.asarray(lp.center), axis=1) > rz + 0.5 * h
        interior, spacing = np.vstack([interior[keep], rp]), np.r_[spacing[keep], rs]

    if len(interior):
        ok = _inside(interior, g)
        interior, spacing = interior[ok], spacing[ok]
    if len(interior):
        a, b = loop_segments(bnd_loops)
        interior = interior[distance_to_segments(interior, a, b) > 0.55 * spacing]

    nodes = np.vstack([bnd, interior])
    tri = Delaunay(nodes).simplices
    cent = nodes[tri].mean(axis=1)
    tri = tri[_inside(cent, g)]
    # orient CCW, drop degenerate
    area = _areas(nodes, tri)
    tri[area < 0] = tri[area < 0][:, [0, 2, 1]]
    tri = tri[np.abs(area) > 1e-10 * h * h]

    n_bnd = len(bnd)
    nodes = _smooth(nodes, tri, n_bnd, smooth_iters)
    used = np.unique(tri)
    remap = -np.ones(len(nodes), dtype=int)
    remap[used] = np.arange(len(used))
    mesh = Mesh(nodes=nodes[used], elements=remap[tri], order=1)
    mesh.boundary_loop = {int(remap[i]): int(loop_id[i]) for i in range(n_bnd) if remap[i] >= 0}
    return mesh


def _inside(pts: np.ndarray, g: Geometry) -> np.ndarray:
    ok = points_in_polygon(pts, g.outer.points)
    for hole in g.holes:
        ok &= ~points_in_polygon(pts, hole.points)
    return ok


def _areas(nodes: np.ndarray, tri: np.ndarray) -> np.ndarray:
    p = nodes[tri]
    a, b = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    return 0.5 * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])


def _smooth(nodes: np.ndarray, tri: np.ndarray, n_fixed: int, iters: int) -> np.ndarray:
    if iters <= 0 or len(nodes) <= n_fixed:
        return nodes
    n = len(nodes)
    edges = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    edges = np.vstack([edges, edges[:, ::-1]])
    deg = np.bincount(edges[:, 0], minlength=n).astype(float)
    free = np.arange(n) >= n_fixed
    free &= deg > 0
    for _ in range(iters):
        s = np.zeros_like(nodes)
        np.add.at(s, edges[:, 0], nodes[edges[:, 1]])
        new = nodes.copy()
        new[free] = s[free] / deg[free, None]
        if (_areas(new, tri) <= 0).any():
            break
        nodes = new
    return nodes


def to_quadratic(mesh: Mesh, g: Geometry) -> Mesh:
    """Insert mid-side nodes (Tri6).  Mid-nodes on circular holes are
    projected onto the circle."""
    nodes = [*mesh.nodes]
    mid: dict[tuple[int, int], int] = {}
    elems = []
    circles = {i: lp for i, lp in enumerate(g.loops) if lp.is_circle and lp.center is not None}
    bl = dict(mesh.boundary_loop)
    on_boundary = {tuple(sorted(mesh.face_nodes(e, f))) for e, f in mesh.boundary_edges()}
    for tri in mesh.elements:
        m = []
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (min(a, b), max(a, b))
            if key not in mid:
                p = (mesh.nodes[a] + mesh.nodes[b]) / 2
                la = bl.get(int(a))
                if key in on_boundary and la in circles and la == bl.get(int(b)):
                    c, r = np.asarray(circles[la].center), circles[la].radius
                    v = p - c
                    p = c + v / max(np.linalg.norm(v), 1e-300) * r
                mid[key] = len(nodes)
                nodes.append(p)
            m.append(mid[key])
        elems.append([*tri, *m])
    out = Mesh(nodes=np.asarray(nodes), elements=np.asarray(elems, dtype=int), order=2,
               boundary_loop=bl)
    # mid-side nodes of boundary edges inherit the loop id
    for (a, b), k in mid.items():
        if (a, b) in on_boundary and a in bl:
            out.boundary_loop[k] = bl[a]
    return out
