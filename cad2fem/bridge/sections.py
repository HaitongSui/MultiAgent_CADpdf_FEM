"""Cross-section properties of arbitrary (multi-cell) polygonal sections.

Area and second moments are exact for the polygon (Green's theorem).  The
Saint-Venant torsion constant is computed with a finite-element solution of
the warping function (Laplace equation with Neumann boundary data) on a
triangular mesh, so it handles solid, open and multi-cell closed sections.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from ..fem.geom import distance_to_segments
from ..fem.mesher import triangulate
from ..models import Geometry, Loop, polygon_area
from .models import SectionProps


def _loop_integrals(p: np.ndarray) -> np.ndarray:
    """[A, Sx(=int y), Sy(=int x), Ixx(=int y^2), Iyy(=int x^2), Ixy] of a CCW polygon."""
    x, y = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    c = x * y1 - x1 * y
    A = c.sum() / 2
    Sy = ((x + x1) * c).sum() / 6
    Sx = ((y + y1) * c).sum() / 6
    Iyy = ((x * x + x * x1 + x1 * x1) * c).sum() / 12
    Ixx = ((y * y + y * y1 + y1 * y1) * c).sum() / 12
    Ixy = ((x * y1 + 2 * x * y + 2 * x1 * y1 + x1 * y) * c).sum() / 24
    return np.array([A, Sx, Sy, Ixx, Iyy, Ixy])


def polygon_properties(g: Geometry) -> dict[str, float]:
    tot = np.zeros(6)
    for k, lp in enumerate(g.loops):
        p = lp.points if polygon_area(lp.points) > 0 else lp.points[::-1]
        tot += _loop_integrals(p) * (1 if k == 0 else -1)
    A, Sx, Sy, Ixx, Iyy, Ixy = tot
    cx, cy = Sy / A, Sx / A
    return {"A": A, "cx": cx, "cy": cy,
            "Ixx": Ixx - A * cy * cy,        # about the horizontal centroidal axis
            "Iyy": Iyy - A * cx * cx,        # about the vertical centroidal axis
            "Ixy": Ixy - A * cx * cy}


def min_wall_thickness(g: Geometry) -> float:
    """Smallest distance between two different boundary loops (or the section
    size when there are no holes)."""
    x0, y0, x1, y1 = g.bbox
    t = min(x1 - x0, y1 - y0)
    loops = g.loops
    for i, a in enumerate(loops):
        for b in loops[i + 1:]:
            t = min(t, float(distance_to_segments(b.points, a.points, np.roll(a.points, -1, 0)).min()))
    return t


def torsion_constant(g: Geometry, cx: float, cy: float, h: float | None = None) -> tuple[float, int]:
    """Saint-Venant J from the warping function (CST elements).  Returns (J, n_elements)."""
    if h is None:
        x0, y0, x1, y1 = g.bbox
        h = min(min_wall_thickness(g) / 4.0, min(x1 - x0, y1 - y0) / 25.0)
    shifted = Geometry(
        outer=Loop(g.outer.points - (cx, cy), g.outer.is_circle,
                   None if g.outer.center is None else (g.outer.center[0] - cx, g.outer.center[1] - cy),
                   g.outer.radius),
        holes=[Loop(lp.points - (cx, cy), lp.is_circle,
                    None if lp.center is None else (lp.center[0] - cx, lp.center[1] - cy), lp.radius)
               for lp in g.holes])
    mesh = triangulate(shifted, h, smooth_iters=4, min_hole_segments=24)
    xy = mesh.nodes
    tri = mesh.elements
    n = len(xy)
    p = xy[tri]
    b = np.stack([p[:, 1, 1] - p[:, 2, 1], p[:, 2, 1] - p[:, 0, 1], p[:, 0, 1] - p[:, 1, 1]], 1)
    c = np.stack([p[:, 2, 0] - p[:, 1, 0], p[:, 0, 0] - p[:, 2, 0], p[:, 1, 0] - p[:, 0, 0]], 1)
    area = 0.5 * (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0])
    ke = (b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :]) / (4 * area)[:, None, None]
    rows = np.repeat(tri, 3, axis=1).ravel()
    cols = np.tile(tri, (1, 3)).ravel()
    K = coo_matrix((ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()

    # Neumann data d(omega)/dn = y n_x - x n_y on every boundary edge
    f = np.zeros(n)
    for e, face in mesh.boundary_edges():
        i, j = mesh.face_nodes(e, face)
        d = xy[j] - xy[i]
        L = math.hypot(*d)
        nx, ny = d[1] / L, -d[0] / L                    # outward normal (CCW elements)
        gi = xy[i, 1] * nx - xy[i, 0] * ny
        gj = xy[j, 1] * nx - xy[j, 0] * ny
        f[i] += L * (2 * gi + gj) / 6
        f[j] += L * (gi + 2 * gj) / 6
    keep = np.arange(1, n)                               # pin omega at node 0
    w = np.zeros(n)
    w[keep] = spsolve(K[keep][:, keep].tocsc(), f[keep])

    dwdx = (b * w[tri]).sum(1) / (2 * area)
    dwdy = (c * w[tri]).sum(1) / (2 * area)
    xc, yc = p[:, :, 0].mean(1), p[:, :, 1].mean(1)
    # J = int (x^2 + y^2 + x dw/dy - y dw/dx) dA ; exact polar moment for the first part
    xx = (p[:, :, 0] ** 2).sum(1) + (p[:, :, 0].sum(1)) ** 2
    yy = (p[:, :, 1] ** 2).sum(1) + (p[:, :, 1].sum(1)) ** 2
    Ip = (area * (xx + yy) / 12).sum()
    J = Ip + (area * (xc * dwdy - yc * dwdx)).sum()
    return float(J), len(tri)


def section_from_geometry(name: str, g_m: Geometry, compute_torsion: bool = True) -> SectionProps:
    """``g_m`` in metres; drawing x -> local y, drawing y -> local z."""
    pp = polygon_properties(g_m)
    x0, y0, x1, y1 = g_m.bbox
    J = torsion_constant(g_m, pp["cx"], pp["cy"])[0] if compute_torsion else float("nan")
    return SectionProps(
        name=name, A=pp["A"], Iyy=pp["Ixx"], Izz=pp["Iyy"], Iyz=pp["Ixy"], J=J,
        width=x1 - x0, height=y1 - y0, cy=pp["cx"] - x0, cz=pp["cy"] - y0,
        n_cells=len(g_m.holes), outline=g_m.outer.points.copy(),
        holes=[h.points.copy() for h in g_m.holes])


def rectangle_section(name: str, b: float, h: float) -> SectionProps:
    """Solid rectangle b (local y) x h (local z) with the series torsion formula."""
    a, c = max(b, h), min(b, h)
    beta = 1 / 3 - 0.21 * (c / a) * (1 - (c / a) ** 4 / 12)
    return SectionProps(name=name, A=b * h, Iyy=b * h ** 3 / 12, Izz=h * b ** 3 / 12, Iyz=0.0,
                        J=beta * a * c ** 3, width=b, height=h, cy=b / 2, cz=h / 2,
                        source="default")
