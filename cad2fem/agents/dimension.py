"""DimensionAgent: determines the drawing scale and converts the part to mm.

Two independent sources are reconciled:

* the title-block scale ``1:N`` (paper mm -> real mm), and
* the dimension texts: each value is matched against measured lengths of the
  outline (bounding box, straight edges, hole diameters/radii); a ratio that
  many dimensions agree on is a reliable calibration.

The dimension calibration wins when the two disagree (drawings are often
marked "NOT TO SCALE" or re-plotted at a different paper size).
"""

from __future__ import annotations

import math
import re
from typing import Callable, Optional

import numpy as np

from ..core import Agent, Blackboard
from ..models import Geometry, Loop
from .annotation import UNIT_TO_MM

PT_TO_MM = 25.4 / 72.0
DIM_RE = re.compile(r"^(?:(\d+)\s*[xX×]\s*)?([ØⲪ⌀∅φΦR]?)\s*(\d+(?:[.,]\d+)?)\s*(?:±\s*[\d.]+|[HhGg]\d+)?$")


def _candidates(g: Geometry, min_frac: float = 0.03) -> dict[str, list[float]]:
    """Measurable lengths of the part (PDF points), by dimension type."""
    xmin, ymin, xmax, ymax = g.bbox
    size = max(xmax - xmin, ymax - ymin)
    lin = [xmax - xmin, ymax - ymin]
    for lp in g.loops:
        if lp.is_circle:
            continue
        p = lp.points
        e = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
        lin.extend(float(v) for v in e)          # straight edges
        bx = lp.bbox
        lin.extend([bx[2] - bx[0], bx[3] - bx[1]])
    for h in g.holes:                             # hole positions from the part edges
        c = np.mean(h.points, axis=0)
        lin.extend([c[0] - xmin, c[1] - ymin, xmax - c[0], ymax - c[1]])
    # ignore facets of flattened arcs and other tiny lengths
    lin = [v for v in lin if v > min_frac * size]
    dia = [2 * h.radius for h in g.loops if h.is_circle and h.radius]
    return {"lin": lin, "dia": dia, "rad": [d / 2 for d in dia]}


def dimension_ratios(texts: list[str], g: Geometry, unit_mm: float) -> list[tuple[float, int]]:
    """All (mm-per-pt ratio, text index) pairs a dimension text could imply."""
    cands = _candidates(g)
    ratios: list[tuple[float, int]] = []
    for k, t in enumerate(texts):
        m = DIM_RE.match(t.strip())
        if not m:
            continue
        val = float(m.group(3).replace(",", ".")) * unit_mm
        kind = {"": "lin", "R": "rad"}.get(m.group(2), "dia")
        ratios.extend((val / L, k) for L in cands[kind] if L > 1e-9)
    return ratios


def support(ratios: list[tuple[float, int]], r: float, rel_tol: float) -> int:
    """Number of distinct dimension texts consistent with ratio ``r``."""
    return len({k for rr, k in ratios if abs(rr - r) <= rel_tol * r})


def calibrate(texts: list[str], g: Geometry, unit_mm: float, rel_tol: float = 0.01):
    """Best-supported scale: returns (mm_per_pt, n_supporting_texts) or (None, 0)."""
    ratios = dimension_ratios(texts, g, unit_mm)
    best, best_n = None, 0
    for r, _ in ratios:
        n = support(ratios, r, rel_tol)
        if n > best_n:
            best_n = n
            best = float(np.median([rr for rr, _ in ratios if abs(rr - r) <= rel_tol * r]))
    return best, best_n


def transform(g: Geometry, s: float) -> Geometry:
    xmin, ymin = g.bbox[0], g.bbox[1]
    origin = np.array([xmin, ymin])

    def tr(lp: Loop) -> Loop:
        c = None
        if lp.center is not None:
            c = tuple(((np.asarray(lp.center) - origin) * s).tolist())
        return Loop((lp.points - origin) * s, lp.is_circle, c,
                    lp.radius * s if lp.radius else None)

    holes = sorted((tr(h) for h in g.holes),
                   key=lambda h: (round(float(h.points[:, 0].mean()), 3),
                                  float(h.points[:, 1].mean())))
    return Geometry(outer=tr(g.outer), holes=holes, scale_mm_per_pt=s)


def scale_part(g: Geometry, texts: list[str], unit_mm: float, title_scale: Optional[float],
               min_support: int = 2, rel_tol: float = 0.01,
               log: Callable[[str, str], None] = lambda level, text: None) -> Geometry:
    """Convert a part from PDF points to mm (origin at its lower-left corner)
    using the title-block scale ``1:title_scale`` and/or the dimension texts."""
    dim_s, n_dim = calibrate(texts, g, unit_mm, rel_tol)
    title_s = PT_TO_MM * title_scale if title_scale else None
    n_title = support(dimension_ratios(texts, g, unit_mm), title_s, rel_tol) if title_s else 0

    if title_s and (n_title >= n_dim or n_dim < min_support):
        s = title_s
        log("INFO", f"title-block scale 1:{title_scale:g} ({n_title} dimension(s) agree)")
    elif dim_s and n_dim >= min_support:
        if title_s:
            log("WARNING", f"title-block scale 1:{title_scale:g} disagrees with dimensions "
                           f"(1:{dim_s / PT_TO_MM:.4g} from {n_dim} dims vs {n_title}) "
                           "- using dimensions")
        else:
            log("INFO", f"scale 1:{dim_s / PT_TO_MM:.4g} derived from {n_dim} dimension(s)")
        s = dim_s
    else:
        s = PT_TO_MM
        log("WARNING", "no scale information - assuming 1:1 on paper")
    if math.isclose(s, 0.0):
        raise ValueError("degenerate scale")
    return transform(g, s)


class DimensionAgent(Agent):
    name = "dimension"
    requires = ("part_pt", "spec", "drawing")
    provides = ("geometry",)

    def __init__(self, min_support: int = 2, rel_tol: float = 0.01) -> None:
        super().__init__(min_support=min_support, rel_tol=rel_tol)

    def run(self, bb: Blackboard) -> None:
        spec = bb["spec"]
        geo = scale_part(bb["part_pt"], [t.text for t in bb["drawing"].lines],
                         UNIT_TO_MM.get(spec.units, 1.0), spec.scale,
                         self.params["min_support"], self.params["rel_tol"],
                         log=lambda level, text: bb.say(self.name, text, level))
        x0, y0, x1, y1 = geo.bbox
        self.info(bb, f"part {x1 - x0:.4g} x {y1 - y0:.4g} mm, area {geo.area:.6g} mm^2, "
                      f"{len(geo.holes)} hole(s)")
        bb.post("geometry", geo, self.name)
