"""GeometryAgent: recognises the part outline and holes among the strokes.

Strategy
--------
1. Keep *outline* strokes: solid (not dashed) and thick.  CAD convention
   draws visible edges with a heavier pen than dimension, hatch and centre
   lines, so a two-class split of line widths separates them.
2. Node all segments (split at intersections) and build a planar graph.
3. For every connected component take its outer boundary (the unbounded
   face).  Components that contain the sheet frame or title-block keywords
   are discarded.
4. The largest remaining loop is the part; loops inside it are holes.
"""

from __future__ import annotations

import re

import numpy as np

from ..core import Agent, Blackboard
from ..fem.geom import PlanarGraph, ccw, points_in_polygon, simplify_collinear, split_segments
from ..models import Geometry, Loop, polygon_area

TITLE_KEYWORDS = re.compile(
    r"\b(TITLE|SCALE|DRAWN|CHECKED|APPROVED|DATE|SHEET|DWG|REV|MATERIAL|WEIGHT|PROJECTION)\b|"
    r"图号|比例|材料|设计|审核|日期|标题", re.I)


def _outline_width_threshold(widths: np.ndarray) -> float:
    """Split stroke widths into thin/thick classes; return the threshold."""
    w = np.unique(np.round(widths, 3))
    if len(w) < 2 or w.max() < 1.4 * max(w.min(), 1e-6):
        return 0.0
    gaps = np.diff(w) / np.maximum(w[:-1], 1e-6)
    k = int(np.argmax(gaps))
    return float((w[k] + w[k + 1]) / 2.0)


class GeometryAgent(Agent):
    name = "geometry"
    requires = ("drawing",)
    provides = ("part_pt",)

    def __init__(self, min_outline_width: float | None = None, snap_tol: float = 0.5,
                 min_hole_frac: float = 1e-4) -> None:
        super().__init__(min_outline_width=min_outline_width, snap_tol=snap_tol,
                         min_hole_frac=min_hole_frac)

    def run(self, bb: Blackboard) -> None:
        d = bb["drawing"]
        tol = self.params["snap_tol"]
        solid = [s for s in d.strokes if not s.dashed]
        if not solid:
            raise ValueError("no solid strokes in drawing")
        thr = self.params["min_outline_width"]
        if thr is None:
            thr = _outline_width_threshold(np.array([s.linewidth for s in solid]))
        outline = [s for s in solid if s.linewidth >= thr - 1e-9]
        self.info(bb, f"{len(outline)}/{len(d.strokes)} strokes classified as outline "
                      f"(width >= {thr:.3g} pt)")

        page_area = d.page_width * d.page_height
        keyword_pts = np.array([t.center for t in d.lines if TITLE_KEYWORDS.search(t.text)]
                               or np.zeros((0, 2)))

        # ---- polylines -> planar graph -> component outer boundaries ------
        circles = [s for s in outline if s.circle]
        segs = [np.array([p0, p1]) for s in outline if not s.circle
                for p0, p1 in zip(s.points[:-1], s.points[1:])]
        loops: list[Loop] = []
        excluded: list[np.ndarray] = []
        if segs:
            graph = PlanarGraph(split_segments(np.array(segs), tol), tol)
            for comp in graph.components():
                faces = graph.faces(comp)
                polys = [graph.polygon(f) for f in faces]
                polys = [p for p in polys if len(p) >= 3]
                if not polys:
                    continue
                areas = [polygon_area(p) for p in polys]
                outer = polys[int(np.argmin(areas))]
                outer_area = abs(min(areas))
                if outer_area < 1e-6:
                    continue  # open chains (leader lines etc.)
                frame = outer_area > 0.5 * page_area
                titled = any(len(keyword_pts) and points_in_polygon(keyword_pts, p).any()
                             for p, a in zip(polys, areas) if a > 0 and abs(a) < 0.5 * page_area)
                if frame or titled:
                    excluded.extend(p for p, a in zip(polys, areas) if 0 < a < 0.5 * page_area)
                    continue
                n_inner = sum(1 for a in areas if a > 0)
                if n_inner > 1:
                    self.info(bb, f"component with {n_inner} internal faces merged to its outer "
                                  "boundary (internal feature lines ignored)")
                loops.append(Loop(simplify_collinear(ccw(outer[::-1].copy()))))
        for s in circles:
            c, r = s.circle
            if any(points_in_polygon(np.array([c]), p)[0] for p in excluded):
                continue
            pts = np.asarray(s.points[:-1])
            loops.append(Loop(ccw(pts), is_circle=True, center=c, radius=r))

        if not loops:
            raise ValueError("no closed outline found on the drawing")

        # ---- part + holes ---------------------------------------------------
        loops.sort(key=lambda lp: lp.area, reverse=True)
        part, rest = loops[0], loops[1:]
        holes, ignored = [], 0
        for lp in rest:
            inside = points_in_polygon(lp.points, part.points).all()
            if not inside:
                ignored += 1
                continue
            if lp.area < self.params["min_hole_frac"] * part.area:
                continue
            if any(points_in_polygon(lp.points[:1], h.points)[0] for h in holes):
                self.warn(bb, "loop nested inside a hole ignored (e.g. thread/chamfer line)")
                continue
            holes.append(lp)
        if ignored:
            self.warn(bb, f"{ignored} closed loop(s) outside the main outline ignored "
                          "(other views / details?) - only the largest view is modelled")
        self.info(bb, f"part outline with {len(part.points)} vertices, {len(holes)} hole(s)")
        bb.post("part_pt", Geometry(outer=part, holes=holes, scale_mm_per_pt=1.0), self.name)
