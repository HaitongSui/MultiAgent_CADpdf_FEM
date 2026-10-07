"""PDFParserAgent: extracts vector strokes and text from a CAD PDF page."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..core import Agent, Blackboard
from ..models import DrawingData, Stroke, TextItem


def _bezier(p0, p1, p2, p3, n: int) -> list[tuple[float, float]]:
    t = np.linspace(0.0, 1.0, n + 1)[1:, None]
    p0, p1, p2, p3 = (np.asarray(p, float) for p in (p0, p1, p2, p3))
    pts = ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t * t * p2 + t ** 3 * p3
    return [tuple(map(float, p)) for p in pts]


def _flatten_path(path: list, flip, bezier_steps: int) -> list[list[tuple[float, float]]]:
    """Turn a pdfplumber path (m/l/c/h commands) into polylines (subpaths)."""
    subpaths: list[list[tuple[float, float]]] = []
    cur: list[tuple[float, float]] = []
    for cmd in path:
        op = cmd[0]
        if op == "m":
            if len(cur) > 1:
                subpaths.append(cur)
            cur = [flip(cmd[1])]
        elif op == "l":
            p = flip(cmd[1])
            if not cur or math.dist(cur[-1], p) > 1e-9:
                cur.append(p)
        elif op == "c" and cur:
            pts = [flip(p) for p in cmd[1:4]]
            cur.extend(_bezier(cur[-1], *pts, n=bezier_steps))
        elif op == "v" and cur:  # first control point = current point
            pts = [flip(p) for p in cmd[1:3]]
            cur.extend(_bezier(cur[-1], cur[-1], *pts, n=bezier_steps))
        elif op == "y" and cur:  # second control point = end point
            pts = [flip(p) for p in cmd[1:3]]
            cur.extend(_bezier(cur[-1], pts[0], pts[1], pts[1], n=bezier_steps))
        elif op == "h" and cur:
            if math.dist(cur[0], cur[-1]) > 1e-9:
                cur.append(cur[0])
    if len(cur) > 1:
        subpaths.append(cur)
    return subpaths


def _circle_fit(pts: np.ndarray) -> tuple[tuple[float, float], float, float]:
    """Algebraic (Kåsa) circle fit; returns centre, radius, max rel. residual."""
    A = np.c_[2 * pts, np.ones(len(pts))]
    b = (pts ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:2]
    r = math.sqrt(max(sol[2] + c @ c, 0.0))
    resid = np.abs(np.linalg.norm(pts - c, axis=1) - r).max() / max(r, 1e-12)
    return (float(c[0]), float(c[1])), r, float(resid)


def group_text_lines(words: list[TextItem], tol: float = 2.0) -> list[TextItem]:
    """Group words that share a baseline and are close horizontally."""
    lines: list[TextItem] = []
    for w in sorted(words, key=lambda w: (-round(w.y0 / tol), w.x0)):
        if lines:
            last = lines[-1]
            gap = w.x0 - last.x1
            if abs(w.y0 - last.y0) <= tol and -1.0 <= gap <= max(w.size, 4.0) * 1.2:
                last.text += " " + w.text
                last.x1 = max(last.x1, w.x1)
                last.y1 = max(last.y1, w.y1)
                continue
        lines.append(TextItem(w.text, w.x0, w.y0, w.x1, w.y1, w.size))
    return lines


class PDFParserAgent(Agent):
    name = "pdf_parser"
    requires = ("pdf_path",)
    provides = ("drawing",)

    def __init__(self, page: int = 0, bezier_steps: int = 16) -> None:
        super().__init__(page=page, bezier_steps=bezier_steps)

    def run(self, bb: Blackboard) -> None:
        import pdfplumber

        path = Path(bb["pdf_path"])
        with pdfplumber.open(path) as pdf:
            page = pdf.pages[self.params["page"]]
            H = float(page.height)
            flip = lambda p: (float(p[0]), H - float(p[1]))  # noqa: E731  (y-down -> y-up)
            drawing = DrawingData(page_width=float(page.width), page_height=H,
                                  page_index=self.params["page"])

            for obj in [*page.lines, *page.rects, *page.curves]:
                if not obj.get("stroke", True):
                    continue
                raw = obj.get("path") or [("m", obj["pts"][0])] + [("l", p) for p in obj["pts"][1:]]
                is_rect = obj.get("object_type") == "rect"
                for sub in _flatten_path(raw, flip, self.params["bezier_steps"]):
                    closed = math.dist(sub[0], sub[-1]) < 1e-6
                    if is_rect and not closed:
                        sub = sub + [sub[0]]
                        closed = True
                    st = Stroke(points=sub, linewidth=float(obj.get("linewidth") or 0.0),
                                dashed=bool(obj.get("dash") and obj["dash"][0]),
                                closed=closed, color=obj.get("stroking_color"))
                    if closed and len(sub) >= 12 and obj.get("object_type") == "curve":
                        c, r, res = _circle_fit(np.asarray(sub[:-1]))
                        if res < 0.01:
                            # Bézier anchors lie exactly on the circle: refit on them
                            anchors = np.asarray([flip(p) for p in obj.get("pts", [])])
                            if len(anchors) >= 4:
                                c2, r2, res2 = _circle_fit(anchors)
                                if res2 < 1e-4 and abs(r2 - r) < 0.01 * r:
                                    c, r = c2, r2
                            st.circle = (c, r)
                    drawing.strokes.append(st)

            # CAD texts rotated 90 deg read bottom-to-top
            for w in page.extract_words(keep_blank_chars=False, use_text_flow=False,
                                        char_dir_rotated="btt", line_dir_rotated="ltr",
                                        extra_attrs=["size"]):
                drawing.texts.append(TextItem(w["text"], float(w["x0"]), H - float(w["bottom"]),
                                              float(w["x1"]), H - float(w["top"]),
                                              float(w.get("size", 0.0))))
            drawing.lines = group_text_lines(drawing.texts)

        n_circ = sum(1 for s in drawing.strokes if s.circle)
        self.info(bb, f"page {drawing.page_index}: {len(drawing.strokes)} strokes "
                      f"({n_circ} circles), {len(drawing.texts)} words, {len(drawing.lines)} text lines")
        if not drawing.strokes:
            self.error(bb, "no vector geometry found - is this a scanned (raster) PDF? "
                           "Vectorise it first (e.g. Inkscape trace / raster-to-vector).")
        bb.post("drawing", drawing, self.name)
