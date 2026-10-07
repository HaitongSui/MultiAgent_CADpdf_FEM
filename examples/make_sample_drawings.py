"""Generate sample vector CAD drawings (PDF) for trying out cad2fem.

    python examples/make_sample_drawings.py examples/drawings

Each sheet follows ordinary drafting practice: thick visible outlines, thin
dimension lines with arrowheads, dashed centre lines, a sheet frame and a
title block with scale/material/thickness, plus analysis notes.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas

MM = 72.0 / 25.4          # points per paper millimetre
THICK, THIN = 0.7 * MM / 2.0, 0.25 * MM / 2.0   # ~0.99 pt and ~0.35 pt pens


class Sheet:
    def __init__(self, path: Path, scale: float, origin_mm=(40.0, 60.0)) -> None:
        self.c = canvas.Canvas(str(path), pagesize=landscape(A4))
        self.W, self.H = landscape(A4)
        self.scale = scale
        self.ox, self.oy = origin_mm[0] * MM, origin_mm[1] * MM

    # model mm -> page points
    def P(self, x: float, y: float) -> tuple[float, float]:
        return self.ox + x / self.scale * MM, self.oy + y / self.scale * MM

    def pen(self, w: float, dash=None) -> None:
        self.c.setLineWidth(w)
        self.c.setDash(*(dash or ([], 0)))

    def polyline(self, pts, close=True) -> None:
        self.pen(THICK)
        p = self.c.beginPath()
        p.moveTo(*self.P(*pts[0]))
        for q in pts[1:]:
            p.lineTo(*self.P(*q))
        if close:
            p.close()
        self.c.drawPath(p, stroke=1, fill=0)

    def circle(self, x, y, r) -> None:
        self.pen(THICK)
        self.c.circle(*self.P(x, y), r / self.scale * MM, stroke=1, fill=0)
        self.centre_mark(x, y, r)

    def centre_mark(self, x, y, r) -> None:
        self.pen(THIN, ([6, 2, 1, 2], 0))
        e = r * 1.3
        self.c.line(*self.P(x - e, y), *self.P(x + e, y))
        self.c.line(*self.P(x, y - e), *self.P(x, y + e))

    def _arrow(self, tip, direction) -> None:
        dx, dy = direction
        L, w = 3 * MM, 0.9 * MM
        bx, by = tip[0] - dx * L, tip[1] - dy * L
        p = self.c.beginPath()
        p.moveTo(*tip)
        p.lineTo(bx - dy * w, by + dx * w)
        p.lineTo(bx + dy * w, by - dx * w)
        p.close()
        self.c.drawPath(p, stroke=0, fill=1)

    def dim_linear(self, a, b, offset, text=None) -> None:
        """Aligned dimension between model points a and b, offset in paper mm."""
        pa, pb = self.P(*a), self.P(*b)
        dx, dy = pb[0] - pa[0], pb[1] - pa[1]
        L = math.hypot(dx, dy)
        ux, uy = dx / L, dy / L
        nx, ny = -uy, ux
        o = offset * MM
        qa = (pa[0] + nx * o, pa[1] + ny * o)
        qb = (pb[0] + nx * o, pb[1] + ny * o)
        self.pen(THIN)
        ext = 2 * MM * (1 if o > 0 else -1)
        self.c.line(pa[0] + nx * MM * (1 if o > 0 else -1), pa[1] + ny * MM * (1 if o > 0 else -1),
                    qa[0] + nx * ext, qa[1] + ny * ext)
        self.c.line(pb[0] + nx * MM * (1 if o > 0 else -1), pb[1] + ny * MM * (1 if o > 0 else -1),
                    qb[0] + nx * ext, qb[1] + ny * ext)
        self.c.line(*qa, *qb)
        self._arrow(qa, (-ux, -uy))
        self._arrow(qb, (ux, uy))
        value = text or f"{math.dist(a, b):g}"
        mx, my = (qa[0] + qb[0]) / 2, (qa[1] + qb[1]) / 2
        self.c.saveState()
        self.c.translate(mx + nx * 1.2 * MM, my + ny * 1.2 * MM)
        ang = math.degrees(math.atan2(uy, ux))
        if ang > 90.0 or ang <= -90.0:   # keep dimension text readable
            ang -= math.copysign(180.0, ang)
        self.c.rotate(ang)
        self.c.setFont("Helvetica", 9)
        self.c.drawCentredString(0, 0 if o > 0 else -8, value)
        self.c.restoreState()

    def dim_leader(self, at, to, text) -> None:
        pa, pb = self.P(*at), self.P(*to)
        self.pen(THIN)
        self.c.line(*pb, *pa)
        d = math.dist(pa, pb)
        self._arrow(pa, ((pa[0] - pb[0]) / d, (pa[1] - pb[1]) / d))
        self.c.line(pb[0], pb[1], pb[0] + 18 * MM, pb[1])
        self.c.setFont("Helvetica", 9)
        self.c.drawString(pb[0] + 1 * MM, pb[1] + 1 * MM, text)

    def frame_and_title(self, fields: list[tuple[str, str]]) -> None:
        m = 10 * MM
        self.pen(THICK)
        self.c.rect(m, m, self.W - 2 * m, self.H - 2 * m, stroke=1, fill=0)
        tw, rh = 120 * MM, 7 * MM
        x0, y0 = self.W - m - tw, m
        self.c.rect(x0, y0, tw, rh * len(fields), stroke=1, fill=0)
        self.pen(THIN)
        for i in range(1, len(fields)):
            self.c.line(x0, y0 + i * rh, x0 + tw, y0 + i * rh)
        self.c.line(x0 + 35 * MM, y0, x0 + 35 * MM, y0 + rh * len(fields))
        self.c.setFont("Helvetica", 8)
        for i, (k, v) in enumerate(reversed(fields)):
            self.c.drawString(x0 + 2 * MM, y0 + i * rh + 2.2 * MM, k)
            self.c.drawString(x0 + 37 * MM, y0 + i * rh + 2.2 * MM, v)

    def notes(self, lines: list[str], at_mm=(20.0, 45.0)) -> None:
        self.c.setFont("Helvetica", 9)
        x, y = at_mm[0] * MM, at_mm[1] * MM
        for i, ln in enumerate(lines):
            self.c.drawString(x, y - i * 4.5 * MM, ln)

    def save(self) -> None:
        self.c.showPage()
        self.c.save()


def plate_with_hole(path: Path, hole: bool = True, title_scale: str = "1:2") -> Path:
    """200 x 100 plate, Ø40 central hole, scale 1:2, fixed left, 50 MPa tension right."""
    s = Sheet(path, scale=2.0, origin_mm=(40.0, 80.0))
    W, H, D = 200.0, 100.0, 40.0
    s.polyline([(0, 0), (W, 0), (W, H), (0, H)])
    if hole:
        s.circle(W / 2, H / 2, D / 2)
        s.dim_leader((W / 2 + D / 2 * math.cos(0.8), H / 2 + D / 2 * math.sin(0.8)),
                     (W / 2 + 45, H + 20), f"Ø{D:g}")
        s.dim_linear((0, H / 2), (W / 2, H / 2), offset=-30.0)
    s.dim_linear((0, 0), (W, 0), offset=-10.0)
    s.dim_linear((W, 0), (W, H), offset=-10.0)
    title = "PLATE_WITH_HOLE" if hole else "PLAIN_PLATE"
    s.frame_and_title([("TITLE", title), ("MATERIAL", "STEEL S355"), ("THICKNESS", "10 mm"),
                       ("SCALE", title_scale), ("UNITS", "mm"), ("DRAWN", "cad2fem example")])
    s.notes(["NOTES:", "1. FIXED: LEFT EDGE", "2. LOAD: TENSION 50 MPa ON RIGHT EDGE",
             "3. MESH SIZE 6"], at_mm=(20.0, 45.0))
    s.save()
    return path


def l_bracket(path: Path) -> Path:
    """L-bracket with inner fillet R10, two Ø12 holes, scale 1:1, aluminium."""
    s = Sheet(path, scale=1.0, origin_mm=(50.0, 70.0))
    A, B, T, R = 150.0, 120.0, 30.0, 10.0
    # outline with an arc fillet at the inner corner (T, T)
    s.pen(THICK)
    p = s.c.beginPath()
    p.moveTo(*s.P(0, 0))
    p.lineTo(*s.P(A, 0))
    p.lineTo(*s.P(A, T))
    p.lineTo(*s.P(T + R, T))
    cx, cy = s.P(T + R, T + R)
    rr = R * MM
    p.arcTo(cx - rr, cy - rr, cx + rr, cy + rr, startAng=270, extent=-90)
    p.lineTo(*s.P(T, B))
    p.lineTo(*s.P(0, B))
    p.close()
    s.c.drawPath(p, stroke=1, fill=0)
    s.circle(15, 100, 6)
    s.circle(130, 15, 6)
    s.dim_linear((0, 0), (A, 0), offset=-12.0)
    s.dim_linear((0, B), (0, 0), offset=-12.0)
    s.dim_linear((A, 0), (A, T), offset=-8.0)
    s.dim_linear((T, B), (0, B), offset=-8.0)
    s.dim_leader((15 + 6 * math.cos(0.7), 100 + 6 * math.sin(0.7)), (45, 112), "2x Ø12")
    s.dim_leader((T + R - R * math.cos(math.pi / 4), T + R - R * math.sin(math.pi / 4)),
                 (60, 60), "R10")
    s.frame_and_title([("TITLE", "L_BRACKET"), ("MATERIAL", "ALUMINIUM 6061-T6"),
                       ("THICKNESS", "8 mm"), ("SCALE", "1:1"), ("UNITS", "mm"),
                       ("DRAWN", "cad2fem example")])
    s.notes(["NOTES:", "1. FIXED: HOLE 1", "2. FORCE 2 kN -Y ON HOLE 2",
             "3. ELEMENT: TRI6, MESH SIZE 5"], at_mm=(20.0, 45.0))
    s.save()
    return path


def main(out: str = "examples/drawings") -> None:
    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    for p in (plate_with_hole(d / "plate_with_hole.pdf"),
              plate_with_hole(d / "plain_plate.pdf", hole=False),
              l_bracket(d / "l_bracket.pdf")):
        print("wrote", p)


if __name__ == "__main__":
    main(*sys.argv[1:])
