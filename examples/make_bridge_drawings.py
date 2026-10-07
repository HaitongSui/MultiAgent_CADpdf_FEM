"""Generate a sample bridge drawing set (one 3-page PDF):

  p1  总体布置图 GENERAL ARRANGEMENT  1:500  - elevation, spans, piers, notes
  p2  主梁横断面 GIRDER CROSS SECTION 1:60   - single-cell box girder
  p3  桥墩断面   PIER SECTION         1:40   - chamfered rectangular pier

    python examples/make_bridge_drawings.py examples/drawings
"""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

sys.path.insert(0, str(Path(__file__).parent))
from make_sample_drawings import MM, THICK, THIN, Sheet  # noqa: E402

pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))


def _font(text: str) -> str:
    return "STSong-Light" if any(ord(ch) > 0x2E80 for ch in text) else "Helvetica"


class CNSheet(Sheet):
    """Sheet on a shared canvas, with CJK-capable text."""

    def __init__(self, c: canvas.Canvas, scale: float, origin_mm) -> None:
        self.c = c
        self.W, self.H = landscape(A4)
        self.scale = scale
        self.ox, self.oy = origin_mm[0] * MM, origin_mm[1] * MM

    def text(self, x_pt, y_pt, s, size=8):
        self.c.setFont(_font(s), size)
        self.c.drawString(x_pt, y_pt, s)

    def frame_and_title(self, fields):
        m = 10 * MM
        self.pen(THICK)
        self.c.rect(m, m, self.W - 2 * m, self.H - 2 * m, stroke=1, fill=0)
        tw, rh = 130 * MM, 7 * MM
        x0, y0 = self.W - m - tw, m
        self.c.rect(x0, y0, tw, rh * len(fields), stroke=1, fill=0)
        self.pen(THIN)
        for i in range(1, len(fields)):
            self.c.line(x0, y0 + i * rh, x0 + tw, y0 + i * rh)
        self.c.line(x0 + 30 * MM, y0, x0 + 30 * MM, y0 + rh * len(fields))
        for i, (k, v) in enumerate(reversed(fields)):
            self.text(x0 + 2 * MM, y0 + i * rh + 2.2 * MM, k)
            self.text(x0 + 32 * MM, y0 + i * rh + 2.2 * MM, v)

    def notes(self, lines, at_mm=(20.0, 70.0)):
        for i, ln in enumerate(lines):
            self.text(at_mm[0] * MM, at_mm[1] * MM - i * 4.6 * MM, ln, 8.5)

    def rect_mm(self, x, y, w, h):
        """Model-space rectangle (outline pen)."""
        self.polyline([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def general_arrangement(c, spans=(30.0, 40.0, 30.0), heights=(8.0, 10.0), depth=2.2,
                        pier_w=1.6, title_scale="1:500", note_spans=True):
    # model units: cm (drawing unit), x along the bridge, y up; deck top at y = 0
    s = CNSheet(c, scale=500.0 / 10.0, origin_mm=(35.0, 150.0))   # 1 cm model -> 1/50 mm paper
    L = sum(spans) * 100
    D = depth * 100
    s.rect_mm(0, -D, L, D)                                   # deck elevation
    x = 0.0
    xs = []
    for span in spans[:-1]:
        x += span * 100
        xs.append(x)
    for xp, hp in zip(xs, heights):
        s.rect_mm(xp - pier_w * 50, -D - hp * 100, pier_w * 100, hp * 100)
    for xa in (0.0, L):                                      # abutments (squat trapezoids)
        sgn = -1 if xa == 0 else 1
        s.polyline([(xa, -D), (xa, -D - 500), (xa + sgn * 600, -D - 500), (xa + sgn * 300, -D)])
    s.pen(THIN, ([3, 2], 0))                                 # ground line
    s.c.line(*s.P(-800, -D - 1300), *s.P(L + 800, -D - 1300))
    xa = 0.0
    for span in spans:
        s.dim_linear((xa, 0), (xa + span * 100, 0), offset=8.0, text=f"{span * 100:g}")
        xa += span * 100
    s.dim_linear((0, 0), (L, 0), offset=16.0, text=f"{L:g}")
    for xp, hp in zip(xs, heights):
        s.dim_linear((xp + pier_w * 50, -D), (xp + pier_w * 50, -D - hp * 100), offset=-6.0,
                     text=f"{hp * 100:g}")
    s.text(*s.P(-200, 150), "A0", 8)
    for k, xp in enumerate(xs, 1):
        s.text(*s.P(xp - 100, 150 + 500), f"P{k}", 8)
    s.text(*s.P(L - 200, 150), f"A{len(spans)}", 8)
    s.frame_and_title([("PROJECT", "XX RIVER BRIDGE 某河大桥"), ("TITLE", "总体布置图 GENERAL ARRANGEMENT"),
                       ("SCALE", title_scale), ("UNITS", "cm"), ("SHEET", "1/3")])
    notes = ["NOTES 说明:", "1. 尺寸单位: cm, 高程单位: m (DIMENSIONS IN cm)"]
    if note_spans:
        notes.append("2. 跨径布置 SPANS: " + "+".join(f"{v:g}" for v in spans) + " m")
    notes += [
        "3. 墩高 PIER HEIGHTS: " + ", ".join(f"P{k}={h:g} m" for k, h in enumerate(heights, 1)),
        "4. 支座 BEARINGS: A0 双向活动, P1 固定, P2 纵向活动, A3 双向活动",
        "5. 主梁 DECK: C50;  桥墩 PIER: C40;  容重 UNIT WEIGHT 26 kN/m3",
        "6. 二期恒载 SECONDARY DEAD LOAD: 45 kN/m",
        "7. 车道荷载 LANE LOAD: qk=10.5 kN/m, Pk=320 kN, 2 LANES",
        "8. 单元长度 ELEMENT SIZE: 2 m",
    ]
    s.notes(notes, at_mm=(20.0, 72.0))
    c.showPage()


def girder_section(c):
    """Single-cell box girder, cm: top 1200 wide, bottom 650, depth 220."""
    s = CNSheet(c, scale=60.0 / 10.0, origin_mm=(75.0, 105.0))   # 1 cm -> 1/6 mm
    outer = [(0, 0), (650, 0), (650, 170), (925, 200), (925, 220), (-275, 220), (-275, 200), (0, 170)]
    cell = [(45, 22), (605, 22), (605, 180), (575, 195), (75, 195), (45, 180)]
    s.polyline(outer)
    s.polyline(cell)
    s.pen(THIN, ([6, 2, 1, 2], 0))
    s.c.line(*s.P(325, -30), *s.P(325, 250))
    s.dim_linear((-275, 220), (925, 220), offset=8.0, text="1200")
    s.dim_linear((0, 0), (650, 0), offset=-8.0, text="650")
    s.dim_linear((925, 0), (925, 220), offset=-14.0, text="220")
    s.frame_and_title([("PROJECT", "XX RIVER BRIDGE 某河大桥"), ("TITLE", "主梁横断面 GIRDER CROSS SECTION"),
                       ("SCALE", "1:60"), ("UNITS", "cm"), ("SHEET", "2/3")])
    s.notes(["NOTES 说明:", "1. 尺寸单位: cm", "2. 单箱单室 SINGLE-CELL BOX, C50"], at_mm=(20.0, 50.0))
    c.showPage()


def pier_section(c):
    """Chamfered rectangle 160 (longitudinal) x 400 (transverse) cm."""
    s = CNSheet(c, scale=40.0 / 10.0, origin_mm=(110.0, 85.0))
    ch = 20
    s.polyline([(ch, 0), (160 - ch, 0), (160, ch), (160, 400 - ch), (160 - ch, 400), (ch, 400),
                (0, 400 - ch), (0, ch)])
    s.dim_linear((0, 0), (160, 0), offset=-8.0, text="160")
    s.dim_linear((160, 0), (160, 400), offset=-10.0, text="400")
    s.frame_and_title([("PROJECT", "XX RIVER BRIDGE 某河大桥"), ("TITLE", "桥墩断面 PIER SECTION"),
                       ("SCALE", "1:40"), ("UNITS", "cm"), ("SHEET", "3/3")])
    s.notes(["NOTES 说明:", "1. 尺寸单位: cm", "2. 倒角 CHAMFER 20x20, C40",
             "3. 横向 TRANSVERSE = vertical on this sheet"], at_mm=(20.0, 50.0))
    c.showPage()


def bridge_set(path: Path, **kw) -> Path:
    c = canvas.Canvas(str(path), pagesize=landscape(A4))
    general_arrangement(c, **kw)
    girder_section(c)
    pier_section(c)
    c.save()
    return path


def main(out: str = "examples/drawings") -> None:
    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    print("wrote", bridge_set(d / "bridge_3span.pdf"))


if __name__ == "__main__":
    main(*sys.argv[1:])
