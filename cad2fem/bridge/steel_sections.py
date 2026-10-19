"""Built-up steel sections from plate lists (as read from detail drawings).

Section frame: y horizontal (member local y), z vertical (member local z),
z = 0 at the underside.  Lengths in metres.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import SectionProps


@dataclass
class Plate:
    b: float      # extent along y
    h: float      # extent along z
    y: float      # centre
    z: float


def _props(name: str, plates: list[Plate], J: float, width: float, depth: float,
           source: str, n_cells: int = 0) -> SectionProps:
    A = sum(p.b * p.h for p in plates)
    yc = sum(p.b * p.h * p.y for p in plates) / A
    zc = sum(p.b * p.h * p.z for p in plates) / A
    Iyy = sum(p.b * p.h ** 3 / 12 + p.b * p.h * (p.z - zc) ** 2 for p in plates)
    Izz = sum(p.h * p.b ** 3 / 12 + p.b * p.h * (p.y - yc) ** 2 for p in plates)
    Iyz = sum(p.b * p.h * (p.y - yc) * (p.z - zc) for p in plates)
    return SectionProps(name, A, Iyy, Izz, Iyz, J, width, depth, yc + width / 2, zc,
                        n_cells=n_cells, source=source)


def steel_box(name: str, *, web_spacing: float, depth: float, top: tuple[float, float],
              bottom: tuple[float, float], web_t: float,
              flange_ribs: tuple[float, float, int] | None = None,
              web_ribs: tuple[float, float, int] | None = None,
              rib_y: float | None = None, source: str = "drawing") -> SectionProps:
    """Single-cell box: ``top``/``bottom`` = (width, thickness); webs centred at
    +-web_spacing/2.  ``flange_ribs`` = (height, thickness, count per flange),
    placed at +-rib_y; ``web_ribs`` = (width, thickness, count per web)."""
    bt, tt = top
    bb, tb = bottom
    hw = depth - tt - tb
    s = web_spacing / 2
    plates = [Plate(bt, tt, 0.0, depth - tt / 2), Plate(bb, tb, 0.0, tb / 2),
              Plate(web_t, hw, -s, tb + hw / 2), Plate(web_t, hw, s, tb + hw / 2)]
    if flange_ribs:
        hr, tr, n = flange_ribs
        ys = _spread(n, rib_y if rib_y is not None else web_spacing / 6)
        for y in ys:
            plates += [Plate(tr, hr, y, depth - tt - hr / 2), Plate(tr, hr, y, tb + hr / 2)]
    if web_ribs:
        br, tr, n = web_ribs
        zs = [tb + hw * (k + 1) / (n + 1) for k in range(n)]
        for z in zs:
            plates += [Plate(br, tr, -s + web_t / 2 + br / 2, z), Plate(br, tr, s - web_t / 2 - br / 2, z)]
    hm = depth - (tt + tb) / 2
    J = 4 * (web_spacing * hm) ** 2 / (web_spacing / tt + web_spacing / tb + 2 * hm / web_t)
    return _props(name, plates, J, max(bt, bb), depth, source, n_cells=1)


def steel_i(name: str, *, depth: float, web_t: float, top: tuple[float, float],
            bottom: tuple[float, float], source: str = "drawing") -> SectionProps:
    bt, tt = top
    bb, tb = bottom
    hw = depth - tt - tb
    plates = [Plate(bt, tt, 0.0, depth - tt / 2), Plate(bb, tb, 0.0, tb / 2),
              Plate(web_t, hw, 0.0, tb + hw / 2)]
    J = (bt * tt ** 3 + bb * tb ** 3 + hw * web_t ** 3) / 3
    return _props(name, plates, J, max(bt, bb), depth, source)


def _spread(n: int, y: float) -> list[float]:
    if n <= 0:
        return []
    if n == 1:
        return [0.0]
    if n == 2:
        return [-y, y]
    return [-y + 2 * y * k / (n - 1) for k in range(n)]
