"""Practice tasks (knowledge gaps) and the Verifier's grounded checks.

Each task names what to read, how to find its sheet in the sheet index, the
JSON the Actor must return, and deterministic checks that compare the claims
with independent evidence (other sheets, physics, arithmetic) - never with
the Actor's own reasoning.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..bridge.arch_reading import ArchReading
from .claims import Claims

WIRE_AREA_MM2 = math.pi * 3.5 ** 2          # one 7 mm wire


@dataclass
class Task:
    id: str
    goal: str
    title_keys: list[str]                  # matched against the sheet index
    priority: int
    example: str                           # JSON shape the Actor must return
    checks: Callable[[Claims, "Context"], list[dict]]
    depends_on: list[str] = field(default_factory=lambda: ["sheet_index"])
    critical: bool = True                  # needed by the model


@dataclass
class Context:
    reading: ArchReading
    pages: dict[int, str] = field(default_factory=dict)   # page -> drawing no (from index)
    known: dict[str, Claims] = field(default_factory=dict)  # other tasks' verified claims (memory)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", "", s)


def _c(name: str, ok: bool, detail: str) -> dict:
    return {"check": name, "ok": bool(ok), "detail": detail}


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def check_index(c: Claims, ctx: Context) -> list[dict]:
    pages = [e.page for e in c.sheet_index]
    out = [_c("every page indexed once", len(pages) == len(set(pages)) and len(pages) >= 1,
              f"{len(pages)} entries, {len(set(pages))} unique pages"),
           _c("drawing numbers present", all(e.drawing_no.strip() for e in c.sheet_index), "")]
    known = {}
    for s in ctx.reading.sheets:                      # e.g. "J-5 構造一般図 (PDF（241101）-14)"
        m = re.match(r"\s*([A-Z]+-\d+).*-(\d+)\)\s*$", unicodedata.normalize("NFKC", s))
        if m:
            known[int(m.group(2))] = m.group(1)
    got = {e.page: norm(e.drawing_no) for e in c.sheet_index}
    for page, no in known.items():
        out.append(_c(f"page {page} is {no} (verified reading)", got.get(page) == norm(no),
                      f"claimed {got.get(page)}"))
    return out


def check_cables(c: Claims, ctx: Context) -> list[dict]:
    r = ctx.reading
    out = []
    n = len(r.cables)
    nums = sorted(cb.number for cb in c.cables)
    out.append(_c("one entry per hanger of the J-8 layout", nums == list(range(1, n + 1)), f"{nums}"))
    specs = {s.strands: s for s in c.cable_specs}
    for s in c.cable_specs:
        exp = s.strands * WIRE_AREA_MM2
        out.append(_c(f"{s.name}: area = strands x 38.48 mm2", abs(s.area_mm2 - exp) <= 0.015 * exp,
                      f"{s.area_mm2:g} vs {exp:.0f}"))
        out.append(_c(f"{s.name}: E plausible for wire", 150 <= s.E_kN_mm2 <= 220, f"{s.E_kN_mm2:g}"))
    by = {cb.number: cb for cb in c.cables}
    out.append(_c("every hanger has a specified cable type", all(cb.strands in specs for cb in c.cables),
                  f"types {sorted(specs)}"))
    if nums == list(range(1, n + 1)):
        sym = all(by[k].strands == by[n + 1 - k].strands for k in range(1, n + 1))
        out.append(_c("cable types symmetric about midspan", sym, ""))
        L8 = {f"{cb.girder_station}-{cb.side}": cb.length_mm / 1000 for cb in r.cables}
        order = r.cable_anchor_order
        if all(by[k].node_length_m for k in by) and len(order) == n:
            d = np.array([L8[order[k - 1]] - by[k].node_length_m for k in range(1, n + 1)])
            out.append(_c("node lengths differ from J-8 hanger lengths by a constant",
                          float(d.std()) < 0.015 and 0.1 < float(d.mean()) < 1.5,
                          f"offset {d.mean() * 1000:.0f} mm, scatter {d.std() * 1000:.1f} mm"))
        if all(by[k].dead_tension_kN for k in by):
            t = np.array([by[k].dead_tension_kN for k in range(1, n + 1)])
            rel = float(np.max(np.abs(t - t[::-1]) / t))
            out.append(_c("dead-load tensions symmetric (<10%)", rel < 0.10, f"max {100 * rel:.1f}%"))
    return out


def _section(c: Claims, name: str):
    return next((s for s in c.sections if s.name == name), None)


def _plates(s, role):
    return [p for p in s.plates if p.role == role]


def _one_per_role(s, roles) -> dict:
    dup = sorted({p.role for p in s.plates if p.role in roles and len(_plates(s, p.role)) > 1})
    return _c("one plate entry per role (governing cross-section, not per panel/piece)", not dup,
              f"repeated: {dup}" if dup else "")


def _depths(s) -> tuple[float, float]:
    d = s.depth_range_mm or [s.depth_mm]
    return min(d), max(d)


def _same_layout_as_verified(s, seg: int, ctx: Context) -> list[dict]:
    """Rib segments of the same depth share the stiffener layout: compare with the
    other segments already verified in memory (independent readings of other sheets)."""
    out = []
    lo, hi = _depths(s)
    for tid, c in sorted(ctx.known.items()):
        other = _section(c, f"RIB_SEG{tid[-1]}") if tid.startswith("rib_seg") and tid != f"rib_seg{seg}" else None
        if other is None or _depths(other) != (lo, hi):
            continue
        for role in ("flange_rib", "web_rib"):
            a, b = _plates(s, role), _plates(other, role)
            if a and b:
                out.append(_c(f"{role} count per flange/web = verified {tid} (same depth {hi:g})",
                              a[0].count == b[0].count, f"{a[0].count} vs {b[0].count}"))
    return out


def check_box(name: str, depth_from_plates: bool, rib_segment: Optional[int] = None):
    def run(c: Claims, ctx: Context) -> list[dict]:
        r = ctx.reading
        s = _section(c, name)
        if s is None:
            return [_c(f"{name} section reported", False, "missing")]
        out = [_one_per_role(s, ("top", "bottom", "web", "flange_rib", "web_rib"))]
        top, bot, web = _plates(s, "top"), _plates(s, "bottom"), _plates(s, "web")
        out.append(_c("top, bottom and two webs reported", bool(top and bot and web) and
                      sum(p.count for p in web) == 2, f"{len(top)}/{len(bot)}/{sum(p.count for p in web)}"))
        if not (top and bot and web):
            return out
        ws = s.web_spacing_mm or 0
        out.append(_c("web spacing = main girder width (J-5/J-8)", abs(ws - r.girder_width_mm) <= 2,
                      f"{ws:g} vs {r.girder_width_mm:g}"))
        out.append(_c("top flange covers the webs (outstand 0-200 mm)",
                      0 <= top[0].width_mm - ws <= 200, f"{top[0].width_mm:g} vs {ws:g}"))
        for p in (*top, *bot, *web):
            out.append(_c(f"{p.role} thickness plausible", 6 <= p.thickness_mm <= 80, f"{p.thickness_mm:g}"))
        if depth_from_plates:
            d = top[0].thickness_mm + web[0].width_mm + bot[0].thickness_mm
            out.append(_c("depth = flange + web + flange", abs(d - s.depth_mm) <= 5, f"{d:g} vs {s.depth_mm:g}"))
            out.append(_c("end block deeper than the midspan girder (J-5)", s.depth_mm > r.girder_depth_mm,
                          f"{s.depth_mm:g} > {r.girder_depth_mm:g}"))
        if rib_segment is not None:
            chain = r.rib_member_lengths_mm
            if chain and s.axis_length_mm:
                exp = chain[rib_segment - 1]
                out.append(_c(f"segment {rib_segment} axis length = J-8 部材長",
                              abs(s.axis_length_mm - exp) <= 1.0, f"{s.axis_length_mm:g} vs {exp:g}"))
            lo, hi = r.rib_depth_mm - 25, r.rib_depth_mm * 1.30
            out.append(_c("rib depth between crown depth (J-5) and +30%", lo <= s.depth_mm <= hi,
                          f"{s.depth_mm:g} in [{lo:g}, {hi:g}]"))
            out += _same_layout_as_verified(s, rib_segment, ctx)
            wr = _plates(s, "web_rib")
            if wr and s.web_rib_pitch_mm:
                n = round(_depths(s)[0] / s.web_rib_pitch_mm) - 1
                out.append(_c("web ribs per web = web depth / pitch - 1", wr[0].count == n,
                              f"{wr[0].count} vs {_depths(s)[0]:g}/{s.web_rib_pitch_mm:g}-1 = {n}"))
            if rib_segment == 5:
                out.append(_c("crown depth = J-5 rib depth", abs(s.depth_mm - r.rib_depth_mm) <= 25,
                              f"{s.depth_mm:g} vs {r.rib_depth_mm:g}"))
        return out
    return run


def check_xbeam(c: Claims, ctx: Context) -> list[dict]:
    s = _section(c, "XBEAM")
    if s is None:
        return [_c("XBEAM section reported", False, "missing")]
    web, bot = _plates(s, "web"), _plates(s, "bottom")
    return [_one_per_role(s, ("top", "bottom", "web")),
            _c("web and bottom flange reported", bool(web and bot), ""),
            _c("depth below main-girder end depth and above 0.5 m", 500 <= s.depth_mm <= 3000, f"{s.depth_mm:g}"),
            _c("web thickness plausible", bool(web) and 6 <= web[0].thickness_mm <= 30,
               f"{web[0].thickness_mm if web else None}"),
            _c("bottom flange width plausible", bool(bot) and 150 <= bot[0].width_mm <= 800,
               f"{bot[0].width_mm if bot else None}")]


# --------------------------------------------------------------------------- #
# the curriculum's task list
# --------------------------------------------------------------------------- #
SECTION_EXAMPLE = ('{"task": "<id>", "sheets": ["J-xx <title>"], "sections": [{"name": "<NAME>", '
                   '"sheet": "J-xx", "view": "...", "depth_mm": 0, "depth_range_mm": [], "web_spacing_mm": 0, '
                   '"axis_length_mm": null, "web_rib_pitch_mm": null, "plates": [{"role": "top|bottom|web|flange_rib|web_rib", '
                   '"width_mm": 0, "thickness_mm": 0, "count": 1, "material": "SM490YB"}]}], '
                   '"lessons": ["<reusable lesson about reading this kind of sheet>"]}')

ONE_SECTION = (" Report the governing cross-section with one plate entry per role: count = pieces in that "
               "one cross-section - 2 webs, flange ribs per flange, web ribs per web (a callout such as "
               "4-RIB that covers both webs is 2 per web) - not one entry per panel or piece length.")

TASKS: list[Task] = [
    Task("sheet_index", "Read the title block of every sheet: page number, 図面番号 and 図面名称.",
         [], 0, '{"task": "sheet_index", "sheet_index": [{"page": 1, "drawing_no": "K-3", "title": "..."}], '
                '"lessons": []}', check_index, depends_on=[]),
    Task("cables", "Read the hanger cable table (ケーブル寸法表) and cable specification (ケーブル仕様): for "
                   "cables 1..18 the strand count, dead-load tension and node length Ls; for each cable type "
                   "the nominal area, elastic modulus, breaking load and weight.",
         ["ケーブル図"], 1,
         '{"task": "cables", "sheets": ["J-81 ケーブル図"], "cables": [{"number": 1, "strands": 19, '
         '"dead_tension_kN": 0, "node_length_m": 0}], "cable_specs": [{"name": "SPWC 19", "strands": 19, '
         '"area_mm2": 0, "E_kN_mm2": 0, "breaking_kN": 0, "weight_kg_m": 0}], "lessons": []}',
         check_cables),
    Task("girder_end", "Read the main-girder (補剛桁) end-block box section: flange plates, webs, "
                       "longitudinal ribs and the section depth. Name the section GIRDER_END." + ONE_SECTION,
         ["補剛桁"], 2, SECTION_EXAMPLE, check_box("GIRDER_END", depth_from_plates=True)),
    *[Task(f"rib_seg{k}", f"Read arch-rib segment {k} (アーチリブ（その{k}）): top/bottom flange plates, web "
                          f"plates and ribs, the section depth (use the H2 / section view, not the curved web "
                          f"PL width), the segment length on the arch axis and the web-rib pitch over the web depth "
                          f"(web_rib_pitch_mm, e.g. 400 from 3@400=1200). Name the section RIB_SEG{k}."
                          + ONE_SECTION,
           [f"アーチリブ(その{k})"], 2 + k, SECTION_EXAMPLE, check_box(f"RIB_SEG{k}", False, rib_segment=k))
      for k in (2, 3, 4, 5)],
    Task("xbeam", "Read the intermediate cross beam (中間横桁) section between the main girders: web, "
                  "bottom flange and depth (top flange is the steel deck). Name the section XBEAM." + ONE_SECTION,
         ["中間横桁"], 8, SECTION_EXAMPLE, check_xbeam),
]


def task_by_id(tid: str) -> Task:
    return next(t for t in TASKS if t.id == tid)
