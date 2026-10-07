"""Bridge-level agents: read a drawing *set* and agree on what the bridge is.

Sheet types
-----------
GENERAL  general arrangement / elevation (总体布置图): spans, piers, bearings, loads
DECK     girder cross-section (主梁横断面): section properties of the spine
PIER     pier section (桥墩断面): section properties of the columns
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..agents.annotation import UNIT_TO_MM, parse_spec
from ..agents.dimension import DIM_RE, PT_TO_MM, scale_part
from ..agents.geometry import extract_part, find_loops
from ..agents.pdf_parser import parse_pdf
from ..core import Agent, Blackboard
from ..models import AnalysisSpec, DrawingData, Geometry, Loop
from .layout_rules import parse_layout
from .models import SUPPORT_KINDS, BridgeLayout, SectionProps
from .sections import rectangle_section, section_from_geometry

SHEET_KINDS = ("GENERAL", "DECK", "PIER", "OTHER")

_PIER = re.compile(r"\bPIERS?\b|桥墩|墩身|墩柱", re.I)
_DECK = re.compile(r"\b(GIRDER|DECK|BOX|SUPERSTRUCTURE)\b|CROSS[\s-]*SECTION|TYPICAL\s+SECTION|"
                   r"横断面|主梁|箱梁|标准断面|跨中断面", re.I)
_GENERAL = re.compile(r"GENERAL\s+ARRANGEMENT|\bELEVATION\b|\bLAYOUT\b|总体布置|桥型布置|立面|"
                      r"\bSPANS?\b|跨径", re.I)


@dataclass
class Sheet:
    source: str
    drawing: DrawingData
    spec: AnalysisSpec = field(default_factory=AnalysisSpec)
    kind: str = "OTHER"

    @property
    def lines(self) -> list[str]:
        return [t.text for t in self.drawing.lines]

    @property
    def title(self) -> str:
        """Sheet title: 'TITLE: X', or the title-block cell right of a 'TITLE' label."""
        lines = self.drawing.lines
        for i, t in enumerate(lines):
            m = re.match(r"\s*(?:TITLE|DRAWING\s+TITLE|图名)\s*[:：]?\s*(.*)$", t.text, re.I)
            if not m:
                continue
            if m.group(1).strip():
                return m.group(1).strip()
            row = [o for o in lines if o is not t and abs(o.y0 - t.y0) < 2.0 and o.x0 > t.x1]
            if row:
                return min(row, key=lambda o: o.x0).text
        return ""


def classify(sheet: Sheet) -> str:
    """Title block decides; otherwise the kind whose keywords occur most often."""
    tests = (("PIER", _PIER), ("DECK", _DECK), ("GENERAL", _GENERAL))
    title = sheet.title
    for kind, pat in tests:
        if title and pat.search(title):
            return kind
    text = "\n".join(sheet.lines)
    counts = {kind: len(pat.findall(text)) for kind, pat in tests}
    kind = max(counts, key=counts.get)
    return kind if counts[kind] else "OTHER"


class SheetReaderAgent(Agent):
    """Reads every page of every input PDF."""

    name = "sheet_reader"
    requires = ("pdf_paths",)
    provides = ("sheets",)

    def run(self, bb: Blackboard) -> None:
        sheets = []
        for path in bb["pdf_paths"]:
            for d in parse_pdf(path):
                sh = Sheet(f"{Path(path).name}#p{d.page_index + 1}", d)
                sh.spec = parse_spec(sh.lines)
                sheets.append(sh)
                if not d.strokes:
                    self.warn(bb, f"{sh.source}: no vector geometry (scanned sheet?)")
        self.info(bb, f"read {len(sheets)} sheet(s) from {len(bb['pdf_paths'])} file(s)")
        bb.post("sheets", sheets, self.name)


class SheetClassifierAgent(Agent):
    """Assigns GENERAL / DECK / PIER / OTHER from the title block and notes.
    ``kinds`` (list, one per sheet) overrides the automatic choice."""

    name = "sheet_classifier"
    requires = ("sheets",)
    provides = ("sheet_kinds",)

    def __init__(self, kinds: Optional[list[str]] = None) -> None:
        super().__init__(kinds=kinds)

    def run(self, bb: Blackboard) -> None:
        sheets: list[Sheet] = bb["sheets"]
        forced = self.params["kinds"]
        for i, sh in enumerate(sheets):
            sh.kind = forced[i].upper() if forced and i < len(forced) else classify(sh)
            self.info(bb, f"{sh.source}: {sh.kind}" + (f" ('{sh.title}')" if sh.title else ""))
        kinds = [sh.kind for sh in sheets]
        for k in ("GENERAL", "DECK"):
            if k not in kinds:
                self.warn(bb, f"no {k} sheet found in the drawing set")
        bb.post("sheet_kinds", kinds, self.name)


def _sheets(bb: Blackboard, kind: str) -> list[Sheet]:
    return [s for s, k in zip(bb["sheets"], bb["sheet_kinds"]) if k == kind]


# --------------------------------------------------------------------------- #
# Elevation: spans and pier heights measured on the general arrangement
# --------------------------------------------------------------------------- #
def read_elevation(sheet: Sheet, log=lambda level, text: None) -> Optional[dict]:
    faces = find_loops(sheet.drawing, mode="faces")
    if not faces:
        return None
    boxes = [lp.bbox for lp in faces]
    w = np.array([b[2] - b[0] for b in boxes])
    h = np.array([b[3] - b[1] for b in boxes])
    slender = np.where(w > 8 * h)[0]
    if not len(slender):
        log("WARNING", "no deck (long slender outline) found on the elevation")
        return None
    k = slender[np.argmax(w[slender])]
    dx0, dy0, dx1, dy1 = boxes[k]
    deck_h = dy1 - dy0
    piers = []
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        if i == k or (y1 - y0) < 1.5 * (x1 - x0):
            continue
        xc = (x0 + x1) / 2
        if abs(y1 - dy0) <= deck_h and dx0 + 2 * (x1 - x0) < xc < dx1 - 2 * (x1 - x0):
            piers.append((xc, y1 - y0))
    piers.sort()

    # scale: title block 1:N and/or dimension texts (in the sheet's units)
    unit_m = UNIT_TO_MM.get(sheet.spec.units, 1.0) / 1000.0
    xs = [dx0] + [p[0] for p in piers] + [dx1]
    cand = [dx1 - dx0] + list(np.diff(xs)) + [p[1] for p in piers]
    ratios = []
    for t in sheet.lines:
        m = DIM_RE.match(t.strip())
        if m and not m.group(2):
            v = float(m.group(3).replace(",", ".")) * unit_m
            ratios += [(v / c, t) for c in cand if c > 1e-9]

    def agree(r):
        return len({t for rr, t in ratios if abs(rr - r) <= 0.01 * r})

    title_s = PT_TO_MM * sheet.spec.scale / 1000.0 if sheet.spec.scale else None
    best = max(ratios, key=lambda rt: agree(rt[0]), default=(None, None))[0]
    n_best, n_title = (agree(best) if best else 0), (agree(title_s) if title_s else 0)
    if title_s and n_title >= n_best:
        s = title_s
    elif best and n_best >= 2:
        s = best
        if title_s:
            log("WARNING", f"elevation: title scale 1:{sheet.spec.scale:g} disagrees with "
                           f"{n_best} dimensions - using dimensions")
    elif title_s:
        s = title_s
    else:
        log("WARNING", "elevation: no usable scale")
        return None
    spans = [float(v * s) for v in np.diff(xs)]
    return {"spans": spans,
            "pier_heights": {f"P{i}": float(ph * s) for i, (_, ph) in enumerate(piers, 1)},
            "deck_depth": float(deck_h * s), "length": float((dx1 - dx0) * s),
            "dims_agree": agree(s), "scale_m_per_pt": s}


class ElevationAgent(Agent):
    name = "elevation"
    requires = ("sheets", "sheet_kinds")
    provides = ("elevation",)

    def run(self, bb: Blackboard) -> None:
        for sh in _sheets(bb, "GENERAL"):
            el = read_elevation(sh, log=lambda level, text: bb.say(self.name, text, level))
            if el:
                self.info(bb, f"{sh.source}: spans {[round(v, 3) for v in el['spans']]} m, "
                              f"pier heights { {k: round(v, 3) for k, v in el['pier_heights'].items()} } m, "
                              f"girder depth {el['deck_depth']:.3g} m "
                              f"({el['dims_agree']} dimension(s) agree)")
                bb.post("elevation", el, self.name)
                return
        self.warn(bb, "could not read spans/piers from an elevation view")


# --------------------------------------------------------------------------- #
# Layout: notes (+ LLM) reconciled with the measured elevation
# --------------------------------------------------------------------------- #
LAYOUT_FIELDS = ("title", "spans", "pier_heights", "bearings", "deck_material", "pier_material",
                 "secondary_dead", "element_size")


class LayoutAgent(Agent):
    name = "layout"
    requires = ("sheets", "sheet_kinds")
    optional = ("elevation", "layout_llm", "bridge_overrides")
    provides = ("layout",)

    def __init__(self, rel_tol: float = 0.02) -> None:
        super().__init__(rel_tol=rel_tol)

    def run(self, bb: Blackboard) -> None:
        sheets: list[Sheet] = bb["sheets"]
        kinds = bb["sheet_kinds"]
        order = sorted(range(len(sheets)), key=lambda i: kinds[i] != "GENERAL")
        lay = parse_layout([t for i in order for t in sheets[i].lines])

        llm: Optional[BridgeLayout] = bb.get("layout_llm")
        if llm is not None:
            lay.source = "rules+llm"
            for f in (*LAYOUT_FIELDS, "lane"):
                if f in llm.explicit and f not in lay.explicit:
                    attrs = ("lane_q", "lane_P", "lanes", "lane_P_span") if f == "lane" else (f,)
                    for a in attrs:
                        setattr(lay, a, copy.deepcopy(getattr(llm, a)))
                    lay.explicit.add(f)
                    self.info(bb, f"{f} taken from LLM")

        el = bb.get("elevation")
        tol = self.params["rel_tol"]
        if el:
            if not lay.spans:
                lay.spans = [round(v, 3) for v in el["spans"]]
                self.info(bb, f"spans taken from elevation: {lay.spans}")
            elif len(lay.spans) != len(el["spans"]) or any(
                    abs(a - b) > tol * a for a, b in zip(lay.spans, el["spans"])):
                self.warn(bb, f"notes give spans {lay.spans} but elevation measures "
                              f"{[round(v, 2) for v in el['spans']]} - keeping notes")
            else:
                self.info(bb, "span arrangement confirmed by the elevation view")
            for p, hm in el["pier_heights"].items():
                if p not in lay.pier_heights:
                    lay.pier_heights[p] = round(hm, 3)
                    self.info(bb, f"{p} height {hm:.3g} m taken from elevation")
                elif abs(lay.pier_heights[p] - hm) > tol * lay.pier_heights[p]:
                    self.warn(bb, f"{p}: notes H={lay.pier_heights[p]:g} m, elevation "
                                  f"{hm:.3g} m - keeping notes")

        if not lay.spans:
            raise ValueError("span arrangement not found (notes or elevation)")
        piers = lay.supports[1:-1]
        missing = [p for p in piers if p not in lay.pier_heights]
        if missing:
            raise ValueError(f"no height for pier(s) {missing}")
        self._default_bearings(bb, lay)
        for k, v in (bb.get("bridge_overrides") or {}).items():
            if v is not None:
                setattr(lay, k, v)
                lay.explicit.add(k)
        self.info(bb, f"{lay.title}: spans {lay.spans} m (L={lay.length:g} m), piers "
                      f"{lay.pier_heights}, bearings {lay.bearings}, deck {lay.deck_material.name}, "
                      f"piers {lay.pier_material.name}")
        bb.post("layout", lay, self.name)

    def _default_bearings(self, bb: Blackboard, lay: BridgeLayout) -> None:
        sup = lay.supports
        unknown = [k for k, v in lay.bearings.items() if k not in sup or v not in SUPPORT_KINDS]
        for k in unknown:
            self.warn(bb, f"bearing entry {k}={lay.bearings.pop(k)} ignored")
        missing = [s for s in sup if s not in lay.bearings]
        if not missing:
            return
        has_fixed = any(v in ("FIXED", "MONOLITHIC") for v in lay.bearings.values())
        piers = sup[1:-1] or sup[:1]
        centre = piers[len(piers) // 2]
        for s in missing:
            lay.bearings[s] = "FIXED" if (s == centre and not has_fixed) else "SLIDING_X"
        self.warn(bb, f"bearings not specified for {missing}; assumed "
                      f"{ {s: lay.bearings[s] for s in missing} }")


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def read_section(sheet: Sheet, name: str, log) -> SectionProps:
    g_pt = extract_part(sheet.drawing, log=log)
    unit_mm = UNIT_TO_MM.get(sheet.spec.units, 1.0)
    g_mm = scale_part(g_pt, sheet.lines, unit_mm, sheet.spec.scale, log=log)
    g_m = Geometry(outer=Loop(g_mm.outer.points / 1000.0),
                   holes=[Loop(h.points / 1000.0, h.is_circle,
                               None if h.center is None else (h.center[0] / 1000, h.center[1] / 1000),
                               None if h.radius is None else h.radius / 1000) for h in g_mm.holes])
    if g_mm.outer.is_circle:
        c = g_mm.outer.center
        g_m.outer = Loop(g_m.outer.points, True, (c[0] / 1000, c[1] / 1000), g_mm.outer.radius / 1000)
    sec = section_from_geometry(name, g_m)
    sec.source = sheet.source
    return sec


class SectionAgent(Agent):
    name = "section"
    requires = ("sheets", "sheet_kinds")
    provides = ("sections",)

    def __init__(self, default_pier: tuple[float, float] = (1.8, 1.8)) -> None:
        super().__init__(default_pier=default_pier)

    def run(self, bb: Blackboard) -> None:
        log = lambda level, text: bb.say(self.name, text, level)  # noqa: E731
        sections: dict[str, SectionProps] = {}
        for kind, name in (("DECK", "DECK"), ("PIER", "PIER")):
            sh = next(iter(_sheets(bb, kind)), None)
            if sh is None:
                continue
            sec = read_section(sh, name, log)
            sections[name] = sec
            self.info(bb, f"{name} ({sh.source}): {sec.width:.3g} x {sec.height:.3g} m, "
                          f"{sec.n_cells} cell(s), A={sec.A:.4g} m2, Iyy={sec.Iyy:.4g}, "
                          f"Izz={sec.Izz:.4g}, J={sec.J:.4g} m4, centroid {sec.cz:.3g} m above soffit")
        if "DECK" not in sections:
            raise ValueError("no girder cross-section sheet (DECK) - cannot build the spine")
        if "PIER" not in sections:
            b, h = self.params["default_pier"]
            sections["PIER"] = rectangle_section("PIER", b, h)
            self.warn(bb, f"no pier section sheet; assuming solid {b} x {h} m rectangle")
        bb.post("sections", sections, self.name)
