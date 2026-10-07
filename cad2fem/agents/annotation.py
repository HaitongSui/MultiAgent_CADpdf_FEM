"""AnnotationAgent: rule-based interpretation of title block and notes.

Turns free text such as::

    MATERIAL: STEEL S355        THICKNESS: 10 mm      SCALE 1:2
    FIXED: LEFT EDGE
    LOAD: TENSION 50 MPa ON RIGHT EDGE
    FORCE 12 kN -Y ON HOLE 2

into an :class:`~cad2fem.models.AnalysisSpec`.
"""

from __future__ import annotations

import re
from typing import Optional

from .. import materials
from ..core import Agent, Blackboard
from ..models import AnalysisSpec, BoundaryCondition, Load, Material

NUM = r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?"

UNIT_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4, "inch": 25.4, "inches": 25.4}
PRESSURE_TO_MPA = {"mpa": 1.0, "n/mm2": 1.0, "n/mm^2": 1.0, "n/mm²": 1.0, "kpa": 1e-3,
                   "pa": 1e-6, "gpa": 1e3, "bar": 0.1, "psi": 0.00689476, "ksi": 6.89476}
FORCE_TO_N = {"n": 1.0, "kn": 1e3, "mn": 1e6, "kgf": 9.80665, "lbf": 4.44822, "kip": 4448.22}

BC_WORDS = r"(FIXED|CLAMPED|ENCASTRE|ENCASTRÉ|PINNED|ROLLER|SYMMETRY|SUPPORT(?:ED)?|固定|约束|对称)"
LOAD_WORDS = r"(PRESSURE|TENSION|TENSILE|TRACTION|COMPRESSION|FORCE|LOAD|压力|拉力|力)"

_LOC_PATTERNS = [
    (re.compile(r"\bALL\s+HOLES\b|\bHOLES\b|所有孔"), lambda m: "HOLES"),
    (re.compile(r"\bHOLE\s*#?\s*(\d+)|孔\s*(\d+)"), lambda m: f"HOLE{m.group(1) or m.group(2)}"),
    (re.compile(r"\b([XY])\s*=\s*(" + NUM + ")"), lambda m: f"{m.group(1)}={m.group(2)}"),
    (re.compile(r"\b(LEFT|RIGHT|TOP|BOTTOM|UPPER|LOWER)\s*(?:EDGE|SIDE|FACE|END|BOUNDARY)|"
                r"\b(LEFT|RIGHT|TOP|BOTTOM)\b|(左|右|上|下)\s*(?:边|侧|端)"),
     lambda m: _side(m.group(1) or m.group(2) or m.group(3))),
]


def _side(word: str) -> str:
    return {"UPPER": "TOP", "LOWER": "BOTTOM", "左": "LEFT", "右": "RIGHT",
            "上": "TOP", "下": "BOTTOM"}.get(word, word)


def _num(s: str) -> float:
    return float(s.replace(",", "."))


def parse_location(text: str) -> Optional[str]:
    up = text.upper()
    for pat, fn in _LOC_PATTERNS:
        m = pat.search(up)
        if m:
            return fn(m)
    return None


def _normal_dof(location: str) -> tuple[int, ...]:
    if location in ("LEFT", "RIGHT") or location.startswith("X="):
        return (1,)
    if location in ("TOP", "BOTTOM") or location.startswith("Y="):
        return (2,)
    return (1, 2)


def parse_bc(stmt: str) -> Optional[BoundaryCondition]:
    up = stmt.upper()
    m = re.search(BC_WORDS, up)
    if not m:
        return None
    loc = parse_location(up)
    if loc is None:
        return None
    word = m.group(1)
    explicit = [1 if d in "X1" else 2 for d in re.findall(r"\bU([XY12])\s*=\s*0(?:\.0*)?\b", up)]
    if explicit:
        dofs = tuple(sorted(set(explicit)))
    elif word in ("ROLLER", "SYMMETRY", "对称"):
        dofs = _normal_dof(loc)
    else:
        dofs = (1, 2)
    kind = {"对称": "SYMMETRY", "固定": "FIXED", "约束": "FIXED", "ENCASTRÉ": "ENCASTRE"}.get(word, word)
    return BoundaryCondition(location=loc, dofs=dofs, kind=kind)


def parse_load(stmt: str) -> Optional[Load]:
    up = stmt.upper()
    if not re.search(LOAD_WORDS, up):
        return None
    loc = parse_location(re.sub(r"^\s*LOAD\s*[:=]?", "", up))
    if loc is None:
        return None
    # pressure-type value
    pm = re.search(r"(" + NUM + r")\s*(MPA|N/MM\^?2|N/MM²|KPA|GPA|PA|BAR|PSI|KSI)\b", up)
    if pm:
        p = _num(pm.group(1)) * PRESSURE_TO_MPA[pm.group(2).lower()]
        # Abaqus convention: positive pressure acts *into* the surface
        if re.search(r"TENSION|TENSILE|TRACTION|PULL|拉", up):
            p = -abs(p)
        return Load(location=loc, kind="PRESSURE", magnitude=p)
    fm = re.search(r"(" + NUM + r")\s*(KN|MN|N|KGF|LBF|KIP)\b", up)
    if fm:
        f = _num(fm.group(1)) * FORCE_TO_N[fm.group(2).lower()]
        d = _parse_direction(up[fm.end():]) or _parse_direction(up) or (0.0, -1.0)
        return Load(location=loc, kind="FORCE", magnitude=abs(f), direction=d)
    return None


def _parse_direction(up: str) -> Optional[tuple[float, float]]:
    m = re.search(r"(?:DIR(?:ECTION)?\s*[:=]?\s*|IN\s+|ALONG\s+|\s|^)([+-])\s*([XY])\b", up)
    if m:
        s = 1.0 if m.group(1) == "+" else -1.0
        return (s, 0.0) if m.group(2) == "X" else (0.0, s)
    m = re.search(r"DIR(?:ECTION)?\s*[:=]?\s*([XY])\b", up)
    if m:
        return (1.0, 0.0) if m.group(1) == "X" else (0.0, 1.0)
    for word, d in (("DOWNWARD", (0.0, -1.0)), ("DOWN", (0.0, -1.0)), ("UPWARD", (0.0, 1.0)),
                    ("UP", (0.0, 1.0)), ("向下", (0.0, -1.0)), ("向上", (0.0, 1.0))):
        if word in up:
            return d
    return None


def parse_spec(lines: list[str]) -> AnalysisSpec:
    spec = AnalysisSpec(notes=list(lines))
    text = "\n".join(lines)
    up = text.upper()
    unit = "mm"

    m = re.search(r"(?:UNITS?|DIMENSIONS?\s+IN|单位)\s*[:=]?\s*(MM|CM|M|INCHES|INCH|IN)\b", up)
    if m:
        unit = m.group(1).lower()
        spec.explicit.add("units")
    spec.units = unit
    to_mm = UNIT_TO_MM.get(unit, 1.0)

    m = re.search(r"(?:SCALE|比例)\s*[:=]?\s*(" + NUM + r")\s*[:/]\s*(" + NUM + ")", up)
    if m:
        spec.scale = _num(m.group(2)) / _num(m.group(1))
        spec.explicit.add("scale")

    m = re.search(r"(?:TITLE|PART\s*NAME|名称)\s*(?:[:=]\s*|\s+)(\S.*)", text, re.I)
    if m:
        spec.title = re.sub(r"[^A-Za-z0-9_\-]+", "_", m.group(1).strip())[:60].strip("_") or "PART"
        spec.explicit.add("title")

    m = (re.search(r"(?:THICKNESS|THK|厚度)\s*[:=]?\s*(" + NUM + r")\s*(MM|CM|M|IN)?\b", up)
         or re.search(r"(?<![\w-])T\s*=\s*(" + NUM + r")\s*(MM|CM|M|IN)?\b", up))
    if m:
        spec.thickness = _num(m.group(1)) * UNIT_TO_MM.get((m.group(2) or unit).lower(), to_mm)
        spec.explicit.add("thickness")

    mat: Optional[Material] = None
    m = re.search(r"(?:MATERIAL|MATL|材料)\s*[:=]?\s*([^\n;]+)", text, re.I)
    if m:
        mat = materials.lookup(m.group(1)) or Material(name=re.sub(r"\W+", "_", m.group(1).strip().upper()))
    if mat is not None:
        spec.explicit.add("material")
    mat = mat or Material()
    m = re.search(r"(?<![A-Z])E\s*[:=]\s*(" + NUM + r")\s*(GPA|MPA|N/MM2|PA)?", up)
    if m:
        mat.E = _num(m.group(1)) * PRESSURE_TO_MPA.get((m.group(2) or "MPA").lower(), 1.0)
        spec.explicit.add("material")
    m = re.search(r"(?:\bNU|Ν|POISSON(?:'S)?(?:\s+RATIO)?)\s*[:=]?\s*(" + NUM + ")", up)
    if m:
        mat.nu = _num(m.group(1))
    spec.material = mat

    if re.search(r"PLANE\s*STRAIN|平面应变", up):
        spec.analysis = "PLANE_STRAIN"
        spec.explicit.add("analysis")
    m = re.search(r"MESH\s*SIZE\s*[:=]?\s*(" + NUM + ")", up)
    if m:
        spec.mesh_size = _num(m.group(1)) * to_mm
        spec.explicit.add("mesh_size")
    if re.search(r"\b(TRI6|CPS6|CPE6|QUADRATIC|SECOND[\s-]ORDER)\b", up):
        spec.element_order = 2
        spec.explicit.add("element_order")

    for line in lines:
        for stmt in re.split(r"[;；]", line):
            if not stmt.strip():
                continue
            ld = parse_load(stmt)
            if ld is not None:
                spec.loads.append(_scale_loc(ld, to_mm))
                continue
            bc = parse_bc(stmt)
            if bc is not None:
                spec.bcs.append(_scale_loc(bc, to_mm))
    return spec


def _scale_loc(item, to_mm: float):
    m = re.match(r"([XY])=(" + NUM + ")$", item.location)
    if m and to_mm != 1.0:
        item.location = f"{m.group(1)}={_num(m.group(2)) * to_mm:g}"
    return item


class AnnotationAgent(Agent):
    """Rule-based reading of the title block and notes."""

    name = "annotation"
    requires = ("drawing",)
    provides = ("spec_rules",)

    def run(self, bb: Blackboard) -> None:
        drawing = bb["drawing"]
        spec = parse_spec([ln.text for ln in drawing.lines])
        self.info(bb, f"material={spec.material.name} (E={spec.material.E:g}, nu={spec.material.nu}), "
                      f"t={spec.thickness:g} mm, scale=1:{spec.scale}, {len(spec.bcs)} BCs, "
                      f"{len(spec.loads)} loads")
        bb.post("spec_rules", spec, self.name)
