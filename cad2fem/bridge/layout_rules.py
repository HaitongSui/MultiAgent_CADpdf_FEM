"""Rule-based reading of general-arrangement notes (English / Chinese).

Examples understood::

    SPANS: 30+40+30 m            跨径布置: 3×30 m
    PIER HEIGHTS: P1=8.0 m, P2=10.0 m          墩高: P1=8.0 m, P2=10.0 m
    BEARINGS: A0 SLIDING, P1 FIXED, P2 SLIDING, A3 SLIDING
    支座: A0 双向活动, P1 固定, P2 纵向活动, A3 双向活动      墩梁固结: P1
    DECK: C50   PIER: C40   UNIT WEIGHT: 26 kN/m3
    SECONDARY DEAD LOAD: 45 kN/m        二期恒载: 45 kN/m
    LANE LOAD: q = 10.5 kN/m, P = 300 kN, 2 LANES  车道荷载: qk=10.5 kN/m, Pk=300 kN, 2车道
    ELEMENT SIZE: 2 m                   单元长度: 2 m
"""

from __future__ import annotations

import re
from typing import Optional

from .models import BridgeLayout, BridgeMaterial

NUM = r"\d+(?:\.\d+)?"

# JTG 3362-2018 concrete moduli (MPa)
CONCRETE_E = {25: 2.80e4, 30: 3.00e4, 35: 3.15e4, 40: 3.25e4, 45: 3.35e4, 50: 3.45e4,
              55: 3.55e4, 60: 3.60e4, 65: 3.65e4, 70: 3.70e4, 75: 3.75e4, 80: 3.80e4}

KIND_WORDS = [
    (r"MONOLITHIC|RIGID(?:LY)?\s*CONNECTED|INTEGRAL|固结|刚接", "MONOLITHIC"),
    (r"FIXED|固定", "FIXED"),
    (r"SLIDING[\s_-]*XY|BI-?DIRECTIONAL|FREE|双向活动|多向活动", "SLIDING_XY"),
    (r"SLIDING[\s_-]*X|GUIDED|LONGITUDINAL(?:LY)?\s*(?:SLIDING|MOVABLE)|UNIDIRECTIONAL|SLIDING|MOVABLE|"
     r"纵向活动|单向活动|活动", "SLIDING_X"),
]


def material(code: str) -> Optional[BridgeMaterial]:
    up = code.upper()
    m = re.search(r"\bC\s*(\d{2})\b", up)
    if m and int(m.group(1)) in CONCRETE_E:
        return BridgeMaterial(f"C{m.group(1)}", CONCRETE_E[int(m.group(1))] * 1e3, 0.2, 26.0)
    m = re.search(r"\b(Q\d{3}[A-Z]*)", up)
    if m or "STEEL" in up or "钢" in code:
        return BridgeMaterial(m.group(1) if m else "STEEL", 2.06e8, 0.3, 78.5)
    return None


def parse_spans(text: str) -> list[float]:
    """'30+40+30' / '3×30' / '4x25+2x30' / '(30+40+30)' -> list of spans."""
    out: list[float] = []
    for term in re.split(r"\+", text.replace("(", "").replace(")", "").replace("（", "").replace("）", "")):
        term = term.strip()
        m = re.fullmatch(r"(\d+)\s*[×xX*]\s*(" + NUM + r")", term)
        if m:
            out += [float(m.group(2))] * int(m.group(1))
        elif re.fullmatch(NUM, term):
            out.append(float(term))
        else:
            return []
    return out


def _unit_factor(unit: Optional[str]) -> float:
    return {"mm": 1e-3, "cm": 1e-2, "m": 1.0, None: 1.0}.get(unit and unit.lower(), 1.0)


def parse_layout(lines: list[str]) -> BridgeLayout:
    lay = BridgeLayout(notes=list(lines))
    text = "\n".join(lines)
    up = text.upper()

    m = (re.search(r"(?:BRIDGE\s*NAME|PROJECT|桥名|工程名称)\s*[:：]?\s*([^\n]+)", text, re.I)
         or re.search(r"(?:TITLE|图名)\s*[:：]?\s*([^\n]+)", text, re.I))
    if m:
        t = re.sub(r"[^\w\-]+", "_", m.group(1).strip()).strip("_")
        if t:
            lay.title, _ = t[:60], lay.explicit.add("title")

    m = re.search(r"(?:SPANS?(?:\s+ARRANGEMENT)?|跨径(?:布置|组合)?|孔跨布置)\s*[:：=]?\s*"
                  r"([\d.+×X*()（）][\d.\s+×X*()（）]*)\s*(MM|CM|M)?\b", up)
    if m:
        spans = parse_spans(m.group(1).strip())
        if spans:
            lay.spans = [s * _unit_factor(m.group(2)) for s in spans]
            lay.explicit.add("spans")

    for line in lines:
        u = line.upper()
        if re.search(r"PIER\s*HEIGHT|墩高|桥墩高度|墩身高度", u):
            pairs = re.findall(r"\b(P\d+)\s*[:=：]?\s*(?:H\s*=\s*)?(" + NUM + r")", u)
            if pairs:
                lay.pier_heights.update({p: float(v) for p, v in pairs})
            else:
                nums = re.findall(NUM, u.split(":", 1)[-1].split("：", 1)[-1])
                lay.pier_heights.update({f"P{i}": float(v) for i, v in enumerate(nums, 1)})
            lay.explicit.add("pier_heights")
        if re.search(r"BEARING|支座|固结", u) and not re.search(r"STIFFNESS|刚度", u):
            for stmt in re.split(r"[,，;；]", re.sub(r"^.*?[:：]", "", u) if re.search(r"[:：]", u) else u):
                tags = re.findall(r"\b([AP]\d+)\b", stmt)
                kind = next((k for pat, k in KIND_WORDS if re.search(pat, stmt)), None)
                if kind is None and re.search(r"墩梁固结|MONOLITHIC", u):
                    kind = "MONOLITHIC"
                for tag in tags:
                    if kind:
                        lay.bearings[tag] = kind
            if lay.bearings:
                lay.explicit.add("bearings")
        m = re.search(r"(?:BEARING\s*STIFFNESS|支座刚度)\s*[:：=]?\s*(" + NUM + r"(?:E[+-]?\d+)?)", u)
        if m:
            lay.bearing_stiffness = float(m.group(1))
        for stmt in re.split(r"[;；]", u):        # "DECK: C50; PIER: C40" on one line
            if re.search(r"\b(DECK|GIRDER|SUPERSTRUCTURE)\b|主梁|箱梁|上部结构", stmt):
                mat = material(stmt)
                if mat:
                    lay.deck_material = mat
                    lay.explicit.add("deck_material")
            if re.search(r"\bPIERS?\b|桥墩|墩身|下部结构", stmt) and \
                    not re.search(r"HEIGHT|墩高|高度", stmt):
                mat = material(stmt)
                if mat:
                    lay.pier_material = mat
                    lay.explicit.add("pier_material")

    m = re.search(r"(?:UNIT\s*WEIGHT|DENSITY|容重|重度)\s*[:：=]?\s*(" + NUM + r")\s*KN/M", up)
    if m:   # applies to concrete members; steel keeps 78.5
        for mat in (lay.deck_material, lay.pier_material):
            if mat.name.startswith("C"):
                mat.gamma = float(m.group(1))
        lay.explicit.add("gamma")
    m = re.search(r"(?:SECONDARY\s*DEAD\s*LOAD|SUPERIMPOSED\s*DEAD\s*LOAD|SDL|二期恒载|二期荷载)"
                  r"\s*[:：=]?\s*(" + NUM + r")\s*KN/M", up)
    if m:
        lay.secondary_dead = float(m.group(1))
        lay.explicit.add("secondary_dead")
    lane_line = next((ln.upper() for ln in lines if re.search(r"LANE\s*LOAD|车道荷载", ln, re.I)), "")
    if lane_line:
        m = re.search(r"\bQ\s*K?\s*[=:：]?\s*(" + NUM + r")\s*KN/M", lane_line) or \
            re.search(r"(" + NUM + r")\s*KN/M", lane_line)
        if m:
            lay.lane_q = float(m.group(1))
        m = re.search(r"\bP\s*K?\s*[=:：]?\s*(" + NUM + r")\s*KN\b(?!/)", lane_line) or \
            re.search(r"(" + NUM + r")\s*KN\b(?!/)", lane_line)
        if m:
            lay.lane_P = float(m.group(1))
        m = re.search(r"(\d+)\s*(?:LANES?|车道)", lane_line)
        if m:
            lay.lanes = int(m.group(1))
        m = re.search(r"(?:SPAN|第)\s*(\d+)\s*(?:跨)?", lane_line)
        if m:
            lay.lane_P_span = int(m.group(1))
        lay.explicit.add("lane")
    m = re.search(r"(?:ELEMENT\s*(?:SIZE|LENGTH)|MESH\s*SIZE|单元长度|单元尺寸)\s*[:：=]?\s*(" + NUM
                  + r")\s*(MM|CM|M)?\b", up)
    if m:
        lay.element_size = float(m.group(1)) * _unit_factor(m.group(2) or "m")
        lay.explicit.add("element_size")
    return lay
