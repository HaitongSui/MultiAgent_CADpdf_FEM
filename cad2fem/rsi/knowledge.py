"""Turn verified memory into model inputs for the arch bridge (test-time reuse)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..bridge.arch_reading import ArchReading
from ..bridge.models import SectionProps
from ..bridge.steel_sections import steel_box, steel_i
from .claims import SectionClaim
from .memory import Memory


@dataclass
class ArchKnowledge:
    sections: dict[str, SectionProps] = field(default_factory=dict)
    rib_segments: list[tuple[float, float, str]] = field(default_factory=list)   # (s0, s1, section) m
    cable_section_by_anchor: dict[str, str] = field(default_factory=dict)
    cable_E: Optional[float] = None                                             # kPa
    design_dead_tension: dict[str, float] = field(default_factory=dict)         # anchor -> kN
    sources: dict[str, str] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not self.sections and not self.cable_section_by_anchor


def _box(s: SectionClaim, name: str, source: str, depth_mm: Optional[float] = None) -> SectionProps:
    get = lambda role: [p for p in s.plates if p.role == role]  # noqa: E731
    top, bot, web = get("top")[0], get("bottom")[0], get("web")[0]
    fr, wr = get("flange_rib"), get("web_rib")
    d = (depth_mm or (sum(s.depth_range_mm) / len(s.depth_range_mm) if s.depth_range_mm else s.depth_mm)) / 1000
    return steel_box(
        name, web_spacing=(s.web_spacing_mm or 1600) / 1000, depth=d,
        top=(top.width_mm / 1000, top.thickness_mm / 1000), bottom=(bot.width_mm / 1000, bot.thickness_mm / 1000),
        web_t=web.thickness_mm / 1000,
        flange_ribs=(fr[0].width_mm / 1000, fr[0].thickness_mm / 1000, fr[0].count) if fr else None,
        web_ribs=(wr[0].width_mm / 1000, wr[0].thickness_mm / 1000, wr[0].count) if wr else None,
        rib_y=0.267, source=source)


def knowledge_from_memory(mem: Memory, r: ArchReading, deck_effective_width: float = 1.0,
                          deck_t: float = 0.012) -> ArchKnowledge:
    k = ArchKnowledge()

    # ---- hangers --------------------------------------------------------------
    cab = mem.claims("cables")
    if cab:
        specs = {s.strands: s for s in cab.cable_specs}
        for s in cab.cable_specs:
            name = "CABLE_" + s.name.replace(" ", "")
            A = s.area_mm2 / 1e6
            tiny = A * 1e-6
            k.sections[name] = SectionProps(name, A, tiny, tiny, 0.0, 2 * tiny, 0.0, 0.0, 0.0, 0.0,
                                            source=f"{cab.sheets[0] if cab.sheets else 'J-81'} (verified)")
        k.cable_E = cab.cable_specs[0].E_kN_mm2 * 1e6 if cab.cable_specs else None
        for cb in cab.cables:
            anchor = r.cable_anchor_order[cb.number - 1]
            k.cable_section_by_anchor[anchor] = "CABLE_" + specs[cb.strands].name.replace(" ", "")
            if cb.dead_tension_kN:
                k.design_dead_tension[anchor] = cb.dead_tension_kN
        k.sources["hangers"] = "J-81 ケーブル図 (verified)"

    # ---- main girder -----------------------------------------------------------
    ge = mem.claims("girder_end")
    if ge:
        s = next(x for x in ge.sections if x.name == "GIRDER_END")
        k.sections["GIRDER_END"] = _box(s, "GIRDER_END", f"{s.sheet} (verified)")
        k.sections["GIRDER"] = _box(s, "GIRDER", f"inferred: {s.sheet} plates at the J-5 depth "
                                    f"{r.girder_depth_mm:g} mm (J-58..J-72 not in the set)",
                                    depth_mm=r.girder_depth_mm)
        k.sources["girder"] = k.sections["GIRDER"].source

    # ---- arch rib (segments 1..9 symmetric about the crown) ----------------------
    read = {}
    for seg in (2, 3, 4, 5):
        c = mem.claims(f"rib_seg{seg}")
        if c:
            s = next(x for x in c.sections if x.name == f"RIB_SEG{seg}")
            read[seg] = _box(s, f"RIB_SEG{seg}", f"{s.sheet} (verified)")
    if read and r.rib_member_lengths_mm:
        chain = r.rib_member_lengths_mm
        n = len(chain)
        s0 = 0.0
        for i, length in enumerate(chain, 1):
            mirror = min(i, n + 1 - i)                      # 1..5 from either springing
            src = mirror if mirror in read else min(read, key=lambda j: abs(j - mirror))
            name = f"RIB_SEG{src}"
            if mirror not in read:
                k.sources[f"rib segment {i}"] = f"inferred from segment {src}"
            k.sections[name] = read[src]
            k.rib_segments.append((s0 / 1000, (s0 + length) / 1000, name))
            s0 += length
        k.sources["arch rib"] = "J-43..J-46 (verified), mirrored about the crown"

    # ---- cross beams --------------------------------------------------------------
    xb = mem.claims("xbeam")
    if xb:
        s = next(x for x in xb.sections if x.name == "XBEAM")
        web = next(p for p in s.plates if p.role == "web")
        bot = next(p for p in s.plates if p.role == "bottom")
        top = next((p for p in s.plates if p.role == "top"), None)
        k.sections["XBEAM"] = steel_i(
            "XBEAM", depth=(sum(s.depth_range_mm) / len(s.depth_range_mm) if s.depth_range_mm else s.depth_mm) / 1000,
            web_t=web.thickness_mm / 1000,
            top=((top.width_mm / 1000, top.thickness_mm / 1000) if top else (deck_effective_width, deck_t)),
            bottom=(bot.width_mm / 1000, bot.thickness_mm / 1000),
            source=f"{s.sheet} (verified); top flange = steel deck {deck_effective_width:g} m x "
                   f"{deck_t * 1000:g} mm (assumed effective width)")
        k.sources["cross beams"] = k.sections["XBEAM"].source
    return k
