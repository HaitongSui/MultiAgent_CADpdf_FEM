"""Steel arch + box-girder composite bridge (アーチ複合橋) from a sheet reading.

Agents
------
ArchGeometryAgent   checks the reading's dimension chains and fits the arch
                    axis so that every hanger length on the drawing is met
ArchAssemblyAgent   builds a 3D frame: two arch planes (girder + rib + V hangers),
                    cross beams, rigid springings, bearings, load cases
ArchValidatorAgent  model sanity checks
ArchVerifierAgent   solves every load case with the built-in frame solver

Units m, kN, kPa.  X along the bridge from bearing S1, Y transverse, Z up;
girder axes at Z = 0 (the road's vertical curve is not modelled).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import least_squares

from ..core import Agent, Blackboard
from .arch_reading import ArchReading
from .frame_solver import rotation, solve, uniform_load_vector
from .models import Beam, BridgeLayout, BridgeMaterial, FrameModel, SectionProps


# --------------------------------------------------------------------------- #
# Values NOT given on the general drawings (J-5 / J-8)
# --------------------------------------------------------------------------- #
@dataclass
class ArchAssumptions:
    steel_E: float = 2.0e8           # kPa  (道路橋示方書)
    steel_nu: float = 0.3
    steel_gamma: float = 77.0        # kN/m3
    cable_E: float = 1.95e8          # kPa, parallel-wire strand
    cable_area: float = 0.0025       # m2 per hanger
    girder_tf: float = 0.030         # m, box flange
    girder_tw: float = 0.016         # m, box web
    rib_width: float = 0.800         # m
    rib_tf: float = 0.032
    rib_tw: float = 0.025
    xbeam_h: float = 1.600           # cross beam (I-section)
    xbeam_bf: float = 0.350
    xbeam_tf: float = 0.022
    xbeam_tw: float = 0.012
    deck_dead_kN_m2: float = 2.5     # steel deck plate + U-ribs + stiffeners
    pavement_gamma: float = 23.0     # kN/m3
    railing_kN_m: float = 3.0        # per side
    element_size: float = 2.5        # m

    def rows(self) -> list[tuple[str, str]]:
        return [(k, f"{v:g}") for k, v in asdict(self).items()]


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def box_section(name: str, b: float, h: float, tf: float, tw: float) -> SectionProps:
    """Thin-walled rectangular box (two flanges b x tf, two webs tw)."""
    hw = h - 2 * tf
    A = 2 * b * tf + 2 * hw * tw
    Iyy = b * h ** 3 / 12 - (b - 2 * tw) * hw ** 3 / 12
    Izz = h * b ** 3 / 12 - hw * (b - 2 * tw) ** 3 / 12
    bm, hm = b - tw, h - tf
    J = 4 * (bm * hm) ** 2 / (2 * bm / tf + 2 * hm / tw)      # Bredt
    return SectionProps(name, A, Iyy, Izz, 0.0, J, b, h, b / 2, h / 2, n_cells=1, source="assumed")


def i_section(name: str, h: float, bf: float, tf: float, tw: float) -> SectionProps:
    hw = h - 2 * tf
    A = 2 * bf * tf + hw * tw
    Iyy = bf * h ** 3 / 12 - (bf - tw) * hw ** 3 / 12
    Izz = 2 * tf * bf ** 3 / 12 + hw * tw ** 3 / 12
    J = (2 * bf * tf ** 3 + hw * tw ** 3) / 3
    return SectionProps(name, A, Iyy, Izz, 0.0, J, bf, h, bf / 2, h / 2, source="assumed")


def cable_section(name: str, A: float) -> SectionProps:
    """Hanger as a beam with negligible bending stiffness."""
    d = math.sqrt(4 * A / math.pi)
    tiny = A * d ** 2 * 1e-4
    return SectionProps(name, A, tiny, tiny, 0.0, 2 * tiny, d, d, d / 2, d / 2, source="assumed")


# --------------------------------------------------------------------------- #
# Geometry from the reading
# --------------------------------------------------------------------------- #
@dataclass
class ArchGeometry:
    stations: dict[str, float]          # label -> x (m)
    x_spring: tuple[float, float]
    params: tuple[float, float, float]  # zs, zc, a2 of the anchor curve
    rib_offset: float                   # rib axis above the anchor curve (m)
    anchor_names: list[str]
    anchor_x: np.ndarray
    cable_station: list[str]
    cable_length: np.ndarray            # drawing (m)
    cable_model_length: np.ndarray      # fitted geometry (m)
    checks: list[dict] = field(default_factory=list)

    @property
    def span(self) -> float:
        return max(self.stations.values())

    def anchor_z(self, x):
        x0, x1 = self.x_spring
        zs, zc, a2 = self.params
        t = np.abs(np.asarray(x, float) - (x0 + x1) / 2) / ((x1 - x0) / 2)
        return zc + a2 * t ** 2 + (zs - zc - a2) * t ** 3

    def rib_z(self, x):
        return self.anchor_z(x) + self.rib_offset

    @property
    def fit_rms_mm(self) -> float:
        return float(1000 * np.sqrt(np.mean((self.cable_model_length - self.cable_length) ** 2)))


def _check(checks, name, value, expected, tol, unit="mm"):
    ok = abs(value - expected) <= tol
    checks.append({"check": name, "value": round(value, 3), "expected": round(expected, 3),
                   "tolerance": tol, "unit": unit, "ok": bool(ok)})
    return ok


def fit_arch(r: ArchReading) -> ArchGeometry:
    checks: list[dict] = []
    if len(r.cross_beam_spacing_mm) != len(r.station_labels) - 1:
        raise ValueError("cross-beam spacing chain does not match the station list")
    xs = np.r_[0.0, np.cumsum(r.cross_beam_spacing_mm)] / 1000
    stations = dict(zip(r.station_labels, map(float, xs)))
    _check(checks, "sum of 横桁間隔 = 支間長", 1000 * xs[-1], r.span_mm, 1.0)
    _check(checks, "sum of ケーブル間隔 = アーチリブ軸線長", sum(r.cable_anchor_chain_mm), r.arch_axis_length_mm, 1.0)
    end = (r.girder_length_mm - r.span_mm) / 2          # girder end to bearing
    x0 = ((r.girder_length_mm - r.arch_span_mm) / 2 - end) / 1000
    x1 = x0 + r.arch_span_mm / 1000
    _check(checks, "arch springs symmetric in the span", 1000 * (xs[-1] - x1), 1000 * x0, 1.0)
    sp = np.array(r.cross_beam_spacing_mm)
    _check(checks, "横桁間隔 symmetric", float(np.abs(sp - sp[::-1]).max()), 0.0, 1.0)

    if len(r.cable_anchor_order) != len(r.cable_anchor_chain_mm) - 1:
        raise ValueError("anchor names do not match the anchor chain")
    s_anchor = np.cumsum(r.cable_anchor_chain_mm)[:-1] / 1000
    by_name = {f"{c.girder_station}-{c.side}": c for c in r.cables}
    missing = [a for a in r.cable_anchor_order if a not in by_name]
    if missing:
        raise ValueError(f"no hanger length for anchors {missing}")
    cables = [by_name[a] for a in r.cable_anchor_order]
    L_draw = np.array([c.length_mm for c in cables]) / 1000
    xg = np.array([stations[c.girder_station] for c in cables])
    asym = np.abs(L_draw - L_draw[::-1]) / L_draw
    _check(checks, "hanger lengths symmetric (max rel. diff)", 100 * float(asym.max()), 0.0, 1.0, "%")

    xq = np.linspace(x0, x1, 20001)
    xc, half = (x0 + x1) / 2, (x1 - x0) / 2

    def zf(x, p):
        zs, zc, a2 = p
        t = np.abs(x - xc) / half
        return zc + a2 * t ** 2 + (zs - zc - a2) * t ** 3

    def anchors(p):
        z = zf(xq, p)
        s = np.r_[0, np.cumsum(np.hypot(np.diff(xq), np.diff(z)))]
        return np.interp(s_anchor, s, xq), s[-1]

    def residual(p):
        xa, length = anchors(p)
        return np.r_[np.hypot(xa - xg, zf(xa, p)) - L_draw, 3 * (length - r.arch_axis_length_mm / 1000)]

    crown_guess = (r.rib_bottom_above_girder_top_mm + r.girder_depth_mm / 2) / 1000
    sol = least_squares(residual, [1.0, crown_guess, 0.0])
    xa, length = anchors(sol.x)
    L_model = np.hypot(xa - xg, zf(xa, sol.x))
    geo = ArchGeometry(stations, (x0, x1), tuple(map(float, sol.x)), r.rib_depth_mm / 2000,
                       list(r.cable_anchor_order), xa, [c.girder_station for c in cables],
                       L_draw, L_model, checks)
    _check(checks, "arch axis length closes (fitted)", 1000 * length, r.arch_axis_length_mm, 5.0)
    _check(checks, "hanger lengths reproduced (RMS)", geo.fit_rms_mm, 0.0, 50.0)
    _check(checks, "crown: anchor height vs 8000 + girder half depth", 1000 * sol.x[1],
           1000 * crown_guess, 300.0)
    if r.cable_angle_at_crown_deg:
        k = int(np.argmin(np.abs(xa - xc)))
        ang = math.degrees(math.atan2(zf(xa[k], sol.x), abs(xa[k] - xg[k])))
        _check(checks, "hanger angle at the crown", ang, r.cable_angle_at_crown_deg, 2.0, "deg")
    return geo


class ArchGeometryAgent(Agent):
    name = "arch_geometry"
    requires = ("arch_reading",)
    provides = ("arch_geometry",)

    def run(self, bb: Blackboard) -> None:
        r: ArchReading = bb["arch_reading"]
        geo = fit_arch(r)
        for c in geo.checks:
            msg = f"{c['check']}: {c['value']:g} vs {c['expected']:g} {c['unit']}"
            (self.info if c["ok"] else self.warn)(bb, ("OK   " if c["ok"] else "FAIL ") + msg)
        zs, zc, _ = geo.params
        self.info(bb, f"arch springs at x={geo.x_spring[0]:.2f} / {geo.x_spring[1]:.2f} m; rib axis "
                      f"{zs + geo.rib_offset:.3f} m (springing) -> {zc + geo.rib_offset:.3f} m (crown) "
                      f"above the girder axis; hanger RMS {geo.fit_rms_mm:.1f} mm")
        for u in r.uncertain:
            self.warn(bb, f"reading uncertain: {u}")
        bb.post("arch_geometry", geo, self.name)


# --------------------------------------------------------------------------- #
# Loads
# --------------------------------------------------------------------------- #
def b_live_load(span_m: float, carriageway_m: float, sidewalk_each_m: float) -> dict:
    """B活荷重 (道路橋示方書 H14), simplified symmetric full-width placement.

    p2 and the sidewalk load are 3.5 kN/m2 for L <= 80 m, 4.3 - 0.01 L for
    80 < L <= 130 m and 3.0 beyond; p1 = 10 kN/m2 (bending) over D = 10 m.
    The main loading width is 5.5 m; the rest of the carriageway takes half.
    No impact factor.
    """
    L = span_m
    p2 = 3.5 if L <= 80 else (4.3 - 0.01 * L if L <= 130 else 3.0)
    main = min(5.5, carriageway_m)
    eff = main + (carriageway_m - main) / 2
    return {"p1": 10.0, "p2": p2, "sidewalk": p2, "D": 10.0,
            "q_uniform": p2 * eff + p2 * 2 * sidewalk_each_m, "q_p1": 10.0 * eff}


def _member_load(m: FrameModel, case: str, e: int, q_global) -> None:
    b = m.beams[e]
    R, L = rotation(m.nodes[b.n1], m.nodes[b.n2], b.vec_y)
    f = uniform_load_vector(R, L, q_global)
    m.element_loads.setdefault(case, {})[e] = m.element_loads.get(case, {}).get(e, 0) + f
    F = m.load_cases.setdefault(case, np.zeros((m.n_nodes, 6)))
    F[b.n1] += f[:6]
    F[b.n2] += f[6:]


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def _stations_with_subdivision(fixed: list[float], h: float) -> list[float]:
    fixed = sorted(set(round(x, 6) for x in fixed))
    out = [fixed[0]]
    for a, b in zip(fixed[:-1], fixed[1:]):
        n = max(1, math.ceil((b - a) / h - 1e-9))
        out += list(np.linspace(a, b, n + 1)[1:])
    return out


def build_arch_frame(r: ArchReading, geo: ArchGeometry, a: ArchAssumptions) -> FrameModel:
    steel = BridgeMaterial("SM490Y", a.steel_E, a.steel_nu, a.steel_gamma)
    cable_mat = BridgeMaterial("CABLE", a.cable_E, 0.3, a.steel_gamma)
    secs = {
        "GIRDER": box_section("GIRDER", r.girder_width_mm / 1000, r.girder_depth_mm / 1000,
                              a.girder_tf, a.girder_tw),
        "RIB": box_section("RIB", a.rib_width, r.rib_depth_mm / 1000, a.rib_tf, a.rib_tw),
        "CABLE": cable_section("CABLE", a.cable_area),
        "XBEAM": i_section("XBEAM", a.xbeam_h, a.xbeam_bf, a.xbeam_tf, a.xbeam_tw),
    }
    lay = BridgeLayout(spans=[geo.span], title="ARCH_BRIDGE", deck_material=steel,
                       pier_material=steel, element_size=a.element_size)
    m = FrameModel(nodes=np.zeros((0, 3)), sections=secs,
                   materials={"STEEL": steel, "CABLE": cable_mat}, layout=lay)
    x0, x1 = geo.x_spring
    mid = geo.span / 2
    gx = _stations_with_subdivision(list(geo.stations.values()) + [x0, x1, mid - 5.0, mid + 5.0],
                                    a.element_size)
    rx = _stations_with_subdivision([x0, x1, *geo.anchor_x], a.element_size)
    half = r.girder_spacing_mm / 2000
    station_x = {round(x, 6): lab for lab, x in geo.stations.items()}

    girder_node: dict[tuple[str, float], int] = {}
    for plane, y in (("G1", half), ("G2", -half)):
        # girder
        nodes = []
        for x in gx:
            label = f"{plane}_{station_x[round(x, 6)]}" if round(x, 6) in station_x else None
            n = m.add_node((x, y, 0.0), label)
            girder_node[(plane, round(x, 6))] = n
            nodes.append(n)
        for p, q in zip(nodes[:-1], nodes[1:]):
            m.beams.append(Beam(p, q, "GIRDER", "STEEL", (0.0, 1.0, 0.0), f"GIRDER_{plane}"))
        m.node_groups[f"GIRDER_{plane}"] = nodes
        # rib
        rib = [m.add_node((x, y, float(geo.rib_z(x)))) for x in rx]
        for p, q in zip(rib[:-1], rib[1:]):
            m.beams.append(Beam(p, q, "RIB", "STEEL", (0.0, 1.0, 0.0), f"RIB_{plane}"))
        m.node_groups[f"RIB_{plane}"] = rib
        rib_at = {round(x, 6): n for x, n in zip(rx, rib)}
        # springings: rib rigidly connected to the girder
        for xs_ in (x0, x1):
            m.rigid.append((girder_node[(plane, round(xs_, 6))], rib_at[round(xs_, 6)]))
        # hangers: girder axis -> anchor (rib bottom flange), anchor rigid to the rib axis
        for name, xa, st in zip(geo.anchor_names, geo.anchor_x, geo.cable_station):
            anchor = m.add_node((xa, y, float(geo.anchor_z(xa))), f"{plane}_{name}")
            m.rigid.append((rib_at[round(float(xa), 6)], anchor))
            g = girder_node[(plane, round(geo.stations[st], 6))]
            m.beams.append(Beam(g, anchor, "CABLE", "CABLE", (0.0, 1.0, 0.0),
                                f"CABLE_{plane}_{name.replace('-', '_')}"))   # solver-safe set name
    # cross beams at every station
    for lab, x in geo.stations.items():
        p, q = girder_node[("G1", round(x, 6))], girder_node[("G2", round(x, 6))]
        m.beams.append(Beam(p, q, "XBEAM", "STEEL", (1.0, 0.0, 0.0), f"XBEAM_{lab}"))
    # bearings (simple support of the tied system): S1 pinned, S2 longitudinally free
    first, last = r.station_labels[0], r.station_labels[-1]
    for plane in ("G1", "G2"):
        m.supports[girder_node[(plane, round(geo.stations[first], 6))]] = (1, 2, 3)
        m.supports[girder_node[(plane, round(geo.stations[last], 6))]] = (2, 3)

    # ---- loads -----------------------------------------------------------------
    m.load_cases["DEAD"] = np.zeros((m.n_nodes, 6))
    for e, b in enumerate(m.beams):
        _member_load(m, "DEAD", e, (0, 0, -m.materials[b.material].gamma * m.sections[b.section].A))
    width = r.deck_width_mm / 1000
    paved = (r.carriageway_width_mm + 2 * r.sidewalk_width_mm) / 1000
    w_deck = (a.deck_dead_kN_m2 * width + a.pavement_gamma * r.pavement_thickness_mm / 1000 * paved
              + 2 * a.railing_kN_m) / 2                       # per girder
    live = b_live_load(geo.span, r.carriageway_width_mm / 1000, r.sidewalk_width_mm / 1000)
    m.load_cases["LIVE_B"] = np.zeros((m.n_nodes, 6))
    for e, b in enumerate(m.beams):
        if not b.group.startswith("GIRDER"):
            continue
        _member_load(m, "DEAD", e, (0, 0, -w_deck))
        _member_load(m, "LIVE_B", e, (0, 0, -live["q_uniform"] / 2))
        xm = (m.nodes[b.n1, 0] + m.nodes[b.n2, 0]) / 2
        if abs(xm - mid) < live["D"] / 2:
            _member_load(m, "LIVE_B", e, (0, 0, -live["q_p1"] / 2))
    m.layout.notes = [f"deck+pavement+railing {w_deck:.2f} kN/m per girder",
                      f"B live: p2={live['p2']:.3f} kPa, uniform {live['q_uniform']:.2f} kN/m, "
                      f"p1 {live['q_p1']:.1f} kN/m over {live['D']:g} m at midspan (whole deck)"]
    return m


class ArchAssemblyAgent(Agent):
    name = "arch_assembly"
    requires = ("arch_reading", "arch_geometry")
    optional = ("arch_assumptions",)
    provides = ("frame",)

    def run(self, bb: Blackboard) -> None:
        a = bb.get("arch_assumptions") or ArchAssumptions()
        m = build_arch_frame(bb["arch_reading"], bb["arch_geometry"], a)
        n_cab = len([b for b in m.beams if b.section == "CABLE"])
        self.info(bb, f"{m.n_nodes} nodes, {len(m.beams)} members ({n_cab} hangers), "
                      f"{len(m.rigid)} rigid links, load cases {list(m.load_cases)}")
        for note in m.layout.notes:
            self.info(bb, note)
        self.warn(bb, "plate thicknesses, rib width, hanger and cross-beam sections and deck dead "
                      "load are ASSUMED (not on J-5/J-8) - see report")
        bb.post("frame", m, self.name)


class ArchValidatorAgent(Agent):
    name = "arch_validator"
    requires = ("frame", "arch_geometry", "arch_reading")
    provides = ("bridge_validation",)

    def run(self, bb: Blackboard) -> None:
        m: FrameModel = bb["frame"]
        geo: ArchGeometry = bb["arch_geometry"]
        r: ArchReading = bb["arch_reading"]
        issues = [f"reading check failed: {c['check']}" for c in geo.checks if not c["ok"]]
        n_cab = len([b for b in m.beams if b.section == "CABLE"])
        if n_cab != 2 * len(r.cables):
            issues.append(f"{n_cab} hanger members for {len(r.cables)} hangers per plane")
        lengths = [np.linalg.norm(m.nodes[b.n2] - m.nodes[b.n1]) for b in m.beams]
        if min(lengths) < 0.05:
            issues.append(f"very short member ({min(lengths):.3f} m)")
        for msg in issues:
            self.warn(bb, msg)
        self.info(bb, "PASSED" if not issues else "PASSED WITH WARNINGS")
        bb.post("bridge_validation", {"ok": not issues, "issues": issues}, self.name)


def summarise_arch(m: FrameModel, geo: ArchGeometry, res) -> dict:
    mid = geo.span / 2
    out: dict = {}
    applied = res.applied[:, :3].sum(axis=0)
    reac = res.reactions[:, :3].sum(axis=0)
    out["applied_kN"], out["reaction_sum_kN"] = applied.tolist(), reac.tolist()
    out["equilibrium_error"] = float(np.linalg.norm(applied + reac) / max(np.abs(applied).sum(), 1e-9))
    out["reactions"] = {m.node_labels.get(n, str(n)): res.reactions[n, :3].tolist() for n in m.supports}
    g1 = m.node_groups["GIRDER_G1"]
    k = int(np.argmin(np.abs(m.nodes[g1, 0] - mid)))
    out["midspan_deflection_mm"] = float(-1000 * res.u[g1[k], 2])
    out["max_girder_deflection_mm"] = float(-1000 * res.u[g1, 2].min())
    cab = {}
    for e, b in enumerate(m.beams):
        if b.group.startswith("CABLE_G1_"):
            cab[b.group.removeprefix("CABLE_G1_").replace("_", "-")] = float(-res.end_forces[e, 0])
    out["hanger_force_kN"] = cab
    rib = [e for e, b in enumerate(m.beams) if b.group == "RIB_G1"]
    gir = [e for e, b in enumerate(m.beams) if b.group == "GIRDER_G1"]
    xm = lambda e: (m.nodes[m.beams[e].n1, 0] + m.nodes[m.beams[e].n2, 0]) / 2  # noqa: E731
    e_rib = min(rib, key=lambda e: abs(xm(e) - mid))
    e_gir = min(gir, key=lambda e: abs(xm(e) - mid))
    out["rib_axial_crown_kN"] = float(-res.end_forces[e_rib, 0])        # tension +
    out["girder_axial_midspan_kN"] = float(-res.end_forces[e_gir, 0])
    # both member ends, so jumps (thrust eccentricity at the springings) are kept
    My = []
    for e in gir:
        b = m.beams[e]
        My += [(float(m.nodes[b.n1, 0]), float(res.end_forces[e, 4])),
               (float(m.nodes[b.n2, 0]), float(-res.end_forces[e, 10]))]
    out["girder_moment"] = My
    out["max_sagging_kNm"] = max(v for _, v in My)
    out["max_hogging_kNm"] = min(v for _, v in My)
    return out


class ArchVerifierAgent(Agent):
    name = "arch_verifier"
    requires = ("frame", "bridge_validation", "arch_geometry")
    provides = ("bridge_results",)

    def run(self, bb: Blackboard) -> None:
        m: FrameModel = bb["frame"]
        geo = bb["arch_geometry"]
        results = {}
        for case in m.load_cases:
            try:
                res = solve(m, case)
            except np.linalg.LinAlgError as exc:
                self.error(bb, f"{case}: {exc}")
                continue
            s = summarise_arch(m, geo, res)
            s["result"] = res
            results[case] = s
            hf = s["hanger_force_kN"].values()
            self.info(bb, f"{case}: midspan deflection {s['midspan_deflection_mm']:.1f} mm, hangers "
                          f"{min(hf):.0f}..{max(hf):.0f} kN, rib N(crown) {s['rib_axial_crown_kN']:.0f} kN, "
                          f"girder N(mid) {s['girder_axial_midspan_kN']:.0f} kN, sum Rz "
                          f"{s['reaction_sum_kN'][2]:.1f} kN")
            if s["equilibrium_error"] > 1e-6:
                self.warn(bb, f"{case}: equilibrium error {s['equilibrium_error']:.1e}")
            if min(hf) < 0:
                self.warn(bb, f"{case}: {sum(v < 0 for v in hf)} hanger(s) in compression")
        bb.post("bridge_results", results, self.name)
