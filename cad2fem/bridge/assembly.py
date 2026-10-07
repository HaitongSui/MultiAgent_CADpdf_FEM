"""Assemble the global spine model and check it.

Topology per support (X = support station, Zb = soffit / bearing level)::

    deck node D (X, 0, 0)  ──rigid──►  bearing top BT (X, 0, Zb)
                                         │ springs on the restrained dofs
    pier top PT (X, 0, Zb) ── pier beams ── pier base (X, 0, Zb - H)  [fixed]
    abutment: BT ── springs ── ground node G (X, 0, Zb)              [fixed]

MONOLITHIC piers connect D rigidly to PT (rigid-frame bridge).
"""

from __future__ import annotations

import math

import numpy as np

from ..core import Agent, Blackboard
from .frame_solver import rotation, solve, uniform_load_vector
from .models import BEARING_DOFS, Beam, BridgeLayout, FrameModel, SectionProps, Spring

ALL = (1, 2, 3, 4, 5, 6)


def default_element_size(lay: BridgeLayout) -> float:
    return min(lay.spans) / 12.0


def build_frame(lay: BridgeLayout, sections: dict[str, SectionProps], h: float) -> FrameModel:
    deck, pier = sections["DECK"], sections["PIER"]
    m = FrameModel(nodes=np.zeros((0, 3)), sections={"DECK": deck, "PIER": pier},
                   materials={"DECK": lay.deck_material, "PIER": lay.pier_material}, layout=lay)
    zb = -deck.cz
    sx = lay.support_x

    # ---- deck spine --------------------------------------------------------
    deck_nodes = [m.add_node((0.0, 0.0, 0.0))]
    span_of_beam = []
    midspan: dict[int, int] = {}
    x0 = 0.0
    for k, L in enumerate(lay.spans, 1):
        n = 2 * max(4, math.ceil(L / (2 * h)))
        for j, x in enumerate(np.linspace(x0, x0 + L, n + 1)[1:], 1):
            node = m.add_node((x, 0.0, 0.0))
            m.beams.append(Beam(deck_nodes[-1], node, "DECK", "DECK", (0.0, 1.0, 0.0), f"DECK_S{k}"))
            span_of_beam.append(k)
            deck_nodes.append(node)
            if j == n // 2:
                midspan[k] = node
                m.node_labels[node] = f"MID_S{k}"
        x0 += L
    m.node_groups["DECK"] = deck_nodes
    deck_at = {s: min(deck_nodes, key=lambda i: abs(m.nodes[i, 0] - x)) for s, x in sx.items()}

    # ---- supports ------------------------------------------------------------
    k_b = lay.bearing_stiffness
    for s, x in sx.items():
        D = deck_at[s]
        m.node_labels[D] = f"DECK_{s}"
        kind = lay.bearings[s]
        if s.startswith("P"):
            H = lay.pier_heights[s]
            n = max(4, math.ceil(H / h))
            zs = np.linspace(zb - H, zb, n + 1)
            col = [m.add_node((x, 0.0, z)) for z in zs]
            for a, b in zip(col[:-1], col[1:]):
                m.beams.append(Beam(a, b, "PIER", "PIER", (1.0, 0.0, 0.0), f"PIER_{s}"))
            m.supports[col[0]] = ALL
            m.node_labels[col[0]] = f"BASE_{s}"
            m.node_labels[col[-1]] = f"TOP_{s}"
            m.node_groups[f"PIER_{s}"] = col
            top = col[-1]
        else:
            if kind == "MONOLITHIC":          # integral abutment: clamp the deck end
                m.supports[D] = ALL
                continue
            top = m.add_node((x, 0.0, zb), f"GROUND_{s}")
            m.supports[top] = ALL
        if kind == "MONOLITHIC":
            m.rigid.append((D, top))
            continue
        bt = m.add_node((x, 0.0, zb), f"BEARING_{s}")
        m.rigid.append((D, bt))
        for dof in BEARING_DOFS[kind]:
            m.springs.append(Spring(bt, top, dof, k_b, f"BRG_{s}"))

    # ---- load cases ------------------------------------------------------------
    def member_load(case: str, e: int, q_global) -> None:
        b = m.beams[e]
        R, L = rotation(m.nodes[b.n1], m.nodes[b.n2], b.vec_y)
        f = uniform_load_vector(R, L, q_global)
        m.element_loads.setdefault(case, {})[e] = m.element_loads.get(case, {}).get(e, 0) + f
        F = m.load_cases.setdefault(case, np.zeros((m.n_nodes, 6)))
        F[b.n1] += f[:6]
        F[b.n2] += f[6:]

    q_deck = lay.deck_material.gamma * deck.A + lay.secondary_dead
    q_pier = lay.pier_material.gamma * pier.A
    for e, b in enumerate(m.beams):
        q = q_deck if b.section == "DECK" else q_pier
        member_load("DEAD", e, (0.0, 0.0, -q))
    if lay.lane_q or lay.lane_P:
        F = m.load_cases.setdefault("LIVE", np.zeros((m.n_nodes, 6)))
        if lay.lane_q:
            for e, b in enumerate(m.beams):
                if b.section == "DECK":
                    member_load("LIVE", e, (0.0, 0.0, -lay.lane_q * lay.lanes))
        if lay.lane_P:
            span = lay.lane_P_span or int(np.argmax(lay.spans)) + 1
            F[midspan[span], 2] -= lay.lane_P * lay.lanes
    return m


class BridgeAssemblyAgent(Agent):
    name = "assembly"
    requires = ("layout", "sections")
    provides = ("frame",)

    def __init__(self, element_size: float | None = None) -> None:
        super().__init__(element_size=element_size)

    def run(self, bb: Blackboard) -> None:
        lay: BridgeLayout = bb["layout"]
        h = self.params["element_size"] or lay.element_size or default_element_size(lay)
        frame = build_frame(lay, bb["sections"], h)
        self.info(bb, f"h={h:.3g} m -> {frame.n_nodes} nodes, {len(frame.beams)} beams "
                      f"({len(frame.beams_in('DECK'))} deck, {len(frame.beams_in('PIER'))} pier), "
                      f"{len(frame.springs)} bearing springs, {len(frame.rigid)} rigid links; "
                      f"load cases {list(frame.load_cases)}")
        bb.post("element_size_used", h, self.name)
        bb.post("frame", frame, self.name)


class BridgeValidatorAgent(Agent):
    """Engineering sanity checks; asks the assembler for a finer spine when a
    span has too few elements."""

    name = "bridge_validator"
    requires = ("frame",)
    optional = ("elevation",)
    provides = ("bridge_validation",)

    def __init__(self, min_elements_per_span: int = 10, max_rework: int = 2) -> None:
        super().__init__(min_elements_per_span=min_elements_per_span, max_rework=max_rework,
                         rework_done=0)

    def run(self, bb: Blackboard) -> None:
        m: FrameModel = bb["frame"]
        lay = m.layout
        deck = m.sections["DECK"]
        issues: list[str] = []

        kinds = [lay.bearings[s] for s in lay.supports]
        if not any(k in ("FIXED", "MONOLITHIC") for k in kinds):
            issues.append("no longitudinally fixed support: the deck can slide (mechanism)")
        n_lat = sum(k in ("FIXED", "SLIDING_X", "MONOLITHIC") for k in kinds)
        if n_lat < 2:
            issues.append("fewer than two supports restrain the deck transversely "
                          "(rotation about the vertical axis may be unrestrained)")
        if not m.load_cases:
            issues.append("no loads")
        for k, L in enumerate(lay.spans, 1):
            ratio = L / deck.height
            if not 8 <= ratio <= 45:
                issues.append(f"span {k}: L/h = {ratio:.1f} is unusual for a girder - "
                              "check the section scale")
        el = bb.get("elevation")
        if el and abs(el["deck_depth"] - deck.height) > 0.05 * deck.height:
            issues.append(f"girder depth {deck.height:.3g} m (section sheet) vs "
                          f"{el['deck_depth']:.3g} m (elevation)")
        elif el:
            self.info(bb, f"girder depth {deck.height:.3g} m consistent between section and elevation")
        for msg in issues:
            self.warn(bb, msg)

        per_span = [len(m.beams_in(f"DECK_S{k}")) for k in range(1, len(lay.spans) + 1)]
        if min(per_span) < self.params["min_elements_per_span"] and \
                self.params["rework_done"] < self.params["max_rework"]:
            self.params["rework_done"] += 1
            h = min(lay.spans) / self.params["min_elements_per_span"]
            bb.remove("bridge_validation")
            self.request(bb, "assembly", f"only {min(per_span)} elements in a span -> h={h:.3g} m",
                         element_size=h)
            return
        self.info(bb, f"{'PASSED' if not issues else 'PASSED WITH WARNINGS'}: "
                      f"{per_span} deck elements per span")
        bb.post("bridge_validation", {"ok": not issues, "issues": issues,
                                      "elements_per_span": per_span}, self.name)


def summarise(m: FrameModel, case: str, res) -> dict:
    lay = m.layout
    deck = m.node_groups["DECK"]
    x = m.nodes[deck, 0]
    uz = res.u[deck, 2]
    sx = np.r_[0.0, np.cumsum(lay.spans)]
    spans = []
    for k in range(len(lay.spans)):
        sel = (x >= sx[k] - 1e-9) & (x <= sx[k + 1] + 1e-9)
        i = np.argmin(uz[sel])
        spans.append({"span": k + 1, "min_uz_m": float(uz[sel][i]), "at_x_m": float(x[sel][i])})
    reac = {}
    for node, _ in m.supports.items():
        label = m.node_labels.get(node, str(node))
        reac[label] = res.reactions[node].tolist()
    # internal bending moment along the deck, sagging positive
    de = m.beams_in("DECK")
    My = [(float(m.nodes[m.beams[e].n1, 0]), float(res.end_forces[e, 4])) for e in de]
    My.append((float(m.nodes[m.beams[de[-1]].n2, 0]), float(-res.end_forces[de[-1], 10])))
    applied = res.applied[:, :3].sum(axis=0)
    total_r = res.reactions[:, :3].sum(axis=0)
    return {
        "spans": spans,
        "reactions": reac,
        "deck_moment": My,                 # (x, M) sagging positive, kN m
        "applied_kN": applied.tolist(),
        "reaction_sum_kN": total_r.tolist(),
        "equilibrium_error": float(np.linalg.norm(applied + total_r) / max(np.abs(applied).sum(), 1e-9)),
        "max_sagging_kNm": float(max(v for _, v in My)),
        "max_hogging_kNm": float(min(v for _, v in My)),
    }


class BridgeVerifierAgent(Agent):
    name = "bridge_verifier"
    requires = ("frame", "bridge_validation")
    provides = ("bridge_results",)

    def run(self, bb: Blackboard) -> None:
        m: FrameModel = bb["frame"]
        results = {}
        for case in m.load_cases:
            try:
                res = solve(m, case)
            except np.linalg.LinAlgError as exc:
                self.error(bb, f"{case}: {exc}")
                continue
            summ = summarise(m, case, res)
            summ["result"] = res
            results[case] = summ
            worst = min(summ["spans"], key=lambda s: s["min_uz_m"])
            self.info(bb, f"{case}: max deflection {-worst['min_uz_m'] * 1000:.2f} mm (span "
                          f"{worst['span']}), M+ {summ['max_sagging_kNm']:.4g} / M- "
                          f"{summ['max_hogging_kNm']:.4g} kN m, sum Rz={summ['reaction_sum_kN'][2]:.6g} kN"
                          f" (applied {-summ['applied_kN'][2]:.6g})")
            if summ["equilibrium_error"] > 1e-6:
                self.warn(bb, f"{case}: equilibrium error {summ['equilibrium_error']:.2e}")
        bb.post("bridge_results", results, self.name)
