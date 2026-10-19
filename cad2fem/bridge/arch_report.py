"""ArchReportAgent: report.md + arch_bridge.png for the arch pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core import Agent, Blackboard
from .arch import ArchAssumptions, ArchGeometry
from .arch_reading import ArchReading
from .models import FrameModel
from .report import GRID, INK, INK_2, SERIES


def midspan_simple_moment(m: FrameModel, res, x_mid: float) -> float:
    """Simple-beam moment at midspan from the applied vertical loads (whole bridge)."""
    x = m.nodes[:, 0]
    fz = res.applied[:, 2]
    R_left = -sum(fz * (x.max() - x)) / (x.max() - x.min())
    left = x < x_mid - 1e-9
    return float(R_left * (x_mid - x.min()) + sum(fz[left] * (x_mid - x[left])))


class ArchReportAgent(Agent):
    name = "arch_report"
    requires = ("outputs", "frame", "arch_geometry", "arch_reading", "out_dir")
    optional = ("bridge_results", "arch_assumptions", "arch_knowledge")
    provides = ("report",)

    def __init__(self, plot: bool = True) -> None:
        super().__init__(plot=plot)

    def run(self, bb: Blackboard) -> None:
        m: FrameModel = bb["frame"]
        geo: ArchGeometry = bb["arch_geometry"]
        r: ArchReading = bb["arch_reading"]
        a: ArchAssumptions = bb.get("arch_assumptions") or ArchAssumptions()
        res = bb.get("bridge_results") or {}
        out = Path(bb["out_dir"])
        zs, zc, _ = geo.params
        L = [f"# {r.bridge_name} - global arch model", ""]
        if self.params["plot"] and (png := self._plot(m, geo, res, out / "arch_bridge.png")):
            L += [f"![arch]({png.name})", ""]
        L += [f"* Type: {r.structure_type}; span {r.span_mm / 1000:g} m, arch span "
              f"{r.arch_span_mm / 1000:g} m, width {r.deck_width_mm / 1000:g} m",
              f"* Read from: {', '.join(r.sheets)}",
              f"* Reader: {r.reader}",
              f"* Arch rib axis: springs at x = {geo.x_spring[0]:.2f} / {geo.x_spring[1]:.2f} m, "
              f"{zs + geo.rib_offset:.3f} m -> {zc + geo.rib_offset:.3f} m above the girder axis "
              f"(fitted to {len(geo.cable_length)} hanger lengths, RMS {geo.fit_rms_mm:.1f} mm)",
              f"* Model: {m.n_nodes} nodes, {len(m.beams)} members, {len(m.rigid)} rigid links; "
              "two arch planes (girder + rib + V hangers), cross beams at every station, "
              "S1 pinned / S2 longitudinally free", "",
              "## Drawing-reading checks", "",
              "| Check | Value | Expected | Tol. | OK |", "|---|---|---|---|---|"]
        L += [f"| {c['check']} | {c['value']:g} {c['unit']} | {c['expected']:g} | ±{c['tolerance']:g} | "
              f"{'yes' if c['ok'] else '**no**'} |" for c in geo.checks]
        L += ["", "| Hanger | Drawing (mm) | Model (mm) | Diff (mm) |", "|---|---|---|---|"]
        L += [f"| {n} | {1000 * d:.1f} | {1000 * mo:.1f} | {1000 * (mo - d):+.1f} |"
              for n, d, mo in zip(geo.anchor_names, geo.cable_length, geo.cable_model_length)]
        kn = bb.get("arch_knowledge")
        if kn is not None and not kn.empty:
            L += ["", "## Values from RSI memory (verified detail sheets)", ""]
            L += [f"* {k}: {v}" for k, v in kn.sources.items()]
            L += ["", "## Values not on the sheets read (ASSUMED where memory has no value)", ""]
        else:
            L += ["", "## Values not on J-5 / J-8 (ASSUMED - replace from detail sheets)", ""]
        L += ["| Parameter | Value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in a.rows()]
        L += ["", "| Section | A (m2) | Iyy (m4) | Izz (m4) | J (m4) | Source |", "|---|---|---|---|---|---|"]
        L += [f"| {k} | {s.A:.5f} | {s.Iyy:.5f} | {s.Izz:.5f} | {s.J:.5f} | {s.source} |"
              for k, s in m.sections.items()]
        L += ["", "Load notes: " + "; ".join(m.layout.notes), ""]
        if r.uncertain:
            L += ["## Reader's doubts", ""] + [f"* {u}" for u in r.uncertain] + [""]
        for case, s in res.items():
            M0 = midspan_simple_moment(m, s["result"], geo.span / 2)
            couple = -s["rib_axial_crown_kN"] * 2 * (zc + geo.rib_offset)
            L += [f"## Load case {case}", "",
                  f"* Applied {-s['applied_kN'][2]:.1f} kN, reactions {s['reaction_sum_kN'][2]:.1f} kN "
                  f"(equilibrium error {s['equilibrium_error']:.1e})",
                  f"* Midspan deflection {s['midspan_deflection_mm']:.1f} mm (max {s['max_girder_deflection_mm']:.1f} mm)",
                  f"* Rib axial force at the crown {s['rib_axial_crown_kN']:.0f} kN, girder (tie) axial "
                  f"force at midspan {s['girder_axial_midspan_kN']:.0f} kN (per plane)",
                  f"* Arch couple (2 planes x N_rib x rise) {couple:.0f} kN m = "
                  f"{100 * couple / M0:.0f}% of the simple-beam midspan moment {M0:.0f} kN m",
                  f"* Girder G1 moment: max {s['max_sagging_kNm']:.0f} / min {s['max_hogging_kNm']:.0f} kN m",
                  "", "| Hanger (G1) | Force (kN) |", "|---|---|"]
            L += [f"| {k} | {v:.1f} |" for k, v in s["hanger_force_kN"].items()]
            L.append("")
        L += ["## Output files", ""] + [f"* {k}: `{v}`" for k, v in bb["outputs"].items()]
        L += ["", "## Agent log", "", "```"] + [str(msg) for msg in bb.messages] + ["```", ""]
        path = out / "report.md"
        path.write_text("\n".join(L), encoding="utf-8")
        self.info(bb, f"wrote {path}")
        bb.post("report", str(path), self.name)

    @staticmethod
    def _plot(m: FrameModel, geo: ArchGeometry, res: dict, path: Path):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None
        plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                             "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
        fig = plt.figure(figsize=(11, 9))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.0, 1.0], hspace=0.5)

        # 1. elevation of plane G1, deformed shape under DEAD
        ax = fig.add_subplot(gs[0])
        dead = res.get("DEAD")
        u = dead["result"].u if dead else None
        scale = 5.0
        for b in m.beams:
            if b.group.startswith("XBEAM") or "_G2" in b.group:
                continue
            p = m.nodes[[b.n1, b.n2]]
            cable = b.section.startswith("CABLE")
            ax.plot(p[:, 0], p[:, 2], color=INK_2, lw=0.8 if cable else 1.6, zorder=2)
            if u is not None:
                q = p + scale * u[[b.n1, b.n2], :3]
                ax.plot(q[:, 0], q[:, 2], color=SERIES[0], lw=0.8 if cable else 1.4, alpha=0.9, zorder=3)
        for mst, slv in m.rigid:
            if m.nodes[mst, 1] > 0 and abs(m.nodes[mst, 0] - m.nodes[slv, 0]) < 1e-6 and \
                    abs(m.nodes[mst, 2]) < 1e-9:                          # springing links
                p = m.nodes[[mst, slv]]
                ax.plot(p[:, 0], p[:, 2], color=INK, lw=2.2, zorder=4)
        ax.plot([], [], color=INK_2, lw=1.6, label="model (plane G1)")
        ax.plot([], [], color=INK, lw=2.2, label="rigid springing link")
        if u is not None:
            ax.plot([], [], color=SERIES[0], lw=1.4, label=f"DEAD deformed (x{scale:g})")
        for n, dofs in m.supports.items():
            ax.plot(*m.nodes[n, [0, 2]], marker="^" if 1 in dofs else "o", ms=8, color=INK,
                    mec="white", mew=1.5, ls="none", zorder=5)
        for lab, x in geo.stations.items():
            if lab.startswith("S") or lab in ("C1", "C5", "C10", "C15", "C19"):
                ax.annotate(lab, (x, -1.6), ha="center", fontsize=7, color=INK_2)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_ylim(-3, 12)
        ax.set_title("Arch plane G1: girder (tie), arch rib and V hangers", loc="left", fontsize=10)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.grid(color=GRID, lw=0.6)
        ax.legend(loc="upper right", frameon=False, fontsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        # 2. hanger forces per load case (grouped bars, G1)
        ax = fig.add_subplot(gs[1])
        names = geo.anchor_names
        k = len(res) or 1
        w = 0.8 / k
        for i, (case, s) in enumerate(res.items()):
            vals = [s["hanger_force_kN"][n] for n in names]
            ax.bar(np.arange(len(names)) + (i - (k - 1) / 2) * w, vals, w * 0.9, color=SERIES[i % 3],
                   label=case)
        ax.set_xticks(np.arange(len(names)), names, rotation=90, fontsize=7)
        ax.set_ylabel("kN (tension +)")
        ax.set_title("Hanger forces, plane G1", loc="left", fontsize=10)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.legend(frameon=False, fontsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        # 3. girder moment
        ax = fig.add_subplot(gs[2])
        ax.axhline(0, color=INK_2, lw=0.8)
        for i, (case, s) in enumerate(res.items()):
            xm = np.array(s["girder_moment"])
            ax.plot(xm[:, 0], xm[:, 1], color=SERIES[i % 3], lw=1.6, label=case)
        for x in geo.x_spring:
            ax.axvline(x, color=GRID, lw=0.8)
        ax.invert_yaxis()
        ax.set_title("Girder G1 bending moment (kN m, sagging +, drawn on the tension side)",
                     loc="left", fontsize=10)
        ax.set_xlabel("X (m)")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.legend(frameon=False, fontsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path
