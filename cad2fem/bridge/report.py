"""BridgeWriterAgent and BridgeReportAgent."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core import Agent, Blackboard
from .models import FrameModel
from .writers import WRITERS, ascii_name

# validated categorical slots 1-3 (pass all-pairs CVD checks); neutral inks
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
INK, INK_2, GRID = "#1f1f1e", "#5c5b55", "#d9d8d2"


class BridgeWriterAgent(Agent):
    name = "bridge_writer"
    requires = ("frame", "bridge_validation", "out_dir")
    provides = ("outputs",)

    def __init__(self, formats: tuple[str, ...] = tuple(WRITERS)) -> None:
        unknown = set(formats) - set(WRITERS)
        if unknown:
            raise ValueError(f"unknown bridge formats {sorted(unknown)}; choose from {sorted(WRITERS)}")
        super().__init__(formats=tuple(formats))

    def run(self, bb: Blackboard) -> None:
        m: FrameModel = bb["frame"]
        out = Path(bb["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        stem = ascii_name(m.layout.title)
        outputs = {}
        for fmt in self.params["formats"]:
            fn, suffix = WRITERS[fmt]
            outputs[fmt] = str(fn(m, out / f"{stem}{suffix}"))
            self.info(bb, f"wrote {outputs[fmt]}")
        bb.post("outputs", outputs, self.name)


class BridgeReportAgent(Agent):
    name = "bridge_report"
    requires = ("outputs", "frame", "bridge_validation", "out_dir")
    optional = ("bridge_results", "elevation", "sheets", "sheet_kinds")
    provides = ("report",)

    def __init__(self, plot: bool = True) -> None:
        super().__init__(plot=plot)

    def run(self, bb: Blackboard) -> None:
        m: FrameModel = bb["frame"]
        lay = m.layout
        out = Path(bb["out_dir"])
        res = bb.get("bridge_results") or {}
        d, p = m.sections["DECK"], m.sections["PIER"]
        L = [f"# cad2fem bridge model - {lay.title}", ""]
        if self.params["plot"] and (png := self._plot(m, res, out / "bridge.png")):
            L += [f"![bridge]({png.name})", ""]
        if bb.has("sheets"):
            L += ["## Drawing set", ""]
            L += [f"* `{s.source}` -> {k}" + (f" ({s.title})" if s.title else "")
                  for s, k in zip(bb["sheets"], bb["sheet_kinds"])]
            L.append("")
        L += ["## Layout", "",
              f"* Spans: {' + '.join(f'{s:g}' for s in lay.spans)} m (total {lay.length:g} m)",
              "* Piers: " + ", ".join(f"{k} H={v:g} m" for k, v in lay.pier_heights.items()),
              "* Bearings: " + ", ".join(f"{k} {v}" for k, v in lay.bearings.items()),
              f"* Materials: deck {lay.deck_material.name} (E={lay.deck_material.E / 1e3:g} MPa), "
              f"piers {lay.pier_material.name} (E={lay.pier_material.E / 1e3:g} MPa), "
              f"unit weight {lay.deck_material.gamma:g} kN/m3",
              f"* Loads: secondary dead {lay.secondary_dead:g} kN/m; lane q={lay.lane_q:g} kN/m, "
              f"P={lay.lane_P:g} kN, x{lay.lanes} lane(s)", "",
              "## Sections", "",
              "| | A (m2) | Iyy (m4) | Izz (m4) | J (m4) | b x h (m) | source |",
              "|---|---|---|---|---|---|---|"]
        for s in (d, p):
            L.append(f"| {s.name} | {s.A:.4f} | {s.Iyy:.4f} | {s.Izz:.4f} | {s.J:.4f} | "
                     f"{s.width:.3g} x {s.height:.3g} | {s.source} |")
        L += ["", f"Model: {m.n_nodes} nodes, {len(m.beams)} beams, {len(m.springs)} bearing springs, "
                  f"{len(m.rigid)} rigid links.", ""]
        for case, r in res.items():
            L += [f"## Load case {case}", "",
                  f"* Applied vertical load {-r['applied_kN'][2]:.6g} kN, reactions "
                  f"{r['reaction_sum_kN'][2]:.6g} kN (equilibrium error {r['equilibrium_error']:.1e})",
                  f"* Deck moment: max sagging {r['max_sagging_kNm']:.5g} kN m, max hogging "
                  f"{r['max_hogging_kNm']:.5g} kN m", "",
                  "| Span | max deflection (mm) | at x (m) |", "|---|---|---|"]
            L += [f"| {s['span']} | {-s['min_uz_m'] * 1e3:.2f} | {s['at_x_m']:.2f} |" for s in r["spans"]]
            L += ["", "| Support node | Rx (kN) | Ry (kN) | Rz (kN) |", "|---|---|---|---|"]
            L += [f"| {k} | {v[0]:.2f} | {v[1]:.2f} | {v[2]:.2f} |" for k, v in r["reactions"].items()]
            L.append("")
        L += ["## Output files", ""] + [f"* {k}: `{v}`" for k, v in bb["outputs"].items()]
        L += ["", "## Agent log", "", "```"] + [str(msg) for msg in bb.messages] + ["```", ""]
        path = out / "report.md"
        path.write_text("\n".join(L))
        self.info(bb, f"wrote {path}")
        bb.post("report", str(path), self.name)

    @staticmethod
    def _plot(m: FrameModel, res: dict, path: Path):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Polygon
        except ImportError:
            return None
        plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                             "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
        fig = plt.figure(figsize=(11, 8.2))
        gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 1.0, 1.0], hspace=0.55, wspace=0.35)
        lay = m.layout

        # ---- 1. model elevation + deformed shape (DEAD) -----------------------
        ax = fig.add_subplot(gs[0, :])
        for b in m.beams:
            p = m.nodes[[b.n1, b.n2]]
            ax.plot(p[:, 0], p[:, 2], color=INK_2, lw=1.0 if b.section == "PIER" else 1.6,
                    solid_capstyle="round")
        if "DEAD" in res:
            u = res["DEAD"]["result"].u
            span = max(float(np.ptp(m.nodes[:, 0])), 1.0)
            scale = 0.06 * span / max(np.abs(u[:, :3]).max(), 1e-12)
            for k, b in enumerate(m.beams):
                p = m.nodes[[b.n1, b.n2]] + scale * u[[b.n1, b.n2], :3]
                ax.plot(p[:, 0], p[:, 2], color=SERIES[0], lw=1.6,
                        label=f"DEAD deformed (x{scale:.0f})" if k == 0 else None)
        for n in m.supports:
            ax.plot(*m.nodes[n, [0, 2]], marker="^", ms=8, color=INK, mec="white", mew=1.5,
                    ls="none", zorder=5)
        for s, x in lay.support_x.items():
            ax.annotate(f"{s} {lay.bearings[s].replace('_', '-').lower()}", (x, 0.6),
                        ha="center", fontsize=8, color=INK_2)
        ax.set_title(f"{ascii_name(lay.title)}: global spine model "
                     f"({' + '.join(f'{v:g}' for v in lay.spans)} m)", loc="left", fontsize=10)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(color=GRID, lw=0.6)
        ax.legend(loc="lower right", frameon=False, fontsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        # ---- 2. deck bending moment per load case -----------------------------
        ax = fig.add_subplot(gs[1, :])
        ax.axhline(0, color=INK_2, lw=0.8)
        for k, (case, r) in enumerate(res.items()):
            xm = np.array(r["deck_moment"])
            c = SERIES[k % len(SERIES)]
            ax.plot(xm[:, 0], xm[:, 1], color=c, lw=1.6, label=case)
            for idx in (np.argmax(xm[:, 1]), np.argmin(xm[:, 1])):
                ax.plot(*xm[idx], "o", ms=5, color=c, mec="white", mew=1.2)
                ax.annotate(f"{xm[idx, 1]:,.0f}", xm[idx], textcoords="offset points",
                            xytext=(4, 6 if xm[idx, 1] > 0 else -12), fontsize=8, color=INK)
        for x in lay.support_x.values():
            ax.axvline(x, color=GRID, lw=0.8, zorder=0)
        ax.invert_yaxis()   # engineering convention: sagging drawn below the axis
        ax.set_title("Deck bending moment (kN m, sagging +, drawn on the tension side)",
                     loc="left", fontsize=10)
        ax.set_xlabel("X (m)")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=max(len(res), 1))
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        # ---- 3. sections ------------------------------------------------------
        for j, key in enumerate(("DECK", "PIER")):
            s = m.sections[key]
            ax = fig.add_subplot(gs[2, j])
            if s.outline is not None:
                ax.add_patch(Polygon(s.outline, closed=True, fc="#e8eef8", ec=INK, lw=1.0))
                for h in s.holes:
                    ax.add_patch(Polygon(h, closed=True, fc="white", ec=INK, lw=1.0))
                ax.plot(*np.r_[s.outline[:, 0].min() + s.cy, s.outline[:, 1].min() + s.cz],
                        "+", color=SERIES[1], ms=10, mew=1.5)
            else:
                ax.add_patch(Polygon([[0, 0], [s.width, 0], [s.width, s.height], [0, s.height]],
                                     closed=True, fc="#e8eef8", ec=INK, lw=1.0, ls="--"))
            ax.autoscale_view()
            ax.set_aspect("equal")
            ax.set_title(f"{key} section ({'assumed' if s.source == 'default' else 'from drawing'})",
                         loc="left", fontsize=10)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
        ax = fig.add_subplot(gs[2, 2])
        ax.axis("off")
        rows = [("", "DECK", "PIER")] + [
            (lbl, f"{getattr(m.sections['DECK'], a):.4g}", f"{getattr(m.sections['PIER'], a):.4g}")
            for lbl, a in (("A (m²)", "A"), ("Iyy (m⁴)", "Iyy"), ("Izz (m⁴)", "Izz"), ("J (m⁴)", "J"),
                           ("width (m)", "width"), ("height (m)", "height"))]
        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                ax.text(j * 0.36, 1 - i * 0.14, cell, transform=ax.transAxes, fontsize=9,
                        color=INK if i else INK_2, weight="bold" if i == 0 else "normal")
        fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path
