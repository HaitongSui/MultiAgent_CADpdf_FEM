"""WriterAgent and ReportAgent: persist the model and a human-readable report."""

from __future__ import annotations

import re
from pathlib import Path

from ..core import Agent, Blackboard
from ..fem.writers import WRITERS


class WriterAgent(Agent):
    name = "writer"
    requires = ("fe_model", "validation", "out_dir")
    provides = ("outputs",)

    def __init__(self, formats: tuple[str, ...] = ("abaqus", "calculix", "nastran", "gmsh", "json")) -> None:
        unknown = set(formats) - set(WRITERS)
        if unknown:
            raise ValueError(f"unknown formats {sorted(unknown)}; choose from {sorted(WRITERS)}")
        super().__init__(formats=tuple(formats))

    def run(self, bb: Blackboard) -> None:
        model = bb["fe_model"]
        out_dir = Path(bb["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", model.spec.title) or "model"
        outputs = {}
        for fmt in self.params["formats"]:
            fn, suffix = WRITERS[fmt]
            outputs[fmt] = str(fn(model, out_dir / f"{stem}{suffix}"))
            self.info(bb, f"wrote {outputs[fmt]}")
        bb.post("outputs", outputs, self.name)


class ReportAgent(Agent):
    name = "report"
    requires = ("outputs", "fe_model", "validation", "out_dir")
    optional = ("verification",)
    provides = ("report",)

    def __init__(self, plot: bool = True) -> None:
        super().__init__(plot=plot)

    def run(self, bb: Blackboard) -> None:
        model = bb["fe_model"]
        spec, mesh, g = model.spec, model.mesh, model.geometry
        out_dir = Path(bb["out_dir"])
        val = bb["validation"]
        ver = bb.get("verification")
        x0, y0, x1, y1 = g.bbox
        lines = [
            f"# cad2fem report - {spec.title}", "",
            f"* Source PDF: `{bb.get('pdf_path')}`",
            f"* Spec source: {spec.source}",
            f"* Part: {x1 - x0:.4g} x {y1 - y0:.4g} mm, area {g.area:.6g} mm^2, "
            f"{len(g.holes)} hole(s), scale {g.scale_mm_per_pt:.5g} mm/pt",
            f"* Material: {spec.material.name} (E={spec.material.E:g} MPa, nu={spec.material.nu:g}), "
            f"thickness {spec.thickness:g} mm, {spec.analysis}",
            f"* Mesh: {mesh.n_nodes} nodes, {mesh.n_elements} Tri{3 * mesh.order} elements, "
            f"min angle {val['stats']['min_angle_deg']:.1f} deg",
            "", "## Boundary conditions", "",
        ]
        lines += [f"* `{n}` dofs {list(d)} = {v:g}  ({s}, {len(mesh.node_sets[n])} nodes)"
                  for n, d, v, s in model.constraints] or ["* none"]
        lines += ["", "## Loads", ""]
        for name, forces in model.nodal_forces.items():
            tot = forces.sum(axis=0)
            p = dict(model.pressures).get(name)
            kind = f"pressure {p:g} MPa" if p is not None else "force"
            lines.append(f"* `{name}`: {kind}, resultant ({tot[0]:.6g}, {tot[1]:.6g}) N")
        if not model.nodal_forces:
            lines.append("* none")
        if ver:
            lines += ["", "## Built-in CST verification", "",
                      f"* max |u| = {ver['max_displacement_mm']:.5g} mm",
                      f"* max von Mises = {ver['max_von_mises_mpa']:.5g} MPa "
                      f"(area-weighted mean {ver['mean_von_mises_mpa']:.5g} MPa)",
                      f"* equilibrium error = {ver['equilibrium_error']:.2e}"]
        lines += ["", "## Output files", ""] + [f"* {k}: `{v}`" for k, v in bb["outputs"].items()]
        lines += ["", "## Agent log", "", "```"] + [str(m) for m in bb.messages] + ["```", ""]

        report_path = out_dir / "report.md"
        if self.params["plot"]:
            png = self._plot(model, ver, out_dir / "mesh.png")
            if png:
                lines.insert(2, f"![mesh]({png.name})\n")
        report_path.write_text("\n".join(lines))
        self.info(bb, f"wrote {report_path}")
        bb.post("report", str(report_path), self.name)

    @staticmethod
    def _plot(model, ver, path: Path):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.tri import Triangulation
        except ImportError:
            return None
        mesh = model.mesh
        tri = Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.corner_elements())
        fig, ax = plt.subplots(figsize=(9, 5))
        if ver:
            tp = ax.tripcolor(tri, facecolors=ver["result"].von_mises, cmap="viridis",
                              edgecolors="none")
            fig.colorbar(tp, ax=ax, label="von Mises (MPa)")
        ax.triplot(tri, lw=0.3, color="k", alpha=0.6)
        colors = iter(["tab:red", "tab:orange", "tab:purple", "tab:brown", "tab:pink"])
        for name, *_ in model.constraints:
            p = mesh.nodes[mesh.node_sets[name]]
            ax.plot(p[:, 0], p[:, 1], "s", ms=3, color="tab:blue", label=name)
        for name in model.nodal_forces:
            p = mesh.nodes[mesh.node_sets[name]]
            ax.plot(p[:, 0], p[:, 1], "^", ms=3, color=next(colors, "tab:red"), label=name)
        ax.set_aspect("equal")
        ax.set_title(f"{model.spec.title}: {mesh.n_elements} elements")
        if model.constraints or model.nodal_forces:
            ax.legend(loc="upper left", bbox_to_anchor=(1.25, 1), fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path

