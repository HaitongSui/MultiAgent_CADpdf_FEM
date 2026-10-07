"""Nastran bulk-data (free-field) writer: CTRIA3/CTRIA6 membrane model."""

from __future__ import annotations

from pathlib import Path

from ...models import FEModel


def _r(v: float) -> str:
    """Nastran real: must carry a decimal point."""
    s = f"{v:.8g}"
    if "e" in s:
        mant, exp = s.split("e")
        if "." not in mant:
            mant += "."
        return f"{mant}e{exp}"
    return s if "." in s else s + "."


def write_nastran(model: FEModel, path: str | Path) -> Path:
    path = Path(path)
    mesh, spec = model.mesh, model.spec
    mat = spec.material
    out = [
        f"$ cad2fem: {spec.title} - units mm, N, MPa, t/mm^3",
        "$ Membrane (plane stress) model: out-of-plane and rotational DOFs held by GRID PS=3456",
    ]
    if spec.analysis == "PLANE_STRAIN":
        out.append("$ WARNING: plane strain requested - written as plane-stress membrane (PSHELL)")
    out += ["SOL 101", "CEND", f"TITLE = {spec.title}", "SUBCASE 1", "  SPC = 1", "  LOAD = 1",
            "  DISPLACEMENT = ALL", "  SPCFORCES = ALL", "  STRESS = ALL", "BEGIN BULK",
            "PARAM,POST,-1"]
    out += [f"GRID,{i + 1},,{_r(x)},{_r(y)},0.,,3456" for i, (x, y) in enumerate(mesh.nodes)]
    card = "CTRIA6" if mesh.order == 2 else "CTRIA3"
    out += [f"{card},{e + 1},1," + ",".join(str(n + 1) for n in el)
            for e, el in enumerate(mesh.elements)]
    out.append(f"PSHELL,1,1,{_r(spec.thickness)}")
    out.append(f"MAT1,1,{_r(mat.E)},,{_r(mat.nu)},{_r(mat.density)}")
    for name, dofs, value, desc in model.constraints:
        out.append(f"$ {desc} ({name})")
        comp = "".join(str(d) for d in sorted(dofs))
        if value:
            out += [f"SPC,1,{int(n) + 1},{comp},{_r(value)}" for n in mesh.node_sets[name]]
        else:
            out += [f"SPC1,1,{comp},{int(n) + 1}" for n in mesh.node_sets[name]]
    for name, forces in model.nodal_forces.items():
        out.append(f"$ {name} (consistent nodal forces)")
        for nid, (fx, fy) in zip(mesh.node_sets[name], forces):
            if fx or fy:
                out.append(f"FORCE,1,{int(nid) + 1},0,1.,{_r(fx)},{_r(fy)},0.")
    out.append("ENDDATA")
    path.write_text("\n".join(out) + "\n")
    return path
