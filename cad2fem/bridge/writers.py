"""Writers for the global bridge model (units m, kN, kPa, t).

* Abaqus  : B33 (Euler-Bernoulli) beams with *BEAM GENERAL SECTION, SPRING2
            bearings, *MPC BEAM rigid links, one perturbation step per load case
* Nastran : CBAR/PBAR, CELAS2 bearings, RBE2 rigid links, one SUBCASE per case
* OpenSees: runnable OpenSeesPy script (elasticBeamColumn, zeroLength, rigidLink)
            that solves every load case and writes the results to JSON
* JSON    : solver-neutral dump
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .models import FrameModel

DOF_NAMES = ("UX", "UY", "UZ", "RX", "RY", "RZ")


def ascii_name(title: str, default: str = "BRIDGE") -> str:
    """Solver-safe ASCII name (Abaqus job names / Nastran titles reject CJK)."""
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", title).strip("_") or default


def _r(v: float) -> str:
    s = f"{v:.9g}"
    if "e" in s:
        mant, exp = s.split("e")
        return (mant if "." in mant else mant + ".") + "e" + exp
    return s if "." in s else s + "."


def _chunks(ids, n=16):
    ids = list(ids)
    return [", ".join(map(str, ids[k:k + n])) for k in range(0, len(ids), n)]


def _groups(m: FrameModel) -> dict[str, list[int]]:
    g: dict[str, list[int]] = defaultdict(list)
    for e, b in enumerate(m.beams):
        g[b.group].append(e)
    return g


# --------------------------------------------------------------------------- #
def write_abaqus(m: FrameModel, path: str | Path) -> Path:
    path = Path(path)
    lay = m.layout
    out = ["*HEADING", f"cad2fem bridge model: {ascii_name(lay.title)} | spans {lay.spans} m | units m, kN, kPa, t",
           "** X longitudinal, Y transverse, Z up; deck spine at girder centroid",
           "*NODE, NSET=NALL"]
    out += [f"{i + 1}, {_r(x)}, {_r(y)}, {_r(z)}" for i, (x, y, z) in enumerate(m.nodes)]
    nb = len(m.beams)
    for group, els in _groups(m).items():
        out.append(f"*ELEMENT, TYPE=B33, ELSET={group}")
        out += [f"{e + 1}, {m.beams[e].n1 + 1}, {m.beams[e].n2 + 1}" for e in els]
    for group, els in _groups(m).items():
        b = m.beams[els[0]]
        s, mat = m.sections[b.section], m.materials[b.material]
        out += [f"*BEAM GENERAL SECTION, ELSET={group}, SECTION=GENERAL, DENSITY={_r(mat.density)}",
                f"{_r(s.A)}, {_r(s.Iyy)}, {_r(s.Iyz)}, {_r(s.Izz)}, {_r(s.J)}",
                ", ".join(_r(v) for v in b.vec_y),          # n1 = member local y
                f"{_r(mat.E)}, {_r(mat.G)}"]
    # bearings: one SPRING2 element per restrained dof
    by_dof: dict[int, list] = defaultdict(list)
    for sp in m.springs:
        by_dof[sp.dof].append(sp)
    eid = nb
    for dof, sps in sorted(by_dof.items()):
        out.append(f"*ELEMENT, TYPE=SPRING2, ELSET=BRG_{DOF_NAMES[dof - 1]}")
        for sp in sps:
            eid += 1
            out.append(f"{eid}, {sp.n1 + 1}, {sp.n2 + 1}")
        out += [f"*SPRING, ELSET=BRG_{DOF_NAMES[dof - 1]}", f"{dof}, {dof}", _r(sps[0].k)]
    if m.rigid:
        out.append("*MPC")
        out += [f"BEAM, {s + 1}, {mm + 1}" for mm, s in m.rigid]
    for name, ids in m.node_groups.items():
        out.append(f"*NSET, NSET={name}")
        out += _chunks(i + 1 for i in ids)
    out.append("*NSET, NSET=SUPPORTS")
    out += _chunks(n + 1 for n in m.supports)
    out += ["*BOUNDARY"] + [f"{n + 1}, {d}, {d}" for n, dofs in m.supports.items() for d in dofs]
    for case, F in m.load_cases.items():
        out += ["**", f"*STEP, NAME={case}, PERTURBATION", f"{case}", "*STATIC", "*CLOAD"]
        for n, f in enumerate(F):
            out += [f"{n + 1}, {d + 1}, {_r(v)}" for d, v in enumerate(f) if abs(v) > 0]
        out += ["*OUTPUT, FIELD", "*NODE OUTPUT", "U, RF", "*ELEMENT OUTPUT", "SF",
                "*NODE PRINT, NSET=SUPPORTS, TOTALS=YES", "RF", "*END STEP"]
    path.write_text("\n".join(out) + "\n")
    return path


# --------------------------------------------------------------------------- #
def write_nastran(m: FrameModel, path: str | Path) -> Path:
    path = Path(path)
    lay = m.layout
    cases = list(m.load_cases)
    out = [f"$ cad2fem bridge model: {ascii_name(lay.title)} - units m, kN, kPa, t", "SOL 101",
           "CEND", f"TITLE = {ascii_name(lay.title)}", "SPC = 100", "DISPLACEMENT = ALL", "SPCFORCES = ALL",
           "FORCE = ALL"]
    for k, case in enumerate(cases, 1):
        out += [f"SUBCASE {k}", f"  LABEL = {case}", f"  LOAD = {k}"]
    out += ["BEGIN BULK", "PARAM,POST,-1"]
    out += [f"GRID,{i + 1},,{_r(x)},{_r(y)},{_r(z)}" for i, (x, y, z) in enumerate(m.nodes)]
    pid = {}
    for name, s in m.sections.items():
        mid = list(m.materials).index(name) + 1 if name in m.materials else 1
        pid[name] = len(pid) + 1
        # PBAR I1: bending in plane 1 (x-v plane) = Izz ; I2 = Iyy
        out.append(f"PBAR,{pid[name]},{mid},{_r(s.A)},{_r(s.Izz)},{_r(s.Iyy)},{_r(s.J)}")
    for k, (name, mat) in enumerate(m.materials.items(), 1):
        out.append(f"MAT1,{k},{_r(mat.E)},,{_r(mat.nu)},{_r(mat.density)}")
    for e, b in enumerate(m.beams):
        vx, vy, vz = b.vec_y
        out.append(f"CBAR,{e + 1},{pid[b.section]},{b.n1 + 1},{b.n2 + 1},{_r(vx)},{_r(vy)},{_r(vz)}")
    eid = len(m.beams)
    for sp in m.springs:
        eid += 1
        out.append(f"CELAS2,{eid},{_r(sp.k)},{sp.n1 + 1},{sp.dof},{sp.n2 + 1},{sp.dof}")
    for mm, s in m.rigid:
        eid += 1
        out.append(f"RBE2,{eid},{mm + 1},123456,{s + 1}")
    for n, dofs in m.supports.items():
        out.append(f"SPC1,100,{''.join(map(str, dofs))},{n + 1}")
    for k, case in enumerate(cases, 1):
        out.append(f"$ load case {case}")
        for n, f in enumerate(m.load_cases[case]):
            if np.any(f[:3]):
                out.append(f"FORCE,{k},{n + 1},0,1.,{_r(f[0])},{_r(f[1])},{_r(f[2])}")
            if np.any(f[3:]):
                out.append(f"MOMENT,{k},{n + 1},0,1.,{_r(f[3])},{_r(f[4])},{_r(f[5])}")
    out.append("ENDDATA")
    path.write_text("\n".join(out) + "\n")
    return path


# --------------------------------------------------------------------------- #
def write_opensees(m: FrameModel, path: str | Path) -> Path:
    """Runnable OpenSeesPy script; results go to <script>_results.json."""
    path = Path(path)
    lay = m.layout
    L = [f'"""OpenSeesPy model of {lay.title} generated by cad2fem (units m, kN, kPa, t).',
         "", "Run:  python " + path.name, '"""', "import json", "import sys", "",
         "import openseespy.opensees as ops", "", "ops.wipe()",
         "ops.model('basic', '-ndm', 3, '-ndf', 6)"]
    L += [f"ops.node({i + 1}, {x!r}, {y!r}, {z!r})" for i, (x, y, z) in enumerate(m.nodes.tolist())]
    L += [f"ops.fix({n + 1}, " + ", ".join("1" if d in dofs else "0" for d in range(1, 7)) + ")"
          for n, dofs in m.supports.items()]
    # geomTransf vecxz must lie in the local x-z plane: use local z = x cross y
    L += ["", "def _cross(a, b):",
          "    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]", "",
          "_tr = {}", "def transf(n1, n2, vy):",
          "    x1, x2 = ops.nodeCoord(n1), ops.nodeCoord(n2)",
          "    ex = [b - a for a, b in zip(x1, x2)]",
          "    ez = _cross(ex, vy)",
          "    key = tuple(round(v, 9) for v in ez)",
          "    if key not in _tr:",
          "        _tr[key] = len(_tr) + 1",
          "        ops.geomTransf('Linear', _tr[key], *ez)",
          "    return _tr[key]", ""]
    for e, b in enumerate(m.beams):
        s, mat = m.sections[b.section], m.materials[b.material]
        props = ", ".join(repr(float(v)) for v in (s.A, mat.E, mat.G, s.J, s.Iyy, s.Izz))
        vy = [float(v) for v in b.vec_y]
        L.append(f"ops.element('elasticBeamColumn', {e + 1}, {b.n1 + 1}, {b.n2 + 1}, {props}, "
                 f"transf({b.n1 + 1}, {b.n2 + 1}, {vy!r}))")
    eid = len(m.beams)
    springs: dict[tuple[int, int], list] = defaultdict(list)
    for sp in m.springs:
        springs[(sp.n1, sp.n2)].append(sp)
    for k, ((n1, n2), sps) in enumerate(springs.items(), 1):
        mats = []
        for j, sp in enumerate(sps):
            tag = 1000 * k + j
            L.append(f"ops.uniaxialMaterial('Elastic', {tag}, {float(sp.k)!r})")
            mats.append(tag)
        eid += 1
        L.append(f"ops.element('zeroLength', {eid}, {n1 + 1}, {n2 + 1}, '-mat', "
                 f"{', '.join(map(str, mats))}, '-dir', {', '.join(str(sp.dof) for sp in sps)})")
    L += [f"ops.rigidLink('beam', {mm + 1}, {s + 1})" for mm, s in m.rigid]
    L += ["", "results = {}", "ops.timeSeries('Linear', 1)"]
    for k, (case, F) in enumerate(m.load_cases.items(), 1):
        L += [f"# ---- load case {case}", f"ops.pattern('Plain', {k}, 1)"]
        L += [f"ops.load({n + 1}, " + ", ".join(repr(float(v)) for v in f) + ")"
              for n, f in enumerate(F) if np.any(f)]
        L += ["ops.constraints('Transformation')", "ops.numberer('RCM')",
              "ops.system('UmfPack')", "ops.test('NormDispIncr', 1e-10, 10)",
              "ops.algorithm('Linear')", "ops.integrator('LoadControl', 1.0)",
              "ops.analysis('Static')",
              "assert ops.analyze(1) == 0, 'analysis failed'", "ops.reactions()",
              f"results[{case!r}] = {{",
              "    'disp': {n: ops.nodeDisp(n) for n in ops.getNodeTags()},",
              f"    'reactions': {{n: ops.nodeReaction(n) for n in {[n + 1 for n in m.supports]!r}}},",
              "}",
              f"ops.remove('loadPattern', {k})", "ops.wipeAnalysis()", "ops.reset()"]
    L += ["", "out = sys.argv[1] if len(sys.argv) > 1 else __file__.replace('.py', '_results.json')",
          "with open(out, 'w') as fh:", "    json.dump(results, fh)", "print('results written to', out)"]
    path.write_text("\n".join(L) + "\n")
    return path


# --------------------------------------------------------------------------- #
def write_json(m: FrameModel, path: str | Path) -> Path:
    path = Path(path)
    lay = m.layout
    data = {
        "units": {"length": "m", "force": "kN", "stress": "kPa", "mass": "t"},
        "layout": {"title": lay.title, "spans": lay.spans, "pier_heights": lay.pier_heights,
                   "bearings": lay.bearings, "secondary_dead_kN_m": lay.secondary_dead,
                   "lane_q_kN_m": lay.lane_q, "lane_P_kN": lay.lane_P, "lanes": lay.lanes},
        "materials": {k: {"E": v.E, "G": v.G, "nu": v.nu, "gamma": v.gamma}
                      for k, v in m.materials.items()},
        "sections": {k: {"A": s.A, "Iyy": s.Iyy, "Izz": s.Izz, "Iyz": s.Iyz, "J": s.J,
                         "width": s.width, "height": s.height, "cz": s.cz, "source": s.source}
                     for k, s in m.sections.items()},
        "nodes": m.nodes.round(9).tolist(),
        "node_labels": {str(k): v for k, v in m.node_labels.items()},
        "beams": [{"nodes": [b.n1, b.n2], "section": b.section, "material": b.material,
                   "vec_y": list(b.vec_y), "group": b.group} for b in m.beams],
        "springs": [{"nodes": [s.n1, s.n2], "dof": s.dof, "k": s.k, "group": s.group}
                    for s in m.springs],
        "rigid_links": [{"master": a, "slave": b} for a, b in m.rigid],
        "supports": {str(k): list(v) for k, v in m.supports.items()},
        "load_cases": {k: v.round(9).tolist() for k, v in m.load_cases.items()},
    }
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    return path


WRITERS = {
    "abaqus": (write_abaqus, "_abaqus.inp"),
    "nastran": (write_nastran, ".bdf"),
    "opensees": (write_opensees, "_opensees.py"),
    "json": (write_json, "_model.json"),
}
