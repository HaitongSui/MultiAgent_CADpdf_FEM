"""Solver-neutral JSON dump of the FE model."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ...models import FEModel


def write_json(model: FEModel, path: str | Path) -> Path:
    path = Path(path)
    mesh, spec = model.mesh, model.spec
    spec_d = asdict(spec)
    spec_d["explicit"] = sorted(spec.explicit)
    data = {
        "units": {"length": "mm", "force": "N", "stress": "MPa"},
        "spec": spec_d,
        "nodes": mesh.nodes.round(10).tolist(),
        "elements": mesh.elements.tolist(),
        "element_type": f"tri{3 * mesh.order}",
        "node_sets": {k: v.tolist() for k, v in mesh.node_sets.items() if k != "NALL"},
        "surfaces": {k: [list(map(int, ef)) for ef in v] for k, v in mesh.surfaces.items()},
        "constraints": [{"nset": n, "dofs": list(d), "value": v, "description": s}
                        for n, d, v, s in model.constraints],
        "pressures": [{"surface": n, "value": p} for n, p in model.pressures],
        "nodal_forces": {k: v.tolist() for k, v in model.nodal_forces.items()},
    }
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    return path
