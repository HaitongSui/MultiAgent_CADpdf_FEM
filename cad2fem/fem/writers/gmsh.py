"""Gmsh MSH 2.2 ASCII writer (view the mesh and named boundary groups)."""

from __future__ import annotations

from pathlib import Path

from ...models import FEModel


def write_gmsh(model: FEModel, path: str | Path) -> Path:
    path = Path(path)
    mesh = model.mesh
    quad = mesh.order == 2
    groups: list[tuple[int, str, list[list[int]]]] = []
    for name, faces in mesh.surfaces.items():
        groups.append((1, name, _edges(mesh, faces, quad)))
    for name, *_ in model.constraints:
        ids = set(int(i) for i in mesh.node_sets[name])
        faces = [(e, f) for e, f in mesh.boundary_edges()
                 if set(mesh.face_nodes(e, f)) <= ids]
        groups.append((1, name, _edges(mesh, faces, quad)))
    for name in model.nodal_forces:
        if name not in mesh.surfaces:
            ids = set(int(i) for i in mesh.node_sets[name])
            faces = [(e, f) for e, f in mesh.boundary_edges() if set(mesh.face_nodes(e, f)) <= ids]
            groups.append((1, name, _edges(mesh, faces, quad)))

    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", str(len(groups) + 1),
           '2 1 "DOMAIN"']
    out += [f'{dim} {k + 2} "{name}"' for k, (dim, name, _) in enumerate(groups)]
    out += ["$EndPhysicalNames", "$Nodes", str(mesh.n_nodes)]
    out += [f"{i + 1} {x:.10g} {y:.10g} 0" for i, (x, y) in enumerate(mesh.nodes)]
    out += ["$EndNodes", "$Elements"]
    elems = []
    eid = 1
    for el in mesh.elements:
        elems.append(f"{eid} {9 if quad else 2} 2 1 1 " + " ".join(str(n + 1) for n in el))
        eid += 1
    for k, (_, _, edges) in enumerate(groups):
        for ed in edges:
            elems.append(f"{eid} {8 if quad else 1} 2 {k + 2} {k + 2} " + " ".join(str(n + 1) for n in ed))
            eid += 1
    out += [str(len(elems)), *elems, "$EndElements"]
    path.write_text("\n".join(out) + "\n")
    return path


def _edges(mesh, faces, quad: bool) -> list[list[int]]:
    res = []
    for e, f in faces:
        a, b = mesh.face_nodes(e, f)
        res.append([a, b, int(mesh.elements[e][3 + f - 1])] if quad else [a, b])
    return res
