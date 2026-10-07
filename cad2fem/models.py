"""Data models shared by all agents through the blackboard.

Units convention after the GeometryAgent: millimetres, newtons, MPa (N/mm^2).
Raw PDF data are in PDF points (1 pt = 1/72 inch) with a y-up origin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

Point = tuple[float, float]


# --------------------------------------------------------------------------- #
# Raw drawing content (PDF space, points, y-up)
# --------------------------------------------------------------------------- #
@dataclass
class Stroke:
    """A stroked polyline from the PDF (Béziers already flattened)."""

    points: list[Point]
    linewidth: float = 0.0
    dashed: bool = False
    closed: bool = False
    color: object = None
    # Set when the path was recognised as a full circle.
    circle: Optional[tuple[Point, float]] = None


@dataclass
class TextItem:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float = 0.0

    @property
    def center(self) -> Point:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)


@dataclass
class DrawingData:
    page_width: float
    page_height: float
    strokes: list[Stroke] = field(default_factory=list)
    texts: list[TextItem] = field(default_factory=list)   # words
    lines: list[TextItem] = field(default_factory=list)   # words grouped into text lines
    page_index: int = 0

    @property
    def full_text(self) -> str:
        return "\n".join(t.text for t in self.lines)


# --------------------------------------------------------------------------- #
# Geometry (model space, mm, y-up)
# --------------------------------------------------------------------------- #
@dataclass
class Loop:
    points: np.ndarray          # (n, 2), CCW, not repeating the first point
    is_circle: bool = False
    center: Optional[Point] = None
    radius: Optional[float] = None

    @property
    def area(self) -> float:
        return abs(polygon_area(self.points))

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        mn = self.points.min(axis=0)
        mx = self.points.max(axis=0)
        return float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1])


@dataclass
class Geometry:
    outer: Loop
    holes: list[Loop] = field(default_factory=list)
    scale_mm_per_pt: float = 1.0

    @property
    def area(self) -> float:
        return self.outer.area - sum(h.area for h in self.holes)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.outer.bbox

    @property
    def loops(self) -> list[Loop]:
        return [self.outer, *self.holes]


def polygon_area(pts: np.ndarray) -> float:
    """Signed area (positive for counter-clockwise)."""
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# --------------------------------------------------------------------------- #
# Analysis specification (what the drawing *means* for FEM)
# --------------------------------------------------------------------------- #
@dataclass
class Material:
    name: str = "STEEL"
    E: float = 210000.0      # MPa
    nu: float = 0.3
    density: float = 7.85e-9  # t/mm^3 (consistent with mm-N-s)


@dataclass
class BoundaryCondition:
    """Displacement constraint on a boundary selection.

    ``dofs`` uses 1 = x, 2 = y.  ``location`` is a selector string understood
    by the BoundaryAgent: LEFT, RIGHT, TOP, BOTTOM, HOLE<n>, HOLES, X=<v>, Y=<v>.
    """

    location: str
    dofs: tuple[int, ...] = (1, 2)
    value: float = 0.0
    kind: str = "FIXED"


@dataclass
class Load:
    """``kind`` is PRESSURE (MPa, positive pushes into the body) or FORCE
    (total force in N distributed over the selection, along ``direction``)."""

    location: str
    kind: str = "PRESSURE"
    magnitude: float = 0.0
    direction: tuple[float, float] = (0.0, 0.0)


@dataclass
class AnalysisSpec:
    title: str = "PART"
    units: str = "mm"
    scale: Optional[float] = None          # drawing scale denominator (1:N -> N)
    thickness: float = 1.0
    analysis: str = "PLANE_STRESS"          # or PLANE_STRAIN
    element_order: int = 1                  # 1 -> Tri3, 2 -> Tri6
    mesh_size: Optional[float] = None
    material: Material = field(default_factory=Material)
    bcs: list[BoundaryCondition] = field(default_factory=list)
    loads: list[Load] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    source: str = "rules"
    # names of the fields above that the drawing stated explicitly
    explicit: set[str] = field(default_factory=set)


# --------------------------------------------------------------------------- #
# Mesh
# --------------------------------------------------------------------------- #
@dataclass
class Mesh:
    nodes: np.ndarray                      # (n, 2)
    elements: np.ndarray                   # (m, 3) or (m, 6), 0-based, CCW
    order: int = 1
    # boundary node id -> loop index (0 = outer, k = hole k)
    boundary_loop: dict[int, int] = field(default_factory=dict)
    node_sets: dict[str, np.ndarray] = field(default_factory=dict)
    # name -> list of (element index, local face 1..3)
    surfaces: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    def corner_elements(self) -> np.ndarray:
        return self.elements[:, :3]

    def element_areas(self) -> np.ndarray:
        tri = self.nodes[self.corner_elements()]
        a = tri[:, 1] - tri[:, 0]
        b = tri[:, 2] - tri[:, 0]
        return 0.5 * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])

    def boundary_edges(self) -> list[tuple[int, int]]:
        """Edges used by exactly one element, as (element index, face 1..3).

        Face f joins corner nodes f and f+1 (Abaqus/CalculiX convention)."""
        counts: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for e, tri in enumerate(self.corner_elements()):
            for f in range(3):
                a, b = int(tri[f]), int(tri[(f + 1) % 3])
                counts.setdefault((min(a, b), max(a, b)), []).append((e, f + 1))
        return [v[0] for v in counts.values() if len(v) == 1]

    def face_nodes(self, e: int, face: int) -> tuple[int, int]:
        tri = self.elements[e]
        return int(tri[face - 1]), int(tri[face % 3])


@dataclass
class FEModel:
    """Everything a writer needs: mesh + named sets + resolved BCs/loads."""

    mesh: Mesh
    spec: AnalysisSpec
    geometry: Geometry
    # (node set name, dofs, value, description)
    constraints: list[tuple[str, tuple[int, ...], float, str]] = field(default_factory=list)
    # (surface name, pressure MPa) - faces are in mesh.surfaces[name]
    pressures: list[tuple[str, float]] = field(default_factory=list)
    # node set name -> (n, 2) equivalent nodal forces for the nodes of that set (in set order)
    nodal_forces: dict[str, np.ndarray] = field(default_factory=dict)

    def total_nodal_forces(self) -> np.ndarray:
        """All loads (forces + pressures) as an (n_nodes, 2) array."""
        f = np.zeros((self.mesh.n_nodes, 2))
        for name, forces in self.nodal_forces.items():
            np.add.at(f, self.mesh.node_sets[name], forces)
        return f
