"""Data models for whole-bridge (global) models.

Units: m, kN, kPa (kN/m^2), t (tonne).  Global axes: X along the bridge,
Y transverse, Z up.  The deck spine runs along X at the girder centroid
(Z = 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

SUPPORT_KINDS = ("FIXED", "SLIDING_X", "SLIDING_XY", "MONOLITHIC")

# dofs (1..6 = ux uy uz rx ry rz) a bearing restrains; rx is held so the single
# spine is torsionally supported (stands in for the bearing pair).
BEARING_DOFS = {
    "FIXED": (1, 2, 3, 4),
    "SLIDING_X": (2, 3, 4),
    "SLIDING_XY": (3, 4),
}


@dataclass
class SectionProps:
    """Cross-section properties in the section-drawing frame:
    drawing x -> member local y, drawing y -> member local z."""

    name: str
    A: float
    Iyy: float            # about local y  (= integral of z^2: vertical bending for the deck)
    Izz: float            # about local z  (= integral of y^2: lateral bending for the deck)
    Iyz: float
    J: float              # Saint-Venant torsion constant
    width: float
    height: float
    cy: float             # centroid from the left edge
    cz: float             # centroid above the bottom (soffit)
    n_cells: int = 0
    outline: Optional[np.ndarray] = None
    holes: list[np.ndarray] = field(default_factory=list)
    source: str = "drawing"


@dataclass
class BridgeMaterial:
    name: str = "C50"
    E: float = 3.45e7     # kPa
    nu: float = 0.2
    gamma: float = 26.0   # unit weight kN/m^3

    @property
    def G(self) -> float:
        return self.E / (2 * (1 + self.nu))

    @property
    def density(self) -> float:   # t/m^3
        return self.gamma / 9.80665


@dataclass
class BridgeLayout:
    spans: list[float] = field(default_factory=list)              # m
    pier_heights: dict[str, float] = field(default_factory=dict)  # "P1" -> m
    bearings: dict[str, str] = field(default_factory=dict)        # "A0"/"P1" -> kind
    deck_material: BridgeMaterial = field(default_factory=BridgeMaterial)
    pier_material: BridgeMaterial = field(default_factory=lambda: BridgeMaterial("C40", 3.25e7))
    secondary_dead: float = 0.0     # kN/m
    lane_q: float = 0.0             # kN/m per lane
    lane_P: float = 0.0             # kN per lane
    lane_P_span: Optional[int] = None   # 1-based span carrying P (default: longest)
    lanes: int = 1
    element_size: Optional[float] = None    # m
    bearing_stiffness: float = 1.0e8        # kN/m and kN m/rad
    title: str = "BRIDGE"
    explicit: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)
    source: str = "rules"

    @property
    def supports(self) -> list[str]:
        n = len(self.spans)
        return ["A0"] + [f"P{i}" for i in range(1, n)] + [f"A{n}"] if n else []

    @property
    def support_x(self) -> dict[str, float]:
        xs = np.r_[0.0, np.cumsum(self.spans)]
        return dict(zip(self.supports, map(float, xs)))

    @property
    def length(self) -> float:
        return float(sum(self.spans))


@dataclass
class Beam:
    n1: int
    n2: int
    section: str
    material: str
    vec_y: tuple[float, float, float]   # direction of member local y
    group: str


@dataclass
class Spring:
    n1: int
    n2: int
    dof: int        # 1..6, global
    k: float
    group: str


@dataclass
class FrameModel:
    nodes: np.ndarray                                     # (n, 3)
    beams: list[Beam] = field(default_factory=list)
    springs: list[Spring] = field(default_factory=list)
    rigid: list[tuple[int, int]] = field(default_factory=list)   # (master, slave)
    supports: dict[int, tuple[int, ...]] = field(default_factory=dict)
    sections: dict[str, SectionProps] = field(default_factory=dict)
    materials: dict[str, BridgeMaterial] = field(default_factory=dict)
    load_cases: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (n, 6)
    # case -> {beam index: global (12,) equivalent nodal loads of member loads}
    element_loads: dict[str, dict[int, np.ndarray]] = field(default_factory=dict)
    node_labels: dict[int, str] = field(default_factory=dict)
    node_groups: dict[str, list[int]] = field(default_factory=dict)
    layout: Optional[BridgeLayout] = None

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def add_node(self, xyz, label: str | None = None) -> int:
        self.nodes = np.vstack([self.nodes, np.asarray(xyz, float)[None]])
        i = len(self.nodes) - 1
        if label:
            self.node_labels[i] = label
        return i

    def beams_in(self, prefix: str) -> list[int]:
        return [k for k, b in enumerate(self.beams) if b.group.startswith(prefix)]
