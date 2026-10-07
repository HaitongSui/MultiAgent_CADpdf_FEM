"""Small material library (mm-N-s-t units: E in MPa, density in t/mm^3)."""

from __future__ import annotations

import re
from typing import Optional

from .models import Material

# (regex matched against the MATERIAL text, name, E, nu, density)
_LIBRARY = [
    (r"\b(S235|S275|S355|Q235|Q345|Q355|A36|SS400|STEEL|ST37|ST52|C45|1045|45#|钢)", "STEEL", 210000.0, 0.30, 7.85e-9),
    (r"\b(SUS|304|316|STAINLESS|不锈钢)", "STAINLESS_STEEL", 193000.0, 0.29, 8.0e-9),
    (r"\b(6061|6063|7075|2024|AL|ALU|ALUMIN(I)?UM|铝)", "ALUMINIUM", 70000.0, 0.33, 2.70e-9),
    (r"\b(TI|TITANIUM|TI-?6AL-?4V|钛)", "TITANIUM", 113800.0, 0.34, 4.43e-9),
    (r"\b(CAST\s*IRON|GG25|HT200|QT\d+|铸铁)", "CAST_IRON", 120000.0, 0.26, 7.20e-9),
    (r"\b(COPPER|CU|铜)", "COPPER", 117000.0, 0.34, 8.96e-9),
    (r"\b(BRASS|黄铜)", "BRASS", 100000.0, 0.34, 8.50e-9),
    (r"\b(CONCRETE|C30|混凝土)", "CONCRETE", 30000.0, 0.20, 2.40e-9),
    (r"\b(PMMA|ACRYLIC)", "PMMA", 3000.0, 0.37, 1.18e-9),
]


def lookup(text: str) -> Optional[Material]:
    up = text.upper()
    for pattern, family, E, nu, rho in _LIBRARY:
        if re.search(pattern, up):
            label = re.sub(r"[^A-Z0-9]+", "_", up).strip("_")[:40]
            return Material(name=label or family, E=E, nu=nu, density=rho)
    return None
