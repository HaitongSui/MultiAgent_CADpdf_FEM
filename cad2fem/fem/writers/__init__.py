"""FE input-file writers.  Each writer takes an FEModel and a path."""

from .abaqus import write_abaqus, write_calculix
from .gmsh import write_gmsh
from .json_writer import write_json
from .nastran import write_nastran

WRITERS = {
    "abaqus": (write_abaqus, "_abaqus.inp"),
    "calculix": (write_calculix, "_ccx.inp"),
    "nastran": (write_nastran, ".bdf"),
    "gmsh": (write_gmsh, ".msh"),
    "json": (write_json, "_model.json"),
}

__all__ = ["WRITERS", "write_abaqus", "write_calculix", "write_gmsh", "write_json", "write_nastran"]
