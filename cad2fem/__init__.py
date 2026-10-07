"""cad2fem - a multi-agent framework that turns vector CAD drawings (PDF)
into finite-element input files (Abaqus, CalculiX, Nastran, Gmsh, JSON)."""

from .pipeline import PipelineConfig, build_agents, run

__version__ = "0.1.0"
__all__ = ["PipelineConfig", "build_agents", "run", "__version__"]
