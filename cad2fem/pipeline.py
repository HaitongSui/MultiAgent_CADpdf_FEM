"""High-level entry point: assemble the agent team and run it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .agents import (AnnotationAgent, BoundaryAgent, DimensionAgent, GeometryAgent,
                     LLMAnnotationAgent, MeshAgent, PDFParserAgent, ReportAgent, SpecAgent,
                     ValidatorAgent, VerifierAgent, WriterAgent)
from .core import Agent, Blackboard, Orchestrator


@dataclass
class PipelineConfig:
    page: int = 0
    use_llm: bool = False
    llm_model: str = "claude-opus-5-5"
    prefer: str = "rules"                     # "rules" | "llm" when both disagree
    formats: tuple[str, ...] = ("abaqus", "calculix", "nastran", "gmsh", "json")
    mesh_size: Optional[float] = None         # mm
    element_order: Optional[int] = None       # 1 | 2
    thickness: Optional[float] = None         # mm
    analysis: Optional[str] = None            # PLANE_STRESS | PLANE_STRAIN
    min_outline_width: Optional[float] = None
    verify: bool = True
    plot: bool = True


def build_agents(cfg: PipelineConfig) -> list[Agent]:
    """Agent order only breaks ties; scheduling is driven by data availability."""
    agents: list[Agent] = [
        PDFParserAgent(page=cfg.page),
        AnnotationAgent(),
    ]
    if cfg.use_llm:
        agents.append(LLMAnnotationAgent(model=cfg.llm_model))
    agents += [
        SpecAgent(prefer=cfg.prefer),
        GeometryAgent(min_outline_width=cfg.min_outline_width),
        DimensionAgent(),
        MeshAgent(),
        BoundaryAgent(),
        ValidatorAgent(),
    ]
    if cfg.verify:
        agents.append(VerifierAgent())
    agents += [WriterAgent(formats=cfg.formats), ReportAgent(plot=cfg.plot)]
    return agents


def run(pdf_path: str | Path, out_dir: str | Path, cfg: Optional[PipelineConfig] = None,
        agents: Optional[list[Agent]] = None) -> Blackboard:
    cfg = cfg or PipelineConfig()
    overrides = {"mesh_size": cfg.mesh_size, "element_order": cfg.element_order,
                 "thickness": cfg.thickness, "analysis": cfg.analysis}
    bb = Blackboard(pdf_path=str(pdf_path), out_dir=str(out_dir),
                    overrides={k: v for k, v in overrides.items() if v is not None})
    Orchestrator(agents or build_agents(cfg)).run(bb)
    return bb
