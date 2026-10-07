"""Whole-bridge pipeline: drawing set (PDFs) -> global spine FE model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from ..core import Agent, Blackboard, Orchestrator
from .agents import ElevationAgent, LayoutAgent, SectionAgent, SheetClassifierAgent, SheetReaderAgent
from .assembly import BridgeAssemblyAgent, BridgeValidatorAgent, BridgeVerifierAgent
from .llm_layout import LLMLayoutAgent
from .report import BridgeReportAgent, BridgeWriterAgent
from .writers import WRITERS


@dataclass
class BridgeConfig:
    formats: tuple[str, ...] = tuple(WRITERS)
    sheet_kinds: Optional[list[str]] = None      # force GENERAL/DECK/PIER per sheet
    element_size: Optional[float] = None         # m
    use_llm: bool = False
    llm_model: str = "claude-opus-5-5"
    verify: bool = True
    plot: bool = True


def build_bridge_agents(cfg: BridgeConfig) -> list[Agent]:
    agents: list[Agent] = [SheetReaderAgent(), SheetClassifierAgent(kinds=cfg.sheet_kinds),
                           ElevationAgent(), SectionAgent()]
    if cfg.use_llm:
        agents.append(LLMLayoutAgent(model=cfg.llm_model))
    agents += [LayoutAgent(), BridgeAssemblyAgent(element_size=cfg.element_size),
               BridgeValidatorAgent()]
    if cfg.verify:
        agents.append(BridgeVerifierAgent())
    agents += [BridgeWriterAgent(formats=cfg.formats), BridgeReportAgent(plot=cfg.plot)]
    return agents


def run_bridge(pdf_paths: Sequence[str | Path], out_dir: str | Path,
               cfg: Optional[BridgeConfig] = None, agents: Optional[list[Agent]] = None) -> Blackboard:
    cfg = cfg or BridgeConfig()
    bb = Blackboard(pdf_paths=[str(p) for p in pdf_paths], out_dir=str(out_dir))
    Orchestrator(agents or build_bridge_agents(cfg)).run(bb)
    return bb
