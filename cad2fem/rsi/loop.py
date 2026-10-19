"""Recursive self-improvement loop for reading a drawing set.

Phase 1 + 2  practice: Curriculum -> Actor -> Verifier -> Memory, repeated
             until no reachable gap is left, attempts run out or the budget
             is spent (the sheet index is the broad first task; the section
             and cable tasks are the deep ones).
Phase 3      test-time reuse: memory is frozen, turned into ArchKnowledge,
             and the arch model is rebuilt with it next to an assumptions-
             only baseline so the effect of what was learned is visible.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..bridge.arch_reading import ArchReading
from ..core import Blackboard, Orchestrator
from .agents import ActorAgent, CurriculumAgent, MemoryAgent, VerifierAgent
from .backends import Backend, auto_backend
from .knowledge import knowledge_from_memory
from .memory import Memory
from .tasks import TASKS, Context

log = logging.getLogger("cad2fem.rsi")


@dataclass
class RSIConfig:
    pdf_dir: str = "cad_pdf"
    reading_file: str = "cad_pdf/readings/J-5_J-8_arch_reading.json"
    memory_dir: str = "memory"
    practice_dir: str = "cad_pdf/readings/practice"   # transcripts replayed when no backend
    work_dir: str = "out_rsi/work"
    out_dir: str = "out_rsi"
    backend: str = "auto"            # auto | api | cli | none
    budget_usd: float = 5.0
    max_attempts: int = 2
    llm_verify: bool = True
    tasks: Optional[list[str]] = None
    relearn: Optional[dict[str, str]] = None   # task -> reviewer's reason; evicted and re-practised


def sheet_pdfs(pdf_dir: str | Path) -> dict[int, Path]:
    out = {}
    for p in Path(pdf_dir).glob("*.pdf"):
        m = re.search(r"-(\d+)\.pdf$", p.name)
        if m:
            out[int(m.group(1))] = p
    return dict(sorted(out.items()))


def audit(mem: Memory, reading: ArchReading) -> list[str]:
    """Re-run the current deterministic checks on every committed claim; evict what fails.
    When the Verifier learns a new check, earlier knowledge is held to it too."""
    snapshot = {t.id: mem.claims(t.id) for t in TASKS if mem.has(t.id)}
    failures = {}
    for t in TASKS:                     # all against the same snapshot: when two verified claims
        c = snapshot.get(t.id)          # disagree, neither is trusted and both are re-practised
        if c is None:
            continue
        ctx = Context(reading=reading, known={k: v for k, v in snapshot.items() if k != t.id})
        failed = [x for x in t.checks(c, ctx) if not x["ok"]]
        if failed:
            failures[t.id] = failed
    for tid, failed in failures.items():
        mem.evict(tid, failed)
        log.warning("[WARNING] memory audit: %s evicted (%s)", tid,
                    "; ".join(f"{x['check']} {x['detail']}" for x in failed))
    return list(failures)


def practice(cfg: RSIConfig, actor: Optional[Backend] = None, verifier: Optional[Backend] = None,
             use_auto: bool = True) -> Blackboard:
    if use_auto and actor is None and cfg.backend != "none":
        actor = auto_backend(cfg.backend)
        verifier = verifier or (auto_backend(cfg.backend) if cfg.llm_verify else None)
    mem = Memory(cfg.memory_dir)
    reading = ArchReading.model_validate_json(Path(cfg.reading_file).read_text("utf-8"))
    for tid, why in (cfg.relearn or {}).items():
        if mem.has(tid):
            mem.evict(tid, [{"check": "reviewer: re-practise", "ok": False, "detail": why}], by="reviewer")
            log.warning("[WARNING] memory: %s evicted by reviewer (%s)", tid, why)
    audit(mem, reading)
    rsi = {"memory": mem, "reading": reading,
           "pdfs": sheet_pdfs(cfg.pdf_dir), "work_dir": cfg.work_dir, "practice_dir": cfg.practice_dir,
           "actor_backend": actor, "verifier_backend": verifier,
           "backends": [b for b in {id(x): x for x in (actor, verifier) if x}.values()]}
    bb = Blackboard(rsi=rsi, memory_version=mem.version)
    agents = [CurriculumAgent(cfg.max_attempts, cfg.budget_usd, cfg.tasks), ActorAgent(),
              VerifierAgent(llm_check=cfg.llm_verify), MemoryAgent()]
    Orchestrator(agents, max_passes=400).run(bb)
    return bb


def reuse(cfg: RSIConfig):
    """Phase 3: frozen memory -> arch model, compared with the assumptions-only baseline."""
    from ..bridge.pipeline import ArchConfig, run_arch

    mem = Memory(cfg.memory_dir, frozen=True)
    reading = ArchReading.model_validate_json(Path(cfg.reading_file).read_text("utf-8"))
    kn = knowledge_from_memory(mem, reading)
    out = Path(cfg.out_dir)
    base = run_arch([], out / "baseline", ArchConfig(reading_file=cfg.reading_file, plot=False))
    bb = Blackboard(pdf_paths=[], out_dir=str(out / "with_memory"), reading_file=cfg.reading_file,
                    arch_knowledge=kn)
    from ..bridge.pipeline import build_arch_agents
    Orchestrator(build_arch_agents(ArchConfig(reading_file=cfg.reading_file))).run(bb)
    return base, bb, kn, mem


def compare(base: Blackboard, learned: Blackboard, kn) -> dict:
    rows = {}
    for case in ("DEAD", "LIVE_B"):
        b, l_ = base["bridge_results"][case], learned["bridge_results"][case]
        rows[case] = {k: (b[k], l_[k]) for k in ("midspan_deflection_mm", "rib_axial_crown_kN",
                                                  "girder_axial_midspan_kN", "max_sagging_kNm",
                                                  "max_hogging_kNm")}
    design = kn.design_dead_tension
    if design:
        hb = base["bridge_results"]["DEAD"]["hanger_force_kN"]
        hl = learned["bridge_results"]["DEAD"]["hanger_force_kN"]
        rows["hangers"] = {a: (hb[a], hl[a], design[a]) for a in design}
    return rows


def write_report(cfg: RSIConfig, practice_bb: Optional[Blackboard], mem: Memory, kn, rows: dict) -> Path:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    eps = mem.episodes()
    L = ["# RSI run report - reading the drawing set to fill model gaps", "",
         "## Practice episodes", "", "| # | Task | Page | Reader | Checks | Independent re-read | Result |",
         "|---|---|---|---|---|---|---|"]
    for k, e in enumerate(eps, 1):
        n_ok = sum(c["ok"] for c in e["checks"])
        L.append(f"| {k} | {e['task']} | {e.get('page') or '-'} | {e['reader']} | {n_ok}/{len(e['checks'])} | "
                 f"{e['llm'].get('status', '-')} | {'EVICTED' if e.get('audit') else 'PASS' if e['ok'] else 'FAIL'} |")
    spent, last = {}, {}
    for e in eps:                         # cumulative per run; a drop means a new run started
        for name, v in (e.get("spent_usd") or {}).items():
            spent[name] = round(spent.get(name, 0) + (v - last.get(name, 0) if v >= last.get(name, 0) else v), 4)
            last[name] = v
    L += ["", f"Model spend (all runs on this memory, per backend): {spent or 'none'}", "",
          "## Verified lessons in memory", ""] + [f"* {ls}" for ls in mem.lessons] + [""]
    L += ["## Knowledge used by the model", ""] + [f"* {k}: {v}" for k, v in kn.sources.items()] + [""]
    if kn.sections:
        L += ["| Section | A (m2) | Iyy (m4) | Izz (m4) | J (m4) | Source |", "|---|---|---|---|---|---|"]
        L += [f"| {n} | {s.A:.5f} | {s.Iyy:.5f} | {s.Izz:.5f} | {s.J:.5f} | {s.source} |"
              for n, s in kn.sections.items()]
        L.append("")
    L += ["## Effect on the model (assumed sections -> verified sections)", "",
          "| Case | Quantity | Assumptions only | With memory |", "|---|---|---|---|"]
    for case in ("DEAD", "LIVE_B"):
        for q, (b, l_) in rows.get(case, {}).items():
            L.append(f"| {case} | {q} | {b:,.1f} | {l_:,.1f} |")
    if "hangers" in rows:
        L += ["", "Hanger dead-load tension (G1): model vs the design table on J-81 (design values include "
                  "cable prestress / length adjustment, so they are a reference, not a pass/fail target).", "",
              "| Hanger | Assumptions only (kN) | With memory (kN) | J-81 design (kN) |", "|---|---|---|---|"]
        L += [f"| {a} | {b:.0f} | {l_:.0f} | {d:.0f} |" for a, (b, l_, d) in rows["hangers"].items()]
    path = out / "rsi_report.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "rsi_comparison.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), "utf-8")
    return path


def run_rsi(cfg: RSIConfig, actor: Optional[Backend] = None, verifier: Optional[Backend] = None,
            use_auto: bool = True):
    pbb = practice(cfg, actor, verifier, use_auto)
    base, learned, kn, mem = reuse(cfg)
    rows = compare(base, learned, kn)
    report = write_report(cfg, pbb, mem, kn, rows)
    return {"practice": pbb, "baseline": base, "learned": learned, "knowledge": kn, "memory": mem,
            "comparison": rows, "report": report}
