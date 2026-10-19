"""Command line: ``python -m cad2fem drawing.pdf -o out/``"""

from __future__ import annotations

import argparse
import logging
import sys

from .core import PipelineError
from .pipeline import PipelineConfig, run


def _logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(message)s")
    for noisy in ("pdfminer", "matplotlib", "httpx", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def bridge_main(argv: list[str]) -> int:
    from .bridge import BridgeConfig, run_bridge
    from .bridge.writers import WRITERS as BRIDGE_WRITERS

    ap = argparse.ArgumentParser(prog="cad2fem bridge",
                                 description="Drawing set (general arrangement + girder section + "
                                             "pier section) -> global bridge spine model")
    ap.add_argument("pdfs", nargs="+", help="one or more PDFs; every page is a sheet")
    ap.add_argument("-o", "--out", default="out_bridge")
    ap.add_argument("--formats", default=",".join(BRIDGE_WRITERS),
                    help=f"comma list of {', '.join(BRIDGE_WRITERS)}")
    ap.add_argument("--sheets", help="force sheet kinds in order, e.g. GENERAL,DECK,PIER")
    ap.add_argument("--element-size", type=float, help="deck/pier element length (m)")
    ap.add_argument("--llm", action="store_true", help="also read notes with Claude")
    ap.add_argument("--llm-model", default="claude-opus-5-5")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    _logging(a.verbose)
    cfg = BridgeConfig(formats=tuple(f.strip() for f in a.formats.split(",") if f.strip()),
                       sheet_kinds=a.sheets.split(",") if a.sheets else None,
                       element_size=a.element_size, use_llm=a.llm, llm_model=a.llm_model,
                       verify=not a.no_verify, plot=not a.no_plot)
    try:
        bb = run_bridge(a.pdfs, a.out, cfg)
    except PipelineError as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 2
    return _summary(bb)


def arch_main(argv: list[str]) -> int:
    from .bridge import ArchConfig, run_arch
    from .bridge.writers import WRITERS as BRIDGE_WRITERS

    ap = argparse.ArgumentParser(prog="cad2fem arch",
                                 description="Scanned arch-bridge general drawings -> global 3D model "
                                             "(sheets are read with Claude's vision)")
    ap.add_argument("pdfs", nargs="*", help="sheet PDFs to read (e.g. the 一般図 and 基本寸法図)")
    ap.add_argument("--reading", help="saved reading JSON (skips the vision call)")
    ap.add_argument("--save-reading", help="write the reading used to this JSON file")
    ap.add_argument("-o", "--out", default="out_arch")
    ap.add_argument("--formats", default=",".join(BRIDGE_WRITERS))
    ap.add_argument("--model", default="claude-opus-5-5", help="vision model")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    if not a.pdfs and not a.reading:
        ap.error("give sheet PDFs to read, or --reading")
    _logging(a.verbose)
    cfg = ArchConfig(formats=tuple(f.strip() for f in a.formats.split(",") if f.strip()),
                     reading_file=a.reading, vision_model=a.model, verify=not a.no_verify,
                     plot=not a.no_plot)
    try:
        bb = run_arch(a.pdfs, a.out, cfg)
    except PipelineError as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 2
    if a.save_reading and bb.has("arch_reading"):
        from pathlib import Path
        Path(a.save_reading).write_text(bb["arch_reading"].model_dump_json(indent=1), encoding="utf-8")
        print(f"reading saved to {a.save_reading}")
    return _summary(bb)


def _summary(bb) -> int:
    print("\nOutputs:")
    for k, v in (bb.get("outputs") or {}).items():
        print(f"  {k:9s} {v}")
    if bb.get("report"):
        print(f"  {'report':9s} {bb['report']}")
    n_warn = len(bb.warnings())
    if n_warn:
        print(f"\n{n_warn} warning(s) - see report.md")
    return 0 if bb.has("outputs") else 1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "bridge":
        return bridge_main(argv[1:])
    if argv and argv[0] == "arch":
        return arch_main(argv[1:])
    ap = argparse.ArgumentParser(prog="cad2fem",
                                 epilog="Whole-bridge models: cad2fem bridge <pdfs...> -h | "
                                        "scanned arch bridges: cad2fem arch <pdfs...> -h",
                                 description="Multi-agent CAD-PDF -> FEM input file generator")
    ap.add_argument("pdf", help="vector CAD drawing (PDF)")
    ap.add_argument("-o", "--out", default="out", help="output directory")
    ap.add_argument("--page", type=int, default=0)
    ap.add_argument("--formats", default="abaqus,calculix,nastran,gmsh,json",
                    help="comma list of abaqus, calculix, nastran, gmsh, json")
    ap.add_argument("--mesh-size", type=float, help="target element size in mm")
    ap.add_argument("--order", type=int, choices=(1, 2), help="element order (1=Tri3, 2=Tri6)")
    ap.add_argument("--thickness", type=float, help="override thickness (mm)")
    ap.add_argument("--plane-strain", action="store_true")
    ap.add_argument("--outline-width", type=float,
                    help="minimum stroke width (pt) treated as part outline (default: auto)")
    ap.add_argument("--llm", action="store_true",
                    help="also interpret notes with Claude (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--llm-model", default="claude-opus-5-5")
    ap.add_argument("--prefer", choices=("rules", "llm"), default="rules")
    ap.add_argument("--no-verify", action="store_true", help="skip the built-in CST solve")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    _logging(a.verbose)

    cfg = PipelineConfig(
        page=a.page, use_llm=a.llm, llm_model=a.llm_model, prefer=a.prefer,
        formats=tuple(f.strip() for f in a.formats.split(",") if f.strip()),
        mesh_size=a.mesh_size, element_order=a.order, thickness=a.thickness,
        analysis="PLANE_STRAIN" if a.plane_strain else None,
        min_outline_width=a.outline_width, verify=not a.no_verify, plot=not a.no_plot)
    try:
        bb = run(a.pdf, a.out, cfg)
    except PipelineError as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 2
    return _summary(bb)


if __name__ == "__main__":
    sys.exit(main())
