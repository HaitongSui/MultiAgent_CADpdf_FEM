"""VisionSheetAgent: reads scanned (raster) drawings with Claude's vision.

Scanned sheets have no vector geometry or text layer, so the rule-based
agents cannot see anything.  This agent renders each page, sends Claude an
overview plus full-resolution tiles (dense dimension text needs them) and
asks for an :class:`ArchReading` as structured output.

If a saved reading (transcription JSON) is supplied it is loaded instead, so
a reviewed reading can be reused without another model call.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional

from ..core import Agent, Blackboard
from .arch_reading import ArchReading

MAX_EDGE = 2576            # Claude's maximum image long edge (pixels)

SYSTEM = (
    "You read Japanese steel arch-bridge drawings (一般図, 基本寸法図) for a pipeline that builds a "
    "global finite-element model. Each sheet comes as an overview image followed by full-resolution "
    "tiles (top-left, top-right, bottom-left, bottom-right, overlapping). Transcribe dimension values "
    "exactly as written, in millimetres. Use dimension chains rather than measuring the picture. "
    "Check chains yourself: segment sums should equal the overall dimension they subdivide. Record "
    "the sheet and view for each group of values in `sources`, and put every value you are unsure "
    "about, with the reason, in `uncertain`. Do not invent values that are not on the sheets; if a "
    "required field is missing, give your best reading and explain it in `uncertain`."
)


def render_page_images(pdf_path: str | Path, page: int = 0, dpi: int = 200) -> list[bytes]:
    """PNG bytes: one overview (<= MAX_EDGE) and four overlapping tiles."""
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        img = pdf.pages[page].to_image(resolution=dpi).original.convert("L")   # drawings are monochrome
    out = []
    ov = img.copy()
    ov.thumbnail((MAX_EDGE, MAX_EDGE))
    out.append(_png(ov))
    W, H = img.size
    tw, th = min(int(W * 0.56), MAX_EDGE), min(int(H * 0.56), MAX_EDGE)
    for x0, y0 in ((0, 0), (W - tw, 0), (0, H - th), (W - tw, H - th)):
        out.append(_png(img.crop((x0, y0, x0 + tw, y0 + th))))
    return out


def _png(im) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def build_content(pages: list[tuple[str, list[bytes]]]) -> list[dict]:
    content: list[dict] = []
    names = ("overview", "tile top-left", "tile top-right", "tile bottom-left", "tile bottom-right")
    for label, images in pages:
        for name, data in zip(names, images):
            content.append({"type": "text", "text": f"{label} - {name}"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                        "data": base64.b64encode(data).decode()}})
    content.append({"type": "text", "text": "Read these sheets into the ArchReading schema."})
    return content


class VisionSheetAgent(Agent):
    name = "vision"
    requires = ("pdf_paths",)
    optional = ("reading_file",)
    provides = ("arch_reading",)

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "high", dpi: int = 200,
                 client=None) -> None:
        super().__init__(model=model, effort=effort, dpi=dpi)
        self._client = client

    def run(self, bb: Blackboard) -> None:
        path: Optional[str] = bb.get("reading_file")
        if path:
            reading = ArchReading.model_validate_json(Path(path).read_text(encoding="utf-8"))
            self.info(bb, f"loaded reading of {len(reading.sheets)} sheet(s) from {path} "
                          f"(reader: {reading.reader or 'unknown'})")
            bb.post("arch_reading", reading, self.name)
            return

        pages = [(Path(p).name, render_page_images(p, dpi=self.params["dpi"])) for p in bb["pdf_paths"]]
        try:
            import anthropic

            client = self._client or anthropic.Anthropic()
            response = client.beta.messages.parse(
                model=self.params["model"],
                max_tokens=16000,
                system=SYSTEM,
                thinking={"type": "adaptive"},
                output_config={"effort": self.params["effort"]},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=[{"role": "user", "content": build_content(pages)}],
                output_format=ArchReading,
            )
        except Exception as exc:
            raise RuntimeError(
                f"vision reading failed ({type(exc).__name__}: {exc}). Set ANTHROPIC_API_KEY, or pass a "
                "saved reading with --reading") from exc
        if response.stop_reason == "refusal" or response.parsed_output is None:
            raise RuntimeError(f"no structured reading returned (stop_reason={response.stop_reason})")
        reading = response.parsed_output
        reading.reader = reading.reader or f"{self.params['model']} vision"
        for u in reading.uncertain:
            self.warn(bb, f"reader unsure: {u}")
        self.info(bb, f"read {len(pages)} sheet(s) with {self.params['model']}")
        bb.post("arch_reading", reading, self.name)
