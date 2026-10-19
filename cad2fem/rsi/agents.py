"""The RSI team, running on the same blackboard/orchestrator as the pipelines.

  Curriculum ──task──► Actor ──claims──► Verifier ──verdict──► Memory
      ▲                                                          │
      └──────────────────── memory_version ◄─────────────────────┘

* CurriculumAgent picks the next practice task from the knowledge gaps,
  past attempts and the sheet index; it stops when nothing useful is left
  or the budget is spent.
* ActorAgent reads the sheet (LLM backend, with the verified lessons in its
  prompt) - or replays a saved practice transcript when no backend exists.
* VerifierAgent never sees the Actor's reasoning: it runs grounded checks
  against other sheets and arithmetic, and optionally asks an independent
  LLM context to re-read the images and confirm each claimed value.
* MemoryAgent records every episode and commits claims + lessons only on PASS.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw

from ..core import Agent, Blackboard
from .backends import Backend, BackendError
from .claims import Claims
from .memory import Memory
from .tasks import TASKS, Context, Task, norm, task_by_id

MAX_EDGE = 2576


# --------------------------------------------------------------------------- #
# images
# --------------------------------------------------------------------------- #
def render_page(pdf: Path, out_dir: Path, dpi: int = 280) -> list[Path]:
    """Overview + four overlapping tiles (each <= MAX_EDGE) as grayscale PNGs."""
    import pdfplumber

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf.stem.replace("（", "_").replace("）", "_")
    paths = [out_dir / f"{stem}_overview.png"] + [out_dir / f"{stem}_tile{k}.png" for k in range(4)]
    if all(p.exists() for p in paths):
        return paths
    with pdfplumber.open(pdf) as doc:
        img = doc.pages[0].to_image(resolution=dpi).original.convert("L")
    ov = img.copy()
    ov.thumbnail((MAX_EDGE, MAX_EDGE))
    ov.save(paths[0])
    W, H = img.size
    tw, th = min(int(W * 0.56), MAX_EDGE), min(int(H * 0.56), MAX_EDGE)
    for k, (x0, y0) in enumerate(((0, 0), (W - tw, 0), (0, H - th), (W - tw, H - th))):
        img.crop((x0, y0, x0 + tw, y0 + th)).save(paths[k + 1])
    return paths


def render_title_blocks(pdfs: dict[int, Path], out_dir: Path, per_image: int = 12) -> list[Path]:
    """Stack the title-block strips of all sheets, labelled with page numbers."""
    import pdfplumber

    out_dir.mkdir(parents=True, exist_ok=True)
    strips = []
    for page, pdf in sorted(pdfs.items()):
        with pdfplumber.open(pdf) as doc:
            im = doc.pages[0].to_image(resolution=200).original.convert("L")
        W, H = im.size
        strips.append((page, im.crop((int(0.69 * W), int(0.925 * H), int(0.995 * W), int(0.968 * H)))))
    paths = []
    for k in range(0, len(strips), per_image):
        chunk = strips[k:k + per_image]
        w = max(s.size[0] for _, s in chunk) + 90
        h = max(s.size[1] for _, s in chunk) + 8
        sheet = Image.new("L", (w, h * len(chunk)), 255)
        d = ImageDraw.Draw(sheet)
        for j, (page, s) in enumerate(chunk):
            sheet.paste(s, (90, j * h))
            d.text((8, j * h + h // 3), f"page {page}", fill=0)
        p = out_dir / f"title_blocks_{k // per_image + 1}.png"
        sheet.save(p)
        paths.append(p)
    return paths


# --------------------------------------------------------------------------- #
# agents
# --------------------------------------------------------------------------- #
class CurriculumAgent(Agent):
    name = "curriculum"
    requires = ("memory_version", "rsi")
    provides = ("task",)

    def __init__(self, max_attempts: int = 2, budget_usd: float = 5.0, tasks: Optional[list[str]] = None) -> None:
        super().__init__(max_attempts=max_attempts, budget_usd=budget_usd, tasks=tasks)

    def run(self, bb: Blackboard) -> None:
        rsi = bb["rsi"]
        mem: Memory = rsi["memory"]
        spent = sum(b.spent_usd for b in rsi["backends"] if b)
        if spent >= self.params["budget_usd"]:
            self.warn(bb, f"budget spent (${spent:.2f}); stopping practice")
            bb.post("rsi_done", {"reason": "budget"}, self.name)
            return
        index = mem.claims("sheet_index")
        pages = {e.page: norm(e.title) for e in index.sheet_index} if index else {}
        allowed = self.params["tasks"]
        unreachable = []
        for t in sorted(TASKS, key=lambda t: t.priority):
            if allowed and t.id not in allowed:
                continue
            if mem.has(t.id) or any(not mem.has(d) for d in t.depends_on):
                continue
            eps = mem.episodes(t.id)
            since = max((k for k, e in enumerate(eps) if e.get("audit")), default=-1)
            attempts = eps[since + 1:]
            if len(attempts) >= self.params["max_attempts"]:
                continue
            page = None
            if t.title_keys:
                page = next((p for p, title in sorted(pages.items())
                             if any(norm(k) in title for k in t.title_keys)), None)
                if page is None:
                    unreachable.append(t.id)
                    continue
            hint = ""
            if eps and not eps[-1]["ok"]:
                last = eps[-1]
                failed = [c for c in last["checks"] if not c["ok"]]
                hint = "Previous attempt failed these checks - re-read carefully: " + "; ".join(
                    f"{c['check']} ({c['detail']})" for c in failed)
            task = {"id": t.id, "page": page, "attempt": len(attempts) + 1, "hint": hint}
            self.info(bb, f"next practice: {t.id}" + (f" on page {page}" if page else "")
                      + (f" (attempt {task['attempt']})" if attempts else ""))
            bb.post("task", task, self.name)
            return
        if unreachable:
            self.warn(bb, f"no sheet in the set for {unreachable}")
        bb.post("rsi_done", {"reason": "no remaining gaps", "unreachable": unreachable}, self.name)


class ActorAgent(Agent):
    name = "actor"
    requires = ("task", "rsi")
    provides = ("claims",)

    def run(self, bb: Blackboard) -> None:
        rsi = bb["rsi"]
        task: dict = bb["task"]
        t = task_by_id(task["id"])
        backend: Optional[Backend] = rsi["actor_backend"]
        images = self._images(rsi, t, task)
        if backend is None:
            path = Path(rsi["practice_dir"]) / f"{t.id}.json"
            if not path.exists():
                raise FileNotFoundError(f"no LLM backend and no practice transcript {path}")
            claims = Claims.model_validate_json(path.read_text(encoding="utf-8"))
            self.info(bb, f"{t.id}: replayed transcript ({claims.reader or 'unknown reader'})")
        else:
            prompt = self._prompt(rsi["memory"], t, task)
            t0 = time.time()
            try:
                raw = backend.ask(prompt, images)
                raw["task"] = t.id
                claims = Claims.model_validate(raw)
            except (BackendError, ValueError) as exc:
                self.warn(bb, f"{t.id}: actor reply unusable ({exc})")
                claims = Claims(task=t.id, reader=f"{backend.name} (failed)")
            claims.reader = claims.reader or f"{backend.name} actor"
            self.info(bb, f"{t.id}: read with {backend.name} in {time.time() - t0:.0f}s "
                          f"(spent so far ${backend.spent_usd:.2f})")
        bb.post("task_images", [str(p) for p in images], self.name)
        bb.post("claims", claims, self.name)

    @staticmethod
    def _images(rsi: dict, t: Task, task: dict) -> list[Path]:
        work = Path(rsi["work_dir"])
        if t.id == "sheet_index":
            return render_title_blocks(rsi["pdfs"], work / "titles")
        return render_page(rsi["pdfs"][task["page"]], work / f"page{task['page']}")

    @staticmethod
    def _prompt(mem: Memory, t: Task, task: dict) -> str:
        lessons = "\n".join(f"- {ls}" for ls in mem.lessons) or "- (none yet)"
        page = f" The sheet is page {task['page']} of the drawing set." if task.get("page") else ""
        return (
            "You are the Actor in a recursive self-improvement loop that reads scanned Japanese steel "
            f"arch-bridge drawings for a finite-element model.{page}\n\nTask: {t.goal}\n\n"
            "Images: an overview of the sheet followed by four overlapping full-resolution tiles. Transcribe "
            "values exactly as written (mm). Record the drawing number and view for what you read.\n\n"
            f"Lessons verified in earlier practice (apply them):\n{lessons}\n\n"
            + (f"{task['hint']}\n\n" if task.get("hint") else "")
            + "Add to `lessons` anything reusable you learned about reading this kind of sheet.\n\n"
            f"Return JSON shaped like:\n{t.example}"
        )


class VerifierAgent(Agent):
    name = "verifier"
    requires = ("claims", "rsi")
    optional = ("task_images",)
    provides = ("verdict",)

    def __init__(self, llm_check: bool = True) -> None:
        super().__init__(llm_check=llm_check)

    def run(self, bb: Blackboard) -> None:
        rsi = bb["rsi"]
        claims: Claims = bb["claims"]
        t = task_by_id(claims.task)
        mem: Memory = rsi["memory"]
        ctx = Context(reading=rsi["reading"],
                      known={t.id: mem.claims(t.id) for t in TASKS if t.id != claims.task and mem.has(t.id)})
        checks = t.checks(claims, ctx)
        backend: Optional[Backend] = rsi["verifier_backend"]
        llm = {"status": "not run (no backend)" if backend is None else "disabled"}
        if backend is not None and self.params["llm_check"] and all(c["ok"] for c in checks):
            llm = self._llm_confirm(backend, claims, [Path(p) for p in bb.get("task_images") or []])
            checks += [{"check": f"independent re-read: {it.get('item', '?')}", "ok": bool(it.get("agree")),
                        "detail": it.get("note", "")} for it in llm.get("items", [])]
        ok = bool(checks) and all(c["ok"] for c in checks)
        for c in checks:
            if not c["ok"]:
                self.warn(bb, f"{claims.task}: FAIL {c['check']} ({c['detail']})")
        self.info(bb, f"{claims.task}: {'PASS' if ok else 'FAIL'} ({sum(c['ok'] for c in checks)}/"
                      f"{len(checks)} checks; independent re-read: {llm.get('status', 'done')})")
        bb.post("verdict", {"task": claims.task, "ok": ok, "checks": checks, "llm": llm}, self.name)

    @staticmethod
    def _llm_confirm(backend: Backend, claims: Claims, images: list[Path]) -> dict:
        items = _claim_items(claims)
        prompt = ("You are an independent checker. Another reader claims the values below were read from "
                  "these drawing images. You do not see their reasoning. Look at the images yourself and, "
                  "for every item, say whether the drawing shows that value.\n\n"
                  "Judge only the numbers (width x thickness in mm, depth, length, area, count) and the "
                  "steel grade. Plate lists write a plate as e.g. '1-FLG PL 1700x30x4226': the piece "
                  "count, the label (FLG/WEB/PL) and the trailing piece length are not part of the claim, "
                  "and other plates may exist besides the ones listed. Agree when the drawing shows the "
                  "claimed size for that member; disagree only when the numbers or grade differ or the "
                  "value is nowhere on the sheet.\n\n"
                  + "\n".join(f"{k + 1}. {it}" for k, it in enumerate(items))
                  + '\n\nReturn {"items": [{"item": "<number and short label>", "agree": true, '
                    '"note": "<what the drawing shows if different>"}]}')
        try:
            out = backend.ask(prompt, images)
            out["status"] = "done"
            return out
        except BackendError as exc:
            return {"status": f"failed: {exc}", "items": []}


def _claim_items(c: Claims, limit: int = 30) -> list[str]:
    items: list[str] = []
    for s in c.sections:
        items.append(f"{s.name} depth {s.depth_mm:g} mm" + (f", axis length {s.axis_length_mm:g} mm"
                                                            if s.axis_length_mm else ""))
        for p in s.plates:
            if p.role in ("top", "bottom", "web"):
                items.append(f"{s.name} {p.role} plate: {p.width_mm:g} x {p.thickness_mm:g} mm"
                             + (f", {p.material}" if p.material else "") + f" ({p.count} per section)")
            else:
                per = "flange" if p.role == "flange_rib" else "web"
                items.append(f"{s.name} {p.role}: {p.width_mm:g} x {p.thickness_mm:g} mm, {p.count} per {per} "
                             f"in the cross-section")
        if s.web_rib_pitch_mm:
            items.append(f"{s.name} web-rib pitch {s.web_rib_pitch_mm:g} mm over the web depth")
    for s in c.cable_specs:
        items.append(f"{s.name}: area {s.area_mm2:g} mm2, E {s.E_kN_mm2:g} kN/mm2")
    for cb in c.cables:
        items.append(f"cable {cb.number}: {cb.strands} wires" + (f", dead tension {cb.dead_tension_kN:g} kN"
                                                                 if cb.dead_tension_kN else ""))
    for e in c.sheet_index[::4]:
        items.append(f"page {e.page}: {e.drawing_no} {e.title}")
    return items[:limit]


class MemoryAgent(Agent):
    name = "memory"
    requires = ("verdict", "claims", "rsi")
    optional = ("task",)
    provides = ("memory_version",)

    def run(self, bb: Blackboard) -> None:
        rsi = bb["rsi"]
        mem: Memory = rsi["memory"]
        v = bb["verdict"]
        claims: Claims = bb["claims"]
        spent = {b.name: round(b.spent_usd, 4) for b in rsi["backends"] if b}
        ep = {"task": claims.task, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
              "page": (bb.get("task") or {}).get("page"), "reader": claims.reader,
              "ok": v["ok"], "checks": v["checks"], "llm": v["llm"], "spent_usd": spent,
              "claims": json.loads(claims.model_dump_json())}
        path = mem.record_episode(ep)
        if v["ok"]:
            added = mem.commit(claims, v["checks"], path.name)
            self.info(bb, f"{claims.task}: committed to memory ({len(added)} new lesson(s))")
        else:
            self.warn(bb, f"{claims.task}: not committed (episode {path.name})")
        bb.post("memory_version", mem.version, self.name)
