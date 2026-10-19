"""RSI loop: curriculum -> actor -> verifier -> memory, and reuse of frozen memory."""

import json
from pathlib import Path

import pytest

from cad2fem.bridge.arch_reading import ArchReading
from cad2fem.rsi import RSIConfig, run_rsi
from cad2fem.rsi.backends import Backend, extract_json
from cad2fem.rsi.claims import Claims
from cad2fem.rsi.knowledge import knowledge_from_memory
from cad2fem.rsi.loop import practice
from cad2fem.rsi.memory import Memory, MemoryFrozen
from cad2fem.rsi.tasks import TASKS, Context, task_by_id

ROOT = Path(__file__).resolve().parents[1]
PRACTICE = ROOT / "cad_pdf" / "readings" / "practice"
READING = ROOT / "cad_pdf" / "readings" / "J-5_J-8_arch_reading.json"

pytestmark = pytest.mark.skipif(not (ROOT / "cad_pdf").is_dir(), reason="drawing set not present")


def _reading() -> ArchReading:
    return ArchReading.model_validate_json(READING.read_text("utf-8"))


def _transcript(task: str) -> Claims:
    return Claims.model_validate_json((PRACTICE / f"{task}.json").read_text("utf-8"))


def _cfg(tmp_path, **kw) -> RSIConfig:
    return RSIConfig(pdf_dir=str(ROOT / "cad_pdf"), reading_file=str(READING),
                     memory_dir=str(tmp_path / "memory"), practice_dir=str(PRACTICE),
                     work_dir=str(tmp_path / "work"), out_dir=str(tmp_path / "out"), **kw)


class FakeBackend(Backend):
    """Answers like the model would: transcripts for the actor, agreement for the checker.
    ``corrupt`` maps task id -> number of first replies to spoil."""

    name = "fake"

    def __init__(self, corrupt=None):
        super().__init__()
        self.corrupt = dict(corrupt or {})
        self.prompts = []

    def ask(self, prompt, images):
        self.calls += 1
        self.spent_usd += 0.01
        self.prompts.append(prompt)
        if "independent checker" in prompt:
            return {"items": [{"item": "1", "agree": True, "note": ""}]}
        t = next(t for t in TASKS if t.goal in prompt)
        d = json.loads(_transcript(t.id).model_dump_json())
        if self.corrupt.get(t.id, 0) > 0:
            self.corrupt[t.id] -= 1
            d["sections"][0]["axis_length_mm"] += 500          # misread 部材長
        return d


def test_checks_accept_the_verified_transcripts():
    ctx = Context(reading=_reading())
    for t in TASKS:
        checks = t.checks(_transcript(t.id), ctx)
        bad = [c for c in checks if not c["ok"]]
        assert checks and not bad, (t.id, bad)


def test_verifier_catches_misread_values():
    ctx = Context(reading=_reading())
    cab = _transcript("cables")
    cab.cable_specs[0].area_mm2 *= 1.2
    assert not all(c["ok"] for c in task_by_id("cables").checks(cab, ctx))

    rib = _transcript("rib_seg3")
    rib.sections[0].axis_length_mm = 8100.0
    bad = [c["check"] for c in task_by_id("rib_seg3").checks(rib, ctx) if not c["ok"]]
    assert any("部材長" in b for b in bad)

    ge = _transcript("girder_end")
    next(p for p in ge.sections[0].plates if p.role == "web").width_mm = 2500
    assert not all(c["ok"] for c in task_by_id("girder_end").checks(ge, ctx))


def test_memory_commit_and_freeze(tmp_path):
    mem = Memory(tmp_path / "m")
    v0 = mem.version
    c = _transcript("xbeam")
    mem.commit(c, [], "001_xbeam.json")
    assert mem.has("xbeam") and mem.version != v0
    assert set(c.lessons) <= set(Memory(tmp_path / "m").lessons)        # persisted
    frozen = Memory(tmp_path / "m", frozen=True)
    assert frozen.claims("xbeam").sections[0].name == "XBEAM"
    with pytest.raises(MemoryFrozen):
        frozen.commit(c, [], "x")
    with pytest.raises(MemoryFrozen):
        frozen.record_episode({"task": "xbeam"})


def test_extract_json_tolerates_fences():
    assert extract_json('ok\n```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert extract_json('prefix {"b": 1} trailing') == {"b": 1}


def test_failed_attempt_is_retried_with_a_hint_and_lessons(tmp_path):
    fake = FakeBackend(corrupt={"rib_seg4": 1})
    cfg = _cfg(tmp_path, tasks=["sheet_index", "rib_seg4"])
    practice(cfg, actor=fake, verifier=fake)
    mem = Memory(cfg.memory_dir)
    eps = mem.episodes("rib_seg4")
    assert [e["ok"] for e in eps] == [False, True]
    assert mem.has("rib_seg4") and mem.claims("rib_seg4").sections[0].axis_length_mm == 8190.0
    actor_prompts = [p for p in fake.prompts if "Task: Read arch-rib segment 4" in p]
    assert len(actor_prompts) == 2
    assert "部材長" in actor_prompts[1]                         # hint from the failed check
    assert mem.claims("sheet_index").lessons[0] in actor_prompts[0]   # earlier lesson reused


def test_offline_loop_fills_the_model_gaps(tmp_path):
    res = run_rsi(_cfg(tmp_path, backend="none"))
    mem: Memory = res["memory"]
    assert all(mem.has(t.id) for t in TASKS)
    assert all(e["ok"] for e in mem.episodes())
    kn = res["knowledge"]
    assert kn.sections["GIRDER_END"].height == pytest.approx(3.038)
    assert kn.sections["GIRDER"].height == pytest.approx(1.95)
    assert kn.cable_section_by_anchor and set(kn.cable_section_by_anchor.values()) == {
        "CABLE_SPWC19", "CABLE_SPWC55", "CABLE_SPWC73"}
    assert kn.rib_segments[-1][1] == pytest.approx(76.8536, abs=1e-3)
    m = res["learned"]["frame"]
    assert {b.section for b in m.beams} >= {"GIRDER_END", "RIB_SEG5", "XBEAM", "CABLE_SPWC73"}
    dead = res["comparison"]["DEAD"]["midspan_deflection_mm"]
    assert dead[1] < dead[0]                  # verified sections are stiffer than the assumptions
    assert Path(res["report"]).exists()


def test_knowledge_from_empty_memory_is_empty(tmp_path):
    assert knowledge_from_memory(Memory(tmp_path / "m"), _reading()).empty


def test_audit_evicts_knowledge_that_fails_a_new_check_and_relearns_it(tmp_path):
    cfg = _cfg(tmp_path, tasks=["sheet_index", "rib_seg5"])
    mem = Memory(cfg.memory_dir)
    mem.commit(_transcript("sheet_index"), [], "seed")
    bad = _transcript("rib_seg5")                         # one rib listed once per panel
    bad.sections[0].plates.append(bad.sections[0].plates[-1].model_copy(update={"count": 4}))
    bad.lessons = ["a lesson that only this claim taught"]
    mem.commit(bad, [], "seed")
    fake = FakeBackend()
    practice(cfg, actor=fake, verifier=fake)
    mem = Memory(cfg.memory_dir)
    eps = mem.episodes("rib_seg5")
    assert eps[0]["audit"] and not eps[0]["ok"] and eps[-1]["ok"]
    assert "a lesson that only this claim taught" not in mem.lessons
    assert len(_plates_by_role(mem.claims("rib_seg5"), "web_rib")) == 1
    actor = [p for p in fake.prompts if "Task: Read arch-rib segment 5" in p]
    assert len(actor) == 1 and "one plate entry per role" in actor[0]          # audit failure as hint


def _plates_by_role(c, role):
    return [p for p in c.sections[0].plates if p.role == role]


def test_rib_layout_must_match_other_verified_segments_of_the_same_depth():
    ctx = Context(reading=_reading(), known={"rib_seg4": _transcript("rib_seg4")})
    c = _transcript("rib_seg5")
    assert all(x["ok"] for x in task_by_id("rib_seg5").checks(c, ctx))
    next(p for p in c.sections[0].plates if p.role == "web_rib").count = 4    # pieces over both webs
    bad = [x["check"] for x in task_by_id("rib_seg5").checks(c, ctx) if not x["ok"]]
    assert "web_rib count per flange/web = verified rib_seg4 (same depth 1200)" in bad
    c.sections[0].web_rib_pitch_mm = None                 # caught by the sibling segment alone
    bad = [x["check"] for x in task_by_id("rib_seg5").checks(c, ctx) if not x["ok"]]
    assert bad == ["web_rib count per flange/web = verified rib_seg4 (same depth 1200)"]


def test_web_rib_count_must_follow_the_pitch_in_the_section_view():
    ctx = Context(reading=_reading())
    c = _transcript("rib_seg4")
    next(p for p in c.sections[0].plates if p.role == "web_rib").count = 4
    bad = [x["check"] for x in task_by_id("rib_seg4").checks(c, ctx) if not x["ok"]]
    assert bad == ["web ribs per web = web depth / pitch - 1"]


def test_reviewer_can_send_a_task_back(tmp_path):
    cfg = _cfg(tmp_path, backend="none", tasks=["sheet_index", "xbeam"])
    practice(cfg)
    cfg.relearn = {"xbeam": "check the bottom flange width"}
    practice(cfg)
    eps = Memory(cfg.memory_dir).episodes("xbeam")
    assert [e["ok"] for e in eps] == [True, False, True]
    assert eps[1]["checks"][0]["detail"] == "check the bottom flange width"
