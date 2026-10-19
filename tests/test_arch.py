"""Arch bridge read from scanned sheets J-5 / J-8 (cad_pdf pages 14 and 17)."""

import glob
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cad2fem.bridge import ArchConfig, run_arch
from cad2fem.bridge.arch import b_live_load, box_section, fit_arch
from cad2fem.bridge.arch_reading import ArchReading

ROOT = Path(__file__).resolve().parents[1]
READING = ROOT / "cad_pdf" / "readings" / "J-5_J-8_arch_reading.json"


def reading() -> ArchReading:
    return ArchReading.model_validate_json(READING.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def arch_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("arch")
    return run_arch([], out, ArchConfig(reading_file=str(READING), plot=False))


def test_reading_is_self_consistent():
    geo = fit_arch(reading())
    failed = [c["check"] for c in geo.checks if not c["ok"]]
    assert not failed, failed
    assert geo.fit_rms_mm < 50
    assert geo.stations["C10"] == pytest.approx(51.35) and geo.span == pytest.approx(102.7)
    assert geo.x_spring == pytest.approx((14.15, 88.55))
    zs, zc, _ = geo.params
    assert 8.5 < zc < 9.2          # rib anchors ~8.9 m above the girder axis at the crown
    assert np.all(np.diff(geo.anchor_x) > 0)


def test_misread_values_are_caught():
    r = reading()
    r.cable_anchor_chain_mm[9] += 100.0                 # 4401.5 -> 4501.5
    checks = {c["check"]: c["ok"] for c in fit_arch(r).checks}
    assert not checks["sum of ケーブル間隔 = アーチリブ軸線長"]
    r = reading()
    r.cables[4].length_mm += 1500.0                     # one hanger length misread
    geo = fit_arch(r)
    assert not {c["check"]: c["ok"] for c in geo.checks}["hanger lengths symmetric (max rel. diff)"]
    assert geo.fit_rms_mm > 100


def test_b_live_load_and_box_section():
    live = b_live_load(102.7, 9.0, 3.75)
    assert live["p2"] == pytest.approx(3.273)
    assert live["q_uniform"] == pytest.approx(3.273 * (5.5 + 3.5 / 2 + 7.5))
    assert live["q_p1"] == pytest.approx(10 * 7.25)
    s = box_section("B", 1.6, 1.95, 0.03, 0.016)
    assert s.A == pytest.approx(2 * 1.6 * 0.03 + 2 * 1.89 * 0.016)
    assert s.Iyy > s.Izz


def test_tied_arch_behaviour(arch_run):
    bb = arch_run
    res = bb["bridge_results"]
    m = bb["frame"]
    assert len([b for b in m.beams if b.section == "CABLE"]) == 36
    for case, s in res.items():
        assert s["equilibrium_error"] < 1e-9
        assert min(s["hanger_force_kN"].values()) > 0, case          # all hangers in tension
        # the girder ties the arch: crown rib compression ~ girder tension at midspan
        assert s["rib_axial_crown_kN"] < 0 < s["girder_axial_midspan_kN"]
        assert abs(s["rib_axial_crown_kN"] + s["girder_axial_midspan_kN"]) < 0.05 * abs(s["rib_axial_crown_kN"])
        f = s["hanger_force_kN"]
        assert f["C6-L"] == pytest.approx(f["C14-R"], rel=0.02)    # symmetric structure & load
    # the S1 bearing carries no longitudinal load from vertical loads (tied system)
    assert abs(res["DEAD"]["reactions"]["G1_S1"][0]) < 1e-6 * res["DEAD"]["reaction_sum_kN"][2]


def test_writers_and_report(arch_run):
    bb = arch_run
    inp = open(bb["outputs"]["abaqus"]).read()
    assert inp.isascii() and "*MPC" in inp and "ELSET=CABLE_G1_C10_L" in inp
    assert "*STEP, NAME=LIVE_B, PERTURBATION" in inp
    report = Path(bb["report"]).read_text(encoding="utf-8")
    assert "ASSUMED" in report and "新田豊島橋" in report


def test_opensees_reproduces_arch(arch_run, tmp_path):
    pytest.importorskip("openseespy")
    bb = arch_run
    out = tmp_path / "os.json"
    p = subprocess.run([sys.executable, bb["outputs"]["opensees"], str(out)], capture_output=True,
                       text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    r = json.loads(out.read_text())
    m = bb["frame"]
    for case, s in bb["bridge_results"].items():
        u = np.array([r[case]["disp"][str(i + 1)] for i in range(m.n_nodes)])
        np.testing.assert_allclose(u, s["result"].u, atol=1e-9)


def test_vision_agent_calls_claude_with_page_images(tmp_path):
    pytest.importorskip("anthropic")
    pages = sorted(glob.glob(str(ROOT / "cad_pdf" / "*-14.pdf")))
    assert pages, "cad_pdf sheet 14 missing"
    calls = []

    def parse(**kw):
        calls.append(kw)
        return SimpleNamespace(parsed_output=reading(), stop_reason="end_turn")

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(parse=parse)))
    bb = run_arch(pages, tmp_path, ArchConfig(formats=("json",), plot=False, verify=False),
                  vision_client=client)
    kw = calls[0]
    images = [c for c in kw["messages"][0]["content"] if c["type"] == "image"]
    assert len(images) == 5 and kw["output_format"] is ArchReading
    assert kw["model"] == "claude-opus-5-5" and kw["fallbacks"] == "default"
    assert bb["arch_reading"].bridge_name == "新田豊島橋（仮称）"
    assert bb.has("outputs")


def test_vision_without_credentials_explains_fallback(tmp_path):
    pytest.importorskip("anthropic")
    pages = sorted(glob.glob(str(ROOT / "cad_pdf" / "*-14.pdf")))

    def parse(**kw):
        raise TypeError("Could not resolve authentication method")

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(parse=parse)))
    from cad2fem.core import PipelineError
    with pytest.raises(PipelineError, match="--reading"):
        run_arch(pages, tmp_path, ArchConfig(formats=(), plot=False), vision_client=client)
