"""Whole-bridge pipeline: drawing set -> global spine model."""

import json
import math
import subprocess
import sys

import numpy as np
import pytest

from cad2fem.bridge import BridgeConfig, run_bridge
from cad2fem.bridge.assembly import build_frame, summarise
from cad2fem.bridge.frame_solver import solve
from cad2fem.bridge.layout_rules import parse_layout, parse_spans
from cad2fem.bridge.models import Beam, BridgeLayout, BridgeMaterial, FrameModel
from cad2fem.bridge.sections import rectangle_section, section_from_geometry
from cad2fem.models import Geometry, Loop


def rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


# ---------------------------------------------------------------- notes ----
def test_span_forms():
    assert parse_spans("30+40+30") == [30, 40, 30]
    assert parse_spans("3×30") == [30, 30, 30]
    assert parse_spans("4X25+2X30") == [25] * 4 + [30] * 2
    assert parse_spans("(30+40+30)") == [30, 40, 30]


def test_layout_notes_bilingual():
    lay = parse_layout(["桥名: 某河大桥", "跨径布置: 3×30 m", "墩高: P1=8.0 m, P2=10.0 m",
                        "支座: A0 双向活动, P1 固定, P2 纵向活动, A3 双向活动",
                        "主梁 DECK: C50;  桥墩 PIER: C40;  容重 26 kN/m3",
                        "二期恒载: 45 kN/m", "车道荷载: qk=10.5 kN/m, Pk=320 kN, 2车道"])
    assert lay.spans == [30, 30, 30]
    assert lay.pier_heights == {"P1": 8.0, "P2": 10.0}
    assert lay.bearings == {"A0": "SLIDING_XY", "P1": "FIXED", "P2": "SLIDING_X", "A3": "SLIDING_XY"}
    assert (lay.deck_material.name, lay.pier_material.name) == ("C50", "C40")
    assert lay.deck_material.E == 3.45e7 and lay.pier_material.E == 3.25e7
    assert (lay.secondary_dead, lay.lane_q, lay.lane_P, lay.lanes) == (45, 10.5, 320, 2)
    assert parse_layout(["SPANS 30+30", "墩梁固结: P1"]).bearings == {"P1": "MONOLITHIC"}


# ------------------------------------------------------------- sections ----
def test_section_properties_and_torsion():
    s = section_from_geometry("R", Geometry(Loop(rect(0, 0, 2.0, 1.6))))
    assert s.A == pytest.approx(3.2) and s.Iyy == pytest.approx(2 * 1.6 ** 3 / 12)
    assert s.Izz == pytest.approx(1.6 * 2 ** 3 / 12) and s.cz == pytest.approx(0.8)
    assert s.J == pytest.approx(rectangle_section("R", 2.0, 1.6).J, rel=0.01)
    a = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    c = section_from_geometry("C", Geometry(Loop(np.c_[np.cos(a), np.sin(a)], True, (0, 0), 1.0)))
    assert c.J == pytest.approx(math.pi / 2, rel=0.005)
    tube = section_from_geometry("T", Geometry(Loop(rect(0, 0, 2, 2)), [Loop(rect(0.1, 0.1, 1.9, 1.9))]))
    assert tube.n_cells == 1
    assert tube.J == pytest.approx(0.7032, rel=0.005)     # converged FE value (Bredt: 0.686)


# ---------------------------------------------------------------- solver ----
def test_cantilever_pier_tip_load():
    E, H, P = 3.0e7, 10.0, 100.0
    sec = rectangle_section("S", 1.0, 2.0)
    m = FrameModel(nodes=np.array([[0, 0, z] for z in np.linspace(0, H, 6)], float),
                   sections={"S": sec}, materials={"M": BridgeMaterial("M", E, 0.2, 0.0)})
    m.beams = [Beam(i, i + 1, "S", "M", (1.0, 0.0, 0.0), "COL") for i in range(5)]
    m.supports = {0: (1, 2, 3, 4, 5, 6)}
    F = np.zeros((6, 6))
    F[5, 0], F[5, 1] = P, P
    m.load_cases["H"] = F
    u = solve(m, "H").u
    # local y = global X -> bending about local z uses Izz; global Y uses Iyy
    assert u[5, 0] == pytest.approx(P * H ** 3 / (3 * E * sec.Izz), rel=1e-9)
    assert u[5, 1] == pytest.approx(P * H ** 3 / (3 * E * sec.Iyy), rel=1e-9)


def test_two_span_continuous_girder_matches_theory():
    lay = BridgeLayout(spans=[30.0, 30.0], pier_heights={"P1": 8.0},
                       bearings={"A0": "SLIDING_X", "P1": "FIXED", "A2": "SLIDING_X"},
                       deck_material=BridgeMaterial("C50", 3.45e7, 0.2, 0.0),
                       pier_material=BridgeMaterial("C40", 3.25e7, 0.2, 0.0), lane_q=10.0)
    m = build_frame(lay, {"DECK": rectangle_section("D", 6.0, 2.0),
                          "PIER": rectangle_section("P", 3.0, 6.0)}, 2.0)
    s = summarise(m, "LIVE", solve(m, "LIVE"))
    q, L = 10.0, 30.0
    assert s["reactions"]["BASE_P1"][2] == pytest.approx(10 / 8 * q * L, rel=0.005)
    assert s["reactions"]["GROUND_A0"][2] == pytest.approx(3 / 8 * q * L, rel=0.005)
    assert s["max_hogging_kNm"] == pytest.approx(-q * L ** 2 / 8, rel=0.005)
    assert s["max_sagging_kNm"] == pytest.approx(9 / 128 * q * L ** 2, rel=0.005)


def test_mechanism_is_detected():
    lay = BridgeLayout(spans=[30.0, 30.0], pier_heights={"P1": 8.0},
                       bearings={"A0": "SLIDING_XY", "P1": "SLIDING_XY", "A2": "SLIDING_XY"})
    m = build_frame(lay, {"DECK": rectangle_section("D", 6.0, 2.0),
                          "PIER": rectangle_section("P", 3.0, 6.0)}, 3.0)
    with pytest.raises(np.linalg.LinAlgError):
        solve(m, "DEAD")


# -------------------------------------------------------------- pipeline ----
def test_drawing_set_is_understood(bridge_run):
    bb = bridge_run
    assert bb["sheet_kinds"] == ["GENERAL", "DECK", "PIER"]
    lay = bb["layout"]
    assert lay.spans == [30, 40, 30] and lay.pier_heights == {"P1": 8.0, "P2": 10.0}
    assert lay.bearings["P1"] == "FIXED" and lay.lanes == 2
    el = bb["elevation"]
    assert el["spans"] == pytest.approx([30, 40, 30], rel=1e-4)
    assert el["pier_heights"]["P2"] == pytest.approx(10.0, rel=1e-4)
    deck, pier = bb["sections"]["DECK"], bb["sections"]["PIER"]
    assert (deck.width, deck.height, deck.n_cells) == (pytest.approx(12.0), pytest.approx(2.2), 1)
    # hand calculation from the drawing (cm): web block + top slab + 2 tapered overhangs - cell
    outline = 650 * 170 + 650 * 50 + 2 * 275 * (50 + 20) / 2
    cell = 560 * 173 - 2 * (30 * 15 / 2)
    assert deck.A == pytest.approx((outline - cell) / 1e4, rel=1e-4)       # 6.582 m2
    assert pier.A == pytest.approx(1.6 * 4.0 - 2 * 0.2 ** 2, rel=1e-4)
    assert pier.Izz == pytest.approx(1.322, rel=2e-3) and pier.Iyy > pier.Izz


def test_loads_equilibrium_and_three_moment_equation(bridge_run):
    bb = bridge_run
    m, res = bb["frame"], bb["bridge_results"]
    deck, pier = bb["sections"]["DECK"], bb["sections"]["PIER"]
    q = 26 * deck.A + 45
    dead = q * 100 + 26 * pier.A * (8 + 10)
    assert -res["DEAD"]["applied_kN"][2] == pytest.approx(dead, rel=1e-9)
    assert -res["LIVE"]["applied_kN"][2] == pytest.approx(2 * 10.5 * 100 + 2 * 320, rel=1e-9)
    for r in res.values():
        assert r["equilibrium_error"] < 1e-9
    # symmetric 30+40+30 continuous beam: M_support = -q (L1^3 + L2^3) / 4 / (2 L1 + 3 L2)
    M = -q * (30 ** 3 + 40 ** 3) / 4 / (2 * 30 + 3 * 40)
    assert res["DEAD"]["max_hogging_kNm"] == pytest.approx(M, rel=0.01)
    assert len(m.beams_in("DECK")) == 16 + 20 + 16 and len(m.springs) == 11


def test_spans_from_elevation_when_notes_are_silent(bridge_pdfs, tmp_path):
    bb = run_bridge([bridge_pdfs["no_span_note"]], tmp_path, BridgeConfig(formats=(), plot=False,
                                                                           verify=False))
    assert bb["layout"].spans == pytest.approx([30, 40, 30], rel=1e-3)
    assert "spans" not in bb["layout"].explicit
    assert any("spans taken from elevation" in msg.text for msg in bb.messages)


def test_validator_requests_finer_spine(bridge_pdfs, tmp_path):
    bb = run_bridge([bridge_pdfs["set"]], tmp_path,
                    BridgeConfig(formats=("json",), plot=False, verify=False, element_size=10.0))
    reqs = [msg for msg in bb.messages if msg.level == "REQUEST"]
    assert reqs and reqs[0].target == "assembly"
    assert min(bb["bridge_validation"]["elements_per_span"]) >= 10


def test_writers(bridge_run):
    bb = bridge_run
    m = bb["frame"]
    inp = open(bb["outputs"]["abaqus"]).read()
    assert inp.count("TYPE=B33") == len({b.group for b in m.beams})
    assert "*BEAM GENERAL SECTION, ELSET=PIER_P1, SECTION=GENERAL" in inp
    assert "TYPE=SPRING2" in inp and "*MPC" in inp and "*STEP, NAME=LIVE, PERTURBATION" in inp
    assert inp.isascii()
    bdf = open(bb["outputs"]["nastran"]).read()
    assert bdf.count("\nCBAR,") == len(m.beams) and bdf.count("\nCELAS2,") == len(m.springs)
    assert bdf.count("\nRBE2,") == len(m.rigid) and "SUBCASE 2" in bdf and bdf.isascii()
    data = json.load(open(bb["outputs"]["json"]))
    assert data["layout"]["spans"] == [30, 40, 30] and len(data["beams"]) == len(m.beams)


def test_opensees_script_reproduces_results(bridge_run):
    pytest.importorskip("openseespy")
    bb = bridge_run
    script = bb["outputs"]["opensees"]
    out = script.replace(".py", "_test.json")
    proc = subprocess.run([sys.executable, script, out], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    r = json.load(open(out))
    m = bb["frame"]
    for case, summ in bb["bridge_results"].items():
        u = np.array([r[case]["disp"][str(i + 1)] for i in range(m.n_nodes)])
        np.testing.assert_allclose(u, summ["result"].u, atol=1e-10)


def test_llm_layout_agent_fills_gaps(bridge_pdfs, tmp_path):
    pytest.importorskip("anthropic")
    from types import SimpleNamespace

    from cad2fem.bridge.llm_layout import LLMLayout, LLMLayoutAgent
    from cad2fem.bridge.pipeline import build_bridge_agents

    parsed = LLMLayout(title=None, spans_m=[30.0, 40.0, 30.0], piers=[], bearings=[],
                       deck_material=None, pier_material=None, secondary_dead_kN_per_m=None,
                       lane_q_kN_per_m=None, lane_P_kN=None, lanes=None, element_size_m=None,
                       remarks=["spans read from the dimension chain"])
    calls = []

    def parse(**kw):
        calls.append(kw)
        return SimpleNamespace(parsed_output=parsed, stop_reason="end_turn")

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(parse=parse)))
    cfg = BridgeConfig(formats=(), plot=False, verify=False)
    agents = build_bridge_agents(cfg)
    agents.insert(4, LLMLayoutAgent(client=client))
    bb = run_bridge([bridge_pdfs["no_span_note"]], tmp_path, cfg, agents=agents)
    assert calls and calls[0]["output_format"] is LLMLayout and calls[0]["fallbacks"] == "default"
    assert bb["layout"].spans == [30.0, 40.0, 30.0] and "spans" in bb["layout"].explicit
    assert any("spans taken from LLM" in msg.text for msg in bb.messages)
