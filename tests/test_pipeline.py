import math
import shutil
import subprocess

import numpy as np
import pytest

from cad2fem import PipelineConfig, run
from cad2fem.agents import ValidatorAgent
from cad2fem.pipeline import build_agents

FAST = dict(plot=False)


def test_plate_with_hole_geometry_and_model(drawings, tmp_path):
    bb = run(drawings["plate"], tmp_path, PipelineConfig(**FAST))
    g = bb["geometry"]
    x0, y0, x1, y1 = g.bbox
    assert (x1 - x0, y1 - y0) == pytest.approx((200.0, 100.0), rel=1e-6)
    (hole,) = g.holes
    assert hole.is_circle and hole.radius == pytest.approx(20.0, rel=1e-4)
    assert hole.center == pytest.approx((100.0, 50.0), abs=1e-3)
    assert g.area == pytest.approx(200 * 100 - math.pi * 400, rel=2e-3)

    model = bb["fe_model"]
    assert model.spec.title == "PLATE_WITH_HOLE"
    assert bb["validation"]["ok"]
    assert model.mesh.element_areas().min() > 0
    # 50 MPa over 100 mm x 10 mm
    np.testing.assert_allclose(model.total_nodal_forces().sum(axis=0), [50000.0, 0.0], rtol=1e-5, atol=1e-6)
    ver = bb["verification"]
    assert ver["equilibrium_error"] < 1e-8
    # Kt (net section, d/W = 0.4) ~ 2.2 -> peak ~180 MPa; coarse CST underestimates a bit
    assert 140 < ver["max_von_mises_mpa"] < 220
    assert set(bb["outputs"]) == {"abaqus", "calculix", "nastran", "gmsh", "json"}


def test_plain_plate_matches_analytic(drawings, tmp_path):
    bb = run(drawings["plain"], tmp_path, PipelineConfig(**FAST))
    res = bb["verification"]["result"]
    mesh = bb["fe_model"].mesh
    right = mesh.nodes[:, 0] > 200 - 1e-6
    # u = sigma L / E (slightly less due to the clamped Poisson contraction)
    assert res.u[right, 0].mean() == pytest.approx(50 * 200 / 210000, rel=0.02)
    cent = mesh.nodes[mesh.elements].mean(axis=1)
    far = (cent[:, 0] > 60) & (cent[:, 0] < 180)
    assert res.stress[far, 0].mean() == pytest.approx(50.0, rel=0.01)


def test_l_bracket_arcs_tri6_force(drawings, tmp_path):
    bb = run(drawings["bracket"], tmp_path, PipelineConfig(**FAST))
    model = bb["fe_model"]
    assert model.mesh.order == 2 and model.mesh.elements.shape[1] == 6
    g = model.geometry
    assert len(g.holes) == 2
    assert [h.center[0] for h in g.holes] == pytest.approx([15.0, 130.0], abs=1e-3)
    assert model.spec.material.name.startswith("ALUMINIUM")
    # inner fillet R10 replaces a sharp corner: area = L-shape - (1 - pi/4) R^2 - 2 holes
    expected = 150 * 30 + 30 * 90 + (1 - math.pi / 4) * 100 - 2 * math.pi * 36
    assert g.area == pytest.approx(expected, rel=2e-3)
    np.testing.assert_allclose(model.total_nodal_forces().sum(axis=0), [0.0, -2000.0], atol=1e-6)
    assert bb["verification"]["reaction_resultant_n"] == pytest.approx([0.0, 2000.0], abs=1e-6)


def test_wrong_title_scale_is_overruled_by_dimensions(drawings, tmp_path):
    bb = run(drawings["wrong_scale"], tmp_path, PipelineConfig(**FAST))
    x0, y0, x1, y1 = bb["geometry"].bbox
    assert (x1 - x0, y1 - y0) == pytest.approx((200.0, 100.0), rel=1e-6)
    assert any("disagrees with dimensions" in m.text for m in bb.warnings())


def test_validator_feedback_loop_refines_mesh(drawings, tmp_path):
    cfg = PipelineConfig(formats=("json",), verify=False, plot=False)
    agents = build_agents(cfg)
    i = next(k for k, a in enumerate(agents) if a.name == "validator")
    agents[i] = ValidatorAgent(min_angle_warn=89.0, max_rework=2)   # impossible -> always rework
    from cad2fem.pipeline import run as run_pipeline
    bb = run_pipeline(drawings["plain"], tmp_path, cfg, agents=agents)
    requests = [m for m in bb.messages if m.level == "REQUEST"]
    assert len(requests) == 2
    assert bb["validation"]["stats"]["rework"] == 2
    mesh_agent = next(a for a in agents if a.name == "mesh")
    assert mesh_agent.params["refine"] == pytest.approx(0.49)


def test_overrides(drawings, tmp_path):
    cfg = PipelineConfig(mesh_size=10.0, element_order=2, thickness=4.0, analysis="PLANE_STRAIN",
                         formats=("abaqus",), **FAST)
    bb = run(drawings["plate"], tmp_path, cfg)
    spec = bb["spec"]
    assert (spec.thickness, spec.element_order, spec.analysis) == (4.0, 2, "PLANE_STRAIN")
    assert bb["mesh_size_used"] == 10.0
    inp = open(bb["outputs"]["abaqus"]).read()
    assert "TYPE=CPE6" in inp


@pytest.mark.skipif(shutil.which("ccx") is None, reason="CalculiX (ccx) not installed")
def test_calculix_solves_generated_deck(drawings, tmp_path):
    bb = run(drawings["plate"], tmp_path, PipelineConfig(formats=("calculix",), **FAST))
    inp = bb["outputs"]["calculix"]
    job = inp[:-4]
    proc = subprocess.run(["ccx", "-i", job], cwd=tmp_path, capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, proc.stdout[-2000:]
    dat = open(job + ".dat").read()
    fx = float(dat.split("total force")[1].split("\n")[2].split()[0])
    assert fx == pytest.approx(-50000.0, rel=1e-4)
