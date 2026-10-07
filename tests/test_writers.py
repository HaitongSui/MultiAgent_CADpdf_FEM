import json
import re

import pytest

from cad2fem import PipelineConfig, run


@pytest.fixture(scope="module")
def outputs(drawings, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    bb = run(drawings["bracket"], out, PipelineConfig(plot=False))
    return bb, {k: open(v).read() for k, v in bb["outputs"].items()}


def test_abaqus_deck(outputs):
    bb, files = outputs
    inp = files["abaqus"]
    mesh = bb["fe_model"].mesh
    assert "*ELEMENT, TYPE=CPS6, ELSET=EALL" in inp
    assert "*MATERIAL, NAME=ALUMINIUM_6061_T6" in inp
    assert re.search(r"\*SOLID SECTION, ELSET=EALL, MATERIAL=\S+\n8,", inp)
    assert "BC1_HOLE1, 1, 2" in inp
    node_block = inp.split("*NODE, NSET=NALL\n")[1].split("*")[0].strip().splitlines()
    assert len(node_block) == mesh.n_nodes
    assert "*OUTPUT, FIELD" in inp and "*END STEP" in inp


def test_abaqus_pressure_as_dsload(drawings, tmp_path):
    bb = run(drawings["plate"], tmp_path, PipelineConfig(formats=("abaqus", "calculix"), plot=False))
    abq = open(bb["outputs"]["abaqus"]).read()
    ccx = open(bb["outputs"]["calculix"]).read()
    assert "*SURFACE, NAME=LOAD1_RIGHT, TYPE=ELEMENT" in abq
    assert "LOAD1_RIGHT, P, -50" in abq
    assert "*CLOAD" not in abq.split("*DSLOAD")[1]          # no double counting
    assert "*DSLOAD" not in ccx and "*CLOAD" in ccx


def test_nastran_deck(outputs):
    bb, files = outputs
    bdf = files["nastran"]
    mesh = bb["fe_model"].mesh
    assert bdf.count("\nGRID,") == mesh.n_nodes
    assert bdf.count("\nCTRIA6,") == mesh.n_elements
    assert "MAT1,1,70000.,,0.33," in bdf
    assert "PSHELL,1,1,8." in bdf
    fy = sum(float(ln.split(",")[6]) for ln in bdf.splitlines() if ln.startswith("FORCE,"))
    assert fy == pytest.approx(-2000.0)
    assert bdf.rstrip().endswith("ENDDATA")


def test_gmsh_and_json(outputs):
    bb, files = outputs
    msh = files["gmsh"]
    mesh = bb["fe_model"].mesh
    n_nodes = int(msh.split("$Nodes\n")[1].split("\n")[0])
    assert n_nodes == mesh.n_nodes
    assert '"BC1_HOLE1"' in msh and '"LOAD1_HOLE2"' in msh
    data = json.loads(files["json"])
    assert data["element_type"] == "tri6" and len(data["elements"]) == mesh.n_elements
    assert data["constraints"][0]["nset"] == "BC1_HOLE1"
