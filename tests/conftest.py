import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

import make_sample_drawings as samples  # noqa: E402


@pytest.fixture(scope="session")
def drawings(tmp_path_factory):
    d = tmp_path_factory.mktemp("drawings")
    return {
        "plate": samples.plate_with_hole(d / "plate_with_hole.pdf"),
        "plain": samples.plate_with_hole(d / "plain_plate.pdf", hole=False),
        "bracket": samples.l_bracket(d / "l_bracket.pdf"),
        "wrong_scale": samples.plate_with_hole(d / "wrong_scale.pdf", title_scale="1:5"),
    }


import make_bridge_drawings as bridge_samples  # noqa: E402


@pytest.fixture(scope="session")
def bridge_pdfs(tmp_path_factory):
    d = tmp_path_factory.mktemp("bridge")
    return {
        "set": bridge_samples.bridge_set(d / "bridge_3span.pdf"),
        "no_span_note": bridge_samples.bridge_set(d / "no_span_note.pdf", note_spans=False),
    }


@pytest.fixture(scope="session")
def bridge_run(bridge_pdfs, tmp_path_factory):
    from cad2fem.bridge import BridgeConfig, run_bridge

    out = tmp_path_factory.mktemp("bridge_out")
    return run_bridge([bridge_pdfs["set"]], out, BridgeConfig(plot=False))
