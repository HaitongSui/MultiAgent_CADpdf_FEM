"""Blackboard / orchestrator behaviour and the (mocked) LLM agent."""

from types import SimpleNamespace

import pytest

from cad2fem.agents.llm_annotation import LLMAnnotationAgent, LLMBoundaryCondition, LLMLoad, LLMSpec
from cad2fem.agents.spec import SpecAgent
from cad2fem.core import Agent, Blackboard, Orchestrator, PipelineError
from cad2fem.models import AnalysisSpec, BoundaryCondition, DrawingData, TextItem


class Producer(Agent):
    name = "producer"
    requires = ("x",)
    provides = ("y",)

    def run(self, bb):
        bb.post("y", bb["x"] * self.params.get("k", 1), self.name)


class Checker(Agent):
    name = "checker"
    requires = ("y",)
    provides = ("ok",)

    def run(self, bb):
        if bb["y"] < 10:
            self.request(bb, "producer", "too small", k=10)
            return
        bb.post("ok", True, self.name)


def test_orchestrator_feedback_and_ordering():
    bb = Blackboard(x=2)
    orch = Orchestrator([Checker(), Producer()])   # order in the list does not matter
    orch.run(bb)
    assert bb["y"] == 20 and bb["ok"]
    assert [n for n, _ in orch.trace] == ["producer", "checker", "producer", "checker"]


def test_orchestrator_wraps_failures():
    class Boom(Agent):
        name = "boom"
        requires = ("x",)

        def run(self, bb):
            raise ValueError("bad input")

    with pytest.raises(PipelineError, match="boom"):
        Orchestrator([Boom()]).run(Blackboard(x=1))


def _drawing(*lines):
    d = DrawingData(page_width=100, page_height=100)
    d.lines = [TextItem(t, 0, 0, 1, 1) for t in lines]
    return d


class FakeClient:
    """Mimics ``client.beta.messages.parse`` of the anthropic SDK."""

    def __init__(self, parsed, stop_reason="end_turn"):
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._parse))
        self._parsed, self._stop = parsed, stop_reason

    def _parse(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(parsed_output=self._parsed, stop_reason=self._stop,
                               model=kw["model"])


LLM_RESULT = LLMSpec(
    title=None, units="mm", scale=None, thickness_mm=12.0, analysis="PLANE_STRESS",
    material_name="Ti-6Al-4V", youngs_modulus_mpa=None, poisson_ratio=None, mesh_size_mm=None,
    element_order=None,
    boundary_conditions=[LLMBoundaryCondition(location="left", dofs=[1, 2], kind="welded")],
    loads=[LLMLoad(location="RIGHT", kind="PRESSURE", magnitude=-50.0, direction_x=0, direction_y=0)],
    remarks=["'welded' interpreted as fully fixed"],
)


def test_llm_agent_and_spec_merge():
    pytest.importorskip("anthropic")
    client = FakeClient(LLM_RESULT)
    bb = Blackboard(drawing=_drawing("Plate is welded on the left, pulled with 50 MPa at right",
                                     "Ti alloy, 12 thick"))
    LLMAnnotationAgent(client=client).run(bb)
    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5" and call["output_format"] is LLMSpec
    assert call["fallbacks"] == "default"
    llm = bb["spec_llm"]
    assert llm.bcs[0].location == "LEFT" and llm.thickness == 12.0
    assert llm.material.name.startswith("TITANIUM") or llm.material.E == 113800.0

    # rules found nothing but a (conflicting) thickness -> rules keep thickness, LLM fills the rest
    rules = AnalysisSpec(thickness=10.0, explicit={"thickness"})
    bb.post("spec_rules", rules, "test")
    SpecAgent().run(bb)
    spec = bb["spec"]
    assert spec.thickness == 10.0
    assert spec.material.E == 113800.0
    assert [b.location for b in spec.bcs] == ["LEFT"] and spec.loads[0].magnitude == -50.0
    assert any("thickness" in m.text for m in bb.warnings())


def test_llm_refusal_falls_back():
    pytest.importorskip("anthropic")
    bb = Blackboard(drawing=_drawing("FIXED LEFT"))
    LLMAnnotationAgent(client=FakeClient(None, stop_reason="refusal")).run(bb)
    assert not bb.has("spec_llm")
    assert any("declined" in m.text for m in bb.warnings())


def test_spec_prefers_rules_for_bcs_unless_asked():
    bb = Blackboard()
    rules = AnalysisSpec(bcs=[BoundaryCondition("LEFT")])
    llm = AnalysisSpec(bcs=[BoundaryCondition("BOTTOM")], source="llm")
    bb.post("spec_rules", rules, "t")
    bb.post("spec_llm", llm, "t")
    SpecAgent().run(bb)
    assert bb["spec"].bcs[0].location == "LEFT"
    SpecAgent(prefer="llm").run(bb)
    assert bb["spec"].bcs[0].location == "BOTTOM"
