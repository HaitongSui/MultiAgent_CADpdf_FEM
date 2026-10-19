"""What an Actor reports after reading a sheet (validated before it is checked)."""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field


class Fact(BaseModel):
    key: str
    value: Union[float, str]
    unit: str = ""
    sheet: str = ""
    view: str = ""
    note: str = ""


class PlateClaim(BaseModel):
    role: Literal["top", "bottom", "web", "flange_rib", "web_rib"]
    width_mm: float = Field(description="Plate width (for webs: the PL width as written)")
    thickness_mm: float
    count: int = 1
    material: str = ""


class SectionClaim(BaseModel):
    name: str = Field(description="e.g. GIRDER_END, RIB_SEG2 ... RIB_SEG5, XBEAM")
    sheet: str
    view: str = ""
    depth_mm: float = Field(description="Section depth used for analysis (outer, flange to flange)")
    depth_range_mm: list[float] = Field(default_factory=list, description="If depth varies along the member")
    web_spacing_mm: Optional[float] = None
    axis_length_mm: Optional[float] = Field(default=None, description="Arch-rib segment length on the axis")
    web_rib_pitch_mm: Optional[float] = Field(default=None, description="Spacing of the web ribs over the web "
                                              "depth in the section view, e.g. 400 from 3@400=1200")
    plates: list[PlateClaim]


class CableClaim(BaseModel):
    number: int
    strands: int
    dead_tension_kN: Optional[float] = None
    node_length_m: Optional[float] = None


class CableSpec(BaseModel):
    name: str
    strands: int
    area_mm2: float
    E_kN_mm2: float
    breaking_kN: Optional[float] = None
    weight_kg_m: Optional[float] = None


class SheetEntry(BaseModel):
    page: int
    drawing_no: str
    title: str


class Claims(BaseModel):
    task: str
    sheets: list[str] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
    sections: list[SectionClaim] = Field(default_factory=list)
    cables: list[CableClaim] = Field(default_factory=list)
    cable_specs: list[CableSpec] = Field(default_factory=list)
    sheet_index: list[SheetEntry] = Field(default_factory=list)
    lessons: list[str] = Field(default_factory=list,
                               description="Reusable reading lessons learned on this task")
    reader: str = ""

    def fact(self, key: str):
        return next((f.value for f in self.facts if f.key == key), None)
