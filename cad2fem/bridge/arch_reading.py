"""What the vision agent reads from arch-bridge general drawings.

The same schema is used for (a) Claude's structured output when it looks at
the sheet images and (b) a saved transcription JSON, so a reading can be
reviewed, versioned and re-used without calling the model again.

All lengths are millimetres, exactly as dimensioned on the drawings.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Source(BaseModel):
    field: str = Field(description="Name of the reading field this source covers")
    sheet: str = Field(description="Drawing number / title, e.g. 'J-8 基本寸法図'")
    view: str = Field(description="View or table on the sheet, e.g. '側面図', '設計条件'")
    note: Optional[str] = Field(default=None, description="How the value was read; doubts")


class CableReading(BaseModel):
    girder_station: str = Field(description="Girder node the hanger starts from, e.g. 'C6'")
    side: Literal["L", "R"] = Field(description="Anchor on the arch: L = towards A1, R = towards A2")
    length_mm: float = Field(description="Hanger length as dimensioned along the hanger")


class ArchReading(BaseModel):
    bridge_name: str
    structure_type: str = Field(description="形式 from the design-condition table")
    sheets: list[str] = Field(description="Sheets that were read")

    # --- overall lengths (side view dimension chain) -----------------------
    bridge_length_mm: float = Field(description="橋長")
    girder_length_mm: float = Field(description="桁長")
    span_mm: float = Field(description="支間長 (bearing to bearing)")
    arch_span_mm: float = Field(description="アーチ支間長 (springing to springing)")
    arch_axis_length_mm: float = Field(description="Arch-rib length along its axis (アーチリブ軸線上)")

    # --- girder stations ----------------------------------------------------
    station_labels: list[str] = Field(description="Bearing/cross-beam stations from A1 to A2, e.g. S1, C1..C19, S2")
    cross_beam_spacing_mm: list[float] = Field(description="横桁間隔 chain between consecutive stations")

    # --- hangers ---------------------------------------------------------------
    cable_anchor_chain_mm: list[float] = Field(
        description="ケーブル間隔 along the arch axis: springing -> first anchor -> ... -> last anchor -> springing")
    cable_anchor_order: list[str] = Field(description="Anchor names in chain order, e.g. C6-L, C6-R, C7-L ...")
    cables: list[CableReading]
    cable_angle_at_crown_deg: Optional[float] = Field(default=None, description="Hanger inclination shown at the crown")
    rib_member_lengths_mm: list[float] = Field(
        default_factory=list, description="部材長 of the arch-rib segments along the axis, springing to springing")

    # --- cross-section -------------------------------------------------------
    deck_width_mm: float = Field(description="総幅員")
    girder_spacing_mm: float = Field(description="Main girder (arch plane) spacing")
    girder_width_mm: float
    girder_depth_mm: float
    rib_depth_mm: float
    rib_bottom_above_girder_top_mm: float = Field(description="Vertical clearance rib to girder at the crown (8000 on the drawings)")

    # --- design conditions table ----------------------------------------------
    live_load: str
    bearings: str
    deck: str
    steel_grades: list[str]
    pavement_thickness_mm: float
    carriageway_width_mm: float
    sidewalk_width_mm: float = Field(description="Each side")
    longitudinal_grades: str

    sources: list[Source] = Field(default_factory=list, description="Where each value was read")
    uncertain: list[str] = Field(default_factory=list,
                                 description="Values the reader is unsure about and why")
    reader: str = Field(default="", description="Who/what produced this reading")
