from cad2fem.agents.annotation import parse_bc, parse_load, parse_location, parse_spec


def test_title_block_and_notes():
    spec = parse_spec([
        "TITLE L_BRACKET", "MATERIAL ALUMINIUM 6061-T6", "THICKNESS", "8 mm", "SCALE 1:2",
        "1. FIXED: HOLE 1", "2. FORCE 2 kN -Y ON HOLE 2", "3. ELEMENT: TRI6, MESH SIZE 5",
    ])
    assert spec.title == "L_BRACKET"
    assert spec.material.E == 70000.0
    assert spec.thickness == 8.0          # "T6" of the alloy must not be read as thickness
    assert spec.scale == 2.0
    assert spec.element_order == 2 and spec.mesh_size == 5.0
    assert [(b.location, b.dofs) for b in spec.bcs] == [("HOLE1", (1, 2))]
    (ld,) = spec.loads
    assert (ld.location, ld.kind, ld.magnitude, ld.direction) == ("HOLE2", "FORCE", 2000.0, (0.0, -1.0))
    assert {"title", "material", "thickness", "scale", "element_order", "mesh_size"} <= spec.explicit


def test_explicit_material_constants_and_plane_strain():
    spec = parse_spec(["MATERIAL: CUSTOM", "E = 200 GPa, NU = 0.28", "PLANE STRAIN", "T = 3"])
    assert spec.material.E == 200000.0 and spec.material.nu == 0.28
    assert spec.analysis == "PLANE_STRAIN" and spec.thickness == 3.0


def test_loads_signs_and_units():
    assert parse_load("LOAD: TENSION 50 MPa ON RIGHT EDGE").magnitude == -50.0
    assert parse_load("PRESSURE 2 bar on top edge").magnitude == 0.2
    ld = parse_load("FORCE 500 N +X AT X=120")
    assert (ld.location, ld.magnitude, ld.direction) == ("X=120", 500.0, (1.0, 0.0))
    assert parse_load("SEE DETAIL A") is None


def test_bc_kinds():
    assert parse_bc("SYMMETRY: BOTTOM EDGE").dofs == (2,)
    assert parse_bc("ROLLER ON LEFT SIDE").dofs == (1,)
    assert parse_bc("SUPPORT AT Y=0, UY=0").dofs == (2,)
    assert parse_bc("FIXED: ALL HOLES").location == "HOLES"


def test_chinese_notes():
    spec = parse_spec(["材料: Q355", "厚度: 12", "固定: 左边", "拉力 30 MPa 作用于 右边"])
    assert spec.material.E == 210000.0 and spec.thickness == 12.0
    assert spec.bcs[0].location == "LEFT"
    assert spec.loads[0].location == "RIGHT" and spec.loads[0].magnitude == -30.0


def test_location_parser():
    assert parse_location("on hole #3") == "HOLE3"
    assert parse_location("UPPER EDGE") == "TOP"
    assert parse_location("nothing here") is None
