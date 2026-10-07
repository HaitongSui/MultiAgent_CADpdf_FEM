# MultiAgent_CADpdf_FEM

**cad2fem**：一个用 Python 实现的多 Agent 框架。它读取 **矢量 CAD 图纸 PDF**，识别零件轮廓、孔、比例尺、材料、厚度、边界条件和载荷，然后生成 **有限元模型输入文件**：

| 格式 | 文件 | 说明 |
|---|---|---|
| Abaqus | `*_abaqus.inp` | CPS3/CPS6（平面应力）或 CPE3/CPE6（平面应变），压力用 `*DSLOAD` |
| CalculiX | `*_ccx.inp` | 同一套关键字格式，压力换算为一致节点力 `*CLOAD`，已用 `ccx 2.21` 实际求解验证 |
| Nastran | `*.bdf` | SOL 101，CTRIA3/CTRIA6 + PSHELL 膜单元，`SPC1`/`FORCE` |
| Gmsh | `*.msh` | MSH 2.2，带边界物理组，可在 Gmsh / ParaView 中查看 |
| JSON | `*_model.json` | 与求解器无关的完整模型（节点、单元、集合、约束、载荷） |

另外输出 `report.md`（含 Agent 日志）和 `mesh.png`（网格 + 校核解的 von Mises 应力云图）。

## 快速开始

```bash
pip install -e ".[dev]"                                   # 或 pip install -r requirements.txt
python examples/make_sample_drawings.py examples/drawings # 生成示例图纸（已附带在仓库中）
python -m cad2fem examples/drawings/plate_with_hole.pdf -o out/plate
python -m cad2fem examples/drawings/l_bracket.pdf -o out/bracket --formats abaqus,nastran
python -m pytest -q
```

常用参数：

```
--mesh-size 4        目标单元尺寸 (mm)，覆盖图纸中的 MESH SIZE
--order 2            二次单元 (Tri6)
--thickness 8        覆盖厚度 (mm)
--plane-strain       平面应变
--outline-width 0.5  轮廓线最小线宽 (pt)，默认自动按线宽聚类
--llm                额外启用 Claude 解读图纸注释（需要 ANTHROPIC_API_KEY）
--prefer llm         规则解析与 LLM 结果冲突时采用 LLM
```

Python API：

```python
from cad2fem import PipelineConfig, run

bb = run("drawing.pdf", "out/", PipelineConfig(element_order=2, use_llm=True))
model = bb["fe_model"]            # 网格 + 集合 + 约束 + 载荷
print(bb["verification"]["max_von_mises_mpa"])
for msg in bb.messages:           # 每个 Agent 的发现、警告和返工请求
    print(msg)
```

## 架构：黑板 + 数据驱动调度

```
                         ┌──────────────── Blackboard（共享产物 + 消息）────────────────┐
 pdf_path ──► PDFParser ──► drawing ──┬─► Annotation (规则) ──► spec_rules ─┐
                                      ├─► LLMAnnotation (Claude, 可选) ─► spec_llm ─┼─► Spec ──► spec
                                      └─► Geometry ──► part_pt ──────────────────────┘     │
                                                         └──► Dimension (比例尺校准) ◄─────┘
                                                                   │ geometry (mm)
                                                                   ▼
                                    ┌──────────── REQUEST: refine ────────────┐
                                    ▼                                         │
                                  Mesh ──► mesh ──► Boundary ──► fe_model ──► Validator ──► validation
                                                                                   │
                                                  Verifier (内置 CST 求解) ◄────────┤
                                                  Writer ──► outputs ──► Report ◄──┘
```

* **Agent 之间不直接调用**：每个 Agent 声明 `requires` / `optional` / `provides`，只通过黑板读写产物、发送消息。
* **Orchestrator** 按数据可用性调度：某 Agent 所需的键都在黑板上、且其输入版本发生变化（或收到改变其参数的 REQUEST）时才运行，下游自动重算。
* **反馈回路**：`ValidatorAgent` 发现网格问题（小角度单元、边界不贴合、孔周分辨率不足、面积误差）时向 `MeshAgent` 发 REQUEST 加密网格（默认最多 2 次），Boundary/Validator 自动重跑，通过后才放行 Writer。

| Agent | 职责 |
|---|---|
| `PDFParserAgent` | pdfplumber 读取线段/矩形/曲线路径（Bézier 离散化），识别整圆（用锚点精确拟合圆心、半径），提取文字并按行合并；旋转 90° 的尺寸文字按 CAD 习惯从下往上读 |
| `AnnotationAgent` | 规则解析标题栏与技术要求：比例、单位、材料（内置材料库）、E/ν、厚度、平面应力/应变、网格尺寸、单元阶次、边界条件、载荷（支持中英文） |
| `LLMAnnotationAgent` | 可选。调用 Claude（`claude-opus-5-5`，结构化输出 + Pydantic schema）理解自由文本注释；无凭据/网络错误/拒答时自动降级为规则结果 |
| `SpecAgent` | 合并规则、LLM、命令行覆盖值；冲突时给出警告 |
| `GeometryAgent` | 按线宽区分轮廓线与尺寸/中心线，线段求交打断、构建平面图，取各连通分量外边界；剔除图框和标题栏；最大闭合环为零件，内部环为孔 |
| `DimensionAgent` | 用尺寸数字（`200`、`Ø40`、`R10`、`2x Ø12`）与实测长度匹配做比例尺校准，并与标题栏比例互相印证；不一致时以多数尺寸为准并告警 |
| `MeshAgent` | 边界按尺寸重采样（保留角点）+ 正三角形点阵 + Delaunay + Laplace 光顺；小孔周围生成渐变同心环加密；可升阶为 Tri6（孔边中节点投影到圆上） |
| `BoundaryAgent` | 把 `LEFT/RIGHT/TOP/BOTTOM/HOLE<n>/HOLES/X=<v>/Y=<v>` 映射为节点集与单元面；压力和集中力换算为一致节点力 |
| `ValidatorAgent` | 单元方向、最小角、边界贴合、孔周分辨率、面积误差、刚体位移约束秩检验、载荷检查；必要时请求返工 |
| `VerifierAgent` | 内置 CST 线弹性求解器做冒烟测试：平衡误差、最大位移、von Mises |
| `WriterAgent` / `ReportAgent` | 写出各格式文件与报告 |

## 图纸约定（规则解析器能识别的写法）

* 必须是 **矢量 PDF**（CAD 直接导出）。扫描件需先矢量化。
* 可见轮廓线用粗线，尺寸线/中心线用细线或虚线（标准制图习惯，自动识别线宽分界）。
* 坐标原点为零件包围盒左下角，单位 mm / N / MPa。
* 孔编号 `HOLE1, HOLE2, …`：按圆心 x 从小到大，再按 y。
* 注释示例：

```
TITLE  L_BRACKET            MATERIAL  ALUMINIUM 6061-T6      THICKNESS  8 mm      SCALE 1:1
1. FIXED: LEFT EDGE                     (也支持 PINNED / ROLLER / SYMMETRY / UX=0)
2. LOAD: TENSION 50 MPa ON RIGHT EDGE   (TENSION/TRACTION 为拉，PRESSURE 为压入表面)
3. FORCE 2 kN -Y ON HOLE 2              (合力，均布到所选边界)
4. MESH SIZE 5; ELEMENT: TRI6; PLANE STRAIN; E = 200 GPa, NU = 0.28
固定: 左边    拉力 30 MPa 作用于 右边    材料: Q355    厚度: 12
```

## 扩展

新增 Agent 只需继承 `Agent` 并声明输入输出，调度器会自动把它接入流程：

```python
from cad2fem.core import Agent
from cad2fem.pipeline import PipelineConfig, build_agents, run

class HoleCountCheck(Agent):
    name = "hole_check"
    requires = ("geometry",)
    provides = ("hole_check",)

    def run(self, bb):
        if len(bb["geometry"].holes) != 2:
            self.warn(bb, "expected 2 holes")
        bb.post("hole_check", True, self.name)

agents = build_agents(PipelineConfig()) + [HoleCountCheck()]
run("drawing.pdf", "out/", agents=agents)
```

适合继续扩展的方向：多视图/多零件、基于 gmsh 的网格 Agent、三维拉伸（C3D 单元）、扫描件的视觉 Agent（把页面图像交给 Claude 识别）、接触/多载荷步。

## 验证

`tests/` 共 22 个测试，覆盖：注释解析（中英文）、几何识别（圆孔半径、圆角面积）、比例尺冲突处理、反馈回路、LLM Agent（mock 客户端）、各格式写出，以及：

* 无孔板受拉：端部位移与解析解 σL/E 误差 < 2%，远场应力 50 MPa 误差 < 1%；
* 生成的 CalculiX 文件经 `ccx` 实际求解，支反力 −50 kN 与施加载荷平衡；三个示例中内置 CST 校核解与 CalculiX 的最大位移一致（0.0474 / 0.0581 mm，支架 Tri6 3.13 mm vs CST 2.99 mm）。

## 局限

* 二维平面问题（平面应力 / 平面应变），单零件、单视图（多视图时取最大的闭合轮廓并告警）。
* 规则解析依赖常见的注释写法；措辞自由的图纸建议加 `--llm`。
* 内置网格器面向板件/支架类零件；复杂几何建议接入 gmsh。
