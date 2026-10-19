# MultiAgent_CADpdf_FEM

**cad2fem**：一个用 Python 实现的多 Agent 框架，读取 **矢量 CAD 图纸 PDF** 生成 **有限元模型输入文件**。有两种模式：

1. **整桥模式（`cad2fem bridge`）**：读取一套桥梁图纸（总体布置图 + 主梁横断面 + 桥墩断面），把主梁、桥墩、支座拼装成 **一个完整的三维整桥鱼骨梁（spine）模型**，见下文 [整桥模型](#整桥模型cad2fem-bridge)。
2. **构件模式（`cad2fem`）**：单张零件/构件图 → 二维平面应力/应变网格模型。
3. **扫描图纸拱桥模式（`cad2fem arch`）**：扫描（光栅）PDF 没有矢量和文字层，由 Claude 看图读出尺寸，组装钢拱复合桥三维模型，见 [扫描图纸与拱桥](#扫描图纸与拱桥cad2fem-arch)。

构件模式输出：

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

## 整桥模型（cad2fem bridge）

```bash
python examples/make_bridge_drawings.py examples/drawings      # 示例图纸集（3 页，已附带）
python -m cad2fem bridge examples/drawings/bridge_3span.pdf -o out/bridge
python -m cad2fem bridge 总体布置图.pdf 主梁断面.pdf 桥墩.pdf -o out/bridge   # 多个文件也可以
python out/bridge/XX_RIVER_BRIDGE_opensees.py                    # 直接用 OpenSees 求解
```

![bridge](docs/bridge_example.png)

**输入**：每个 PDF 的每一页都是一张"图"，由 Agent 按标题栏自动分类（也可用 `--sheets GENERAL,DECK,PIER` 指定）：

| 图纸 | 读取内容 |
|---|---|
| 总体布置图 GENERAL | 立面图（实测跨径、墩位、墩高、梁高）+ 说明：跨径布置、墩高、支座类型、材料、二期恒载、车道荷载、单元长度 |
| 主梁横断面 DECK | 截面轮廓（含箱室），按比例尺和尺寸标注换算 → A、Iyy、Izz、J、形心高度 |
| 桥墩断面 PIER | 墩身截面 → 截面特性（缺省时假定矩形并告警） |

**整桥模型的组成**（单位 m、kN、kPa；X 顺桥向，Y 横桥向，Z 竖向）：

* 主梁：沿桥轴线的梁单元，位于截面形心高度，每跨偶数个单元（跨中有节点）；
* 桥墩：竖向梁单元，墩底固结；
* 支座：主梁节点 →（刚臂，长度 = 形心到梁底距离）→ 支座顶节点 →（弹簧）→ 墩顶 / 桥台地面节点。
  `FIXED` 固定（约束 X、Y、Z），`SLIDING_X` 纵向活动（约束 Y、Z），`SLIDING_XY` 双向活动（约束 Z）；
  另外都约束绕桥轴的转动，代替实际的一对支座提供抗扭；`MONOLITHIC` 墩梁固结（刚构桥）用刚性连接；
* 荷载工况：`DEAD`（梁、墩自重 + 二期恒载）、`LIVE`（车道均布荷载 qk 满布 + 集中力 Pk 作用在最大跨跨中，乘车道数），
  均换算为一致节点力（含端弯矩），各求解器结果一致。

**输出**：

| 格式 | 文件 | 说明 |
|---|---|---|
| Abaqus | `*_abaqus.inp` | B33 梁 + `*BEAM GENERAL SECTION`、SPRING2 支座、`*MPC BEAM` 刚臂，每个工况一个 perturbation step |
| Nastran | `*.bdf` | CBAR/PBAR、CELAS2、RBE2，每个工况一个 SUBCASE |
| OpenSees | `*_opensees.py` | 可直接运行的 OpenSeesPy 脚本，求解所有工况，结果写入 JSON |
| JSON | `*_model.json` | 求解器无关的完整模型 |
| 报告 | `report.md`、`bridge.png` | 跨径/截面/支座汇总、各跨挠度、支反力、主梁弯矩图 |

**Agent 团队**：`SheetReader` → `SheetClassifier` → `Elevation`（立面实测）/ `Section`（截面特性）/ `LLMLayout`（可选）→ `Layout`（说明与立面实测互相校核）→ `Assembly` → `BridgeValidator`（单元密度不足时请求 `Assembly` 加密）→ `BridgeVerifier`（内置三维框架求解器）→ `Writer` → `Report`。

**交叉校核**（体现多 Agent 的意义）：
* 说明中的跨径、墩高 vs 立面图实测值（不一致告警；说明缺失时直接用实测值）；
* 立面图比例 vs 尺寸标注（如 `3000`、`4000`、`10000` cm）；
* 横断面图梁高 vs 立面图梁高；
* 支座布置是否构成机构（无纵向固定支座、横向约束不足）；求解时奇异刚度会被识别并报错。

**验证**（`tests/test_bridge.py`，共 13 项）：
* 示例三跨 30+40+30 m 连续箱梁：从图纸识别出的跨径、墩高、截面尺寸与绘制值一致；箱梁面积与手算 6.582 m² 一致；
* 恒载中支点负弯矩与三弯矩方程（Clapeyron）解相差 < 1%；两跨连续梁的支反力、跨中/支点弯矩与理论解相差 < 0.5%；
* 墩顶受水平力的悬臂位移与 PH³/3EI 一致（两个方向分别校核截面主轴方向）；
* 生成的 OpenSees 脚本实际运行，各工况位移与内置求解器一致（差值 < 1e-10 m）；
* 扭转常数：圆截面误差 < 0.5%，矩形与级数公式相差 < 1%。

## 扫描图纸与拱桥（cad2fem arch）

`cad_pdf/` 中的 47 张图纸是扫描件（每页一张约 3310×2340 像素的图片，没有矢量和文字层），内容是日本的
**新田豊島橋（仮称）**：钢单纯箱梁 + 拱复合桥，支间长 102.7 m，拱支间长 74.4 m，两片拱面各有 18 根 V 形吊索。
矢量流程读不了这类图纸，因此增加了看图 Agent：

```bash
# 有 ANTHROPIC_API_KEY 时：Claude 直接看图（每页一张概览 + 4 张原分辨率分块），并保存读图结果
python -m cad2fem arch "cad_pdf/PDF（241101）-14.pdf" "cad_pdf/PDF（241101）-17.pdf" \
       --save-reading my_reading.json -o out/arch
# 复用已审核的读图结果（不调用模型）
python -m cad2fem arch --reading cad_pdf/readings/J-5_J-8_arch_reading.json -o out/arch
```

![arch](docs/arch_example.png)

* `cad_pdf/readings/J-5_J-8_arch_reading.json`：J-5 構造一般図（PDF-14）与 J-8 基本寸法図（PDF-17）的读图结果，
  由 Claude 在会话中看图转录，与 `VisionSheetAgent` 的结构化输出同一格式（`ArchReading`），每组数值注明出自哪张图、哪个视图，读不准的地方单独列出。
* **几何 Agent 先校核读数，再拟合拱轴线**：
  * 横梁间距之和 = 支间长 102,700；吊索间距之和 = 拱肋轴线长 76,853.6（两项都精确闭合）；
  * 左右对称性；
  * 用 18 根吊索的标注长度拟合拱轴线（RMS 26 mm），拟合出的拱顶吊索倾角 76.05°，图上标注为 76°。
  * 读错一个数字会被这些校核发现（见测试 `test_misread_values_are_caught`）。
* **模型**：两片拱面（主梁 = 系杆、拱肋、V 形吊索）、每个测站一道横梁、起拱点刚臂、S1 铰支 / S2 纵向活动；
  恒载 + 道路桥示方书 B 活荷载（简化：满宽均布 p2 + 跨中 10 m 的 p1，不计冲击）。
* **验证**：OpenSees 结果与内置求解器一致（位移差 < 1e-9 m）；拱顶拱肋压力与跨中主梁拉力平衡（系杆拱特征）；吊索全部受拉、左右对称。
* **重要**：板厚、吊索截面、横梁截面**不在 J-5/J-8 上**。单独运行 `cad2fem arch` 时这些用假定值，报告中单列；
  由 `cad2fem rsi` 从细节图读出并验证后再代入，见下一节。桥面系恒载仍为假定值。

## RSI 多 Agent：自我改进地读细节图、补齐模型资料（cad2fem rsi）

J-5/J-8 只给出几何，板厚、吊索截面、横梁截面都要从细节图读取。`cad2fem rsi` 把这件事做成
RSI（recursive self-improvement）循环。模型的权重不变，变的是经过验证的记忆：

```
 Curriculum ──task──► Actor ──claims──► Verifier ──verdict──► Memory ──memory_version──┐
     ▲   (挑缺口/页码)    (看图转录)      (确定性校核 + 独立复读)   (只存通过的)            │
     └─────────────────────────────────────────────────────────────────────────────┘
阶段 1 广度：图纸目录（47 页标题栏）→ 阶段 2 深度：吊索、主梁端部、拱肋第 2–5 段、中间横梁
阶段 3 复用：冻结记忆 → ArchKnowledge → 重建拱桥模型，与“全部假定值”的基线对比
```

* **Curriculum** 从记忆中找缺口（依赖满足、尝试次数未用完、预算未花完），用 Actor 自己读出的图纸目录定位页码；
  上次失败的校核项作为提示带入下一次尝试。
* **Actor** 每次在全新、独立的上下文中看图（概览 + 4 张原分辨率分块）。提示中带上记忆里已验证的读图经验（lessons）。
* **Verifier** 不看 Actor 的推理，只用独立证据判断：
  * 其他图纸：腹板间距 = J-5 主梁宽，拱肋段长 = J-8 部材长链；
  * 算术：吊索面积 = 丝数 × 38.48 mm²，Ls = J-8 吊索长 − 常数，梁高 = 翼缘 + 腹板 + 翼缘，腹板肋数 = 腹板高 / 肋间距 − 1；
  * 对称性、同高度拱肋段的加劲肋布置一致（与记忆中其他已验证段比较）；
  * 另开一个独立上下文按条复读数值。
* **Memory**（`knowledge.json` + `lessons.md` + `episodes/`，纯文本，可在 git 中审阅）：
  * 只提交通过的结论与经验；
  * 每次启动先用当前的校核**审计**已有记忆。校核加强后，旧结论不再合格就被逐出并重新练习；两条已验证结论互相矛盾时，两条都逐出重读；
  * 审核人可用 `--relearn 任务=原因` 把某项打回重练。

**为什么不需要 API 密钥**：`cad2fem` 是独立的 Python 进程，必须自己调用模型。本环境里的 Claude Code 命令行
`claude -p` 可以无界面运行，使用当前登录，每次调用都是一个新的独立上下文，用 Read 工具看图并返回 JSON。
`--backend auto` 的选择顺序：有 `ANTHROPIC_API_KEY` 时用 Messages API，否则用 `claude` 命令行，两者都没有时回放已保存的读图记录。
每个后端都记录花费，超过 `--budget`（默认 $5）就停止练习。

```bash
# 实际看图练习（claude 命令行或 API），记忆写入 memory/
python -m cad2fem rsi --backend auto --budget 5 --memory memory -o out_rsi
# 不调用模型：复用仓库里已经练好的记忆，直接重建模型
python -m cad2fem rsi --backend none --memory cad_pdf/readings/rsi_memory -o out_rsi
# 审核人打回某项
python -m cad2fem rsi --memory memory --relearn "rib_seg4=web rib count not verified"
```

**实测（`cad_pdf/readings/rsi_memory/`，报告 `docs/rsi_report.md`）**：用 `claude -p` 读了 8 项资料，共 22 个 episode，
记忆花费 $1.89。另有一次因复读提示过严而中止的运行，约 $0.75。过程中暴露并修正的问题：

1. **独立复读太死板**：图上写 `1-FLG PL 1700x30x4226`，复读者因为写法与“PL 1700x30”不同而判为不一致。
   改为只比较尺寸和钢种。
2. **同一构件按每块板列出**：拱肋第 5 段的肋板被重复列出，面积多了 11%。新增“每种板件只报一条”的校核，
   审计时逐出主梁端部、拱肋第 5 段、横梁三项，重读后通过。
3. **腹板肋数量的口径**：读者把 `4-RIB`（两块腹板合计的件数）当作每块腹板 4 根，第 4、5 段两次都这样读，
   互相一致，所以一致性校核拦不住。新增 `web_rib_pitch_mm` 字段：断面图上 3@400=1200，所以每块腹板 2 根。
   改为算术校核，复读也覆盖肋板。审核人打回后重读，结果是每块腹板 2 根。

最终记忆中的全部截面（3 种吊索、主梁、拱肋 4 段、横梁）与会话中人工核对的读图逐项相同（A、I、J 差 0.0%）。
模型对比（每片拱面，恒载）：

| 量 | 全部假定值 | 用验证过的截面 |
|---|---|---|
| 跨中挠度 DEAD / LIVE_B | 328.5 / 139.7 mm | 182.8 / 68.4 mm |
| 拱顶拱肋轴力 DEAD | −9,908 kN | −11,494 kN |
| 吊索恒载拉力 | 25–843 kN | 61–696 kN |

J-81 上的设计恒载张力含预张和长度调整，只作参考对比，不作为通过/不通过的标准。
主梁中段（J-58–J-72 不在图纸集中）由 J-57 的板件按 J-5 梁高 1950 推定，拱肋第 1/9 段由第 2 段推定，报告中均单独标注。

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

全部 54 个测试。其中 `tests/test_rsi.py` 覆盖 RSI 循环：离线回放、Verifier 拦截读错的数值、记忆冻结、失败后带提示重试、审计逐出并重新学习、审核人打回、用模拟后端测试 Actor 和 Verifier。构件模式的测试覆盖：注释解析（中英文）、几何识别（圆孔半径、圆角面积）、比例尺冲突处理、反馈回路、LLM Agent（mock 客户端）、各格式写出，以及：

* 无孔板受拉：端部位移与解析解 σL/E 误差 < 2%，远场应力 50 MPa 误差 < 1%；
* 生成的 CalculiX 文件经 `ccx` 实际求解，支反力 −50 kN 与施加载荷平衡；三个示例中内置 CST 校核解与 CalculiX 的最大位移一致（0.0474 / 0.0581 mm，支架 Tri6 3.13 mm vs CST 2.99 mm）。

## 局限

拱桥模式：
* 细节图只读了吊索、主梁端部、拱肋第 2–5 段、中间横梁；桥面恒载、主梁中段（图纸集中缺少）为假定/推定值；道路纵断面竖曲线、拱肋横向联结（JU1–JU8）、拱肋与主梁在起拱区的渐变连接都做了简化；
* RSI 循环已用 `claude -p` 实测。Messages API 后端（`AnthropicBackend`、`VisionSheetAgent`）在本环境没有密钥，只用模拟客户端测试过。

整桥模式：
* 单根主梁鱼骨模型（直线桥、无斜交/曲线），全桥等截面（变高度梁需扩展为分段截面）；
* 墩底固结，未考虑桩土相互作用；支座用刚度弹簧模拟；
* 活载为满布车道荷载 + 一个集中力，不是影响线最不利加载包络；未考虑预应力、施工阶段、收缩徐变；
* Abaqus / Nastran 文件按格式规范生成，但本环境中没有这两个求解器，未实际运行验证（OpenSees 已验证）。

构件模式：

* 二维平面问题（平面应力 / 平面应变），单零件、单视图（多视图时取最大的闭合轮廓并告警）。
* 规则解析依赖常见的注释写法；措辞自由的图纸建议加 `--llm`。
* 内置网格器面向板件/支架类零件；复杂几何建议接入 gmsh。
