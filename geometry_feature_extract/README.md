# 门静脉血管分析完整流程 (PPG Prediction)

## 📋 项目概述

本项目是一个门静脉（Portal Vein）血管分析系统，基于STL格式的3D血管模型进行自动化处理。系统包含两大流水线：
- **单患者处理流水线**（Step 1-6）：从STL模型到可视化输出
- **跨患者分析流水线**（Step A-B）：特征与临床指标的相关性分析

---

## 🏗️ 整体架构

```
PPG_Prediction/
├── 单患者处理流程
│   ├── main.py                      [主入口]
│   ├── extract_centerline.py        [Step 1: 中心线提取]
│   ├── smooth_centerline.py         [Step 2: 中心线平滑]
│   ├── segment_vessels.py           [Step 3: 解剖分段]
│   ├── extract_features.py          [Step 4: 统计特征 + 调用 system_features 写 unified_features.json]
│   ├── system_features.py           [Step 4 内部: 系统/联合特征 (Murray, Poiseuille 阻力, 侧支负担)]
│   ├── extract_profiles.py          [Step 5: 剖面特征 (内切半径过滤 + 端点掩码)]
│   ├── export_visualization.py      [Step 5.5: 可视化导出]
│   └── visualize_segments.py        [Step 6: VTK交互可视化]
│
├── 跨患者分析流程
│   ├── correlation_analysis.py      [Step A: 统计 + 系统特征 vs PPG 相关性, 自动解读 + 推荐特征]
│   └── profile_correlation.py       [Step B: 剖面特征相关性]
│
├── 辅助工具
│   ├── utils.py                     [公共工具库]
│   ├── compute angle.py             [SV-SMV夹角计算]
│   └── readme.md                    [本文件]
```

---

## 📊 流水线详解

### 【第一阶段】单患者处理（按顺序执行）

#### **Step 1: 中心线提取** (`extract_centerline.py`)
**输入**：STL 文件
**输出**：`CenterlinePoints.txt`

**核心流程**：
1. STL → 体素化（pitch=0.5mm）
2. 距离变换（Distance Transform）
3. 3D 骨架化（Lee94算法）
4. 图构建 → 增强剪枝
5. BFS建树

**剪枝策略**（自适应去除噪声）：
- 物理长度阈值：末端分支 < 8mm 被剪除
- 相对长度阈值：末端分支 < 总长×5% 被剪除
- 半径判据：末端分支最大半径 < 父主干半径×40% 被剪除
- 分支点合并：距离 < 5mm 的相邻分支点合并

**关键参数**：
```python
pitch=0.5                      # 体素化分辨率(mm)
min_branch_length_mm=8.0       # 最小分支物理长度
min_relative_length=0.05       # 最小相对长度比例
min_radius_ratio=0.4           # 最小半径比例
merge_bp_distance_mm=5.0       # 分支点合并距离
```

---

#### **Step 2: 中心线平滑** (`smooth_centerline.py`)
**输入**：`CenterlinePoints.txt`
**输出**：`newCenterlist.txt`

**核心流程**：
1. 读取原始中心线树 → 构建邻接表
2. 分类节点：端点 vs 分支点
3. 提取所有"段"（关键点之间的路径）
4. 逐段样条平滑（UnivariateSpline）
5. 邻接表重组 → BFS建树

**平滑参数**：
- `smooth_factor=500`：越大越平滑
- `n_mult=3`：采样密度倍数
- `w_key=1e3`：关键点（端点/分支点）权重
- `w_mid=10`：普通点权重

---

#### **Step 3: 解剖分段** (`segment_vessels.py`)
**输入**：`newCenterlist.txt`（平滑后的中心线）
**输出**：`centerline_profiles.json`

**支持的解剖结构**：
| 缩写 | 全名 | 说明 |
|------|------|------|
| MPV | 门静脉主干 | SV/SMV汇合点至肝内左右支分叉点的唯一拓扑路径 |
| SV | 脾静脉 | 从汇合点向患者左侧延伸的入流支 |
| SMV | 肠系膜上静脉 | 从汇合点向患者脚侧延伸的入流支 |
| LPV | 肝门左静脉 | 肝侧左支 |
| RPV | 肝门右静脉 | 肝侧右支 |
| TIPS | 支架（术后） | 仅术后患者 |
| LGV | 胃左静脉 | 术前代偿（可选） |
| PGV | 胃后静脉 | 术前代偿（可选） |

**分段判别准则**：
1. 根据样本文件夹名称是否包含 `#` 确定是否为 TIPS 术后样本。
2. 在中心线树中全局枚举 SV/SMV 汇合点与 LPV/RPV 肝内分叉点组合。
3. SV、SMV 和 MPV 必须从汇合点经三个不同拓扑端口连接；LPV 和 RPV 必须从肝内分叉点分离。
4. 在统一患者坐标系中，以左向、脚向、头向及左右支空间分离度计算解剖一致性。
5. MPV 定义为两个解剖锚点之间的唯一图路径，不通过最长或最直血管判断。
6. LGV/PGV 根据额外分支附着于 MPV/汇合点或 SV 中远段的位置识别。
7. TIPS 仅在术后样本中启用，依据肝内附着位置、走向、直线性及可用时的沿程管径均匀性辅助识别。
8. 输出最优与次优候选差值；候选接近或违反方向先验时设置 `needs_manual_review=true`。

默认坐标系为 DICOM-LPS：`+X` 指向患者左侧，`+Z` 指向头侧。若 STL 使用 RAS，调用
`segment_vessels(..., coordinate_system="RAS")`。血管长度不参与解剖命名，曲折度仅用于
TIPS/侧支的辅助形态评价和下游几何特征计算。

---

#### **Step 4: 统计特征 + 系统特征提取** (`extract_features.py` + `system_features.py`)
**输入**：`centerline_profiles.json`（分段）+ 平滑后中心线 + STL + `pointwise_profiles.json`
**输出**：
- `unified_features.json`（**新, 推荐**, 单文件统一所有特征 — 训练用）
- `portal_vein_features.json`（兼容旧版扁平 schema, 让旧的 correlation 脚本继续可用）
- `sv_smv_angle.json`（夹角详情, 兼容旧脚本）

**每段计算的 9 个标量特征 (`statistical` 块)**：
| 特征 | 计算方式 | 物理意义 |
|------|---------|---------|
| `length` | 路径弧长 | 段长度(mm) |
| `tortuosity` | 弧长/弦长 | 弯曲程度 |
| `mean_curvature` | 平均离散曲率 | 平均弯曲强度(1/mm) |
| `max_curvature` | 最大离散曲率 | 最大弯曲强度(1/mm) |
| `mean_diameter` | 平均截面直径 | 平均通径(mm) |
| `max_diameter` | 最大截面直径 | 最大通径(mm) |
| `mean_area` | 平均截面积 | 平均通流面积(mm²) |
| `area_cv` | 面积变异系数 | 截面积变化剧烈度 |
| `mean_circularity` | 平均圆度 = 4π×A/P² | 截面形状规则度 |

**全局特征 (`global` 块)**：
- `total_centerline_length`：已分配解剖血管段总长
- `sv_smv_diameter_ratio`：SV/SMV 平均直径比
- `sv_smv_angle`：SV-SMV 夹角(度)
- `has_lgv` / `has_pgv` / `has_compensation_vessel` / `has_tips`：二值存在性

**无效截面处理**：
- 端点掩码和求交失败点保留原索引并置 0，统计时统一按 `area > 0` 排除

##### 系统/联合特征 (`system` 块, 由 `system_features.py` 计算)

> 单根血管 ↔ PPG 关系往往较弱; 系统特征捕捉**血管之间的几何关系**, 文献证据更强。

**(A) 角度特征** (基于段端单位方向向量, 默认 10mm 拟合)
| 字段 | 含义 |
|------|------|
| `angle_sv_smv` | SV-SMV 汇合角 |
| `angle_mpv_lpv` / `angle_mpv_rpv` | MPV 入射方向 vs LPV/RPV 出射方向 |
| `angle_lpv_rpv` | LPV 与 RPV 张开角 |
| `angle_mpv_bifurc_total` | LPV-RPV 子分支真实开角 |
| `mpv_bifurc_planarity_deg` | LPV-RPV 分叉平面相对 MPV 轴的非平面度 (理想 T 形≈0°) |
| `angle_mpv_tips` | TIPS 与局部门静脉母血管夹角 (支持内部交点) |

**(B) 直径 / 面积比 (Murray 定律 & 不对称)**

交点直径和面积分别取每根血管从共享交点向内的第一个有效截面
(`area > 0` 且 `eq_diameter > 0`)；不再设置固定毫米偏移、共同测量距离或局部窗口。

| 字段 | 公式 | 物理含义 |
|------|------|---------|
| `confluence_murray3_ratio` | D_MPV³ / (D_SV³ + D_SMV³) | 理想≈1, 偏离反映重塑 |
| `confluence_murray3_deviation` | \|ratio − 1\| | 单调使用 |
| `confluence_area_ratio` | A_MPV / (A_SV + A_SMV) | 汇合处面积守恒 |
| `mpv_bifurc_murray3_ratio/deviation` | 同上 (LPV/RPV) | 肝侧分叉 Murray 偏离 |
| `mpv_bifurc_area_ratio` | A_MPV / (A_LPV + A_RPV) | |
| `sv_smv_diameter_asymmetry` | (D_SV − D_SMV)/(D_SV + D_SMV) | 脾侧扩张 → 升高 |
| `sv_mpv_diameter_ratio` / `smv_mpv_diameter_ratio` | | 静脉曲张/EV 相关 (Mostafa 2015) |
| `lpv_rpv_diameter_asymmetry` | (D_LPV − D_RPV)/(D_LPV + D_RPV) | 肝内阻力不对称 |
| `lgv_mpv_diameter_ratio` / `pgv_mpv_diameter_ratio` | 侧支/MPV | 侧支扩张程度 |
| `splenic_dominance_index` | r_SV⁴ / (r_SV⁴ + r_SMV⁴) | 脾侧流量分配近似 |

**(C) 长度 / 弯曲度联合**
| 字段 | 公式 | 物理含义 |
|------|------|---------|
| `splenoportal_path_chord_ratio` | (L_SV + L_MPV) / chord | 脾门→门静脉分叉的总弯曲度 |
| `collateral_length_mpv_ratio` | Σ L_collat / L_MPV | 侧支总长度归一化 |
| `diameter_weighted_tortuosity` | Σ τᵢ Dᵢ⁴ / Σ Dᵢ⁴ | 大血管主导的整树弯曲 |

**(D) 1D Hagen-Poiseuille 阻力 (无 CFD)**
> R_seg = ∫ dl / r⁴ (省略 8μ/π 常数因子, 单位 mm⁻³)。使用最终 Voronoi 截面的 `hydraulic_diameter/2`，并按实际有效覆盖长度计算等效半径。

| 字段 | 含义 |
|------|------|
| `mpv/sv/smv/lpv/rpv/tips_resistance_integral` | 各段阻力积分 |
| `inflow_parallel_resistance` | (1/R_SV + 1/R_SMV)⁻¹ |
| `inflow_resistance_asymmetry` | (R_SV − R_SMV)/(R_SV + R_SMV) |
| `mpv_effective_radius` | r_eff = (L/R)¹ᐟ⁴, 直接体现 ΔP 效应 |
| `tips_inflow_resistance_ratio` | R_TIPS / R_inflow (术后) |

**(E) 拓扑 / 不对称 / MPV 形态**
| 字段 | 含义 |
|------|------|
| `collateral_burden_score` | Σ (D²·L)_collat / (D²·L)_MPV (体积加权侧支负担) |
| `n_collaterals_detected` | 检测到的侧支根数 |
| `branchpoint_density_per_cm` | 全树分叉点 / MPV 长度 (cm) |
| `mpv_taper_coefficient` | (D_proximal − D_distal)/L_MPV |
| `mpv_proximal_diameter` / `mpv_distal_diameter` | MPV 两端直径 |
| `mpv_min_max_diameter_ratio` | min/max 直径 (MPV 沿线狭窄指标) |
| `tree_area_conservation_mean_dev` | 各分叉处面积守恒平均偏离 |

其中 MPV 近端固定为 SV/SMV 汇合侧，远端固定为 LPV/RPV 分叉侧；两端直径分别取首尾有效 5 mm 的中位数，不受中心线路径存储方向影响。

**文献依据**：Peng *QIMS* 2019, Qi *Hepatology* 2014 / *Radiology* 2018, Kassab *AJP-Heart* 2006, Maruyama *QIMS* 2021, Mostafa *CEG* 2015, Berzigotti *J Hepatol* 2016, Ciurică *Hypertension* 2019。

---

#### **Step 5: 剖面特征提取** (`extract_profiles.py`)
**输入**：`centerline_profiles.json` + STL
**输出**：`pointwise_profiles.json`

**每个中心线点提取的6个特征**：
| 特征 | 计算方式 |
|------|---------|
| `area` | 正交平面与STL网格的交集面积(mm²) |
| `eq_diameter` | 等效直径 = 2√(A/π) |
| `perimeter` | 截面轮廓周长(mm) |
| `circularity` | 4π×A/P² |
| `curvature` | 中心线在该点的曲率(1/mm) |
| `inscribed_radius` | 内切圆半径 |

**截面计算核心**（高曲率无自交）：
1. 用 **4 mm 物理尺度高斯平滑**的中心线切线确定唯一截平面
2. 构造正交基 (u, v)，用垂直平面截断 STL 网格 (`trimesh.intersections.mesh_plane`)
3. 投影到 2D → polygonize → 选择包含当前中心线点的管腔多边形
4. 将多边形限制到当前中心线采样点的 **3D Voronoi 归属单元**；前后 5 mm 的局部邻点不参与限制，以免中心线采样噪声把截面切成窄条
5. 非局部回弯的中心线点通过垂直平分面裁掉不属于当前弧长位置的部分，避免急弯处截平面跨入同一血管的另一段
6. 截面额外计算形状指标，但这些指标只作为逐点特征，不参与删点：
   - `aspect_ratio` = √(λ_max / λ_min) (PCA 长短轴比, 圆/正方=1, 椭圆≈1.4)
   - `circularity` = 4πA/P² (1=完美圆, 0=极不规则)

方法依据：中心线法向截面见 [Liu et al., 2019](https://doi.org/10.1109/icist.2019.8836866)；
中心线/Voronoi 血管计算几何框架见 [Antiga et al., 2003](https://doi.org/10.1109/tmi.2003.812261)。
本项目在此基础上使用“沿弧长局部豁免的中心线 Voronoi 归属”，专门处理曲率半径接近
管腔半径时正交截面族不可避免的非局部相交。

**交点处理规则**：
1. **共享端点**：组成交点的每根血管都使用最终的 Voronoi 归属面积 `area` 检测自己的端点面积比例，不使用未裁剪的原始面积 `raw_area`。若交点在索引0、临界点为20，则再向血管内部扩展6个采样点，将 `0..26`（含端点）置0；此端点判断随后结束。
2. **段内侧支**：侧枝自身使用同一端点面积比例检测。连续主干不置0，而是把侧枝中心线加入主干的半径加权网络 Voronoi 竞争集合；共同节点后5 mm的侧枝局部点不参与限制，只裁掉归属于侧枝远端的截面部分。主干和侧枝半径差异通过 power distance 处理，避免细侧枝把粗主干截面压凹。

置零发生在最终 pointwise 重采样之后，不做插值。除此之外，成功求得的截面不再经过尺寸、形状、MAD、变化率或“小截面”过滤。

**新增剖面字段**:
- `inscribed_radius`：直接从距离场返回真实值, 此前版本是 0 占位
- `n_section_success`：几何求交成功的截面数
- `n_endpoint_junction_zeroed`：汇合端区间置0的点数
- `n_side_branch_junction_zeroed`：兼容字段，网络 Voronoi 策略下固定为0
- `n_total_side_branch_network_voronoi`：使用侧支网络 Voronoi 的主干-侧枝关系数

**可视化端**直接以 `pointwise_profiles.json` 为截面有效性的唯一数据源：数值为0的点不画，其他点全部按保存的中心线位置、法线和 Voronoi 参数重建，不再执行独立的显示过滤。显示轮廓投影面积与 pointwise `area` 一致。

---

#### **Step 5.5: 可视化导出** (`export_visualization.py`)
**输入**：STL + 分段JSON + 逐点剖面JSON
**输出**：`vis_interactive.html`（交互式3D） + `vis_overview.png`（静态多角度）

**可视化内容**：
- STL网格（半透明灰色）
- 分段中心线（按段着色）
- 最大截面位置标记：
  - 实线彩色环 = STL真实截面轮廓
  - 虚线圆 = 等效圆（radius = √(A/π)）
  - 中心点 + 面积/直径标签
- 段信息表格（包含最大面积数值）
- 图例 + 坐标轴

**失败降级**：
- 若分段失败，自动显示原始中心线（红色标题提示）

---

#### **Step 6: VTK 交互可视化** (`visualize_segments.py`)
**输入**：`centerline_profiles.json` + 中心线 + STL
**输出**：无文件输出（实时交互）

**快捷键操作**：
```
R:     重置视角              1-8: 切换各段可见性 (1=MPV, 2=SV, ...)
M:     切换血管模型          C: 切换原始中心线
L:     切换标签              B: 切换分支点 (默认隐藏)
X:     切换最大截面圈        W: 线框↔实体模式
+/-:   调整透明度            S: 截图
Q:     退出
```

**交互特性**：
- 鼠标旋转/平移/缩放血管模型
- 悬停显示点坐标和属性信息
- 实时切换各解剖结构的显隐

---

### 【第二阶段】跨患者相关性分析

#### **Step A: 统计特征相关性分析** (`correlation_analysis.py`)
**前提条件**：
```
patient_001/
  ├── vessel.stl
  ├── portal_vein_features.json    ← Step 4 输出
  └── label/
      ├── PVP.txt                   ← 门静脉压力(mmHg)
      └── PCG.txt                   ← 压力梯度(mmHg)
```

**分析流程**：
1. 从所有患者收集特征 → 特征矩阵 (N_samples × 72_features)
2. 丢弃缺失率 > 50% 的特征
3. 标准化处理
4. Spearman秩相关分析 vs 临床指标（PVP或PCG）
5. p值校正（FDR）
6. 可视化：
   - 相关性条形图 + p值标记
   - 热力图（样本×特征）
   - 散点图（选中特征 vs 目标量）
   - 火山图（|ρ| vs -log10(p)）

**输出**：
```
correlation_pvp/
  ├── correlation_results.csv        [所有特征相关性]
  ├── recommended_features.csv       [推荐入选训练特征 (新)]
  ├── correlation_heatmap.png        [热力图]
  ├── scatter_plots.png              [散点图网格]
  ├── top_features_bar.png           [Top 排名条形图]
  ├── feature_importance.png         [分组分析 + Top-10 表格]
  └── analysis_report.txt            [文字报告 + 自动相关性解读]
```

**自动相关性解读 (新)**: `analysis_report.txt` 末尾会附带:
- Top-K 特征清单 (强 / 中 / 弱 分级, 默认阈值 |ρ|≥0.4 / 0.25)
- "系统特征 vs 单段特征" 对比 (谁更强相关)
- 推荐入选训练的特征列表 (|ρ|≥0.25 ∧ p<0.05 ∧ 覆盖率≥70%) → 单独写到 `recommended_features.csv`
- 模型策略建议 (强相关够 → 基线; 不够 → L1/GBDT; 整体弱 → 排查端点掩码 / 样本量)

---

#### **Step B: 剖面特征逐点相关性分析** (`profile_correlation.py`)
**前提条件**：
```
patient_001/
  ├── pointwise_profiles.json             ← Step 5 输出
  └── label/
      └── PVP.txt or PCG.txt
```

**分析流程**：
1. 收集所有患者的逐点剖面特征 → (N_points × 6_features)
2. 按分支（MPV/SV/SMV等）分别分析
3. 对每个位置单独过滤NaN后做Spearman相关
4. 识别显著相关的位置（p<0.05）
5. 可视化：
   - 逐点相关性曲线（显著区域高亮）
   - 剖面热力图（样本按target值排序）
   - 高/低target组的剖面对比
   - 峰值相关性汇总

**输出**：
```
profile_correlation_pvp/
  ├── pointwise_correlation.png      [逐点相关曲线]
  ├── profile_heatmap.png            [热力图]
  ├── group_comparison.png           [高低组对比]
  ├── peak_correlations.csv          [峰值汇总]
  └── profile_report.txt             [文字报告]
```

**特征键**：`area`, `eq_diameter`, `circularity`, `curvature`, `perimeter`, `inscribed_radius`

---

## 🛠️ 文件夹命名规则

| 规则 | 解释 | 示例 |
|------|------|------|
| `20210909WuJinHeng` | TIPS术前 | 有效 ✓ |
| `20210909WuJinHeng#` | TIPS术后（含#） | 有效 ✓ |
| `*@*` 或 `*!*` | 包含@或! | 跳过 ✗ |

术前/术后标签会影响解剖分段：
- 术前：支持 LGV/PGV 代偿识别
- 术后：激活 TIPS 支架识别

---

## 📁 文件输出结构

每个患者文件夹最终包含：

```
patient_001/
├── vessel.stl                                 [输入]
├── CenterlinePoints.txt                       [Step 1输出: 原始中心线]
├── newCenterlist.txt                          [Step 2输出: 平滑中心线]
├── centerline_profiles.json                   [Step 3输出: 分段信息]
├── pointwise_profiles.json                    [Step 5输出: 逐点剖面 (含真实 inscribed_radius)]
├── portal_vein_features.json                  [Step 4输出: 扁平 schema, 兼容旧脚本]
├── unified_features.json                      [Step 4输出: 统一特征 (训练用) ★]
├── sv_smv_angle.json                          [补充: SV-SMV夹角]
├── vis_interactive.html                       [Step 5.5输出: 交互式3D (最大截面圈与文献值对齐)]
├── vis_overview.png                           [Step 5.5输出: 多角度静态图]
├── centerline_screenshot.png                  [Step 6输出: VTK截图]
└── label/
    ├── PVP.txt                                [输入: 门静脉压力]
    └── PCG.txt                                [输入: 压力梯度]
```

---
`你好`
## 🚀 使用指南

### 基本使用

```python
from main import process_stl_files, PipelineSteps, DEFAULT_PARAMS, run_correlation_analysis

# ===== 配置 =====
ROOT_FOLDER = r"F:\PCG data\dataset\test4all_sample"
TARGET = "PVP"  # or "PCG"
MODE = "all"    # "all" | "process" | "correlate"

# ===== 步骤开关 =====
steps = PipelineSteps()
steps.extract_centerline = True
steps.smooth_centerline = True
steps.segment_vessels = True
steps.extract_features = True
steps.extract_profiles = True
steps.export_visualization = True
steps.visualize = False  # 批量时建议关闭

# ===== 参数调整 =====
params = dict(DEFAULT_PARAMS)
# params['min_branch_length_mm'] = 10.0  # 调整最小分支长度

# ===== 执行 =====
if MODE in ("all", "process"):
    process_stl_files(ROOT_FOLDER, params=params, steps=steps, clean_old=True)

if MODE in ("all", "correlate"):
    run_correlation_analysis(
        ROOT_FOLDER,
        target=TARGET,
        run_statistical=True,
        run_profile=True,
        drop_features_above_missing=0.5,
        min_branch_coverage=0.3
    )
```

### 调试单个患者

```python
from extract_centerline import extract_centerline
from smooth_centerline import smooth_centerline
from segment_vessels import segment_vessels
from extract_features import extract_all_features
from extract_profiles import extract_profiles

stl_path = r"F:\PCG data\patient_001\vessel.stl"

# 逐步执行各模块
extract_centerline(stl_path, pitch=0.5, min_branch_length_mm=8.0)
smooth_centerline(stl_path)
segment_vessels(stl_path, post_tips=False)
extract_all_features(stl_path)
extract_profiles(stl_path)
```

---

### ⚡ 快速截面验证模式 (跳过耗时统计/相关性, 只看截面对不对)

中心线已经跑过、只想反复迭代截面算法时, 用这套设置:

```python
from main import process_stl_files, PipelineSteps, DEFAULT_PARAMS

ROOT_FOLDER = r"E:\zhengzhou_vkan3"
MODE = "process"   # ← 跳过 Step A/B 跨患者相关性 (最耗时的部分)

steps = PipelineSteps()
# ---- 上游 (中心线/分段已跑过, 关掉避免重算) ----
steps.extract_centerline   = False     # Step 1: 已有 CenterlinePoints.txt
steps.smooth_centerline    = False     # Step 2: 已有 newCenterlist.txt
steps.segment_vessels      = False     # Step 3: 已有 centerline_profiles.json
# ---- 跳过统计/系统特征 (~5-10s/患者, 验截面用不到) ----
steps.extract_features     = False     # Step 4: 关掉, 不影响截面验证
# ---- 核心: 截面提取 + 可视化 ----
steps.extract_profiles     = True      # Step 5: 重新算截面 (新算法生效)
steps.export_visualization = True      # Step 5.5: 出 vis_interactive.html 看环画对没有
# ---- VTK 弹窗会逐患者阻塞, 批量验证关掉 ----
steps.visualize            = False

process_stl_files(
    ROOT_FOLDER,
    params=dict(DEFAULT_PARAMS),
    steps=steps,
    clean_old=False,           # ← 关键! True 会删掉已算的中心线/分段, 必须设为 False
)
```

**单患者快速验证**:

```python
from extract_profiles import extract_profiles
from export_visualization import export_patient_visualization

stl = r"F:\patient_001\vessel.stl"
extract_profiles(stl)                       # ~20-30s
export_patient_visualization(stl,            # ~10-15s
                              export_html=True, export_png=False)  # PNG 慢, HTML 够看
# 浏览器打开 vis_interactive.html, 转动看每段最大截面圈是否合理
```

**性能对比** (跳过 Step 1-4 + Step 6 + Step A/B):
- 完整流程: ~80-145 s/患者
- 验证模式: ~30-45 s/患者 (省掉中心线 30-60s + 分段 10-20s + 统计 5-10s + 相关性整体批次)

---

## 📊 JSON输出格式详解

### `centerline_profiles.json`（分段信息）
```json
{
  "segments": {
    "mpv": {
      "nodeids": [0, 1, 2, 3, 4],
      "start_nodeid": 0,
      "end_nodeid": 4,
      "length": 125.34
    },
    "sv": {...},
    "smv": {...},
    ...
  }
}
```

### `portal_vein_features.json`（旧, 扁平 schema, 兼容老脚本）
> 实际是 flat dict: 每个键形如 `mpv_mean_diameter`, `confluence_murray3_ratio`, `sv_smv_angle`, `_meta` 单独一节。
> 这一文件的存在仅为了向后兼容, 未来推荐统一读 `unified_features.json`。

```json
{
  "mpv_length": 125.34,
  "mpv_mean_diameter": 12.5,
  "...": "...",
  "sv_smv_diameter_ratio": 1.23,
  "sv_smv_angle": 87.3,
  "confluence_murray3_ratio": 0.96,
  "inflow_resistance_asymmetry": -0.21,
  "_meta": { "patient_id": "...", "is_post_tips": false }
}
```

### `unified_features.json`（**新**, 统一文件 — 训练用）
> 所有特征聚合到一个 JSON, `_index` 块自带字段说明, 训练管线一次加载即可。

```json
{
  "_schema_version": "v1",
  "_meta": {
    "patient_id": "...", "is_post_tips": false,
    "has_compensation": false, "compensation_type": null
  },
  "_index": {
    "statistical": { "description": "每段9个标量特征",
                      "feature_keys": ["length", "tortuosity", "..."] },
    "system":      { "description": "系统/联合特征",
                      "groups": { "A_angles": [...], "B_diameter_area_ratio": [...],
                                  "C_length_tortuosity": [...], "D_hydraulic": [...],
                                  "E_topology": [...] },
                      "labels_cn": { "confluence_murray3_ratio": "汇合处Murray³比", "...": "..." } },
    "global":      { "description": "全局/树级标量",
                      "feature_keys": ["total_centerline_length", "has_lgv", "..."] },
    "sv_smv_angle":   { "description": "SV-SMV 汇合几何细节" },
    "pointwise":      { "description": "逐点剖面",
                         "feature_keys": ["position", "area", "eq_diameter", "...",
                                           "inscribed_radius", "section_valid"],
                         "mask_explanation": "端点和求交失败位置保留索引并置 0" },
    "segments_meta":  { "description": "每段路径的几何概览" }
  },
  "statistical": {
    "mpv": { "length": 125.34, "tortuosity": 1.08, "mean_diameter": 12.5,
              "...": "..." },
    "sv":  { "..." },
    "smv": { "..." }
  },
  "system": {
    "angle_sv_smv": 87.3,
    "confluence_murray3_ratio": 0.96,
    "confluence_area_ratio": 1.04,
    "splenic_dominance_index": 0.61,
    "mpv_resistance_integral": 0.0023,
    "inflow_resistance_asymmetry": -0.21,
    "mpv_effective_radius": 5.42,
    "collateral_burden_score": 0.18,
    "mpv_taper_coefficient": 0.012,
    "...": "..."
  },
  "global": {
    "total_centerline_length": 350.5,
    "sv_smv_diameter_ratio": 1.23,
    "sv_smv_angle": 87.3,
    "has_lgv": 0, "has_pgv": 0, "has_compensation_vessel": 0, "has_tips": 0
  },
  "sv_smv_angle": {
    "angle_degrees": 87.3,
    "confluence_point_physical": [12.3, 45.6, 78.9],
    "branch1_direction": [0.1, 0.7, -0.7],
    "branch2_direction": [-0.4, 0.5, -0.7]
  },
  "segments_meta": {
    "mpv": { "length_mm": 125.34, "tortuosity": 0.07, "endpoints_id": [12, 78], "...": "..." }
  },
  "pointwise": {
    "mpv": {
      "position": [0.0, 0.01, "..."],
      "arc_length_mm": [0.0, 1.25, "..."],
      "total_length_mm": 125.34,
      "area": [122.5, 123.1, 123.8, "..."],
      "eq_diameter": [12.4, 12.43, 12.47, "..."],
      "perimeter": [...], "circularity": [...], "curvature": [...],
      "inscribed_radius": [6.1, 6.2, 6.0, "..."],
      "endpoint_junction_mask": [0.0, 0.0, 0.0, "..."],
      "side_branch_contamination_mask": [0.0, 0.0, 0.0, "..."],
      "side_branch_network_voronoi": true,
      "n_endpoint_junction_zeroed": 2, "n_section_success": 200
    }
  },
  "pointwise_meta": {
    "n_points": 200,
    "section_filter_policy": "endpoint_zero_only",
    "n_total_endpoint_junction_zeroed": 2,
    "n_total_side_branch_junction_zeroed": 0,
    "n_total_side_branch_network_voronoi": 1
  }
}
```

`unified_features.json` 中的逐点剖面会先删除端点置零和求交失败点，再沿剩余有效弧段线性重采样回原始点数。例如原始 200 点删除前 30 点后，剩余 170 点会在其内部插值增加 30 点，最终仍为 200 点；不会在已删除端点处补零或向外外推。

**字段定位指南** (训练时如何索引):
| 想要 | 路径 |
|------|------|
| MPV 平均直径 | `unified["statistical"]["mpv"]["mean_diameter"]` |
| 汇合 Murray-3 偏离 | `unified["system"]["confluence_murray3_deviation"]` |
| 入流阻力不对称 | `unified["system"]["inflow_resistance_asymmetry"]` |
| MPV 沿线截面积曲线 (200 点) | `unified["pointwise"]["mpv"]["area"]` |
| MPV 长度 | `unified["statistical"]["mpv"]["length"]` 或 `unified["segments_meta"]["mpv"]["length_mm"]` |
| 是否术后 | `unified["_meta"]["is_post_tips"]` |
| 任意系统特征中文名 | `unified["_index"]["system"]["labels_cn"][字段名]` |

### `pointwise_profiles.json`（逐点剖面）
```json
{
  "segments": {
    "mpv": [
      {
        "nodeid": 0,
        "area": 120.5,
        "eq_diameter": 12.4,
        "perimeter": 39.2,
        "circularity": 0.93,
        "curvature": 0.008,
        "inscribed_radius": 6.2
      },
      {...}
    ],
    "sv": [...]
  }
}
```

---

## 🔧 辅助工具

### `compute angle.py`（SV-SMV夹角独立计算）
```python
from compute_angle import compute_sv_smv_angle

result = compute_sv_smv_angle(
    stl_path=r"F:\patient_001\vessel.stl",
    n_fit_points=10
)
# 返回 {'angle_degrees': 87.3, ...}
```

### `utils.py`（公共工具库）
核心函数：
- `voxelize_stl()` — STL体素化
- `load_tree()` / `save_tree()` — 中心线树I/O
- `classify_nodes()` — 节点分类（端点/分支点）
- `find_path()` — BFS路径查找
- `path_to_coords()` — 路径转坐标序列
- `path_physical_length()` — 物理长度计算

---

## ⚙️ 默认参数总表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pitch` | 0.5 | 体素化分辨率(mm) |
| `min_branch_length_mm` | 8.0 | 最小分支物理长度(mm) |
| `min_relative_length` | 0.05 | 最小相对长度比例 |
| `min_radius_ratio` | 0.4 | 最小半径比例 |
| `merge_bp_distance_mm` | 5.0 | 分支点合并距离(mm) |
| `n_fit_points` | 10 | 曲率拟合点数 |
| `n_profile_points` | 200 | 剖面采样点数 |
| `curvature_window` | 7 | 曲率滑动窗口大小 |
| `sample_step` | 3 | 采样步长 |

---

## 📈 性能与可靠性

### 依赖库
```
numpy, scipy, scikit-image, networkx, trimesh, pandas, matplotlib, plotly, kaleido, vtk
```

### 错误处理
- 各步骤独立异常捕获，不影响后续步骤
- 自动降级：若分段失败，可视化仍可生成（仅显示原始中心线）
- NaN处理：截面统计跳过端点区域，避免边界效应

### 性能参考
单患者处理耗时（仅参考，取决于模型复杂度）：
- Step 1（中心线提取）：~30-60s
- Step 2（平滑）：~5-10s
- Step 3（分段）：~10-20s
- Step 4（统计特征）：~5-10s
- Step 5（剖面特征）：~20-30s
- Step 5.5（可视化）：~10-15s
- **总计**：~80-145s/患者

---

## 🎯 关键概念

### 曲折度（Tortuosity）
$$\tau = 1 - \frac{\text{弦长}}{\text{弧长}}$$
- 0 = 完全笔直
- 接近1 = 高度弯曲

### 曲率（Curvature）
基于离散导数的二阶差分：对每个点计算的弯曲强度（1/mm）

### 圆度（Circularity）
$$C = \frac{4\pi A}{P^2}$$
- 1 = 完美圆形
- < 1 = 不规则形状

### 等效直径
$$D_{eq} = 2\sqrt{\frac{A}{\pi}}$$
其中A为截面积，等效于相同面积的圆的直径

---

## 📝 代码作者与维护

项目用途：门静脉血管定量分析与临床预测

关键人员：
- 中心线提取 & 分段算法
- 特征提取 & 统计分析
- 可视化与交互设计

---

## 📄 许可证

该项目用于学术研究，请遵守相关数据保护法规。

---

**最后更新**: 2026年7月
- 新增系统/联合特征 (Murray, Hagen-Poiseuille 阻力, 侧支负担), 统一 `unified_features.json`
- 截面使用平滑切线与中心线 Voronoi 归属，解决高曲率非局部相交
- 成功截面只在汇合端和已知侧支临界区间置0，不再执行其他过滤或插值
- Web 直接读取 pointwise 记录并复用同一 Voronoi 轮廓，不再独立删减
- 自动相关性解读 + 推荐特征 CSV
- 新增"快速截面验证模式"使用指南

