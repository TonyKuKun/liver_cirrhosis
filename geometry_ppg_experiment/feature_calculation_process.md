# 六个全局几何指标计算流程说明

本文档用于核对 `extract_features.py` 中六个全局特征的计算流程是否符合公式要求。当前脚本读取每个病例目录中的：

- `centerline_pointwise_profiles.json`：每段血管 100 个重采样点的逐点截面积、弧长、曲率、坐标等。
- `centerline_profiles.json`：每段血管的中心线路径、端点、长度等。
- `newCenterlist.txt` 或 `CenterlinePoints.txt`：中心线节点坐标。
- `sv_smv_angle.json`：SMV-SV 汇流角的已有拟合结果，优先用于特征 5。
- `label/PVP.txt`：PVP 标签。

输出六个特征：

```text
R_total
D_Murray
R_collateral
Ratio_SMV_SV
theta_SMV_SV
Ratio_LPV_RPV
```

对应代码位置：

- 特征列定义：`extract_features.py` 的 `FEATURE_COLUMNS`
- 总阻力：`compute_r_total`
- Murray 偏离：`compute_d_murray`
- 侧支等效阻力：`compute_r_collateral`
- SMV/SV 直径比：`compute_ratio_smv_sv`
- SMV-SV 汇流角：`compute_theta_smv_sv`
- LPV/RPV 直径比：`compute_ratio_lpv_rpv`

---

## 通用基础计算

### 1. 截面积转等效半径

对任意采样点 `i`，读取截面积 `A_i`，计算等效半径：

```math
r_i = \sqrt{\frac{A_i}{\pi}}
```

实现函数：

```text
radii_from_area(area)
```

只使用有限且大于 0 的面积值。无效面积对应的半径记为缺失。

### 2. 微段长度

每段血管有逐点弧长 `arc_length_mm`。相邻点之间的微段长度为：

```math
\Delta L_i = |s_{i+1} - s_i|
```

如果 `arc_length_mm` 缺失或长度不匹配，则使用 `total_length_mm` 在采样点上等距生成弧长；如果 `total_length_mm` 也无效，则按点序生成替代弧长。

实现函数：

```text
profile_arrays(seg_data)
segment_resistance_visc(seg_data)
```

---

## 特征 1：门脉系统总阻力 `R_total`

### 1.1 每段沿程黏性阻力

对参与总阻力拓扑的血管段分别计算，包括：

```text
SMV, SV, LGV, MPV, LPV, RPV, TIPS
```

其中 LGV 和 TIPS 仅在存在且有效时参与后续并联。每段沿程黏性阻力为：

```math
R_{\text{visc}} = \sum_i \frac{\Delta L_i}{r_i^4}
```

代码中使用第 `i` 个点半径 `r_i` 和 `i -> i+1` 的微段长度：

```text
segment_resistance_visc(seg_data)
```

注意：当前公式没有乘血液黏度和 `8/pi` 等常数，因此脚本也没有加入这些物理常数，只计算相对阻力特征。

### 1.2 局部损失因子 `Phi_local`

对每段血管每个采样点检查三类异常。

#### A. 突扩 / 突缩

相邻截面积比：

```math
AR = \frac{A_{i+1}}{A_i}
```

若突扩：

```math
AR > 1.3
```

则：

```math
\zeta_{\text{exp}} = \left(1 - \frac{A_i}{A_{i+1}}\right)^2
```

若突缩：

```math
AR < 0.7
```

则：

```math
\zeta_{\text{con}} = 0.5 \times \left(1 - \frac{A_{i+1}}{A_i}\right)
```

#### B. 高曲率点

读取逐点曲率 `K_i`，计算：

```math
K_i r_i
```

若：

```math
K_i r_i > 0.1
```

则：

```math
\zeta_{\text{bend}} = K_i r_i
```

#### C. 突扩 + 高曲率复合点

若同一点同时满足突扩和高曲率：

```math
\zeta_{\text{combined}}
= \zeta_{\text{exp}}
+ \zeta_{\text{bend}}
+ \zeta_{\text{exp}}\zeta_{\text{bend}}
```

此时不再单独加入 `zeta_exp` 和 `zeta_bend`，避免重复计数。

#### 汇总

```math
\Phi_{\text{local}} = \sum_j \zeta_j
```

实现函数：

```text
local_loss_factor(seg_data)
```

### 1.3 每段有效阻力

```math
R_{\text{effective}}
= R_{\text{visc}} \times (1 + \lambda \Phi_{\text{local}})
```

当前：

```text
lambda = 1.0
```

实现函数：

```text
compute_effective_resistance(seg_data)
```

兜底说明：

如果某段血管在 `centerline_pointwise_profiles.json` 中存在，但逐点面积全无效或无法支持上述严格计算，则 `R_total` 会尝试读取 `portal_vein_features.json` 中对应的预计算沿程阻力：

```text
mpv_resistance_integral
smv_resistance_integral
sv_resistance_integral
lpv_resistance_integral
rpv_resistance_integral
tips_resistance_integral
lgv_resistance_integral
```

该兜底值只代表沿程阻力积分，无法重新计算局部损失 `Phi_local`，因此 report 中会标记为：

```text
fallback_precomputed_resistance_integral
```

需要核查时，应优先看 `feature_extraction_report.json` 中每段的 `status`，区分严格公式计算和兜底来源。

### 1.4 按拓扑合成为总阻力

下方入口侧按 SMV、SV 和可选 LGV 并联。SMV 和 SV 认为是基础入口血管；如果存在胃左静脉 LGV，并且其直接连接门静脉，则将 LGV 也加入同一并联组：

```math
\frac{1}{R_{\text{inflow}}}
= \frac{1}{R_{\text{effective,SMV}}}
+ \frac{1}{R_{\text{effective,SV}}}
+ I_{\text{LGV}}\frac{1}{R_{\text{effective,LGV}}}
```

其中 `I_LGV=1` 表示 LGV 存在且有效，`I_LGV=0` 表示没有 LGV 或 LGV 无效。然后与 MPV 串联：

```math
R_{\text{prehepatic}}
= R_{\text{inflow}}
+ R_{\text{effective,MPV}}
```

上方出口侧按现有的 LPV、RPV 和 TIPS 手术管并联。某一支缺失时直接忽略，不让整个 `R_total` 缺失：

```math
\frac{1}{R_{\text{upper}}}
=
\sum_{b \in \{\text{LPV, RPV, TIPS}\}_{\text{available}}}
\frac{1}{R_{\text{effective},b}}
```

最终：

```math
R_{\text{total}}
= R_{\text{prehepatic}}
+ R_{\text{upper}}
```

实现函数：

```text
compute_r_total(sources)
```

严格性说明：

- 当前实现要求 SMV、SV 两支有效，才计算 `R_inflow`。
- LGV 有效时加入 SMV/SV/LGV 三支并联；LGV 缺失时不影响 `R_total`。
- 上方 LPV、RPV、TIPS 三者中只要至少一支有效，就计算上方并联阻力；缺失分支忽略。
- 如果 MPV 缺失，或 SMV/SV 基础入口组无法计算，或 LPV/RPV/TIPS 全部无效，则 `R_total` 缺失。

---

## 特征 2：Murray 定律综合偏离度 `D_Murray`

### 单个分叉点偏离度

对每个关键分叉点 `j`，母支半径为 `r_0`，两个子支半径为 `r_1, r_2`：

```math
d_j
= \left|1 - \frac{r_0^3}{r_1^3 + r_2^3}\right|
```

所有有效分叉点取算术平均：

```math
D_{\text{Murray}}
= \frac{1}{N}\sum_{j=1}^{N} d_j
```

实现函数：

```text
compute_d_murray(sources)
```

### 当前纳入的关键分叉点

#### 1. SMV-SV 汇合到 MPV

视为：

```text
母支：MPV
子支：SMV、SV
```

半径取值：

- `r_0`：MPV 在 SMV/SV 汇合点附近的点半径。
- `r_1`：SMV 连接 MPV 的端点半径。
- `r_2`：SV 连接 MPV 的端点半径。

#### 2. MPV 分叉到 LPV/RPV

视为：

```text
母支：MPV
子支：LPV、RPV
```

半径取值：

- `r_0`：MPV 在 LPV/RPV 分叉点附近的点半径。
- `r_1`：LPV 连接 MPV 的端点半径。
- `r_2`：RPV 连接 MPV 的端点半径。

#### 3. 主要侧支起源处

对 `lgv`、`pgv`，如果存在，则寻找其最近的父血管连接点。当前候选父血管包括：

```text
mpv, sv, smv, lpv, rpv
```

视为：

```text
母支：父血管
子支 1：侧支
子支 2：父血管延续方向的一小段
```

半径取值：

- `r_0`：父血管连接点半径。
- `r_1`：侧支连接端点半径。
- `r_2`：父血管连接点后方约 8% 路径位置的半径，作为父血管延续支。

说明：公式文字提到“主要侧支发出处”，但没有给出侧支分叉的第二子支如何定义。这里将父血管延续方向作为第二子支，是为了能套用 `r0^3 = r1^3 + r2^3` 的二分叉形式。这个点需要你重点确认。

---

## 特征 3：侧支网络总等效阻力 `R_collateral`

当前主要侧支候选只包括胃左静脉和胃后静脉：

```text
lgv, pgv
```

### 3.1 单条侧支沿程黏性阻力

对每条侧支 `c`：

```math
R_{\text{visc,coll}}^{(c)}
= \sum_i \frac{\Delta L_i}{r_i^4}
```

实现函数：

```text
segment_resistance_visc(seg_data)
```

### 3.2 侧支入口局部损失

#### 入口角损失

```math
\zeta_{\text{angle}}
= k_{\text{angle}}(1 - \cos\theta)
```

当前：

```text
k_angle = 1.0
```

角度 `theta` 的计算：

- 侧支起始段切向量：使用侧支连接端附近多个点拟合方向。
- 主干局部切向量：使用父血管连接点附近的中心线切向量。
- 为避免中心线存储方向正反导致角度变成补角，当前使用无向夹角，即取 `abs(cos theta)` 后计算 `theta`。

说明：公式写的是侧支起始段切向量与主干局部切向量的夹角，没有明确是否要求有向夹角。当前实现采用无向几何夹角，通常更符合“夹角”定义。若你希望严格保留 0-180 度有向结果，可以改回普通点积夹角。

#### 起始段弯曲损失

```math
\zeta_{\text{curvature}}
= K_{\text{start}} r_{\text{coll_start}}
```

其中：

- `K_start`：侧支连接端点处曲率。
- `r_coll_start`：侧支连接端点处半径。

#### 总入口损失

```math
\zeta_{\text{entrance}}
= \zeta_{\text{angle}}
+ \zeta_{\text{curvature}}
```

### 3.3 单条侧支有效阻力

```math
R_{\text{eff,coll}}^{(c)}
= R_{\text{visc,coll}}^{(c)}
\times (1 + \lambda_{\text{coll}}\zeta_{\text{entrance}})
```

当前：

```text
lambda_coll = 1.0
```

### 3.4 多条侧支并联

```math
\frac{1}{R_{\text{collateral}}}
= \sum_c \frac{1}{R_{\text{eff,coll}}^{(c)}}
```

如果只有一条有效侧支，则：

```math
R_{\text{collateral}}
= R_{\text{eff,coll}}^{(1)}
```

如果没有有效侧支，则 `R_collateral` 记为缺失。

实现函数：

```text
compute_r_collateral(sources)
```

---

## 特征 4：SMV/SV 等效直径比 `Ratio_SMV_SV`

公式：

```math
D_{\text{SMV}}
= 2\sqrt{\frac{A_{\text{SMV}}}{\pi}}
```

```math
D_{\text{SV}}
= 2\sqrt{\frac{A_{\text{SV}}}{\pi}}
```

```math
\text{Ratio}_{\text{SMV/SV}}
= \frac{D_{\text{SMV}}}{D_{\text{SV}}}
```

实现流程：

1. 找到 SMV 连接 MPV 的端点。
2. 找到 SV 连接 MPV 的端点。
3. 分别沿远离汇合点方向取 20-30 mm 范围内的截面积中位数。
4. 面积换算为等效直径。
5. 计算 `D_SMV / D_SV`。

实现函数：

```text
compute_ratio_smv_sv(sources)
area_by_distance_from_attachment(seg_data, side, 20.0, 30.0)
```

说明：公式写的是“汇合前约 2-3 厘米处正常段测量”。当前实现用 20-30 mm 范围内的中位数代表该正常段。

---

## 特征 5：SMV-SV 汇流夹角 `theta_SMV_SV`

公式：

```math
\theta_{\text{SMV-SV}}
= \arccos
\left(
\frac{
\vec{v}_{\text{SMV}}\cdot\vec{v}_{\text{SV}}
}{
|\vec{v}_{\text{SMV}}||\vec{v}_{\text{SV}}|
}
\right)
```

结果单位为度。

实现流程：

1. 优先读取病例目录中的 `sv_smv_angle.json`。
2. 使用其中的 `angle_degrees` 作为 `theta_SMV_SV`。
3. 如果该文件不存在或角度无效，则使用中心线拟合向量计算：
   - `v_SMV`：SMV 上游方向指向汇合点的拟合向量。
   - `v_SV`：SV 上游方向指向汇合点的拟合向量。
4. 用 arccos 点积公式得到角度。

实现函数：

```text
compute_theta_smv_sv(sources)
```

说明：由于 `sv_smv_angle.json` 已经包含拟合角，当前实现优先使用它，避免只用相邻两个中心线点造成角度噪声。

---

## 特征 6：LPV/RPV 等效直径比 `Ratio_LPV_RPV`

公式：

```math
D_{\text{LPV}}
= 2\sqrt{\frac{A_{\text{LPV}}}{\pi}}
```

```math
D_{\text{RPV}}
= 2\sqrt{\frac{A_{\text{RPV}}}{\pi}}
```

```math
\text{Ratio}_{\text{LPV/RPV}}
= \frac{D_{\text{LPV}}}{D_{\text{RPV}}}
```

实现流程：

1. 找到 LPV 连接 MPV 的端点。
2. 找到 RPV 连接 MPV 的端点。
3. 对每支取起始正常段：
   - 如果连接端是该段 `start`，取 0-20% 弧长范围面积中位数。
   - 如果连接端是该段 `end`，取 80-100% 弧长范围面积中位数。
4. 面积换算为等效直径。
5. 计算 `D_LPV / D_RPV`。

实现函数：

```text
compute_ratio_lpv_rpv(sources)
```

说明：公式写的是“主干分叉后 LPV/RPV 各自起始段正常处测量”。当前实现用分叉端附近 20% 段面积中位数代表起始正常段。

---

## 缺失值处理规则

### 样本跳过规则

根据样本名中的 `#` 和实际 TIPS 管识别结果进行一致性检查：

- 样本名带 `#`，但没有识别到 TIPS 管：跳过。
- 样本名不带 `#`，但识别到 TIPS 管：跳过。
- 一致时才写入 `features.csv`。

对应 report 状态：

```text
skipped_hash_without_tips_tube
skipped_tips_tube_without_hash
```

### `R_total`

需要基础入口和主干可计算：

```text
smv, sv, mpv
```

LGV 有效时加入下方并联。上方使用 LPV、RPV、TIPS 中所有有效分支并联，至少需要一支有效。

### `D_Murray`

只要至少一个关键分叉点可计算，就输出所有有效分叉点偏离度的平均值。

### `R_collateral`

只使用有效侧支。没有有效侧支则缺失。

### 三个流量表征项

- `Ratio_SMV_SV`：SMV 或 SV 测量面积缺失则缺失。
- `theta_SMV_SV`：优先角度文件，若文件和中心线角都无效则缺失。
- `Ratio_LPV_RPV`：LPV 或 RPV 起始段面积缺失则缺失。

---

## 当前输出文件

重新运行后，主要输出为：

- `features.csv`：每例六个原始特征。
- `feature_extraction_report.json`：每例每个特征的中间项和状态。
- `feature_pvp_correlations.csv`：六个特征与 PVP 的 Pearson/Spearman 相关性。
- `features_zscore.csv`：附加六个 Z-score 标准化特征。
- `feature_pvp_scatter.png`：特征与 PVP 散点图。
- `feature_correlation_heatmap.png`：相关矩阵热图。

---

## 建议重点核查的问题

1. `D_Murray` 中 LGV/PGV 侧支发出处是否应纳入；如果纳入，父血管延续支作为第二子支是否符合你的定义。
2. LGV 同时参与 `R_total` 下方并联和 `R_collateral` 侧支阻力是否符合你的建模设定。
3. 侧支入口角损失是否应使用无向夹角，还是严格使用有向 0-180 度夹角。
4. `Ratio_SMV_SV` 的 20-30 mm 中位数是否符合“约 2-3 厘米处正常段”的医学测量习惯。
5. `Ratio_LPV_RPV` 的起始 20% 中位数是否符合“起始段正常处”的测量习惯。
