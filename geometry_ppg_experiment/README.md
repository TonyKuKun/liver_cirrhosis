# Portal Geometry PVP Correlation Experiment

本目录用于从每个病例 `features/unified_features.json` 等新生成特征文件中提取 6 个全局门静脉几何指标，并分析这些指标与 PVP 的相关性。

## 数据来源

默认数据根目录：

```text
F:\PCG data\dataset\test4all_sample
```

每个病例优先读取：

- `features/unified_features.json`
- `features/pointwise_profiles.json`
- `features/segment_assignments.json`
- `features/newcenterline.txt`
- `label/PVP.txt`

如果新路径不存在，脚本会兼容旧版根目录文件。

## 运行方法

提取特征：

```text
python extract_features.py --local-loss-lambda 0
```

生成三组主相关性结果和两种改进方案的参数扫描：

```text
python run_correlation.py
```

默认输出目录：

```text
result
```

## TIPS 样本过滤规则

为了避免 TIPS 状态和样本名不一致造成混淆：

- 样本名带 `#` 但未识别到 TIPS 管：跳过。
- 样本名不带 `#` 但识别到 TIPS 管：跳过。
- 一致时才写入 `features.csv`。

## 六个指标

### 1. `R_total`

含义：门静脉系统总等效阻力。

每段血管先计算沿程阻力：

```math
r_i = \sqrt{\frac{A_i}{\pi}}
```

```math
R_{\text{visc}} = \sum_i \frac{\Delta L_i}{r_i^4}
```

采样与有效长度的处理：

- `pointwise_profiles.json` 中的 `arc_length_mm` 是中心线弧长坐标，200 个点按血管弧长等距布置，不是按数组下标把总长度随意均匀分配。
- 计算时优先使用 `unified_features.json` 的 `pointwise.<segment>`。该数据已经删除首尾无效截面，并在保留弧长区间内重新采样；例如 `_point_filter.effective_length_mm` 记录了删除后的有效长度。
- 因此每个微段使用

```math
\Delta L_i = |s_{i+1} - s_i|
```

其中 `s_i` 为有效弧长坐标，实际有效长度为 `s_last - s_first`，不能直接用原始 `total_length_mm`。只有旧格式缺少有效弧长时，才根据正面积点的首尾位置从 `total_length_mm` 缩放回退。

局部损失项：

```math
R_{\text{effective}} = R_{\text{visc}}(1 + \lambda \Phi_{\text{local}})
```

其中 `lambda` 可调。本次最终结果使用：

```text
lambda = 0
```

即只使用沿程阻力，不加入局部突扩、突缩和弯曲损失放大项。

拓扑整合：

下方入口侧：

```math
R_{\text{inflow}} = \text{parallel}(\text{SMV}, \text{SV}, \text{optional LGV})
```

主干串联：

```math
R_{\text{prehepatic}} = R_{\text{inflow}} + R_{\text{MPV}}
```

上方出口侧：

```math
R_{\text{upper}} = \text{parallel}(\text{available LPV}, \text{available RPV}, \text{available TIPS})
```

总阻力：

```math
R_{\text{total}} = R_{\text{prehepatic}} + R_{\text{upper}}
```

### 2. `D_Murray`

含义：Murray 定律综合偏离度，反映分叉处血管半径匹配偏离程度。

单个分叉点：

```math
d_j = \left|1 - \frac{r_0^3}{r_1^3 + r_2^3}\right|
```

总体：

```math
D_{\text{Murray}} = \frac{1}{N}\sum_{j=1}^{N} d_j
```

当前纳入：

- SMV/SV 汇合至 MPV。
- MPV 分叉至 LPV/RPV。
- LGV/PGV 侧支起源处，若存在。

### 3. `R_collateral`

含义：侧支网络总等效阻力。

当前只统计：

```text
LGV, PGV
```

单条侧支：

```math
R_{\text{visc,coll}}^{(c)} = \sum_i \frac{\Delta L_i}{r_i^4}
```

入口损失：

```math
\zeta_{\text{entrance}} = k_{\text{angle}}(1 - \cos\theta) + K_{\text{start}}r_{\text{start}}
```

有效阻力：

```math
R_{\text{eff,coll}}^{(c)} =
R_{\text{visc,coll}}^{(c)}(1 + \lambda_{\text{coll}}\zeta_{\text{entrance}})
```

侧支网络并联：

```math
\frac{1}{R_{\text{collateral}}} =
\sum_c \frac{1}{R_{\text{eff,coll}}^{(c)}}
```

### 4. `Ratio_SMV_SV`

含义：SMV 和 SV 等效直径比。

在 SMV/SV 汇合前约 20-30 mm 正常段取截面积，计算：

```math
D_{\text{SMV}} = 2\sqrt{\frac{A_{\text{SMV}}}{\pi}}
```

```math
D_{\text{SV}} = 2\sqrt{\frac{A_{\text{SV}}}{\pi}}
```

```math
\text{Ratio}_{\text{SMV/SV}} =
\frac{D_{\text{SMV}}}{D_{\text{SV}}}
```

### 5. `theta_SMV_SV`

含义：SMV-SV 汇流夹角，单位为度。

```math
\theta_{\text{SMV-SV}} =
\arccos
\left(
\frac{
\vec{v}_{\text{SMV}}\cdot\vec{v}_{\text{SV}}
}{
|\vec{v}_{\text{SMV}}||\vec{v}_{\text{SV}}|
}
\right)
```

优先读取 `unified_features.json` 中的 `sv_smv_angle`，否则由中心线拟合向量计算。

### 6. `Ratio_LPV_RPV`

含义：LPV 和 RPV 等效直径比。

在 MPV 分叉后 LPV/RPV 起始正常段取截面积：

```math
D_{\text{LPV}} = 2\sqrt{\frac{A_{\text{LPV}}}{\pi}}
```

```math
D_{\text{RPV}} = 2\sqrt{\frac{A_{\text{RPV}}}{\pi}}
```

```math
\text{Ratio}_{\text{LPV/RPV}} =
\frac{D_{\text{LPV}}}{D_{\text{RPV}}}
```

## 实验结果

### 参数与样本

- `local_loss_lambda = 0`：`R_total` 只使用沿程阻力。
- `local_loss_max_lambda = 0`：主结果不加入局部损失峰值项。
- `resistance_peak_alpha = 0`：主结果不加入阻力贡献峰值放大项。
- `stenosis_relative_threshold = 0.8`：只在改进一中用于定义最窄点附近的连续狭窄区。
- `collateral_loss_lambda = 1`：主结果中的 `R_collateral` 使用标准入口损失系数。
- 最终样本总数：87。
- TIPS：34。
- 非 TIPS：53。
- `n_used` 是该分组中指标与 PVP 均非缺失、实际进入相关性计算的样本数。
- 相关系数后的 `*` 表示对应相关性达到 `p < 0.05`。

各指标有效样本数：`R_total=87`、`D_Murray=87`、`R_collateral=42`、`Ratio_SMV_SV=87`、`theta_SMV_SV=87`、`Ratio_LPV_RPV=76`。

### 合并分析

| 指标 | n_used | Pearson r | Pearson p | Spearman r | Spearman p |
|---|---:|---:|---:|---:|---:|
| R_total | 87 | -0.1154 | 0.2873 | -0.1710 | 0.1132 |
| D_Murray | 87 | -0.1907 | 0.0768 | -0.0653 | 0.5481 |
| R_collateral | 42 | -0.2322 | 0.1390 | -0.4295* | 0.0045 |
| Ratio_SMV_SV | 87 | 0.0143 | 0.8953 | 0.0329 | 0.7626 |
| theta_SMV_SV | 87 | -0.0177 | 0.8706 | 0.0187 | 0.8633 |
| Ratio_LPV_RPV | 76 | 0.0192 | 0.8694 | 0.0505 | 0.6647 |

### TIPS 分析

| 指标 | n_used | Pearson r | Pearson p | Spearman r | Spearman p |
|---|---:|---:|---:|---:|---:|
| R_total | 34 | -0.4372* | 0.0097 | -0.5011* | 0.0025 |
| D_Murray | 34 | -0.0544 | 0.7600 | 0.0325 | 0.8554 |
| R_collateral | 1 |  |  |  |  |
| Ratio_SMV_SV | 34 | -0.3849* | 0.0246 | -0.3294 | 0.0571 |
| theta_SMV_SV | 34 | -0.0669 | 0.7070 | -0.0882 | 0.6198 |
| Ratio_LPV_RPV | 23 | -0.2996 | 0.1648 | -0.2808 | 0.1943 |

### 非 TIPS 分析

| 指标 | n_used | Pearson r | Pearson p | Spearman r | Spearman p |
|---|---:|---:|---:|---:|---:|
| R_total | 53 | -0.1584 | 0.2572 | -0.0911 | 0.5165 |
| D_Murray | 53 | -0.0980 | 0.4850 | -0.0660 | 0.6384 |
| R_collateral | 41 | -0.2442 | 0.1240 | -0.4353* | 0.0045 |
| Ratio_SMV_SV | 53 | -0.0331 | 0.8140 | -0.0035 | 0.9801 |
| theta_SMV_SV | 53 | 0.2050 | 0.1408 | 0.1982 | 0.1548 |
| Ratio_LPV_RPV | 53 | 0.2257 | 0.1042 | 0.2025 | 0.1459 |

### 主结果解读

1. TIPS 组的 `R_total` 与 PVP 显著负相关：Pearson `r=-0.4372, p=0.0097`，Spearman `r=-0.5011, p=0.0025`。
2. 合并组的 `R_collateral` 与 PVP 存在显著 Spearman 负相关：`r=-0.4295, p=0.0045`。
3. 非 TIPS 组的 `R_collateral` 同样存在显著 Spearman 负相关：`r=-0.4353, p=0.0045`。
4. TIPS 组的 `Ratio_SMV_SV` 在 Pearson 分析中显著，但 Spearman `p=0.0571`，稳定性弱于 `R_total`。
5. TIPS 组只有 1 例存在自然侧枝，不能单独分析 `R_collateral`。

## 系数实验

下面的系数扫描属于当前数据集上的探索性分析。系数是根据同一批数据比较的，不能视为独立验证结果。

### `R_total` 局部损失系数

`R_total` 的显著相关主要出现在 TIPS 组，因此下表列出 TIPS 组全部系数扫描结果。合并组和非 TIPS 组在这些系数下均未获得稳定显著相关，完整数据保存在 `result/local_loss_tuning.csv`。

| local_loss_lambda | n_used | Pearson r | Pearson p | Spearman r | Spearman p |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 34 | -0.4372 | 0.0097 | -0.5011 | 0.0025 |
| 0.005 | 34 | -0.4223 | 0.0128 | -0.4951 | 0.0029 |
| 0.01 | 34 | -0.4067 | 0.0170 | -0.4581 | 0.0064 |
| 0.025 | 34 | -0.3668 | 0.0329 | -0.4195 | 0.0135 |
| 0.05 | 34 | -0.3235 | 0.0620 | -0.3535 | 0.0403 |
| 0.075 | 34 | -0.2972 | 0.0878 | -0.3228 | 0.0626 |
| 0.1 | 34 | -0.2799 | 0.1089 | -0.2925 | 0.0932 |
| 0.15 | 34 | -0.2587 | 0.1395 | -0.2711 | 0.1210 |
| 0.2 | 34 | -0.2464 | 0.1600 | -0.2620 | 0.1344 |
| 0.25 | 34 | -0.2384 | 0.1745 | -0.2613 | 0.1355 |
| 0.5 | 34 | -0.2210 | 0.2090 | -0.2352 | 0.1805 |
| 1.0 | 34 | -0.2116 | 0.2295 | -0.2058 | 0.2429 |
| 2.0 | 34 | -0.2068 | 0.2407 | -0.2006 | 0.2552 |
| 5.0 | 34 | -0.2038 | 0.2476 | -0.1968 | 0.2646 |
| 10.0 | 34 | -0.2028 | 0.2500 | -0.1968 | 0.2646 |
| 20.0 | 34 | -0.2023 | 0.2512 | -0.1968 | 0.2646 |
| 50.0 | 34 | -0.2020 | 0.2519 | -0.1968 | 0.2646 |
| 100.0 | 34 | -0.2019 | 0.2522 | -0.1968 | 0.2646 |

结论：`local_loss_lambda=0` 时 Pearson 和 Spearman 相关均最强，因此主结果采用 `0`。随着系数增大，相关性持续减弱。

### `R_collateral` 侧枝入口损失系数

TIPS 组 `R_collateral` 只有 `n=1`，不能进行系数比较。以下分别给出合并组和非 TIPS 组结果。

#### 合并组

| collateral_loss_lambda | n_used | Pearson r | Pearson p | Spearman r | Spearman p |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 42 | -0.1944 | 0.2173 | -0.4226 | 0.0053 |
| 0.005 | 42 | -0.1948 | 0.2164 | -0.4236 | 0.0052 |
| 0.01 | 42 | -0.1952 | 0.2155 | -0.4236 | 0.0052 |
| 0.025 | 42 | -0.1963 | 0.2128 | -0.4283 | 0.0047 |
| 0.05 | 42 | -0.1981 | 0.2085 | -0.4286 | 0.0046 |
| 0.075 | 42 | -0.1998 | 0.2045 | -0.4303 | 0.0045 |
| 0.1 | 42 | -0.2014 | 0.2008 | -0.4286 | 0.0046 |
| 0.25 | 42 | -0.2098 | 0.1824 | -0.4278 | 0.0047 |
| 0.5 | 42 | -0.2199 | 0.1618 | -0.4244 | 0.0051 |
| 1.0 | 42 | -0.2322 | 0.1390 | -0.4295 | 0.0045 |
| 2.0 | 42 | -0.2442 | 0.1192 | -0.4387 | 0.0037 |
| 5.0 | 42 | -0.2558 | 0.1020 | -0.4483 | 0.0029 |
| 10.0 | 42 | -0.2610 | 0.0951 | -0.4568 | 0.0024 |
| 20.0 | 42 | -0.2639 | 0.0913 | -0.4524 | 0.0026 |
| 50.0 | 42 | -0.2657 | 0.0889 | -0.4490 | 0.0029 |
| 100.0 | 42 | -0.2664 | 0.0881 | -0.4490 | 0.0029 |

#### 非 TIPS 组

| collateral_loss_lambda | n_used | Pearson r | Pearson p | Spearman r | Spearman p |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 41 | -0.2076 | 0.1929 | -0.4308 | 0.0049 |
| 0.005 | 41 | -0.2079 | 0.1920 | -0.4319 | 0.0048 |
| 0.01 | 41 | -0.2083 | 0.1912 | -0.4319 | 0.0048 |
| 0.025 | 41 | -0.2094 | 0.1889 | -0.4368 | 0.0043 |
| 0.05 | 41 | -0.2111 | 0.1851 | -0.4371 | 0.0043 |
| 0.075 | 41 | -0.2128 | 0.1816 | -0.4387 | 0.0041 |
| 0.1 | 41 | -0.2144 | 0.1783 | -0.4369 | 0.0043 |
| 0.25 | 41 | -0.2225 | 0.1620 | -0.4359 | 0.0044 |
| 0.5 | 41 | -0.2322 | 0.1440 | -0.4296 | 0.0051 |
| 1.0 | 41 | -0.2442 | 0.1240 | -0.4353 | 0.0045 |
| 2.0 | 41 | -0.2557 | 0.1066 | -0.4443 | 0.0036 |
| 5.0 | 41 | -0.2669 | 0.0916 | -0.4542 | 0.0029 |
| 10.0 | 41 | -0.2719 | 0.0855 | -0.4596 | 0.0025 |
| 20.0 | 41 | -0.2747 | 0.0822 | -0.4551 | 0.0028 |
| 50.0 | 41 | -0.2765 | 0.0801 | -0.4516 | 0.0030 |
| 100.0 | 41 | -0.2771 | 0.0794 | -0.4516 | 0.0030 |

结论：`collateral_loss_lambda=0` 时已经存在显著 Spearman 相关；系数增大到 `10` 时，合并组和非 TIPS 组的 Spearman 相关最强。当前主结果仍采用预先设定的 `1`，而 `10` 只能作为当前数据上的探索性候选值，避免直接按同一数据集的最小 p 值确定模型参数。

## 两种进一步改进实验

下面两种方法分别单独测试，以便判断相关性变化来自哪一个新增机制。方法一测试时令 `lambda1=lambda2=0`；方法二测试时令 `alpha=0`。主结果仍保留未调参的基线公式，避免直接用同一批数据选择系数后再把它当成验证结果。

结果标注规则：

- 相关系数后的 `*` 表示对应 `p < 0.05`。
- <span style="text-decoration: underline double;">双下划线</span>表示同一分组、同一种相关方法中的最佳值。
- <span style="text-decoration: underline;">单下划线</span>表示同一分组、同一种相关方法中的第二佳值。
- 相关性的优劣按 `|r|` 判断；同一张扫描表内 `n_used` 固定，因此与按 p 值判断的顺序一致。完全并列时优先系数更小的组合，并在正文说明并列情况。
- 完整扫描 CSV 保留原始数值，不写入星号或 HTML 标记，方便后续程序读取。

### 改进一：阻力贡献峰值放大

对每段血管除原始沿程阻力外，再计算：

```math
R_{\text{visc,max}} = \max_i \left(\frac{1}{r_i^4}\right)
```

图片中的 `L_stenotic` 没有给出判定阈值。为保证实验可复现，本实验将它定义为：包含 `R_visc,max` 所在点，且 `1/r_i^4` 不低于峰值 80% 的连续采样点覆盖长度。`R_visc,max` 在全部采样点上取最大值，狭窄区左右边界取相邻采样点的中点。

```math
L_{\text{stenotic}} =
\sum_{i \in \text{contiguous peak region}} \Delta L_i,
\qquad
\frac{1}{r_i^4} \ge 0.8R_{\text{visc,max}}
```

改进后的单段阻力为：

```math
R_{\text{effective}} =
\left(
R_{\text{visc,sum}}
+ \alpha R_{\text{visc,max}}L_{\text{stenotic}}
\right)
\left(1+\lambda_1\Phi_{\text{local,sum}}+\lambda_2\Phi_{\text{local,max}}\right)
```

本实验只测试峰值阻力项，因此 `lambda1=lambda2=0`。TIPS 组完整结果如下：

| alpha | n_used | Pearson r | Pearson p | Spearman r | Spearman p |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 34 | -0.4372 | 0.0097 | -0.5011 | 0.0025 |
| 0.005 | 34 | -0.4374 | 0.0097 | -0.5011 | 0.0025 |
| 0.01 | 34 | -0.4376 | 0.0096 | -0.5011 | 0.0025 |
| 0.025 | 34 | -0.4381 | 0.0095 | -0.5011 | 0.0025 |
| 0.05 | 34 | -0.4390 | 0.0094 | -0.5002 | 0.0026 |
| 0.075 | 34 | -0.4399 | 0.0092 | -0.5039 | 0.0024 |
| 0.1 | 34 | -0.4407 | 0.0091 | -0.5013 | 0.0025 |
| 0.15 | 34 | -0.4422 | 0.0088 | -0.5063 | 0.0022 |
| 0.2 | 34 | -0.4437 | 0.0086 | -0.5025 | 0.0025 |
| 0.25 | 34 | -0.4451 | 0.0084 | -0.5025 | 0.0025 |
| 0.5 | 34 | -0.4507 | 0.0075 | -0.5007 | 0.0026 |
| 1.0 | 34 | -0.4576 | 0.0065 | -0.5056 | 0.0023 |
| 2.0 | 34 | -0.4632 | 0.0058 | -0.4907 | 0.0032 |
| 5.0 | 34 | -0.4628 | 0.0058 | -0.4922 | 0.0031 |
| 10.0 | 34 | -0.4572 | 0.0066 | -0.5011 | 0.0025 |
| 20.0 | 34 | -0.4508 | 0.0075 | -0.4979 | 0.0027 |
| 50.0 | 34 | -0.4447 | 0.0084 | -0.4683 | 0.0052 |
| 100.0 | 34 | -0.4421 | 0.0089 | -0.4697 | 0.0051 |

方法一结论：

- Pearson 最优为 `alpha=2`：`r=-0.4632, p=0.0058`。
- Spearman 最优为 `alpha=0.15`：`r=-0.5063, p=0.0022`，相对基线仅小幅改善。
- `alpha=1` 时 Pearson 为 `r=-0.4576, p=0.0065`，Spearman 为 `r=-0.5056, p=0.0023`，是同时略微改善两者的平衡候选值。
- 合并组最优 Spearman 仍只有 `r=-0.1904, p=0.0773`，非 TIPS 组最优 Spearman 为 `r=-0.1550, p=0.2677`，均未达到显著。

TIPS 组排名汇总：

| 排名依据 | 名次 | alpha | n_used | r | p |
|---|---:|---:|---:|---:|---:|
| Pearson | 1 | 2.0 | 34 | <span style="text-decoration: underline double;">-0.4632*</span> | <span style="text-decoration: underline double;">0.0058</span> |
| Pearson | 2 | 5.0 | 34 | <span style="text-decoration: underline;">-0.4628*</span> | <span style="text-decoration: underline;">0.0058</span> |
| Spearman | 1 | 0.15 | 34 | <span style="text-decoration: underline double;">-0.5063*</span> | <span style="text-decoration: underline double;">0.0022</span> |
| Spearman | 2 | 1.0 | 34 | <span style="text-decoration: underline;">-0.5056*</span> | <span style="text-decoration: underline;">0.0023</span> |

### 改进二：局部损失累积值与峰值并用

在每个采样位置先合并该位置的突扩、突缩和弯曲损失，得到 `zeta_i`，然后分别计算：

```math
\Phi_{\text{local,sum}} = \sum_i \zeta_i
```

```math
\Phi_{\text{local,max}} = \max_i \zeta_i
```

改进后的局部损失部分为：

```math
R_{\text{effective}} =
R_{\text{visc}}
\left(
1+\lambda_1\Phi_{\text{local,sum}}
+\lambda_2\Phi_{\text{local,max}}
\right)
```

参数扫描约束为 `lambda2 >= lambda1`。先固定 `lambda1=0`，观察只加入峰值项时的 TIPS 组结果：

| lambda1 | lambda2 | n_used | Pearson r | Pearson p | Spearman r | Spearman p |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.0 | 34 | -0.4372 | 0.0097 | -0.5011 | 0.0025 |
| 0.0 | 0.005 | 34 | -0.4372 | 0.0097 | -0.5011 | 0.0025 |
| 0.0 | 0.01 | 34 | -0.4371 | 0.0097 | -0.5060 | 0.0023 |
| 0.0 | 0.025 | 34 | -0.4369 | 0.0098 | -0.5060 | 0.0023 |
| 0.0 | 0.05 | 34 | -0.4365 | 0.0099 | -0.5082 | 0.0022 |
| 0.0 | 0.075 | 34 | -0.4359 | 0.0100 | -0.5089 | 0.0021 |
| 0.0 | 0.1 | 34 | -0.4353 | 0.0101 | -0.5184 | 0.0017 |
| 0.0 | 0.15 | 34 | -0.4337 | 0.0104 | -0.5152 | 0.0018 |
| 0.0 | 0.2 | 34 | -0.4319 | 0.0108 | -0.5105 | 0.0020 |
| 0.0 | 0.25 | 34 | -0.4299 | 0.0112 | -0.5037 | 0.0024 |
| 0.0 | 0.5 | 34 | -0.4188 | 0.0137 | -0.5183 | 0.0017 |
| 0.0 | 1.0 | 34 | -0.3974 | 0.0199 | -0.4800 | 0.0041 |
| 0.0 | 2.0 | 34 | -0.3654 | 0.0336 | -0.3878 | 0.0234 |
| 0.0 | 5.0 | 34 | -0.3168 | 0.0679 | -0.3323 | 0.0548 |
| 0.0 | 10.0 | 34 | -0.2861 | 0.1010 | -0.3032 | 0.0813 |
| 0.0 | 20.0 | 34 | -0.2651 | 0.1298 | -0.2634 | 0.1323 |
| 0.0 | 50.0 | 34 | -0.2500 | 0.1538 | -0.2398 | 0.1719 |
| 0.0 | 100.0 | 34 | -0.2445 | 0.1634 | -0.2351 | 0.1808 |

下面列出每个 `lambda1` 下 Spearman p 值最小的 `lambda2`，用于检查“累积项和峰值项并用”是否进一步改善：

| lambda1 | 最优 lambda2 | n_used | Pearson r | Pearson p | Spearman r | Spearman p |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.1 | 34 | -0.4353 | 0.0101 | -0.5184 | 0.0017 |
| 0.005 | 0.005 | 34 | -0.4222 | 0.0129 | -0.4951 | 0.0029 |
| 0.01 | 0.2 | 34 | -0.4009 | 0.0188 | -0.4752 | 0.0045 |
| 0.025 | 0.15 | 34 | -0.3644 | 0.0341 | -0.4291 | 0.0113 |
| 0.05 | 0.2 | 34 | -0.3226 | 0.0628 | -0.3668 | 0.0329 |
| 0.075 | 0.2 | 34 | -0.2973 | 0.0877 | -0.3325 | 0.0547 |
| 0.1 | 1.0 | 34 | -0.2803 | 0.1083 | -0.3065 | 0.0779 |
| 0.25 | 0.25 | 34 | -0.2394 | 0.1728 | -0.2672 | 0.1265 |
| 0.5 | 20.0 | 34 | -0.2344 | 0.1820 | -0.2518 | 0.1509 |
| 1.0 | 50.0 | 34 | -0.2298 | 0.1912 | -0.2441 | 0.1641 |

方法二结论：

- 最优 Spearman 组合为 `lambda1=0, lambda2=0.1`：`r=-0.5184, p=0.0017`，比基线略好。
- 同一组合的 Pearson 为 `r=-0.4353, p=0.0101`，略弱于基线；Pearson 最优仍是 `lambda1=lambda2=0`。
- `lambda1>0` 后，累积项很快削弱相关性，因此当前数据不支持同时给累积损失较大权重。
- 合并组最优 Spearman 为 `r=-0.1749, p=0.1052`，非 TIPS 组最优 Spearman 为 `r=0.1295, p=0.3553`，均不显著。

TIPS 组排名汇总：

| 排名依据 | 名次 | lambda1 | lambda2 | n_used | r | p |
|---|---:|---:|---:|---:|---:|---:|
| Pearson | 1 | 0.0 | 0.0 | 34 | <span style="text-decoration: underline double;">-0.4372*</span> | <span style="text-decoration: underline double;">0.0097</span> |
| Pearson | 2 | 0.0 | 0.005 | 34 | <span style="text-decoration: underline;">-0.4372*</span> | <span style="text-decoration: underline;">0.0097</span> |
| Spearman | 1 | 0.0 | 0.1 | 34 | <span style="text-decoration: underline double;">-0.5184*</span> | <span style="text-decoration: underline double;">0.0017</span> |
| Spearman | 2 | 0.0 | 0.5 | 34 | <span style="text-decoration: underline;">-0.5183*</span> | <span style="text-decoration: underline;">0.0017</span> |

### 两种改进的比较

方法一的 `alpha=1` 能同时小幅改善 TIPS 组 Pearson 和 Spearman，`alpha=2` 对 Pearson 改善最多，但 Spearman 略弱于基线。方法二的 `lambda1=0, lambda2=0.1` 对 Spearman 的改善更明显，但 Pearson 略有下降。因此，若后续准备独立验证，可优先比较方法一 `alpha=1`、方法一 `alpha=2` 和方法二 `lambda1=0, lambda2=0.1`，而不应直接在当前数据上选定唯一最优参数。

## 两种改进用于 `R_collateral`

侧枝仍只统计 LGV 和 PGV。每条侧枝独立计算改进后的有效阻力，最后按并联公式合成 `R_collateral`。现有侧枝入口损失项固定为主结果使用的 `lambda_coll=1`，因此两种方法的零系数结果都严格等于当前 `R_collateral` 基线。TIPS 组仍只有 1 例存在自然侧枝，无法计算相关性；下面只比较合并组和非 TIPS 组。

### 侧枝改进一：阻力贡献峰值放大

单条侧枝使用：

```math
R_{\text{eff,coll}}^{(c)}(\alpha)=
\left(R_{\text{visc}}^{(c)}+\alpha R_{\text{visc,max}}^{(c)}L_{\text{stenotic}}^{(c)}\right)
\left(1+\lambda_{\text{coll}}\zeta_{\text{entrance}}^{(c)}\right)
```

`L_stenotic` 继续使用峰值 80% 连续区间定义，扫描的 18 个 `alpha` 与 `R_total` 实验完全相同。

排名汇总：

| 分组 | 排名依据 | 名次 | alpha | n_used | r | p |
|---|---|---:|---:|---:|---:|---:|
| 合并 | Pearson | 1 | 0.0 | 42 | <span style="text-decoration: underline double;">-0.2322</span> | <span style="text-decoration: underline double;">0.1390</span> |
| 合并 | Pearson | 2 | 0.005 | 42 | <span style="text-decoration: underline;">-0.2321</span> | <span style="text-decoration: underline;">0.1390</span> |
| 合并 | Spearman | 1 | 0.0 | 42 | <span style="text-decoration: underline double;">-0.4295*</span> | <span style="text-decoration: underline double;">0.0045</span> |
| 合并 | Spearman | 2 | 0.005 | 42 | <span style="text-decoration: underline;">-0.4295*</span> | <span style="text-decoration: underline;">0.0045</span> |
| 非 TIPS | Pearson | 1 | 0.0 | 41 | <span style="text-decoration: underline double;">-0.2442</span> | <span style="text-decoration: underline double;">0.1240</span> |
| 非 TIPS | Pearson | 2 | 0.005 | 41 | <span style="text-decoration: underline;">-0.2441</span> | <span style="text-decoration: underline;">0.1240</span> |
| 非 TIPS | Spearman | 1 | 0.0 | 41 | <span style="text-decoration: underline double;">-0.4353*</span> | <span style="text-decoration: underline double;">0.0045</span> |
| 非 TIPS | Spearman | 2 | 0.005 | 41 | <span style="text-decoration: underline;">-0.4353*</span> | <span style="text-decoration: underline;">0.0045</span> |

其中 `alpha=0、0.005、0.01、0.025` 的 Spearman 结果完全相同，表中按系数较小优先的规则展示前两项。Pearson 在 `alpha=0` 最强，随着 `alpha` 增大总体减弱。因此侧枝数据不支持使用阻力峰值放大，主结果应继续取 `alpha=0`。

### 侧枝改进二：局部损失累积值与峰值并用

为了保留已有入口损失基线，单条侧枝使用加性局部损失形式：

```math
R_{\text{eff,coll}}^{(c)}(\lambda_1,\lambda_2)=
R_{\text{visc}}^{(c)}
\left(
1+\lambda_{\text{coll}}\zeta_{\text{entrance}}^{(c)}
+\lambda_1\Phi_{\text{local,sum}}^{(c)}
+\lambda_2\Phi_{\text{local,max}}^{(c)}
\right)
```

保持 `lambda2 >= lambda1`，共扫描 129 组组合。排名汇总如下：

| 分组 | 排名依据 | 名次 | lambda1 | lambda2 | n_used | r | p |
|---|---|---:|---:|---:|---:|---:|---:|
| 合并 | Pearson | 1 | 0.0 | 100.0 | 42 | <span style="text-decoration: underline double;">-0.3787*</span> | <span style="text-decoration: underline double;">0.0134</span> |
| 合并 | Pearson | 2 | 0.005 | 100.0 | 42 | <span style="text-decoration: underline;">-0.3785*</span> | <span style="text-decoration: underline;">0.0134</span> |
| 合并 | Spearman | 1 | 0.075 | 10.0 | 42 | <span style="text-decoration: underline double;">-0.4616*</span> | <span style="text-decoration: underline double;">0.0021</span> |
| 合并 | Spearman | 2 | 0.05 | 10.0 | 42 | <span style="text-decoration: underline;">-0.4604*</span> | <span style="text-decoration: underline;">0.0022</span> |
| 非 TIPS | Pearson | 1 | 0.0 | 100.0 | 41 | <span style="text-decoration: underline double;">-0.3947*</span> | <span style="text-decoration: underline double;">0.0106</span> |
| 非 TIPS | Pearson | 2 | 0.005 | 100.0 | 41 | <span style="text-decoration: underline;">-0.3946*</span> | <span style="text-decoration: underline;">0.0107</span> |
| 非 TIPS | Spearman | 1 | 0.025 | 5.0 | 41 | <span style="text-decoration: underline double;">-0.4783*</span> | <span style="text-decoration: underline double;">0.0016</span> |
| 非 TIPS | Spearman | 2 | 0.0 | 5.0 | 41 | <span style="text-decoration: underline;">-0.4754*</span> | <span style="text-decoration: underline;">0.0017</span> |

非 TIPS 组的 `(lambda1=0.005, lambda2=5)` 与第二名 Spearman 结果完全相同。两组的 Pearson 最优都出现在 `lambda2=100`，但这是扫描边界，说明当前范围内 Pearson 仍随峰值权重增大，不能据此认定 `100` 是稳定最优值。Spearman 最优则位于中间范围 `lambda2=5~10`。

作为兼顾两组、两种相关方法且形式更简单的候选，`lambda1=0, lambda2=10` 的结果为：

| 分组 | n_used | Pearson r | Pearson p | Spearman r | Spearman p |
|---|---:|---:|---:|---:|---:|
| 合并 | 42 | -0.3673* | 0.0167 | -0.4565* | 0.0024 |
| 非 TIPS | 41 | -0.3836* | 0.0133 | -0.4746* | 0.0017 |

与零系数基线相比，方法二不仅保留了显著 Spearman 负相关，还使 Pearson 从不显著变为显著。不过这些系数是在同一数据集上扫描得到的，`lambda2=5、10、100` 都只能作为独立数据验证的候选，不能直接作为最终固定参数。

## 结果文件

- `result/features.csv`：每例 6 个指标。
- `result/feature_extraction_report.json`：每例中间计算项、有效长度和跳过原因。
- `result/correlation_tables.csv`：合并、TIPS、非 TIPS 三组相关性长表。
- `result/correlation_metrics.json`：机器可读相关性结果。
- `result/local_loss_tuning.csv`：`R_total` 局部损失系数完整调参记录。
- `result/collateral_loss_tuning.csv`：`R_collateral` 侧枝入口损失系数完整调参记录。
- `result/resistance_peak_tuning.csv`：改进一的 `alpha` 完整扫描结果。
- `result/local_loss_sum_max_tuning.csv`：改进二的 `lambda1/lambda2` 全部 129 组扫描结果。
- `result/collateral_resistance_peak_tuning.csv`：侧枝改进一的 `alpha` 完整扫描结果。
- `result/collateral_local_loss_sum_max_tuning.csv`：侧枝改进二的 `lambda1/lambda2` 全部 129 组扫描结果。
