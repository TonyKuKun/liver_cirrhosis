# 2026-06-07 最终实验结果与消融分析

## 实验设置

- 数据：`F:\PCG data\dataset\test4all_sample`
- 有效样本：72 scans / 50 subjects
- 验证：subject-level 5-fold
- PVP 评价指标：MAE、RMSE、R2
- 主监督损失：严格使用 L2/MSE，即 `mean((pred - label_norm)^2)`
- `--disable_physics_losses`：关闭所有物理约束和 spread 项，只保留纯 MSE

本轮消融已按“一次只改变一个变量”重新整理，分为两类：

1. 模型消融：以最终 8 血管模型为 reference，逐项删除或改变一个模块。
2. Loss 消融：以纯 MSE 为 reference，逐项加入一个简化物理约束或一个明确的约束族。

结果文件：

- 模型消融：`ablation/runs/final_20260607_onefactor_model/full/comparison.csv`
- Loss 消融：`ablation/runs/final_20260607_onefactor_loss/full/comparison.csv`
- baseline：`baseline/runs/final_20260607_baselines_72_seed40/summary.json`

## 最终 PVP 模型

当前推荐最终版本是 8 血管方案：

```text
8-vessel layout
+ corrected PGV-SV graph
+ CenterlinePoints position-aware GNN
+ Q / flow as intermediate state
+ GlobalFlowCorrector
+ PhysicsResidualNet
+ learnable physics parameters
+ core geometry channels only
+ simplified physics loss
- hard OrganFlowScaleNet disabled by default
- unreliable raw branch lengths not used by default
```

核心几何通道：

```text
area, hydraulic_diameter, inscribed_radius, curvature,
solidity, circularity, dA_ds_norm
```

不默认使用：

```text
torsion, perimeter, r_insc_to_r_eq_ratio, n_components
```

## Baseline 对照

| Baseline | Features | MAE | RMSE | R2 |
|---|---:|---:|---:|---:|
| physics + AdaBoost | 32 | 3.420 | 4.286 | 0.531 |
| physics + RandomForest | 32 | 3.518 | 4.352 | 0.516 |
| physics + ExtraTrees | 32 | 3.532 | 4.372 | 0.512 |
| combined + ElasticNetCV | 1088 | 3.641 | 4.481 | 0.487 |

最终神经网络 PVP 模型优于最强 baseline：

```text
Final model:       MAE 2.873, RMSE 3.794, R2 0.618
Best baseline:     MAE 3.420, RMSE 4.286, R2 0.531
Absolute gain:     MAE -0.547, RMSE -0.492, R2 +0.087
```

## 模型消融

Reference：`full_model`

| Variant | 改变的变量 | MAE | RMSE | R2 | 结论 |
|---|---|---:|---:|---:|---|
| full_model | none | 2.873 | 3.794 | 0.618 | 最终 reference |
| with_organ_flow_scale | 加入硬器官流量缩放 | 2.883 | 3.805 | 0.616 | 变化极小，不建议默认硬启用 |
| no_global_flow_corrector | 删除全局修正 | 3.456 | 4.530 | 0.467 | 明确有用，必须保留 |
| no_flow_graph | 删除 GNN | 3.352 | 4.416 | 0.496 | 明确有用，必须保留 |
| no_physics_residual | 删除 residual head | 3.031 | 4.074 | 0.560 | 有用，建议保留 |
| fixed_physics_params | 固定物理参数 | 2.901 | 3.998 | 0.584 | 可学习物理参数更好 |
| all_profile_channels | 加入全部剖面通道 | 3.244 | 4.517 | 0.474 | 噪声增加，不建议 |
| use_unreliable_raw_lengths | 使用不可靠原始长度 | 2.962 | 3.953 | 0.588 | 不建议默认使用 |
| six_vessel_layout | 合并代偿/TIPS 为 6 血管 | 3.069 | 4.025 | 0.571 | 不如 8 血管 |
| three_vessel_layout | 代偿+MPV+SV 三血管 | 3.088 | 3.966 | 0.595 | 更紧凑但 MAE 不如 8 血管 |

模型结论：

- 8 血管方案仍然最优。
- GlobalFlowCorrector 和位置感知 GNN 是最重要的两个结构模块。
- PhysicsResidualNet 在严格 MSE 训练下是有用的，之前“去掉 residual 更好”的结论被本轮严格消融推翻。
- 6 血管和 3 血管能减少零输入，但损失了解剖细节，最终不如 8 血管。
- 全部几何通道和不可靠原始长度都会带来噪声。

## Loss 消融

Reference：`loss_mse_only`

| Variant | Loss | MAE | RMSE | R2 | 结论 |
|---|---|---:|---:|---:|---|
| loss_mse_only | MSE | 2.883 | 3.967 | 0.589 | 纯监督 baseline |
| loss_mse_plus_continuity | MSE + 管内流量连续 | 2.883 | 3.967 | 0.589 | 基本无变化 |
| loss_mse_plus_split | MSE + 分流守恒 | 2.873 | 3.794 | 0.618 | 有正向作用 |
| loss_mse_plus_pressure_mono | MSE + 压降单调 | 2.883 | 3.967 | 0.589 | 基本无变化 |
| loss_mse_plus_flow_conservation | MSE + 连续 + 分流 | 2.873 | 3.794 | 0.618 | 主要收益来自分流 |
| loss_mse_plus_all_simple_physics | MSE + 全部简化物理约束 | 2.873 | 3.794 | 0.618 | 最终默认 |

Loss 结论：

- 当前真正有效的简化物理约束是分流/汇合处流量守恒。
- `area * velocity` 管内连续项几乎不改变结果，因为模型内部速度本来由 `Q / area` 得到，这个约束接近恒成立。
- 压力单调项目前没有带来独立增益，可能因为预测头主要由 flow/global/residual 特征驱动。
- 最终保留全部简化物理 loss，主要原因是它不伤害性能，并提供更好的流量一致性和可解释性。

## 当前物理约束

中间状态统一选择流量 `Q`，速度只作为物理计算派生量：

```text
velocity = Q / area
```

保留的简化物理约束：

1. 管内流量连续：同一血管沿程 `area * velocity` 应尽量稳定。
2. 分流守恒：MPV 汇合处满足 `Q_MPV ~= Q_SMV + Q_SV`，代偿/TIPS 作为分流路径参与约束。
3. 压降单调：累计压力代理沿程不应反向下降。

最终结果显示，分流守恒是最有实际贡献的约束。

## Radiology 投稿前建议

模型还可以这样加强：

1. 做 repeated 5-fold CV，至少 5 个 seed，报告 mean 和 95% CI。
2. 对 PVP 的 MAE/RMSE/R2 做 paired bootstrap CI，并和 physics + AdaBoost 做配对比较。
3. 单独报告 pre-TIPS、post-TIPS、LGV、PGV、TIPS 亚组。
4. 对 CenterlinePoints 做质量分层，验证位置感知 GNN 是否在中心线质量高的样本中收益更大。
5. 对 CSPH 不建议现在宣称超过文献，应该作为 secondary endpoint，重点强调连续 PVP 估计和可解释血流中间量。

当前论文叙事建议：

```text
An anatomy-aware, physics-informed portal venous graph model improves continuous PVP estimation over conventional vascular-feature baselines. The main gain comes from global flow correction, CenterlinePoints-aware graph refinement, and flow-split conservation, while compact vessel layouts and noisy geometric channels reduce performance.
```
