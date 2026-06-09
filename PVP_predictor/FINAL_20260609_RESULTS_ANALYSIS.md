# PVP Predictor 2026-06-09 最终实验分析

本文件完全基于 2026-06-09 重新运行的结果。

## 实验范围

- 主模型：`runs/final_20260609_pvp_l2_shunt`
- baseline：`baseline/runs/final_20260609_baselines`
- 消融：`ablation/runs/final_20260609/full`
- 架构图：`docs/figures/model_architecture.png`

当前模型只做 PVP 回归，只有一个 PVP 预测头。训练目标只保留 L2/MSE 和分流 loss。

## 主模型表现

参考配置为 8-vessel + organ global features + GlobalFlowCorrector + FlowGraphRefiner + single PVP head + L2 + core_confluence shunt loss。

- 5-fold：MAE 3.153 +/- 0.540, RMSE 3.986, R2 0.585
- OOF：MAE 3.169, RMSE 4.051, bias -0.002
- CSV/XLSX：`oof_predictions.csv` 与 `oof_predictions.xlsx` 字段一致。

| fold | train | val | best_epoch | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 57 | 15 | 52 | 3.480 | 4.117 | 0.549 |
| 1 | 58 | 14 | 72 | 2.487 | 2.993 | 0.753 |
| 2 | 58 | 14 | 21 | 2.643 | 3.593 | 0.638 |
| 3 | 57 | 15 | 6 | 3.956 | 4.871 | 0.397 |
| 4 | 58 | 14 | 60 | 3.197 | 4.354 | 0.589 |

## baseline 排名

| baseline | features | n | MAE | RMSE | R2 | bias |
| --- | --- | --- | --- | --- | --- | --- |
| physics/adaboost | 32 | 72 | 3.420 | 4.286 | 0.531 | -0.212 |
| physics/random_forest | 32 | 72 | 3.518 | 4.352 | 0.516 | -0.112 |
| physics/extra_trees | 32 | 72 | 3.532 | 4.372 | 0.512 | -0.378 |
| combined/elasticnet_cv | 1088 | 72 | 3.580 | 4.418 | 0.502 | 0.318 |
| combined/extra_trees | 1088 | 72 | 3.672 | 4.515 | 0.480 | -0.122 |
| combined/hist_gradient_boosting | 1088 | 72 | 3.677 | 4.484 | 0.487 | -0.097 |
| geometry/extra_trees | 897 | 72 | 3.685 | 4.550 | 0.472 | -0.054 |
| physics/hist_gradient_boosting | 32 | 72 | 3.699 | 4.645 | 0.449 | -0.086 |
| geometry/elasticnet_cv | 897 | 72 | 3.704 | 4.480 | 0.488 | 0.264 |
| aux/extra_trees | 159 | 72 | 3.733 | 4.536 | 0.475 | 0.119 |

主模型 MAE 3.153，优于最好的传统 baseline `physics/adaboost` 的 MAE 3.420。baseline 仍可作为论文中的传统机器学习对照。

## 消融结果

| variant | category | MAE | RMSE | R2 | dMAE | direction |
| --- | --- | --- | --- | --- | --- | --- |
| full_model | reference | 3.153 | 3.986 | 0.585 |  | reference |
| no_organ_global_features | module | 3.184 | 4.030 | 0.568 | 0.031 | reference_consistently_better |
| no_global_flow_corrector | module | 3.473 | 4.562 | 0.460 | 0.320 | reference_consistently_better |
| no_flow_graph | module | 3.229 | 4.304 | 0.517 | 0.076 | reference_consistently_better |
| fixed_physics_params | module | 2.954 | 3.850 | 0.612 | -0.199 | variant_consistently_better |
| all_profile_channels | geometry | 3.212 | 4.237 | 0.527 | 0.060 | reference_consistently_better |
| use_unreliable_raw_lengths | geometry | 3.029 | 4.027 | 0.582 | -0.124 | mixed_or_noisy |
| six_vessel_layout | layout | 3.141 | 4.174 | 0.552 | -0.012 | mixed_or_noisy |
| three_vessel_layout | layout | 3.198 | 4.057 | 0.567 | 0.045 | reference_consistently_better |
| loss_l2_only | loss | 2.969 | 3.916 | 0.601 | -0.184 | variant_consistently_better |
| loss_l2_plus_core_split | loss | 3.153 | 3.986 | 0.585 | 0.000 | mixed_or_noisy |
| loss_l2_plus_full_split | loss | 2.934 | 3.891 | 0.606 | -0.219 | variant_consistently_better |

### 结构模块

GlobalFlowCorrector、FlowGraphRefiner 和默认筛选后的 profile channels 对参考配置有正贡献。organ global features 的收益很小但方向仍支持保留。fixed physics 参数在本次 5-fold 中优于可学习 physics 参数，这一点和模型直觉不完全一致，应在论文中作为复核点，而不是直接过度解释。

### Loss

本次 loss 消融显示：

| loss variant | MAE | RMSE | R2 | dMAE |
| --- | --- | --- | --- | --- |
| loss_l2_only | 2.969 | 3.916 | 0.601 | -0.184 |
| loss_l2_plus_core_split | 3.153 | 3.986 | 0.585 | 0.000 |
| loss_l2_plus_full_split | 2.934 | 3.891 | 0.606 | -0.219 |

`loss_l2_plus_full_split` 本次分数最好，`loss_l2_only` 也优于参考配置。当前代码仍只保留 L2 和分流 loss 两类目标；分流 loss 的使用需要按复现实验结果谨慎选择，不能再引用旧版结论。

## 结论

1. 当前 PVP_predictor 已整理为单任务 PVP 回归：一个 PVP head，loss 只剩 L2 和分流 loss。
2. 主模型重新运行结果为 MAE 3.153 +/- 0.540，OOF bias 接近 0。
3. 传统 baseline 最优为 `physics/adaboost`，MAE 3.420。
4. 消融最佳单项为 `loss_l2_plus_full_split`，MAE 2.934。
5. 所有保留结果均来自 2026-06-09 重新运行。
