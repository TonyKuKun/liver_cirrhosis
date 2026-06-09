# PVP Predictor

本目录是当前保留的门静脉压力 PVP 预测模型。2026-06-09 已重新跑完主模型、传统机器学习 baseline、完整消融和架构图；文档和结果均按当前单任务 PVP 回归模型重写。

## 当前模型

- 输入：8 条血管分支 MPV、SV、SMV、LPV、RPV、TIPS、LGV、PGV。
- 几何：默认使用稳定截面特征，排除不可靠原始长度和噪声计数特征。
- 全局状态：肝体积、脾体积、脾肝体积比作为病人级上下文。
- 模型头：只保留一个 PVP prediction head。
- Loss：只保留 L2/MSE PVP loss 和可选分流 loss。默认复现实验使用 `lambda_shunt=0.03`、`split_loss_mode=core_confluence`。
- 输出：`oof_predictions.csv` 和 `oof_predictions.xlsx` 使用同一组字段，Excel 与 CSV 结果一致。

## 2026-06-09 主模型结果

数据集为 `F:\PCG data\dataset\test4all_sample`，排除 17 个 `00` 前缀样本后共 72 个有效样本，subject-level 5-fold，seed 40。

- 主模型：MAE 3.153 +/- 0.540, RMSE 3.986, R2 0.585
- OOF overall：MAE 3.169, RMSE 4.051, bias -0.002, n=72
- 结果目录：`runs/final_20260609_pvp_l2_shunt`

| fold | train | val | best_epoch | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 57 | 15 | 52 | 3.480 | 4.117 | 0.549 |
| 1 | 58 | 14 | 72 | 2.487 | 2.993 | 0.753 |
| 2 | 58 | 14 | 21 | 2.643 | 3.593 | 0.638 |
| 3 | 57 | 15 | 6 | 3.956 | 4.871 | 0.397 |
| 4 | 58 | 14 | 60 | 3.197 | 4.354 | 0.589 |

## Baseline 对照

传统机器学习 baseline 使用同一份 `splits.json`。本次最优 baseline 是 `physics/adaboost`，MAE 3.420、RMSE 4.286、R2 0.531。

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

完整结果见：

- `baseline/runs/final_20260609_baselines/summary.csv`
- `baseline/runs/final_20260609_baselines/summary.json`
- `baseline/runs/final_20260609_baselines/oof_predictions.csv`

## 消融结论

参考配置 `full_model` 是当前默认复现实验：8-vessel + organ global features + single PVP head + L2 + core_confluence shunt loss。完整消融共 12 个配置，均已在 5-fold 下跑完。

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

这次重新运行后，`loss_l2_plus_full_split` 在 MAE/RMSE/R2 上最好，`loss_l2_only` 也优于当前参考配置；这说明分流约束需要在论文中按负/混合结果谨慎报告。结构模块里，GlobalFlowCorrector 和 FlowGraphRefiner 对当前参考配置有帮助；learnable physics 参数在本次 seed 下弱于 fixed physics 参数，需要作为后续重点复核项。

完整消融结果见：

- `ablation/runs/final_20260609/full/comparison.csv`
- `ablation/runs/final_20260609/full/comparison.json`
- `ablation/runs/final_20260609/full/analysis.md`

## 复现实验命令

主模型：

```powershell
conda run -n pytorch python train.py ^
  --data_root "F:\PCG data\dataset\test4all_sample" ^
  --out_dir runs\final_20260609_pvp_l2_shunt ^
  --n_folds 5 --epochs 300 --seed 40 ^
  --lambda_shunt 0.03 --split_loss_mode core_confluence
```

baseline：

```powershell
conda run -n pytorch python baseline/run_baselines.py ^
  --data_root "F:\PCG data\dataset\test4all_sample" ^
  --split_json runs\final_20260609_pvp_l2_shunt\splits.json ^
  --out_dir baseline\runs\final_20260609_baselines ^
  --n_points 200 --seed 40
```

消融：

```powershell
conda run -n pytorch python ablation/ablations.py ^
  --suite all --stage full ^
  --out_root ablation/runs/final_20260609 ^
  --full_n_folds 5 --full_epochs 300 --seed 40 --force
```

架构图：

```powershell
conda run -n pytorch python docs/draw_model_architecture.py
```

## 目录

- `dataset.py`：血管 STL、CenterlinePoints 和肝脾体积读取。
- `model.py`：当前 PVP 网络，只保留单一 PVP 预测头。
- `loss.py`：L2/MSE 与分流 loss。
- `train.py`：5-fold 训练、预测 CSV/XLSX 和 summary 输出。
- `baseline/`：传统机器学习 baseline。
- `ablation/`：结构、输入布局和 loss 消融。
- `docs/figures/model_architecture.png`：本次重新生成的架构图。
- `FINAL_20260609_RESULTS_ANALYSIS.md`：中文最终分析。
