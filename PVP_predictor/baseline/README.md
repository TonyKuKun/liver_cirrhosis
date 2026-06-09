# PVP Baselines

本目录保存传统机器学习 baseline。2026-06-09 已使用主模型同一份 `splits.json` 重新运行，旧版备份模型目录已删除。

## 当前结果

- 目录：`baseline/runs/final_20260609_baselines`
- 最优 baseline：`physics/adaboost`
- 指标：MAE 3.420, RMSE 4.286, R2 0.531

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

## 重新运行

```powershell
conda run -n pytorch python baseline/run_baselines.py ^
  --data_root "F:\PCG data\dataset\test4all_sample" ^
  --split_json runs\final_20260609_pvp_l2_shunt\splits.json ^
  --out_dir baseline\runs\final_20260609_baselines ^
  --n_points 200 --seed 40
```

输出：

- `summary.csv`
- `summary.json`
- `oof_predictions.csv`
- `per_group_summary.json`
- `feature_importance.csv`
