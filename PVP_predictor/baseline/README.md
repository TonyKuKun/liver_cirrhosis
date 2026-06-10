# PVP Baselines

Traditional machine-learning baselines were rerun on 2026-06-10 with the same `splits.json` as the final neural model.

## Current Results

- Directory: `baseline/runs/final_20260610_baselines`
- Best baseline: `physics/adaboost`
- Metrics: MAE 3.420, RMSE 4.286, R2 0.531

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

## Re-run

Run from `PVP_predictor`:

```powershell
conda run -n pytorch python baseline/run_baselines.py ^
  --data_root "F:\PCG data\dataset\test4all_sample" ^
  --split_json runs\final_20260610_pvp_l2_shunt\splits.json ^
  --out_dir baseline\runs\final_20260610_baselines ^
  --n_points 200 --seed 40
```

Outputs:

- `summary.csv`
- `summary.json`
- `oof_predictions.csv`
- `per_group_summary.json`
- `feature_importance.csv`
