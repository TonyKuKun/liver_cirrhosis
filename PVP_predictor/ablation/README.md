# PVP Predictor Ablations

This directory contains the module, geometry, layout, and loss ablations for the final 2026-06-10 single-head PVP model.

## Current Full Results

- Directory: `ablation/runs/final_20260610/full`
- Variants: 14
- Reference: `full_model`
- Reference metrics: MAE 2.685, RMSE 3.605, R2 0.643
- Best configuration: `full_model`

| variant | category | MAE | RMSE | R2 | dMAE |
| --- | --- | --- | --- | --- | --- |
| full_model | reference | 2.685 | 3.605 | 0.643 | 0.000 |
| loss_l2_plus_core_split | loss | 2.685 | 3.605 | 0.643 | 0.000 |
| fixed_physics_params | module | 2.696 | 3.616 | 0.644 | 0.011 |
| loss_l2_only | loss | 2.700 | 3.618 | 0.641 | 0.014 |
| loss_l2_plus_full_split | loss | 2.703 | 3.619 | 0.641 | 0.018 |
| with_physics_residual | module | 2.809 | 3.837 | 0.615 | 0.123 |
| use_unreliable_raw_lengths | geometry | 2.910 | 3.977 | 0.586 | 0.224 |
| all_profile_channels | geometry | 2.930 | 4.208 | 0.542 | 0.244 |
| no_dropout_regularizer | module | 2.942 | 3.721 | 0.634 | 0.256 |
| six_vessel_layout | layout | 3.026 | 3.984 | 0.592 | 0.341 |
| no_organ_global_features | module | 3.104 | 4.093 | 0.548 | 0.419 |
| three_vessel_layout | layout | 3.132 | 4.032 | 0.580 | 0.446 |
| no_flow_graph | module | 3.171 | 4.455 | 0.490 | 0.486 |
| no_global_flow_corrector | module | 3.479 | 4.536 | 0.458 | 0.794 |

## Re-run

Run from `PVP_predictor`:

```powershell
conda run -n pytorch python ablation/ablations.py ^
  --suite all --stage full ^
  --out_root ablation/runs/final_20260610 ^
  --full_n_folds 5 --full_epochs 300 --seed 40 --force
```

Output files:

- `ablation/runs/final_20260610/full/manifest.json`
- `ablation/runs/final_20260610/full/comparison.csv`
- `ablation/runs/final_20260610/full/comparison.json`
- `ablation/runs/final_20260610/full/analysis.md`
