# Ablation Experiments

This folder runs model and loss ablations for the PVP predictor.  The point is
to answer why the full model is or is not better than strong traditional
baselines.

## Run All Ablations

```bash
conda run -n pytorch python ablation/run_ablations.py \
  --data_root "F:\PCG data\dataset\test4all_sample" \
  --out_root runs/ablations_v1
```

This trains one folder per variant and writes:

- `manifest.json`: exact commands and hypotheses;
- `comparison.csv`: MAE/RMSE/R2 for each variant and deltas vs `full_model`;
- `comparison.json`: machine-readable version of the comparison table;
- `analysis.md`: short automatic interpretation.

## Fast Smoke Run

```bash
conda run -n pytorch python ablation/run_ablations.py \
  --out_root runs/ablation_smoke \
  --variants full_model module_no_aux loss_main_only \
  --n_folds 2 \
  --epochs 1 \
  --patience 1 \
  --print_every 1
```

## Main Questions

- `module_no_aux`: is the model mainly using TIPS/status/system shortcuts?
- `module_no_branch_embed`: do learned pointwise geometry embeddings help?
- `module_no_physics_baseline`: does the Poiseuille pressure-drop anchor help?
- `module_no_q_scale`: do organ-volume flow priors matter?
- `module_no_flow_features`: do learned flow fractions and junction features matter?
- `loss_main_only`: are physics losses helping or over-regularizing this small dataset?
- `loss_no_tail_weight` and `train_no_extreme_sampler`: are tail-focused tricks improving high/low PVP behavior?

