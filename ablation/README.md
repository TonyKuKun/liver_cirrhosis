# Ablation Experiments

This folder contains the final PVP ablation utilities.

## Scripts

```text
ablations.py
report.py
```

## Current Design

The default ablation reference is the current best PVP model:

```text
8-vessel layout
liver/spleen volumes as global features
no organ flow scaling
pure L2/MSE loss
```

The split-flow physics loss is kept as an explicit control. The final retained
version is the narrow core-confluence constraint:

```text
--lambda_press 0.03 --split_loss_mode core_confluence
```

Other physics losses are set to zero in the final ablation suite.

## Run

Loss ablation:

```bash
conda run -n pytorch python ablation/ablations.py --suite loss --stage full --out_root ablation/runs/loss_ablation_core_split_20260607 --full_n_folds 5 --full_epochs 300 --seed 40 --force
```

Full architecture ablation template:

```bash
conda run -n pytorch python ablation/ablations.py --suite all --stage full --out_root ablation/runs/final_20260607 --full_n_folds 5 --full_epochs 300 --seed 40 --force
```

## Retained Results

Primary loss result:

```text
ablation/runs/loss_ablation_core_split_20260607/full/comparison.csv
ablation/runs/loss_ablation_core_split_20260607/full/comparison.json
ablation/runs/loss_ablation_core_split_20260607/full/analysis.md
```

Architecture diagnostic result:

```text
ablation/runs/arch_ablation_l2_fullsplit_20260607/full/comparison.csv
ablation/runs/arch_ablation_l2_fullsplit_20260607/full/comparison.json
ablation/runs/arch_ablation_l2_fullsplit_20260607/full/analysis.md
```

Latest best:

```text
L2 only + organ global
MAE  2.8098
RMSE 3.8433
R2   0.6142
```
