# Retained Experiment Results

Only the final retained PVP result folders are kept here.

## Primary Loss Decision

```text
loss_ablation_core_split_20260607/full
```

This is the latest loss ablation after narrowing the split-flow constraint to
the MPV/SMV/SV core confluence.

```text
L2 only:                  MAE 2.8098, RMSE 3.8433, R2 0.6142
L2 + core split loss:     MAE 2.8865, RMSE 3.9411, R2 0.5945
L2 + previous full split: MAE 3.0217, RMSE 4.1400, R2 0.5560
```

Conclusion: pure L2/MSE with liver and spleen global features remains the
final default.

## Architecture Diagnostic

```text
arch_ablation_l2_fullsplit_20260607/full
```

This folder keeps the prior architecture screen. It is retained for module
diagnostics and historical comparison.

Key finding: liver/spleen global features, GlobalFlowCorrector, and
PhysicsResidualNet were useful in that screen; GNN and broad split loss were
not reliable default improvements.
