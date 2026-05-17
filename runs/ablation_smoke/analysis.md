# Ablation Analysis

Full model: MAE 12.071, RMSE 13.647, R2 -3.729.

## Largest Accuracy Drops
- No completed ablation was worse than the full model.

## Ablations That Improve MAE
- module_no_aux: delta MAE -0.002. This suggests the removed component may be noisy, over-regularizing, or acting as a shortcut.
- loss_main_only: delta MAE -0.000. This suggests the removed component may be noisy, over-regularizing, or acting as a shortcut.

## Interpretation Rules
- If removing a module increases MAE, that module is carrying useful predictive signal.
- If removing a module improves MAE, inspect whether it is overfitting the 62-sample dataset.
- If `module_no_aux` barely changes performance, the model is learning mainly geometry/physics; if it collapses, it depends heavily on system/status shortcuts.
- If `module_no_branch_embed` barely changes performance, pointwise learned geometry is not adding much beyond hand-coded physics signals.
- If `loss_main_only` improves performance, the physics losses may be too strong for the current data scale.
