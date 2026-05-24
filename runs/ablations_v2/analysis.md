# Ablation Analysis

## Largest Accuracy Drops
- No completed ablation was worse than the full model.

## Ablations That Improve MAE
- No completed ablation improved MAE over the full model.

## Interpretation Rules
- If removing a module increases MAE, that module is carrying useful predictive signal.
- If removing a module improves MAE, inspect whether it is overfitting the 62-sample dataset.
- If `module_no_aux` barely changes performance, the model is learning mainly geometry/physics; if it collapses, it depends heavily on system/status shortcuts.
- If `module_no_branch_embed` barely changes performance, pointwise learned geometry is not adding much beyond hand-coded physics signals.
- If `loss_main_only` improves performance, the physics losses may be too strong for the current data scale.
