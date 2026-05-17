# Ablation Analysis

Full model: MAE 3.249, RMSE 4.164, R2 0.543.

## Largest Accuracy Drops
- train_no_extreme_sampler: delta MAE +0.191, delta RMSE +0.202, delta R2 -0.046. Component: Extreme-value oversampling.
- module_no_gnn: delta MAE +0.114, delta RMSE +0.076, delta R2 -0.028. Component: VesselGraphNet.
- module_no_branch_embed: delta MAE +0.100, delta RMSE +0.208, delta R2 -0.079. Component: Learned pointwise geometry embeddings.
- module_no_q_scale: delta MAE +0.083, delta RMSE +0.209, delta R2 -0.063. Component: SplenicFlowEstimator.
- loss_no_physio: delta MAE +0.033, delta RMSE +0.254, delta R2 -0.072. Component: WSS/Re physiological range loss.
- module_no_residual: delta MAE +0.018, delta RMSE +0.057, delta R2 -0.018. Component: PhysicsResidualNet.
- loss_no_press: delta MAE +0.000, delta RMSE +0.000, delta R2 -0.000. Component: Pressure consistency loss.
- loss_no_spread: delta MAE +0.000, delta RMSE -0.000, delta R2 +0.000. Component: Anti-shrinkage spread loss.

## Ablations That Improve MAE
- loss_main_only: delta MAE -0.098. This suggests the removed component may be noisy, over-regularizing, or acting as a shortcut.
- module_no_physics_baseline: delta MAE -0.081. This suggests the removed component may be noisy, over-regularizing, or acting as a shortcut.
- module_no_flow_features: delta MAE -0.047. This suggests the removed component may be noisy, over-regularizing, or acting as a shortcut.
- loss_no_residual_penalty: delta MAE -0.034. This suggests the removed component may be noisy, over-regularizing, or acting as a shortcut.
- module_data_mlp_only: delta MAE -0.025. This suggests the removed component may be noisy, over-regularizing, or acting as a shortcut.
- loss_no_murray: delta MAE -0.023. This suggests the removed component may be noisy, over-regularizing, or acting as a shortcut.
- module_no_aux: delta MAE -0.004. This suggests the removed component may be noisy, over-regularizing, or acting as a shortcut.
- loss_no_mono: delta MAE -0.000. This suggests the removed component may be noisy, over-regularizing, or acting as a shortcut.

## Interpretation Rules
- If removing a module increases MAE, that module is carrying useful predictive signal.
- If removing a module improves MAE, inspect whether it is overfitting the 62-sample dataset.
- If `module_no_aux` barely changes performance, the model is learning mainly geometry/physics; if it collapses, it depends heavily on system/status shortcuts.
- If `module_no_branch_embed` barely changes performance, pointwise learned geometry is not adding much beyond hand-coded physics signals.
- If `loss_main_only` improves performance, the physics losses may be too strong for the current data scale.
