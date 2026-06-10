# PVP Predictor Ablation Runs

Current retained run:

- `final_20260610/full`

The comparison files in that directory summarize 14 full 5-fold variants. The reference `full_model` is the final model:

- MAE 2.685 +/- 0.746
- RMSE 3.605
- R2 0.643

Key worse variants:

- `loss_l2_only`: MAE 2.700
- `with_physics_residual`: MAE 2.809
- `no_dropout_regularizer`: MAE 2.942
- `no_organ_global_features`: MAE 3.104
- `no_flow_graph`: MAE 3.171
- `no_global_flow_corrector`: MAE 3.479
