# PVP predictor ablation analysis

- full_model: MAE 2.8727, RMSE 3.7941, R2 0.6182
- Variants with supportive one-factor evidence versus full_model:
  - no_global_flow_corrector: dMAE 0.5837, dRMSE 0.7359, dR2 -0.1511
  - no_flow_graph: dMAE 0.4789, dRMSE 0.6215, dR2 -0.1222
  - all_profile_channels: dMAE 0.3716, dRMSE 0.7233, dR2 -0.1444
  - three_vessel_layout: dMAE 0.2157, dRMSE 0.1720, dR2 -0.0229
  - six_vessel_layout: dMAE 0.1963, dRMSE 0.2313, dR2 -0.0469
  - no_physics_residual: dMAE 0.1578, dRMSE 0.2803, dR2 -0.0578
  - use_unreliable_raw_lengths: dMAE 0.0893, dRMSE 0.1591, dR2 -0.0297
  - fixed_physics_params: dMAE 0.0286, dRMSE 0.2039, dR2 -0.0343
  - with_organ_flow_scale: dMAE 0.0100, dRMSE 0.0104, dR2 -0.0019

Interpretation note: decisions use MAE, RMSE, and R2 together. Smoke runs are for wiring and stability; full 5-fold runs are used for module/loss decisions.