# PVP predictor ablation analysis

- full_model: MAE 3.0217, RMSE 4.1400, R2 0.5560
- Variants where full_model is consistently better across MAE/RMSE/R2:
  - no_global_flow_corrector: dMAE 0.3953, dRMSE 0.4186, dR2 -0.0928
  - no_physics_residual: dMAE 0.2465, dRMSE 0.0780, dR2 -0.0184
  - three_vessel_layout: dMAE 0.0841, dRMSE 0.0541, dR2 -0.0200
  - no_organ_global_features: dMAE 0.0624, dRMSE 0.0822, dR2 -0.0279
- Variants consistently better than full_model across MAE/RMSE/R2:
  - loss_l2_only: dMAE -0.2119, dRMSE -0.2966, dR2 0.0582
  - use_unreliable_raw_lengths: dMAE -0.1265, dRMSE -0.1176, dR2 0.0238
  - no_flow_graph: dMAE -0.0234, dRMSE -0.1006, dR2 0.0219
- Mixed/noisy variants where the three metrics disagree:
  - fixed_physics_params: dMAE -0.0995, dRMSE 0.0522, dR2 -0.0084
  - six_vessel_layout: dMAE -0.0172, dRMSE 0.0469, dR2 -0.0076
  - loss_l2_plus_split: dMAE 0.0000, dRMSE 0.0000, dR2 0.0000
  - all_profile_channels: dMAE 0.0361, dRMSE -0.2205, dR2 0.0460

Interpretation note: decisions use MAE, RMSE, and R2 together. Smoke runs are for wiring and stability; full 5-fold runs are used for module/loss decisions.
