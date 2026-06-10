# PVP predictor ablation analysis

- full_model: MAE 2.6853, RMSE 3.6048, R2 0.6432
- Variants where full_model is consistently better across MAE/RMSE/R2:
  - no_global_flow_corrector: dMAE 0.7941, dRMSE 0.9312, dR2 -0.1847
  - no_flow_graph: dMAE 0.4860, dRMSE 0.8497, dR2 -0.1528
  - three_vessel_layout: dMAE 0.4465, dRMSE 0.4276, dR2 -0.0629
  - no_organ_global_features: dMAE 0.4186, dRMSE 0.4882, dR2 -0.0948
  - six_vessel_layout: dMAE 0.3410, dRMSE 0.3794, dR2 -0.0510
  - no_dropout_regularizer: dMAE 0.2563, dRMSE 0.1158, dR2 -0.0091
  - all_profile_channels: dMAE 0.2444, dRMSE 0.6027, dR2 -0.1015
  - use_unreliable_raw_lengths: dMAE 0.2244, dRMSE 0.3721, dR2 -0.0571
  - with_physics_residual: dMAE 0.1234, dRMSE 0.2323, dR2 -0.0279
  - loss_l2_plus_full_split: dMAE 0.0177, dRMSE 0.0142, dR2 -0.0027
  - loss_l2_only: dMAE 0.0145, dRMSE 0.0127, dR2 -0.0018
- Mixed/noisy variants where the three metrics disagree:
  - loss_l2_plus_core_split: dMAE 0.0000, dRMSE 0.0000, dR2 0.0000
  - fixed_physics_params: dMAE 0.0106, dRMSE 0.0115, dR2 0.0008

Interpretation note: decisions use MAE, RMSE, and R2 together. Smoke runs are for wiring and stability; full 5-fold runs are used for module/loss decisions.