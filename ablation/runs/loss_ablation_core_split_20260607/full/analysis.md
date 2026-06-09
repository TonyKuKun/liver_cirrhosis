# PVP predictor ablation analysis

- loss_l2_only: MAE 2.8098, RMSE 3.8433, R2 0.6142
- Variants where loss_l2_only is consistently better across MAE/RMSE/R2:
  - loss_l2_plus_full_split: dMAE 0.2119, dRMSE 0.2966, dR2 -0.0582
  - loss_l2_plus_core_split: dMAE 0.0767, dRMSE 0.0977, dR2 -0.0197

Interpretation note: decisions use MAE, RMSE, and R2 together. Smoke runs are for wiring and stability; full 5-fold runs are used for module/loss decisions.
