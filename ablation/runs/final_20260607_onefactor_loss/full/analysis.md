# PVP predictor ablation analysis

- loss_mse_only: MAE 2.8825, RMSE 3.9667, R2 0.5895
- Variants with supportive one-factor evidence versus loss_mse_only:
  - loss_mse_plus_split: dMAE -0.0098, dRMSE -0.1726, dR2 0.0287
  - loss_mse_plus_flow_conservation: dMAE -0.0098, dRMSE -0.1726, dR2 0.0287
  - loss_mse_plus_all_simple_physics: dMAE -0.0098, dRMSE -0.1726, dR2 0.0287
- Variants without supportive one-factor evidence versus loss_mse_only:
  - loss_mse_plus_continuity: dMAE 0.0000, dRMSE 0.0000, dR2 0.0000
  - loss_mse_plus_pressure_mono: dMAE 0.0000, dRMSE 0.0000, dR2 0.0000

Interpretation note: decisions use MAE, RMSE, and R2 together. Smoke runs are for wiring and stability; full 5-fold runs are used for module/loss decisions.