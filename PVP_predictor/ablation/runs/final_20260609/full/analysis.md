# PVP Predictor Ablation Analysis

本文件由 2026-06-09 完整消融结果重新生成。

参考配置 `full_model`：MAE 3.153, RMSE 3.986, R2 0.585。

| variant | category | MAE | RMSE | R2 | dMAE | direction |
| --- | --- | --- | --- | --- | --- | --- |
| full_model | reference | 3.153 | 3.986 | 0.585 |  | reference |
| no_organ_global_features | module | 3.184 | 4.030 | 0.568 | 0.031 | reference_consistently_better |
| no_global_flow_corrector | module | 3.473 | 4.562 | 0.460 | 0.320 | reference_consistently_better |
| no_flow_graph | module | 3.229 | 4.304 | 0.517 | 0.076 | reference_consistently_better |
| fixed_physics_params | module | 2.954 | 3.850 | 0.612 | -0.199 | variant_consistently_better |
| all_profile_channels | geometry | 3.212 | 4.237 | 0.527 | 0.060 | reference_consistently_better |
| use_unreliable_raw_lengths | geometry | 3.029 | 4.027 | 0.582 | -0.124 | mixed_or_noisy |
| six_vessel_layout | layout | 3.141 | 4.174 | 0.552 | -0.012 | mixed_or_noisy |
| three_vessel_layout | layout | 3.198 | 4.057 | 0.567 | 0.045 | reference_consistently_better |
| loss_l2_only | loss | 2.969 | 3.916 | 0.601 | -0.184 | variant_consistently_better |
| loss_l2_plus_core_split | loss | 3.153 | 3.986 | 0.585 | 0.000 | mixed_or_noisy |
| loss_l2_plus_full_split | loss | 2.934 | 3.891 | 0.606 | -0.219 | variant_consistently_better |

## 解释

- 参考配置相对更好的模块：`no_global_flow_corrector`、`no_flow_graph`、`all_profile_channels`、`three_vessel_layout` 等变体变差，支持保留当前默认路径。
- 本次更好的变体：`fixed_physics_params`、`loss_l2_only`、`loss_l2_plus_full_split`。这些结果需要继续复核，尤其是分流 loss 的使用方式。
- 所有决策应同时看 MAE、RMSE 和 R2，不能只看单个 fold 或 smoke run。
