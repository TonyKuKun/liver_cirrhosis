# PVP Predictor Ablations

本目录保存当前 PVP 模型的结构、输入布局和 loss 消融。2026-06-09 已重新运行完整 5-fold。

## 当前完整结果

- 目录：`ablation/runs/final_20260609/full`
- 配置数：12
- 参考配置：`full_model`
- 参考结果：MAE 3.153, RMSE 3.986, R2 0.585
- 最优配置：`loss_l2_plus_full_split`，MAE 2.934

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

## 重新运行

```powershell
conda run -n pytorch python ablation/ablations.py ^
  --suite all --stage full ^
  --out_root ablation/runs/final_20260609 ^
  --full_n_folds 5 --full_epochs 300 --seed 40 --force
```

输出文件：

- `ablation/runs/final_20260609/full/manifest.json`
- `ablation/runs/final_20260609/full/comparison.csv`
- `ablation/runs/final_20260609/full/comparison.json`
- `ablation/runs/final_20260609/full/analysis.md`
