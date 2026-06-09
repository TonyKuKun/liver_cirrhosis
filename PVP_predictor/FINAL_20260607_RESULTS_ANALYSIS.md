# 2026-06-07 PVP 最终消融结论

## 当前实验设置

- 数据：`F:\PCG data\dataset\test4all_sample`
- 有效样本：72 scans
- 验证：subject-level 5-fold
- 指标：MAE、RMSE、R2
- 肝/脾体积：从 STL 最大联通区域计算，缓存于 `runs/stl_largest_component_volume_cache.json`
- 最终血管布局：8 血管，即 `mpv, sv, smv, lpv, rpv, tips, lgv, pgv`
- 肝脾体积使用方式：只作为全局特征，不再单独修正入口流量
- 当前物理 loss 候选：只保留 L2 和分流 loss；其他物理 loss 已从最终实验中移除

## 架构诊断消融结果

结果文件：

```text
ablation/runs/arch_ablation_l2_fullsplit_20260607/full/comparison.csv
ablation/runs/arch_ablation_l2_fullsplit_20260607/full/analysis.md
```

| Variant | MAE | RMSE | R2 | 结论 |
|---|---:|---:|---:|---|
| L2 only + organ global | 2.810 | 3.843 | 0.614 | 本轮最佳 |
| L2 + split loss + organ global | 3.022 | 4.140 | 0.556 | 分流 loss 使三项指标变差 |
| no organ global features | 3.084 | 4.222 | 0.528 | 肝脾体积作为全局特征有效 |
| no global flow corrector | 3.417 | 4.559 | 0.463 | GlobalFlowCorrector 明确有效 |
| no physics residual | 3.268 | 4.218 | 0.538 | PhysicsResidualNet 明确有效 |
| no flow graph | 2.998 | 4.039 | 0.578 | GNN 当前有轻微负贡献 |
| use unreliable raw lengths | 2.895 | 4.022 | 0.580 | 原始长度仍需谨慎，结果提示可作为候选而非默认 |
| three-vessel layout | 3.106 | 4.194 | 0.536 | 8 血管优于 3 血管 |
| six-vessel layout | 3.005 | 4.187 | 0.548 | 指标混合，不替代 8 血管 |
| all profile channels | 3.058 | 3.919 | 0.602 | RMSE/R2 变好但 MAE 变差，暂不作为默认 |

## 最新 Loss 消融结果：核心合流分流 Loss

根据后续分析，旧版分流 loss 过宽，同时约束了代偿血管、TIPS 和肝内分支。现在新增 `core_confluence` 模式，只约束核心合流：

```text
MPV ~= SMV + SV
```

结果文件：

```text
ablation/runs/loss_ablation_core_split_20260607/full/comparison.csv
ablation/runs/loss_ablation_core_split_20260607/full/analysis.md
```

| Variant | MAE | RMSE | R2 | 结论 |
|---|---:|---:|---:|---|
| L2 only + organ global | 2.810 | 3.843 | 0.614 | 仍然最佳 |
| L2 + core confluence split | 2.887 | 3.941 | 0.595 | 比旧分流 loss 好，但仍弱于纯 L2 |
| L2 + full split | 3.022 | 4.140 | 0.556 | 最差 |

结论：把分流 loss 收窄到 MPV/SMV/SV 后，负面影响明显减小，但仍没有带来正收益。因此最终默认仍采用纯 L2，核心合流分流 loss 只作为可选物理约束实验保留。

## 关键判断

1. 最好的结果是 `L2 only + organ global`，MAE 2.810、RMSE 3.843、R2 0.614。
2. 分流 loss 在严格 5-fold 中没有带来收益。新版核心合流分流 loss 比旧版更好，但仍使 MAE 增加 0.077、RMSE 增加 0.098、R2 下降 0.020。
3. 肝脾体积不适合作为单独的 Q 修正边界条件；目前最有效的方式是作为全局特征输入。
4. GlobalFlowCorrector 和 PhysicsResidualNet 是当前最稳定有效的两个模块。
5. CenterlinePoints-aware GNN 的解剖信息方向是合理的，但当前实现对指标没有正收益，需要下一步重做边定义或边权，而不是作为默认必需模块。
6. 8 血管布局仍然优于 3 血管压缩布局；6 血管结果接近但指标不一致，暂不替代 8 血管。

## 当前推荐默认方案

```text
8-vessel layout
+ core profile geometry only
+ liver/spleen volumes as global features
+ GlobalFlowCorrector
+ PhysicsResidualNet
+ pure L2/MSE loss
- no organ boundary Q scaling
- no explicit split loss by default
- no conductance / pressure balance / monotonic pressure loss
- no unreliable raw branch lengths by default
```

推荐训练命令：

```bash
conda run -n pytorch python train.py ^
  --data_root "F:\PCG data\dataset\test4all_sample" ^
  --out_dir runs\final_pvp_8v_organ_global_l2 ^
  --no_organ_flow_scale ^
  --lambda_press 0
```

如需复现分流 loss 对照：

```bash
conda run -n pytorch python train.py ^
  --data_root "F:\PCG data\dataset\test4all_sample" ^
  --out_dir runs\pvp_8v_organ_global_l2_core_split ^
  --no_organ_flow_scale ^
  --lambda_press 0.03 ^
  --split_loss_mode core_confluence
```

## 下一步改进建议

- 分流 loss 不应直接作为默认项；如果继续研究，需要降低权重或改成只在 MPV/SMV/SV 同时高质量存在时启用。
- GNN 应重点改边权：用 `CenterlinePoints` 里的真实连接点、连接位置和距离，而不是只靠固定解剖邻接。
- 原始长度这次意外变好，但考虑到 SMV/LPV/RPV 分支截断误差大，建议只作为学习特征候选，不进入硬物理公式。
- Radiology 投稿时建议主表报告 `L2 only + organ global`，补充材料报告 `L2 + core confluence split loss` 作为物理约束未改善性能的负结果。
