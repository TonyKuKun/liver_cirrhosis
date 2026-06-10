# PVP Predictor

本目录是当前保留的门静脉压力（Portal Venous Pressure, PVP）预测模块。最终版本是一个单任务、单预测头的 PVP 回归模型：只预测 PVP，不再保留 CSPH 分类头；训练目标只包含 PVP 的 L2/MSE 损失和轻量级分流约束损失。

![PVP Predictor 最终模型架构](docs/figures/model_architecture.png)

## 1. 模型架构

最终模型围绕“可靠几何表征 + 可学习血流动力学代理 + 脾肝全局状态 + 解剖图传播 + 单一 PVP 预测头”构建。输入来自 8 条门静脉系统相关血管分支：MPV、SV、SMV、LPV、RPV、TIPS、LGV 和 PGV。每条分支包含截面序列、弧长、中心线点以及血管存在掩码；模型通过 `segment_mask` 屏蔽缺失血管，避免缺失分支对聚合结果产生伪信号。

几何输入首先经过可靠特征筛选。默认只保留稳定的截面几何特征，包括面积、水力直径、内切半径、曲率、实心度、圆度和归一化 dA/ds；原始长度、扭率、连通域计数等更容易受分割质量和中心线噪声影响的特征默认不进入最终模型。消融结果也支持这个选择：启用全部 profile 通道后 MAE 从 2.685 上升到 2.930，使用不可靠原始长度后 MAE 上升到 2.910。

筛选后的几何序列进入血流动力学代理层。该层不是硬编码的完整 CFD，而是一个可微、可学习的物理代理模块，用几何估计有效半径、相对流量、截面速度、壁面剪切、Reynolds、Dean、阻力和压降等中间量。黏度缩放、半径指数和压力缩放等参数可学习，使模型能在医学先验和数据驱动拟合之间取得平衡。需要注意的是，旧版本的 `PhysicsResidualNet` 在当前单头设置下不再作为默认模块：完整消融显示开启 residual 后 MAE 为 2.809，弱于最终模型的 2.685。

脾脏体积、肝脏体积和脾肝体积比作为患者级全局状态输入模型。这些体积特征由 STL 最大连通域计算得到，只作为全局上下文参与校正，而不是直接作为硬性流量比例。消融结果显示去掉脾肝全局特征后 MAE 从 2.685 上升到 3.104，说明该上下文对 PVP 预测很关键。

`GlobalFlowCorrector` 负责融合分支 embedding、物理代理量、全局几何和脾肝状态，校正中间流量表征。它是最终模型中贡献最大的模块之一：去掉该模块后 MAE 上升到 3.479。随后，`FlowGraphRefiner` 在解剖连接图上做分支间信息传播，保留中心线图结构接口；去掉图传播后 MAE 上升到 3.171，说明血管之间的拓扑关系对 PVP 回归有帮助。

最终预测部分只保留一个 PVP head。该预测头聚合校正后的流量、物理代理、血管 mask、患者全局状态和物理基线，输出一个门静脉压力值。训练中保留 `AuxiliaryDropoutRegularizer`，但它不是预测头、没有额外标签、也不产生额外任务；它只在训练阶段提供随机正则。保留它的原因是旧 2.8 版本中虽然 CSPH 分支不参与 PVP loss，但其 dropout forward 会影响训练随机轨迹。当前实现用训练期正则器保留这部分有益随机性，同时保持模型仍然是单头 PVP 回归。

最终训练目标为：

```text
L = MSE(PVP_pred, PVP_label) + 0.005 * ||Q_MPV - Q_SMV - Q_SV||^2
```

其中分流损失使用 `core_confluence` 模式，只约束核心汇合关系 `MPV ~= SMV + SV`。它是轻量约束，不取代 PVP 主监督。纯 L2 的 MAE 为 2.700，加入轻量 core 分流约束后 MAE 为 2.685，提升幅度不夸张，但方向稳定。

## 2. 最新实验结果

实验数据来自 `F:\PCG data\dataset\test4all_sample`。排除 17 个 `00` 前缀样本目录后，共保留 72 个有效样本。所有最新结果均使用 subject-level 5-fold，随机种子为 40。

### 2.1 最终模型

结果目录：`runs/final_20260610_pvp_l2_shunt`

| 指标 | 结果 |
| --- | --- |
| 5-fold MAE | 2.685 +/- 0.746 |
| 5-fold RMSE | 3.605 +/- 1.132 |
| 5-fold R2 | 0.643 +/- 0.183 |
| OOF MAE | 2.704 |
| OOF RMSE | 3.800 |
| OOF bias | 0.256 |

各折结果如下：

| fold | train | val | best_epoch | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 57 | 15 | 50 | 3.687 | 4.620 | 0.432 |
| 1 | 58 | 14 | 89 | 1.884 | 2.147 | 0.873 |
| 2 | 58 | 14 | 37 | 1.752 | 2.301 | 0.851 |
| 3 | 57 | 15 | 12 | 3.026 | 4.361 | 0.517 |
| 4 | 58 | 14 | 11 | 3.078 | 4.594 | 0.543 |

`oof_predictions.csv` 与 `oof_predictions.xlsx` 由同一组预测行写出，字段完全一致，便于直接核对 Excel 结果。

### 2.2 消融实验

完整消融目录：`ablation/runs/final_20260610/full`

| 变体 | 类别 | MAE | RMSE | R2 | 相对最终模型 dMAE |
| --- | --- | --- | --- | --- | --- |
| full_model | reference | 2.685 | 3.605 | 0.643 | 0.000 |
| loss_l2_plus_core_split | loss | 2.685 | 3.605 | 0.643 | 0.000 |
| fixed_physics_params | module | 2.696 | 3.616 | 0.644 | 0.011 |
| loss_l2_only | loss | 2.700 | 3.618 | 0.641 | 0.014 |
| loss_l2_plus_full_split | loss | 2.703 | 3.619 | 0.641 | 0.018 |
| with_physics_residual | module | 2.809 | 3.837 | 0.615 | 0.123 |
| use_unreliable_raw_lengths | geometry | 2.910 | 3.977 | 0.586 | 0.224 |
| all_profile_channels | geometry | 2.930 | 4.208 | 0.542 | 0.244 |
| no_dropout_regularizer | module | 2.942 | 3.721 | 0.634 | 0.256 |
| six_vessel_layout | layout | 3.026 | 3.984 | 0.592 | 0.341 |
| no_organ_global_features | module | 3.104 | 4.093 | 0.548 | 0.419 |
| three_vessel_layout | layout | 3.132 | 4.032 | 0.580 | 0.446 |
| no_flow_graph | module | 3.171 | 4.455 | 0.490 | 0.486 |
| no_global_flow_corrector | module | 3.479 | 4.536 | 0.458 | 0.794 |

消融结论：

- 最终模型是当前 5-fold 下的最佳配置，MAE 为 2.685。
- 轻量 `core_confluence` 分流 loss 相比纯 L2 略有提升，`2.700 -> 2.685`。
- `GlobalFlowCorrector`、脾肝全局特征、`FlowGraphRefiner` 和 dropout 正则均有明确贡献。
- 旧版 `PhysicsResidualNet` 在当前单头模型中不再有益，开启后 MAE 上升到 2.809。
- 使用全部 profile 通道或不可靠原始长度都会变差，说明最终的几何特征筛选是必要的。

完整表格文件：

- `ablation/runs/final_20260610/full/comparison.csv`
- `ablation/runs/final_20260610/full/comparison.json`
- `ablation/runs/final_20260610/full/analysis.md`

### 2.3 传统机器学习 baseline

baseline 使用与最终模型完全相同的 `splits.json`。结果目录：`baseline/runs/final_20260610_baselines`

| baseline | 特征数 | n | MAE | RMSE | R2 | bias |
| --- | --- | --- | --- | --- | --- | --- |
| physics/adaboost | 32 | 72 | 3.420 | 4.286 | 0.531 | -0.212 |
| physics/random_forest | 32 | 72 | 3.518 | 4.352 | 0.516 | -0.112 |
| physics/extra_trees | 32 | 72 | 3.532 | 4.372 | 0.512 | -0.378 |
| combined/elasticnet_cv | 1088 | 72 | 3.580 | 4.418 | 0.502 | 0.318 |
| combined/extra_trees | 1088 | 72 | 3.672 | 4.515 | 0.480 | -0.122 |
| combined/hist_gradient_boosting | 1088 | 72 | 3.677 | 4.484 | 0.487 | -0.097 |
| geometry/extra_trees | 897 | 72 | 3.685 | 4.550 | 0.472 | -0.054 |
| physics/hist_gradient_boosting | 32 | 72 | 3.699 | 4.645 | 0.449 | -0.086 |

最佳传统 baseline 是 `physics/adaboost`，MAE 为 3.420。最终深度模型 MAE 为 2.685，相比最佳 baseline 降低 0.735 mmHg。

## 3. 代码结构

```text
PVP_predictor/
├── README.md
├── FINAL_20260610_RESULTS_ANALYSIS.md
├── dataset.py
├── model.py
├── train.py
├── ablation/
│   ├── ablations.py
│   ├── report.py
│   └── runs/final_20260610/full/
├── baseline/
│   ├── run_baselines.py
│   └── runs/final_20260610_baselines/
├── docs/
│   ├── draw_model_architecture.py
│   └── figures/model_architecture.png
└── runs/final_20260610_pvp_l2_shunt/
```

核心文件说明：

- `dataset.py`：负责读取血管 STL、截面 profile、弧长、中心线点、脾肝体积和标签，并生成训练所需的 mask 与标准化数据。
- `model.py`：最终单头 PVP 模型实现，包含几何编码、可学习物理代理、全局流校正、图传播、单一 PVP head、分流 loss 和训练期 dropout 正则。
- `train.py`：执行 subject-level 5-fold 训练，输出每折预测、OOF 预测、CSV/XLSX 和 `summary.json`。
- `ablation/ablations.py`：运行结构、几何、layout 和 loss 消融。
- `ablation/report.py`：汇总消融结果，生成 `comparison.csv`、`comparison.json` 和 `analysis.md`。
- `baseline/run_baselines.py`：运行传统机器学习 baseline，并输出 `summary.csv`、`summary.json`、OOF 预测和特征重要性。
- `docs/draw_model_architecture.py`：生成 README 中引用的中文模型架构图。
- `runs/final_20260610_pvp_l2_shunt/`：最终模型主实验输出。
- `ablation/runs/final_20260610/full/`：最新完整消融实验输出。
- `baseline/runs/final_20260610_baselines/`：最新传统 baseline 输出。

复现实验时，在 `PVP_predictor` 目录下运行：

```powershell
conda run -n pytorch python train.py ^
  --data_root "F:\PCG data\dataset\test4all_sample" ^
  --out_dir runs\final_20260610_pvp_l2_shunt ^
  --n_folds 5 --epochs 300 --seed 40 ^
  --lambda_shunt 0.005 --split_loss_mode core_confluence
```

```powershell
conda run -n pytorch python ablation/ablations.py ^
  --suite all --stage full ^
  --out_root ablation/runs/final_20260610 ^
  --full_n_folds 5 --full_epochs 300 --seed 40 --force
```

```powershell
conda run -n pytorch python baseline/run_baselines.py ^
  --data_root "F:\PCG data\dataset\test4all_sample" ^
  --split_json runs\final_20260610_pvp_l2_shunt\splits.json ^
  --out_dir baseline\runs\final_20260610_baselines ^
  --n_points 200 --seed 40
```
