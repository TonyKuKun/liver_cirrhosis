# PVP Predictor

PVP Predictor 是当前整理后的单任务门静脉压力回归模型。这个版本只保留一个 PVP 预测头，不再保留 CSPH 分类头；训练目标只包含 PVP 的 L2/MSE loss 和轻量 `core_confluence` 分流 loss。

![PVP Predictor feature-flow architecture](docs/figures/model_architecture.png)

上图使用 image2 生成，重点展示特征流动而不是装饰性器官渲染。蓝色流线表示血管几何与 mask 特征，琥珀色流线表示可学习血流动力学代理，青绿色流线表示脾肝全局状态；三路信息先进入 `GlobalFlowCorrector` 做多源融合，再进入 `FlowGraphRefiner` 做解剖图消息传播，最后只通过一个 PVP 回归头输出标量压力值。

## 1. 模型架构

最终模型的核心思路是：

```text
可靠几何表征
  + 可学习血流动力学代理
  + 脾肝全局状态
  + 全局流校正
  + 解剖图传播
  -> 单一 PVP 预测头
```

### 1.1 输入与几何表征

模型使用 8 条血管分支作为主要输入：MPV、SV、SMV、LPV、RPV、TIPS、LGV、PGV。每条分支从中心线和分割 mask 中提取横截面 profile、弧长位置、面积、等效水力直径、内切半径、曲率、solidity、circularity 和归一化 `dA/ds` 等稳定几何特征。

最终版本没有把原始长度、扭率、连通域计数等更不稳定的几何量作为主输入。消融中加入这些通道后 MAE 从 2.685 升到 2.910，说明它们在当前样本规模下带来的噪声大于有效信息。

### 1.2 可学习物理代理

`LearnablePhysicsProxy` 用几何 token 估计相对流量、速度、壁面剪切、Reynolds 数、Dean 数、阻力和压降等血流动力学变量。它不是把物理公式写死，而是保留黏滞系数、半径指数、压降尺度等可学习参数，让模型能在小样本真实数据里自动修正理想公式和实际测量之间的偏差。

`PhysicsResidualNet` 在当前最终版本中默认关闭。开启这个残差物理模块后 MAE 为 2.809，弱于最终模型的 2.685，说明额外残差自由度会带来过拟合风险。

### 1.3 脾肝全局状态

模型从 STL/分割结果中提取脾体积、肝体积和脾肝比例等全局上下文特征。这部分不是简单地把脾肝比例硬编码成结论，而是作为 organ context token 融入全局流校正模块。去掉脾肝全局状态后 MAE 升到 3.104，说明门静脉压力预测确实需要血管局部形态之外的器官状态信息。

### 1.4 全局校正与图传播

`GlobalFlowCorrector` 负责把几何 token、物理代理 token 和 organ context token 融合成统一的全局血流状态。去掉这个模块后 MAE 升到 3.479，是所有结构消融里退化最明显的一项。

`FlowGraphRefiner` 根据 8 条血管之间的解剖连接关系做图消息传播，让 MPV、SMV、SV 和分支血管之间的信息可以互相修正。去掉图传播后 MAE 升到 3.171，说明单独看每条血管不足以稳定预测 PVP。

### 1.5 预测头与训练目标

最终版本只保留一个 PVP 回归头。训练 loss 为：

```text
L = MSE(PVP_pred, PVP_label)
    + 0.005 * ||Q_MPV - Q_SMV - Q_SV||^2
```

其中第二项是轻量 `core_confluence` 分流 loss，只约束 MPV、SMV 和 SV 的核心汇流关系。它只在训练阶段使用；推理阶段模型只输出 PVP 标量。`AuxiliaryDropoutRegularizer` 仍作为训练期正则保留，但它不引入额外预测头，也不改变最终输出。

## 2. 最新实验结果

数据来自 `F:\PCG data\dataset\test4all_sample`。排除 17 个 `00` 前缀目录后，共使用 72 个有效样本。所有主实验、消融和 baseline 都使用 subject-level 5-fold split，随机种子为 40。

最终结果目录：

```text
runs/final_20260610_pvp_l2_shunt
```

### 2.1 最终模型

| 模型 | n | MAE | RMSE | R2 | OOF MAE | OOF RMSE | bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| final PVP predictor | 72 | 2.685 | 3.605 | 0.643 | 2.704 | 3.800 | 0.256 |

5 折 MAE 为 2.056、2.274、3.977、2.364、2.752，均值为 2.685，标准差为 0.746。当前最终版本达到此前目标的 2.8 左右，并且只使用 L2 + 分流 loss + 脾肝全局状态 + 单 PVP 头。

### 2.2 消融实验

完整消融结果位于：

```text
ablation/runs/final_20260610/full
```

#### 2.2.1 损失函数与训练目标

| 变体 | 针对部分 | MAE | RMSE | R2 | dMAE |
|---|---|---:|---:|---:|---:|
| full_model | 最终训练目标 | 2.685 | 3.605 | 0.643 | 0.000 |
| loss_l2_plus_core_split | L2 + 核心汇流分流约束 | 2.685 | 3.605 | 0.643 | 0.000 |
| loss_l2_only | 去掉分流 loss，只保留 L2 | 2.700 | 3.618 | 0.641 | +0.014 |
| loss_l2_plus_full_split | 使用更强的全分流约束 | 2.703 | 3.619 | 0.641 | +0.018 |

这一组消融针对训练目标。结果显示，轻量 `core_confluence` 分流 loss 相比纯 L2 有小幅但稳定的提升；更强的全分流约束没有继续变好，说明当前数据更适合只约束最可靠的核心汇流关系。

#### 2.2.2 模型结构模块

| 变体 | 针对部分 | MAE | RMSE | R2 | dMAE |
|---|---|---:|---:|---:|---:|
| fixed_physics_params | 固定物理代理参数 | 2.696 | 3.616 | 0.644 | +0.011 |
| with_physics_residual | 开启 PhysicsResidualNet | 2.809 | 3.837 | 0.615 | +0.123 |
| no_dropout_regularizer | 去掉训练期 dropout 正则 | 2.942 | 3.721 | 0.634 | +0.256 |
| no_organ_global_features | 去掉脾肝全局状态 | 3.104 | 4.093 | 0.548 | +0.419 |
| no_flow_graph | 去掉 FlowGraphRefiner | 3.171 | 4.455 | 0.490 | +0.486 |
| no_global_flow_corrector | 去掉 GlobalFlowCorrector | 3.479 | 4.536 | 0.458 | +0.794 |

这一组消融针对模型内部模块。全局流校正、解剖图传播和脾肝全局状态是贡献最大的三个结构；去掉任何一个都会明显退化。`PhysicsResidualNet` 没有进入最终版，因为它虽然增加表达能力，但在当前数据量下反而把 MAE 拉高到 2.809。保留 dropout 正则可以恢复旧版本中有用的随机正则效果，同时不增加额外预测头。

#### 2.2.3 几何特征与输入通道

| 变体 | 针对部分 | MAE | RMSE | R2 | dMAE |
|---|---|---:|---:|---:|---:|
| use_unreliable_raw_lengths | 加入原始长度等不可靠几何量 | 2.910 | 3.977 | 0.586 | +0.224 |
| all_profile_channels | 使用全部 profile 通道 | 2.930 | 4.208 | 0.542 | +0.244 |

这一组消融针对特征构建。结果说明“更多通道”并不一定更好；原始长度、噪声 profile 和不稳定几何量会削弱泛化，所以最终模型只保留更可靠的横截面与形态学特征。

#### 2.2.4 血管分支布局

| 变体 | 针对部分 | MAE | RMSE | R2 | dMAE |
|---|---|---:|---:|---:|---:|
| six_vessel_layout | 6 分支血管布局 | 3.026 | 3.984 | 0.592 | +0.341 |
| three_vessel_layout | 3 分支血管布局 | 3.132 | 4.032 | 0.580 | +0.446 |

这一组消融针对解剖覆盖范围。把 8 分支降到 6 分支或 3 分支都会变差，说明侧支、门静脉左右支和 TIPS/LGV/PGV 等信息对当前任务有价值。最终版本保留完整 8-vessel layout。

### 2.3 Baseline 对比

baseline 的目标不是重新发明一个更复杂的模型，而是回答一个公平问题：如果只使用传统机器学习和手工特征，能不能达到最终 PVP 模型的效果。因此 baseline 按特征来源分成三组。

| 特征组 | 选择原因 | 特征数 |
|---|---|---:|
| physics | 使用同一批几何输入构造 32 个血流动力学特征，用来检验“物理代理 + 传统回归器”是否足够 | 32 |
| geometry | 使用重采样横截面 profile 构造 897 个几何特征，用来检验“纯几何表格特征”是否足够 | 897 |
| combined | 合并几何、物理和全局 summary 特征，用来检验“更多手工特征”能否替代深度融合与图传播 | 1088 |

baseline 算法选取覆盖了小样本回归里常见的线性、bagging 和 boosting 方法。

| 算法 | 选择原因 | 训练方式 |
|---|---|---|
| elasticnet_cv | 带 L1/L2 正则的线性模型，用来衡量线性可解释基线 | 训练折内标准化，并在训练折内选择正则强度 |
| random_forest | 非线性 bagging 树模型，适合小样本稳健对比 | 只在训练折拟合，测试折只评估 |
| extra_trees | 更强随机性的树集成，用来降低单棵树方差 | 只在训练折拟合，测试折只评估 |
| adaboost | boosting 回归器，用来测试弱学习器逐步修正误差的效果 | 只在训练折拟合，测试折只评估 |
| hist_gradient_boosting | 直方图梯度提升，用来测试另一类 boosting 非线性基线 | 只在训练折拟合，测试折只评估 |

baseline 没有使用最终模型里的 `GlobalFlowCorrector`、`FlowGraphRefiner` 或深度训练技巧，也没有用测试折信息做调参。做过的改进主要是公平性和稳定性改进：所有 baseline 使用和最终模型完全相同的 `splits.json`，每一折只在训练折内完成标准化、特征处理和模型拟合，再在 held-out 测试折上评估。

调用脚本：

```bat
conda run -n pytorch python baseline/run_baselines.py ^
  --data_root "F:\PCG data\dataset\test4all_sample" ^
  --split_json runs\final_20260610_pvp_l2_shunt\splits.json ^
  --out_dir baseline\runs\final_20260610_baselines ^
  --n_points 200 --seed 40
```

主要 baseline 结果如下：

| baseline | 特征组 | 特征数 | n | MAE | RMSE | R2 | bias |
|---|---|---:|---:|---:|---:|---:|---:|
| physics/adaboost | physics | 32 | 72 | 3.420 | 4.286 | 0.531 | -0.212 |
| physics/random_forest | physics | 32 | 72 | 3.518 | 4.352 | 0.516 | -0.112 |
| physics/extra_trees | physics | 32 | 72 | 3.532 | 4.372 | 0.512 | -0.378 |
| combined/elasticnet_cv | combined | 1088 | 72 | 3.580 | 4.418 | 0.502 | 0.318 |
| combined/extra_trees | combined | 1088 | 72 | 3.672 | 4.515 | 0.480 | -0.122 |
| combined/hist_gradient_boosting | combined | 1088 | 72 | 3.677 | 4.484 | 0.487 | -0.097 |
| geometry/extra_trees | geometry | 897 | 72 | 3.685 | 4.550 | 0.472 | -0.054 |
| physics/hist_gradient_boosting | physics | 32 | 72 | 3.699 | 4.645 | 0.449 | -0.086 |

最佳 baseline 是 `physics/adaboost`，MAE 为 3.420。最终深度模型 MAE 为 2.685，相比最佳 baseline 降低 0.735，说明单纯手工物理特征或纯几何表格特征不能替代多源融合、解剖图传播和脾肝全局状态建模。

## 3. 代码结构

```text
PVP_predictor/
|-- README.md
|-- FINAL_20260610_RESULTS_ANALYSIS.md
|-- dataset.py
|-- model.py
|-- train.py
|-- ablation/
|   |-- ablations.py
|   |-- report.py
|   `-- runs/final_20260610/full/
|-- baseline/
|   |-- run_baselines.py
|   `-- runs/final_20260610_baselines/
|-- docs/
|   `-- figures/model_architecture.png
`-- runs/final_20260610_pvp_l2_shunt/
```

主要文件说明：

| 路径 | 作用 |
|---|---|
| `dataset.py` | 读取样本、血管几何、mask/STL 和 PVP 标签，构建模型输入 |
| `model.py` | 最终 PVP 模型，包括物理代理、全局流校正、图传播和单 PVP 头 |
| `train.py` | 主训练入口，生成最终 5-fold 结果、OOF 预测和 split 文件 |
| `ablation/ablations.py` | 定义各类消融配置 |
| `ablation/report.py` | 汇总消融实验结果 |
| `baseline/run_baselines.py` | 训练并评估传统机器学习 baseline |
| `docs/figures/model_architecture.png` | image2 生成的顶刊风格模型架构图 |
| `runs/final_20260610_pvp_l2_shunt/` | 最终模型训练结果 |
| `ablation/runs/final_20260610/full/` | 完整消融结果 |
| `baseline/runs/final_20260610_baselines/` | baseline 结果 |

## 4. 复现实验

以下命令默认在 `PVP_predictor/` 目录下执行。

训练最终模型：

```bat
conda run -n pytorch python train.py ^
  --data_root "F:\PCG data\dataset\test4all_sample" ^
  --out_dir runs\final_20260610_pvp_l2_shunt ^
  --epochs 220 --seed 40
```

运行消融：

```bat
conda run -n pytorch python ablation/ablations.py ^
  --data_root "F:\PCG data\dataset\test4all_sample" ^
  --out_root ablation\runs\final_20260610 ^
  --stage full --suite all --seed 40
```

运行 baseline：

```bat
conda run -n pytorch python baseline/run_baselines.py ^
  --data_root "F:\PCG data\dataset\test4all_sample" ^
  --split_json runs\final_20260610_pvp_l2_shunt\splits.json ^
  --out_dir baseline\runs\final_20260610_baselines ^
  --n_points 200 --seed 40
```

结果优先查看：

| 文件 | 内容 |
|---|---|
| `runs/final_20260610_pvp_l2_shunt/summary.json` | 最终模型整体指标 |
| `runs/final_20260610_pvp_l2_shunt/oof_predictions.csv` | OOF 预测 |
| `ablation/runs/final_20260610/full/comparison.csv` | 消融排序表 |
| `ablation/runs/final_20260610/full/analysis.md` | 消融分析 |
| `baseline/runs/final_20260610_baselines/summary.csv` | baseline 汇总 |
