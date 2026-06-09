# PVP Predictor

PVP Predictor 是一个用于预测门静脉压力（Portal Vein Pressure, PVP）的物理约束几何深度学习项目。它的输入不是普通的二维图像，而是从 CT 后处理得到的门静脉系统中心线、逐点血管截面几何、血管拓扑、器官体积以及少量系统级临床/解剖特征。模型的目标也不只是输出一个 PVP 数值，而是在预测的同时给出流量分配、压力降、壁面切应力、注意力权重等可解释的中间结果。

这个项目的基本思想是：**用物理模型提供可靠骨架，用神经网络学习真实人体中偏离理想公式的部分**。因此，模型不是把所有 CT 特征简单堆进一个黑盒回归器，而是沿着“几何 -> 流量 -> 水动力 -> 压力基线 -> 神经校正”的路径逐步推理。

![Overall pipeline](picture/01_overall_pipeline.png)

## 项目背景与临床问题

门静脉高压是肝硬化、门静脉血栓（PVT）、侧支循环形成以及 TIPS 术后评估中的核心问题。PVP 可以直接反映门静脉系统的压力负荷，但直接测量具有侵入性，不适合频繁随访或大规模筛查。

CT 影像则提供了另一条路线：门静脉主干、脾静脉、肠系膜上静脉、肝内左右门静脉、TIPS 支架和侧支血管的形态，都与血流阻力和压力负荷相关。比如管腔变窄会提高阻力，非圆形残腔会破坏理想圆管假设，侧支或 TIPS 会改变分流路径，脾/肝体积也能间接反映门静脉流量与肝内阻力状态。

![Clinical portal vein system](picture/02_clinical_portal_vein.png)

在本项目中，一位患者的门静脉系统被抽象为 8 条血管段：

| 缩写 | 含义 | 在模型中的作用 |
| --- | --- | --- |
| `mpv` | Main Portal Vein，门静脉主干 | PVP 物理基线的关键路径 |
| `sv` | Splenic Vein，脾静脉 | 与 SMV 一起构成入口汇流 |
| `smv` | Superior Mesenteric Vein，肠系膜上静脉 | 与 SV 一起决定门静脉入口流量来源 |
| `lpv` | Left Portal Vein，左门静脉 | 肝内出口分支之一 |
| `rpv` | Right Portal Vein，右门静脉 | 肝内出口分支之一 |
| `tips` | TIPS 支架/分流通道 | 术后人工低阻力分流路径 |
| `lgv` | Left Gastric Vein，胃左静脉侧支 | 侧支循环与代偿分流 |
| `pgv` | Posterior Gastric Vein，胃后静脉侧支 | 侧支循环与代偿分流 |

## 当前挑战

**1. 标签昂贵，样本规模有限。** PVP 不是常规无创指标，真实标签获取成本高。小样本条件下，纯神经网络容易过拟合，也容易把高压患者预测回均值。

**2. 门静脉不是理想圆管。** PVT、偏心狭窄、海绵样变和分裂管腔会让普通等效直径失真。若直接套用圆管 Poiseuille 公式，阻力可能被明显低估。

**3. 血流量 Q 不可直接从 CT 读出。** CT 能给出几何，但真实门静脉流量受脾脏体积、肝脏体积、肝内阻力、侧支循环和 TIPS 共同影响。模型必须估计患者级流量尺度。

**4. 解剖结构经常缺失或重构。** 某些患者没有 TIPS，某些侧支不存在，某些血管段中心线质量不足。模型需要用 `segment_mask` 和 `point_valid` 稳定处理这些情况。

**5. 分叉分流必须守恒。** SV 和 SMV 汇入 MPV，MPV 又向 LPV/RPV/TIPS 或侧支分流。若每条血管独立预测流量，很容易违反质量守恒。

**6. 纯物理模型和纯数据模型都不够。** Poiseuille 解释了半径、长度、流量与压力降的主关系，但真实人体还有弯曲、入口效应、湍动趋势、术后支架和侧支代偿等非理想因素。

## 方法概览

模型采用 **physics-anchored residual learning**。也就是说，模型先通过几何和流量估计得到一个物理压力基线，再让神经网络学习全局校正和局部残差：

```text
pvp_pred = physics_baseline + predictor_correction + residual_correction
```

这种结构有两个好处。第一，模型从训练初期就站在合理物理基线上，而不是从随机 MLP 开始猜压力。第二，神经网络的任务被缩小为“校正物理公式不够准确的地方”，更适合小样本医学场景。

核心创新包括：

- **形态感知有效半径 `r_eff`**：结合水力直径、内切半径和实心度，适应规则圆管、偏心狭窄和 PVT 后不规则残腔。
- **解剖约束图网络**：8 条血管段作为图节点，消息只沿真实相邻关系传播。
- **守恒流量估计**：在入口、汇合出口、肝内分叉三个 junction 上用 masked softmax 分配流量。
- **患者级 Q scale**：用脾/肝体积估计患者个体化流量尺度。
- **可微 Poiseuille 水动力层**：逐点计算速度、WSS、Re、阻力和压力降。
- **物理约束损失**：同时约束 PVP 误差、Murray 偏离、压力残差、半径平滑、生理范围和压力单调性。

## 输入几何特征

模型的核心输入是逐点血管几何。每位患者有 8 条候选血管段，每条血管中心线被重采样为 `N` 个点，每个点包含 11 个几何通道。因此原始几何输入张量为：

```text
profiles:      (B, 8, N, 11)
profiles_norm: (B, 8, N, 11)
point_valid:   (B, 8, N)
segment_mask:  (B, 8)
```

其中 `B` 是 batch size，`8` 是血管段数量，`N` 是中心线采样点数，`11` 是每个点的几何特征数量。

![Data and geometric features](picture/03_data_features.png)

### 11 个逐点几何通道

| 特征 | 含义 | 为什么重要 |
| --- | --- | --- |
| `area` | 横截面积 | 直接影响速度 `v = Q / A`；面积变小通常意味着狭窄和更高阻力。 |
| `hydraulic_diameter` | 水力直径 `4A/P` | 比普通等效直径更适合非圆形管腔，是 Poiseuille 物理路径的重要尺度。 |
| `perimeter` | 横截面周长 | 反映管腔边界复杂度；同样面积下，周长越复杂，越可能偏离理想圆管。 |
| `curvature` | 中心线曲率 | 表示血管弯曲程度，影响 Dean number 和弯曲流动风险。 |
| `torsion` | 中心线空间扭转 | 补充 3D 走行复杂性，帮助模型理解血管不是平面曲线。 |
| `inscribed_radius` | 最大内切半径 | 描述真正可通行的最小尺度，对偏心狭窄和残余管腔尤其关键。 |
| `solidity` | 截面紧实度 `A/A_convex` | 衡量截面是否规则、饱满；PVT 后残腔或分裂管腔通常 solidity 较低。 |
| `r_insc_to_r_eq_ratio` | 内切半径与等效半径关系 | 判断管腔是否接近圆形；越偏离，圆管假设越不可靠。 |
| `dA_ds_norm` | 面积沿中心线变化率 | 捕捉急剧狭窄、扩张或支架入口处的几何突变。 |
| `circularity` | 圆形度 `4πA/P²` | 直接衡量横截面接近圆形的程度，辅助判断 Poiseuille 假设可信度。 |
| `n_components` | 横截面连通区域数 | 识别分裂管腔、复杂 PVT 或无效截面；正常单腔通常为 1。 |

这些特征可以分成两类：

| 类别 | 特征 | 作用 |
| --- | --- | --- |
| 物理计算核心特征 | `area`, `hydraulic_diameter`, `curvature`, `inscribed_radius` | 直接进入速度、Re、Dean number、有效半径和压力降计算。 |
| 形态可信度与残差学习特征 | `perimeter`, `torsion`, `solidity`, `r_insc_to_r_eq_ratio`, `dA_ds_norm`, `circularity`, `n_components` | 告诉模型当前管腔是否适合用理想公式解释，以及哪里需要神经网络补偿。 |

### `r_eff`：面向非圆形管腔的有效半径

当前模型没有简单地把直径除以 2 作为半径，而是使用形态感知有效半径：

```text
alpha = 1 - solidity
r_eff = (1 - alpha) * (0.5 * hydraulic_diameter)
        + alpha * inscribed_radius
```

直观理解是：当管腔接近规则圆形时，`solidity` 较高，模型更信任 `hydraulic_diameter`；当管腔不规则、偏心或分裂时，`solidity` 降低，模型更多参考 `inscribed_radius`，从而避免把一个狭窄残腔误当作宽圆管。

这个 `r_eff` 在三处保持一致：Poiseuille 前向计算、分支阻力先验、平滑正则项。这样做能避免训练目标和前向物理路径使用不同半径定义。

## 模型结构

### 1. 分支几何编码与解剖图网络

每条血管段的 `profiles_norm` 首先进入共享的 `GeometryEncoder`。编码器使用一维卷积沿中心线读取局部几何变化，再用 GroupNorm 避免 padding 和无效点污染归一化统计。随后 `AttentionPool` 在有效点上做加权汇聚，把每条血管的 `(N, H)` 序列压缩为一个 `(H,)` 分支向量。

这些分支向量再被堆叠成 8 个节点，送入 `VesselGraphNet`。图网络只沿解剖上相邻的血管传播消息，例如 MPV 与 SV/SMV、LPV/RPV、TIPS、LGV/PGV 相连。这样，MPV 的表示能感知入口汇流、肝内分叉和侧支/TIPS 分流，而不会把不直接相邻的分支强行连在一起。

![Encoder and anatomical graph](picture/04_encoder_graph.png)

这一模块输出两个重要结果：

- `branch_embed: (B, 8, H)`，用于后续流量估计和最终预测。
- `attn_weights: (B, 8, N)`，用于解释模型在每条中心线上关注的位置。

### 2. 流量估计与 Poiseuille 水动力层

血流量 `Q` 是连接几何和压力的关键变量，但它不能直接从 CT 中读取。因此模型分两步处理：

1. `SplenicFlowEstimator` 根据脾脏和肝脏体积估计患者级 `q_scale`。
2. `FlowRateEstimator` 在三个 junction 上估计相对流量分配。

流量分配使用三个局部 softmax：

| Junction | 分流关系 | 约束 |
| --- | --- | --- |
| Inflow | `SV` vs `SMV` | 两者共同构成入口流量 |
| Confluence outflow | `MPV` vs `LGV` vs `PGV` | 主干与侧支竞争分流 |
| Bifurcation outflow | `LPV` vs `RPV` vs `TIPS` | 肝内左右支和 TIPS 分流共享 MPV 流量 |

每个 softmax 的 logits 由三部分组成：Murray law 直径先验、分支阻力先验、可学习神经修正。缺失血管通过 mask 排除，因此不会被分到流量。

![Flow and physics module](picture/05_flow_physics.png)

得到每条血管的 `Q_scaled` 后，`PoiseuilleHydrodynamics` 沿中心线逐点计算：

| 输出 | 含义 |
| --- | --- |
| `velocity_m_per_s` | 局部流速 |
| `wss_pa` | 壁面切应力 |
| `reynolds` | Reynolds 数 |
| `local_R_pa_s_per_m4` | 单位长度局部阻力 |
| `cum_R_pa_s_per_m3` | 沿中心线累计阻力 |
| `pressure_drop_pa` | 局部累计压力降 |
| `dean` | 弯曲流动相关 Dean number |
| `area_gradient` | 面积变化率 |

这些输出既参与最终 PVP 预测，也可以用于可视化和模型诊断。

### 3. 物理基线、神经校正与损失函数

模型的物理基线来自门静脉到肝内分支的压力降：

```text
baseline_pa = dP_MPV + mean(dP_LPV, dP_RPV)
```

随后模型把以下信息拼成 `fused features`：8 条血管的图嵌入、每条血管的 Q、junction 物理特征、`q_scale`、物理基线和 26 维辅助特征。`Predictor MLP` 学习全局校正，`PhysicsResidualNet` 读取高 Re、高 WSS、低 solidity、剧烈面积变化等指标，学习局部非理想流动带来的残差。

![Prediction and loss module](picture/06_prediction_loss.png)

训练损失由多项组成：

| 损失项 | 作用 |
| --- | --- |
| `main` | PVP 主预测误差，使用加权 Huber，增强高压尾部学习。 |
| `murray` | 限制流量分配不要无约束偏离 Murray 先验。 |
| `press` | 约束左右肝内分支压力残差。 |
| `smooth` | 约束 `r_eff` 沿中心线平滑变化。 |
| `physio` | 限制 WSS、Re 等水动力量落在合理生理范围。 |
| `mono` | 约束压力降沿中心线单调增加。 |
| `residual` | 避免神经残差过大，防止完全覆盖物理基线。 |
| `spread` | 防止预测结果塌缩到均值。 |

## 输出与可解释性

一次 forward 不只返回 PVP，还返回完整的中间状态：

| 输出 | 解释 |
| --- | --- |
| `pvp_pred` | 最终 PVP 预测，处于归一化标签空间。 |
| `pvp_baseline_pa` / `pvp_baseline_norm` | Poiseuille 路径给出的物理压力基线。 |
| `pvp_physics` / `pvp_residual` | baseline+MLP 校正，以及局部 residual 校正。 |
| `Q`, `flow_out` | 每条血管相对流量、junction 分流比例、delta 和 mask。 |
| `hemo_per_seg` | 每条血管逐点速度、WSS、Re、阻力、压降等。 |
| `junction` | Murray 偏离、左右压力残差、侧支/TIPS/肝内分流比例等。 |
| `attn_weights` | 每条血管中心线上的注意力权重。 |
| `branch_embed` | 图网络后的血管节点表示。 |

这些输出让模型诊断更直接：如果某位患者预测为高压，可以进一步看是 MPV 压降高、肝内分支阻力高、TIPS 分流不足、侧支负荷重，还是 residual correction 在提示非理想流动风险。

## 训练

```bash
python train.py \
  --data_root /path/to/patient_dataset \
  --out_dir ./runs/current \
  --n_points 200 \
  --n_folds 5 \
  --exclude_00_prefix_samples \
  --epochs 300 \
  --batch_size 8 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --patience 40
```

训练流程会保存：

| 文件 | 内容 |
| --- | --- |
| `normalization.pt` | profile、aux、label 的训练集归一化统计 |
| `splits.json` | 交叉验证划分 |
| `fold_*/best.pt` | 每个 fold 的最佳模型 |
| `fold_*/history.csv` | 每个 epoch 的训练/验证指标 |
| `oof_predictions.csv` | out-of-fold 预测结果 |
| `oof_group_summary.json` | 分组诊断统计 |
| `summary.json` | 交叉验证总体指标 |

## 推理与可视化

```bash
python visualize.py \
  --checkpoint_dir ./runs/current \
  --data_root /path/to/patient_dataset \
  --out_dir ./inference_out \
  --fold 0 \
  --make_plots
```

可视化工具可以导出每个患者的血流动力学结果，并生成注意力曲线、流量分配对照图、STL/PLY 表面映射以及与 CFD 结果的对照统计。

## 文件结构

```text
PVP_predictor/
  dataset.py       # 数据发现、逐点重采样、mask、归一化、STL 体积
  model.py         # 几何编码、图网络、流量估计、水动力、预测头、损失
  train.py         # K-fold 训练、subject split、极端值采样、checkpoint
  diagnostics.py   # OOF 预测汇总、分组统计、兼容加载
  visualize.py     # 推理、hemodynamics 导出、attention/flow 图、STL 映射
  picture/         # README 中使用的模型与临床示意图
  tests/           # 关键行为测试
```

## 一句话总结

PVP Predictor 的核心不是用神经网络替代血流动力学，而是把门静脉系统的几何、拓扑、流量守恒和 Poiseuille 关系变成模型的默认推理路径，再让神经网络只学习真实临床场景中偏离理想物理公式的那一部分。
