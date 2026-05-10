# PVP Predictor v3 — Physics-Informed Geometric Deep Learning

基于门静脉系统几何（CT 提取的 centerline + 截面）预测门静脉压力（Portal Vein Pressure, PVP）的物理先验深度学习模型。

## v3 相对 v1/v2 的主要改动

### 一、输入特征精简（102 → 11 标量 + 4 通道几何）

旧版本一次性把 JSON 里所有 `statistical / system / global` 特征（共 102 维）喂给模型，包含大量派生量（比值、对称性、偏差、预先计算的阻力积分等）。

v3 的输入选择原则：

| 类别 | 来源 | 处理 |
|------|------|------|
| 逐点几何（4 通道） | `pointwise.{area, eq_diameter, curvature, inscribed_radius}` | **作为模型输入** |
| 弧长 (mm) | `pointwise.arc_length_mm` | **作为模型输入** |
| 4 个角度 + 1 个 planarity | `system.{angle_*, mpv_bifurc_planarity_deg}` | **作为模型输入**（非 1D profile 可推导，必须显式给） |
| 2 个拓扑计数 | `branchpoint_density_per_cm`, `n_collaterals_detected` | **作为模型输入** |
| 4 个二值 flag | `has_lgv`, `has_pgv`, `has_compensation_vessel`, `has_tips` | **作为模型输入** |
| 13 个比值/不对称性/偏差 | `*_ratio`, `*_asymmetry`, `*_deviation`, `splenic_dominance_index` 等 | **不作为输入**，模型自己计算 |
| 10 个阻力积分 | `*_resistance_integral`, `inflow_parallel_resistance` 等 | **不作为输入**，物理层用 Hagen-Poiseuille 直接积分 |
| 9 个分支统计量 (length, tortuosity, mean/max diameter, mean_area …) | `statistical.<branch>` | **不作为输入**，从 per-point profile 可推导 |

> 派生量保存在 `extras_for_eval` 里，训练后做 sanity comparison 用。

### 二、物理先验层（PoiseuilleHydrodynamics）

旧版的 `PhysicsPriorLayer` 是 8 个零散的"物理特征"，命名相互独立，单位混乱，**没有共享底层物理量**。WSS、velocity、pressure 互不约束。

v3 的核心约束：**每个分支只学一个标量 Q（流量）**。所有 per-point 血流场——速度、剪应力、雷诺数、阻力、压力降——都由 Q + 几何通过 Hagen-Poiseuille 推导：

| 物理量 | 公式 | 单位 |
|--------|------|------|
| `velocity_m_per_s` | `Q / A` | m/s |
| `wss_pa` | `4 μ Q / (π r³)` | Pa |
| `reynolds` | `v D / ν` | dimensionless |
| `local_R_pa_s_per_m4` | `8 μ / (π r⁴)` | Pa·s/m⁴/m |
| `cum_R_pa_s_per_m3` | `∫ R'(s) ds` | Pa·s/m³ |
| `pressure_drop_pa` | `Q · cum_R(s)` | Pa |

血流物理常数（37℃ 血液）：μ = 3.5 mPa·s，ρ = 1060 kg/m³，参考流量 Q_ref ≈ 800 mL/min。

**物理量内部一致性是结构性的，无法在训练中漂移开。**

### 三、流量参数化 = 质量守恒（不靠损失）

v3 的 `FlowRateEstimator`：

```
   target_logit_i  = 3 · log(d_i)        (Murray-3 先验)
   actual_logit_i  = target_logit_i + δ_i (模型学习的修正)
   split fraction  = softmax(logits, 缺失分支屏蔽为 -inf)
   Q_i             = split_i × Q_mpv
```

- `Q_sv + Q_smv = Q_mpv` 与 `Q_lpv + Q_rpv + Q_tips = Q_mpv` 由 softmax **结构性保证**，不需要质量守恒损失项
- 模型初始化时输出层置零 → 默认完全等于 Murray-3 先验，从解剖学合理的起点开始训练
- `δ_i` 的范数即"偏离 Murray 的程度"——这本身就是有生理意义的量（如肝硬化导致脾静脉相对优势）

### 四、6 项物理损失，都对应真实流体定律

| 损失项 | 物理含义 | 默认权重 |
|--------|----------|----------|
| `L_main` | Huber loss on PVP | 1.0 |
| `L_murray` | `‖δ_logits‖²`，偏离 Murray-3 的强度（先验正则） | 0.10 |
| `L_press` | 汇合处 `\|log(R_mpv / R_inflow_parallel)\|` 的标量残差，分叉处 `var(log R_branches)` | 0.05 |
| `L_smooth` | radius profile 的二阶导平方（几何先验） | 0.01 |
| `L_physio` | WSS 在 [0.05, 5] Pa、Re 在 [0, 1500] 之外的归一化 hinge | 0.01 |
| `L_mono` | `∫R(s) ds` 应单调（Poiseuille 已结构性保证，做兜底检查） | 0.05 |

注意：质量守恒不是损失项（已被参数化保证），让损失专注捕捉真正的流体力学。

### 五、可解释性：所有中间量带单位、可导出 STL

- `model.forward()` 返回的 `hemo_per_seg` 是 6 个 dict，每个含 10+ 个命名字段，单位在名字后缀（`_pa`, `_m_per_s`, `_pa_s_per_m3`）
- `visualize.export_patient_hemodynamics()` 导出每病人的 .npz，可直接加载到 ParaView/VTK
- `visualize.map_centerline_to_stl()` 将逐点血流场（如 wss_pa）映射到 STL 表面，输出 PLY，可与 CFD 仿真结果**定量对比**
- `visualize.compare_with_cfd()` 计算模型预测与 CFD ground truth 的 Pearson 相关
- `visualize.plot_attention()` 显示每个分支沿 centerline 的注意力分布（哪些点驱动了 PVP 预测）
- `visualize.plot_flow_splits()` 对比模型 Q 分流 vs Murray-3 先验，展示模型学到了什么解剖偏离

## 文件结构

```
PVP_predictor/
├── dataset.py        Dataset class with selective inputs and NaN-aware resampling
├── model.py          PoiseuilleHydrodynamics + FlowRateEstimator + JunctionPhysics
                      + PortalPressureNet + PhysicsInformedLoss
├── train.py          K-fold CV trainer (stratified by post-TIPS)
├── visualize.py      Inference, NPZ export, STL overlay, CFD comparison, plots
└── README.md         (this file)
```

## 训练

```bash
python train.py \
    --data_root /path/to/patients \
    --out_dir   ./runs/v3_full \
    --n_folds 5 --epochs 300 --batch_size 8 \
    --lr 1e-3 --weight_decay 1e-4 --patience 40
```

输出（`out_dir/`）：

```
fold_0/best.pt, history.csv      最佳模型 + 训练历史
fold_1/best.pt, history.csv
...
normalization.pt                 数据归一化统计量（推理时加载）
splits.json                      每折病人分配
summary.json                     跨折平均 MAE/RMSE/R²
```

## 推理与可视化

```bash
# 跑全部病人推理 + 可视化
python visualize.py \
    --checkpoint_dir ./runs/v3_full \
    --data_root /path/to/patients \
    --out_dir ./inference_out \
    --fold 0 --make_plots
```

输出 `inference_out/`：

- `<patient>.hemodynamics.npz` — per-point 血流场（按段命名，带单位）
- `<patient>.attention.png` — 每分支注意力曲线
- `<patient>.flow_splits.png` — 模型 Q 分流 vs Murray 先验对比
- `diagnostics.json` — 整队列的 junction physics residuals 表

## 与 CFD 对比

```python
from visualize import (
    run_inference, export_patient_hemodynamics,
    map_centerline_to_stl, compare_with_cfd,
)

# 1. 推理 + 导出
results = run_inference('./runs/v3_full', '/path/to/patients', patient_name='Patient_X')
export_patient_hemodynamics(results[0], '/tmp/Patient_X.model.npz')

# 2. 把模型预测 WSS 涂到 STL 表面（在 ParaView 里看）
map_centerline_to_stl(results[0], '/path/to/Patient_X_mpv.stl',
                      scalar_name='wss_pa',
                      ply_out_path='/tmp/Patient_X_mpv_pred.ply',
                      segment_filter=['mpv'])

# 3. 与 CFD 仿真定量对比（CFD 数据需先导出为相同 NPZ 格式）
corr = compare_with_cfd('/tmp/Patient_X.model.npz',
                        '/tmp/Patient_X.cfd_truth.npz',
                        fields=['velocity_m_per_s', 'wss_pa', 'pressure_drop_pa'])
print(corr)  # {'mpv_velocity_m_per_s': {'pearson': 0.87, 'mae': 0.12, ...}, ...}
```

## 设计哲学小结

1. **选择性输入**——能从基本量推导的，让模型自己算；不能（如角度）才显式给。
2. **物理量耦合**——每段一个 Q，所有 per-point 场都由 Poiseuille 导出，不会互相矛盾。
3. **质量守恒结构化**——softmax 流量分配天然满足 Kirchhoff 流量守恒。
4. **Murray 先验 + 学习偏离**——模型从解剖合理的起点出发，"偏离 Murray 的程度"即可解释生理变量。
5. **损失里都是真实物理**——压力连续、生理范围、几何平滑度，每一项都有清晰意义。
6. **输出单位明确**——所有中间量名字带 SI 单位后缀，可直接和 CFD 比。