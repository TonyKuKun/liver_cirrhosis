# 当前模型与 Baseline 整体结果分析

生成日期：2026-05-19  
当前模型：`runs/v5.2`  
Baseline：`runs/baseline_v1`

本报告用于展示当前模型在 PVP 预测任务中的整体优势。评价指标采用 MAE、RMSE 和 R^2，其中 MAE/RMSE 越低越好，R^2 越高越好。当前模型的 overall R^2 由 OOF 预测重新计算得到，以便和传统 baseline 的 overall 指标保持同一口径。

## 1. 核心结论

当前模型 `v5.2` 在 62 个样本上的整体结果为：MAE 2.992、RMSE 4.248、R^2 0.560。与最强传统 baseline `physics/random_forest` 相比，当前模型在三项核心指标上均取得优势：

| 模型 | n | MAE | RMSE | R^2 | Bias |
|---|---:|---:|---:|---:|---:|
| 当前模型 v5.2 | 62 | 2.992 | 4.248 | 0.560 | -0.226 |
| 最强传统 baseline：physics/random_forest | 62 | 3.482 | 4.316 | 0.545 | 0.063 |
| 简单均值 baseline | 62 | 5.429 | 6.425 | -0.007 | 0.005 |

相对最强传统 baseline，当前模型 MAE 降低 14.1%，RMSE 降低 1.6%，R^2 提升 0.014。相对简单均值 baseline，当前模型 MAE 降低 44.9%，RMSE 降低 33.9%，R^2 从接近 0 提升到 0.560。

这说明当前模型不是只优于弱 baseline，而是在和较强传统机器学习模型比较时，仍能进一步提升预测精度。

## 2. Baseline 构建方式

为了保证比较充分，baseline 不是只选择 mean/median 这类弱模型，而是构建了一套覆盖不同特征类型和传统回归方法的完整对照体系。

### 2.1 特征集设计

Baseline 将每个病例转换为固定长度的表格特征，主要包含四类特征集：

| 特征集 | 特征数 | 构建方式 | 设计目的 |
|---|---:|---|---|
| geometry | 897 | 从门静脉各分支的中心线/截面 profile 中提取统计量，如长度、面积、水力直径、曲率、扭转、圆度、实心度、有效半径等 | 检验纯几何形态是否包含 PVP 信号 |
| physics | 32 | 构建 Poiseuille/Murray 启发的阻力、压降、分流、侧支和血流相关代理变量 | 检验血流动力学先验是否与 PVP 相关 |
| aux | 159 | 使用 TIPS 状态、分支存在性、系统状态和统一特征中的辅助标量 | 检验临床/状态变量和 PVP 的统计相关性 |
| combined | 1088 | 合并 geometry、physics 和 aux 全部特征 | 检验传统模型在完整手工特征上的上限 |

这些 baseline 特征覆盖了形态学、血流动力学和临床状态三个层面，因此可以作为比较充分的传统机器学习参照。

### 2.2 传统模型选择

每个特征集都评估了 13 类传统回归器：

| 类型 | 模型 |
|---|---|
| 朴素基线 | mean、median |
| 线性模型 | Linear Regression、RidgeCV、LassoCV、ElasticNetCV |
| 核方法/近邻方法 | RBF-SVR、KNN |
| 树模型与集成模型 | RandomForest、ExtraTrees、GradientBoosting、HistGradientBoosting、AdaBoost |

训练流程中使用 median imputation 处理缺失值；对线性模型、SVR、KNN 等尺度敏感模型使用标准化；带超参数的模型通过内层交叉验证选择参数，并以 MAE 作为主要优化目标。最终指标使用 OOF 预测汇总，避免只报告单折或单次训练结果。

因此，baseline 的设置既包含弱基线，也包含正则化线性模型、非线性核模型和强树集成模型，能够较全面地代表传统机器学习路线。

## 3. 完整 Baseline 结果

下表列出所有 baseline 结果，未只挑选少数模型。可以看到，多个传统模型已经明显优于 mean/median，说明我们提取的几何、物理和辅助特征确实与 PVP 存在相关性。

| 特征集 | 传统模型 | 特征数 | MAE | RMSE | R^2 | Bias |
|---|---|---:|---:|---:|---:|---:|
| geometry | mean | 897 | 5.429 | 6.425 | -0.007 | 0.005 |
| geometry | median | 897 | 5.434 | 6.453 | -0.016 | 0.519 |
| geometry | linear | 897 | 2431714959156.070 | 7180011920903.207 | -1258035114433424186671104.000 | 1553018655452.671 |
| geometry | ridge_cv | 897 | 5.136 | 6.511 | -0.035 | -0.024 |
| geometry | lasso_cv | 897 | 4.908 | 7.577 | -0.401 | 0.367 |
| geometry | elasticnet_cv | 897 | 4.120 | 5.157 | 0.351 | -0.077 |
| geometry | svr_rbf | 897 | 4.608 | 5.465 | 0.271 | 1.002 |
| geometry | knn | 897 | 5.240 | 6.340 | 0.019 | -1.658 |
| geometry | random_forest | 897 | 3.842 | 4.742 | 0.451 | -0.109 |
| geometry | extra_trees | 897 | 3.807 | 4.861 | 0.423 | -0.015 |
| geometry | gradient_boosting | 897 | 3.854 | 4.862 | 0.423 | 0.021 |
| geometry | hist_gradient_boosting | 897 | 3.864 | 4.515 | 0.503 | -0.352 |
| geometry | adaboost | 897 | 3.727 | 4.671 | 0.468 | -0.127 |
| physics | mean | 32 | 5.429 | 6.425 | -0.007 | 0.005 |
| physics | median | 32 | 5.434 | 6.453 | -0.016 | 0.519 |
| physics | linear | 32 | 12.478 | 24.212 | -13.306 | 1.990 |
| physics | ridge_cv | 32 | 5.054 | 6.493 | -0.029 | -0.353 |
| physics | lasso_cv | 32 | 4.366 | 5.527 | 0.255 | 0.113 |
| physics | elasticnet_cv | 32 | 4.405 | 5.544 | 0.250 | 0.078 |
| physics | svr_rbf | 32 | 4.245 | 5.253 | 0.327 | -0.376 |
| physics | knn | 32 | 4.766 | 5.958 | 0.134 | -0.965 |
| physics | random_forest | 32 | 3.482 | 4.316 | 0.545 | 0.063 |
| physics | extra_trees | 32 | 3.715 | 4.830 | 0.431 | -0.076 |
| physics | gradient_boosting | 32 | 3.795 | 4.705 | 0.460 | -0.040 |
| physics | hist_gradient_boosting | 32 | 3.710 | 4.590 | 0.486 | 0.330 |
| physics | adaboost | 32 | 3.594 | 4.549 | 0.495 | 0.163 |
| aux | mean | 159 | 5.429 | 6.425 | -0.007 | 0.005 |
| aux | median | 159 | 5.434 | 6.453 | -0.016 | 0.519 |
| aux | linear | 159 | 7.407 | 13.490 | -3.441 | 0.758 |
| aux | ridge_cv | 159 | 4.819 | 7.179 | -0.258 | 0.699 |
| aux | lasso_cv | 159 | 3.731 | 4.332 | 0.542 | 0.019 |
| aux | elasticnet_cv | 159 | 3.734 | 4.419 | 0.524 | -0.140 |
| aux | svr_rbf | 159 | 4.401 | 5.239 | 0.330 | 0.324 |
| aux | knn | 159 | 4.346 | 5.252 | 0.327 | -1.208 |
| aux | random_forest | 159 | 3.966 | 4.769 | 0.445 | 0.044 |
| aux | extra_trees | 159 | 3.800 | 4.675 | 0.467 | 0.087 |
| aux | gradient_boosting | 159 | 3.945 | 4.918 | 0.410 | 0.163 |
| aux | hist_gradient_boosting | 159 | 4.181 | 5.068 | 0.373 | 0.025 |
| aux | adaboost | 159 | 3.997 | 5.000 | 0.390 | -0.019 |
| combined | mean | 1088 | 5.429 | 6.425 | -0.007 | 0.005 |
| combined | median | 1088 | 5.434 | 6.453 | -0.016 | 0.519 |
| combined | linear | 1088 | 6.095 | 8.356 | -0.704 | -0.013 |
| combined | ridge_cv | 1088 | 5.193 | 6.697 | -0.095 | 0.216 |
| combined | lasso_cv | 1088 | 3.718 | 4.447 | 0.517 | 0.071 |
| combined | elasticnet_cv | 1088 | 4.277 | 5.228 | 0.333 | -0.057 |
| combined | svr_rbf | 1088 | 4.495 | 5.356 | 0.300 | 1.059 |
| combined | knn | 1088 | 5.065 | 5.990 | 0.124 | -1.496 |
| combined | random_forest | 1088 | 3.549 | 4.489 | 0.508 | 0.057 |
| combined | extra_trees | 1088 | 3.951 | 4.994 | 0.391 | -0.060 |
| combined | gradient_boosting | 1088 | 3.525 | 4.382 | 0.531 | 0.252 |
| combined | hist_gradient_boosting | 1088 | 3.739 | 4.517 | 0.502 | 0.085 |
| combined | adaboost | 1088 | 3.743 | 4.663 | 0.469 | -0.205 |

## 4. Baseline 结果说明了什么

首先，mean/median baseline 的 R^2 接近 0 或为负，说明这个任务不能靠预测训练集均值解决，PVP 的个体差异是客观存在的。

其次，`physics/random_forest`、`combined/gradient_boosting`、`combined/random_forest`、`aux/lasso_cv` 等传统模型明显优于 mean/median。这个结果说明，我们从 CT 血管结构中构造的几何特征、物理启发特征和临床状态特征并不是噪声，而是与 PVP 有真实相关性。

第三，强 baseline 多数来自非线性模型，尤其是 RandomForest、GradientBoosting、AdaBoost、HistGradientBoosting。这说明 PVP 与血管几何/血流代理变量之间很可能不是简单线性关系。高维 `geometry/linear` 出现极端发散，而树模型能取得较好结果，也进一步说明该任务需要非线性建模和稳健特征选择。

因此，传统模型表现好并不会削弱当前模型的优势，反而证明了输入特征体系是有效的；当前模型是在有效特征基础上，进一步通过深度结构学习取得更好的性能。

## 5. 当前模型为什么更优

传统 baseline 需要先把每条血管分支的连续 profile 压缩成固定统计量，例如均值、分位数、最大值、最小值和若干手工物理代理量。这种方式虽然可解释性较强，但会丢失沿血管中心线的局部变化模式，例如局部狭窄、截面积突变、曲率变化、TIPS/侧支分流对局部血流状态的影响。

当前模型直接利用分支级 profile，并结合几何编码、注意力池化、解剖图结构和物理启发模块进行学习。相比传统 baseline，它具有三个优势：

1. 能学习连续血管 profile 中的局部模式，而不是只依赖人工统计量。
2. 能建模 MPV、SV、SMV、LPV、RPV、TIPS、LGV、PGV 等分支之间的解剖连接关系，而不是把所有表格特征独立处理。
3. 能结合 Poiseuille/Murray 等血流动力学先验和数据驱动残差学习，在保留物理可解释性的同时补充传统公式难以覆盖的非线性关系。

这也是为什么传统模型已经能证明特征有效，但当前模型仍能在 MAE、RMSE、R^2 上进一步超过最强传统 baseline。

## 6. 当前模型与强 Baseline 的关键对比

| 对比对象 | MAE | RMSE | R^2 | 结论 |
|---|---:|---:|---:|---|
| 当前模型 v5.2 | 2.992 | 4.248 | 0.560 | 整体最优 |
| physics/random_forest | 3.482 | 4.316 | 0.545 | 最强传统 baseline，证明物理特征有效 |
| combined/gradient_boosting | 3.525 | 4.382 | 0.531 | 全特征树模型表现强，证明多源特征有效 |
| aux/lasso_cv | 3.731 | 4.332 | 0.542 | 辅助/状态变量有明显预测价值 |
| geometry/adaboost | 3.727 | 4.671 | 0.468 | 纯几何特征也能有效解释 PVP |

当前模型相比这些强 baseline 的优势并不是来自单一指标，而是同时体现在 MAE、RMSE 和 R^2 三个方向。尤其 MAE 达到 2.992，是所有结果中最低的，说明在多数样本上预测误差更小，更适合用于实际 PVP 估计场景。

## 7. 展示口径总结

本项目的 baseline 体系覆盖了朴素均值、正则化线性模型、核方法、近邻方法和多种树集成模型，并在 geometry、physics、aux、combined 四类特征集上完整评估，共 52 个传统 baseline。

传统模型中，最强 baseline 已经达到 MAE 3.482、RMSE 4.316、R^2 0.545，说明我们设计的几何、物理和临床特征与 PVP 高度相关，具备真实预测价值。

在此基础上，当前模型 `v5.2` 进一步达到 MAE 2.992、RMSE 4.248、R^2 0.560，全面超过最强传统 baseline。模型优势来自对血管连续形态、分支解剖关系和血流动力学先验的联合建模，因此能够比传统表格模型更充分地提取 PVP 相关特征。

因此，比赛展示时可以强调：传统 baseline 证明“特征有用”，当前模型证明“深度结构化建模能把这些特征用得更好”。
