# 特征半径/面积 QC

- 数据根目录：`F:\PCG data\dataset\test4all_sample`
- 扫描已分段样本：104
- MPV/SMV/SV 缺有效面积序列：4

## 问题类型计数

| 问题 | 样本数 |
|---|---:|
| `R_total_not_finite:missing_required_tree_resistance` | 4 |
| `Ratio_SMV_SV_not_finite:missing_smv_sv_diameter` | 4 |
| `sv_missing_valid_area` | 2 |
| `smv_missing_valid_area` | 2 |

## 特征状态计数

### R_total

| status | 样本数 |
|---|---:|
| `ok` | 100 |
| `missing_required_tree_resistance` | 4 |

### D_Murray

| status | 样本数 |
|---|---:|
| `ok` | 104 |

### Ratio_SMV_SV

| status | 样本数 |
|---|---:|
| `ok` | 100 |
| `missing_smv_sv_diameter` | 4 |

### R_collateral

| status | 样本数 |
|---|---:|
| `no_valid_collateral` | 57 |
| `ok` | 47 |

### Ratio_LPV_RPV

| status | 样本数 |
|---|---:|
| `ok` | 83 |
| `missing_lpv_rpv_diameter` | 21 |

## 需检查样本

| 样本 | 问题 |
|---|---|
| `20210412FanYuYing@xueshuan` | sv_missing_valid_area; R_total_not_finite:missing_required_tree_resistance; Ratio_SMV_SV_not_finite:missing_smv_sv_diameter |
| `20210603LiQin` | sv_missing_valid_area; R_total_not_finite:missing_required_tree_resistance; Ratio_SMV_SV_not_finite:missing_smv_sv_diameter |
| `20211208DuanXiuXia` | smv_missing_valid_area; R_total_not_finite:missing_required_tree_resistance; Ratio_SMV_SV_not_finite:missing_smv_sv_diameter |
| `20230408LiuTongLi` | smv_missing_valid_area; R_total_not_finite:missing_required_tree_resistance; Ratio_SMV_SV_not_finite:missing_smv_sv_diameter |
