# 数据集剖面特征 QC

- 数据根目录：`F:\PCG data\dataset\test4all_sample`
- 扫描样本：104
- 完全通过样本：67
- 有问题样本：37

## 检查口径

- MPV、SMV、SV 为必查血管；名字含 `#` 的样本额外必查 TIPS。
- LPV/RPV/LGV/PGV/TIPS 这类非必需血管：不存在不算错，存在则检查剖面字段。
- 剖面字段来自 `centerline_pointwise_profiles.json`，汇总统计来自 `portal_vein_features.json`。

## 问题类型计数

| 问题代码 | 数量 |
|---|---:|
| `lpv_invalid_profile_arrays` | 15 |
| `lpv_bad_summary_stats` | 15 |
| `mpv_invalid_profile_arrays` | 12 |
| `mpv_bad_summary_stats` | 12 |
| `smv_invalid_profile_arrays` | 11 |
| `smv_bad_summary_stats` | 11 |
| `rpv_invalid_profile_arrays` | 10 |
| `rpv_bad_summary_stats` | 10 |
| `lgv_invalid_profile_arrays` | 6 |
| `lgv_bad_summary_stats` | 6 |
| `sv_invalid_profile_arrays` | 3 |
| `sv_bad_summary_stats` | 3 |
| `tips_invalid_profile_arrays` | 2 |
| `tips_bad_summary_stats` | 2 |
| `pgv_invalid_profile_arrays` | 1 |
| `pgv_bad_summary_stats` | 1 |

## 按血管统计

| 血管 | 问题 | 数量 |
|---|---|---:|
| `mpv` | `mpv_invalid_profile_arrays` | 12 |
| `mpv` | `mpv_bad_summary_stats` | 12 |
| `sv` | `sv_invalid_profile_arrays` | 3 |
| `sv` | `sv_bad_summary_stats` | 3 |
| `smv` | `smv_invalid_profile_arrays` | 11 |
| `smv` | `smv_bad_summary_stats` | 11 |
| `lpv` | `lpv_invalid_profile_arrays` | 15 |
| `lpv` | `lpv_bad_summary_stats` | 15 |
| `rpv` | `rpv_invalid_profile_arrays` | 10 |
| `rpv` | `rpv_bad_summary_stats` | 10 |
| `tips` | `tips_invalid_profile_arrays` | 2 |
| `tips` | `tips_bad_summary_stats` | 2 |
| `lgv` | `lgv_invalid_profile_arrays` | 6 |
| `lgv` | `lgv_bad_summary_stats` | 6 |
| `pgv` | `pgv_invalid_profile_arrays` | 1 |
| `pgv` | `pgv_bad_summary_stats` | 1 |

## 半径/截面积字段为空的血管

| 样本 | 血管 | 空字段 |
|---|---|---|
| `0017989647TaQing` | `rpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `0019864392GuChangChun` | `mpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `0019864392GuChangChun` | `rpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `0020022521HouZhengXu` | `lpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;inscribed_radius;owned_radius;anchor_radius` |
| `20201207XuErMin@@@@` | `lpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;inscribed_radius;owned_radius;anchor_radius` |
| `20210305XuErMin@@@@` | `lpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;inscribed_radius;owned_radius;anchor_radius` |
| `20210331ZhangJun` | `mpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20210331ZhangJun` | `lpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20210412FanYuYing@xueshuan` | `sv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;inscribed_radius;owned_radius;anchor_radius` |
| `20210412FanYuYing@xueshuan` | `lpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20210412FanYuYing@xueshuan` | `rpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;inscribed_radius;owned_radius;anchor_radius` |
| `20210412FanYuYing@xueshuan` | `lgv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20210504FanYuYing#@xueshuan` | `lgv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20210510HanShengLi#` | `lgv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;inscribed_radius;owned_radius;anchor_radius` |
| `20210603LiQin` | `sv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;inscribed_radius;owned_radius;anchor_radius` |
| `20210616LiQin#` | `lgv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;inscribed_radius;owned_radius;anchor_radius` |
| `20210702WangTuanJie` | `lpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;inscribed_radius;owned_radius;anchor_radius` |
| `20210909WuJinHeng` | `rpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20210930XieFengE` | `lpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;inscribed_radius;owned_radius;anchor_radius` |
| `20211002WangTuanJie#` | `lpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;inscribed_radius;owned_radius;anchor_radius` |
| `20211006ZhaoShuangYing@centerline` | `mpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20211208DuanXiuXia` | `smv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;inscribed_radius;owned_radius;anchor_radius` |
| `20211208DuanXiuXia` | `lpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20211208DuanXiuXia` | `rpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;inscribed_radius;owned_radius;anchor_radius` |
| `20211208DuanXiuXia` | `tips` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20211208DuanXiuXia` | `lgv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;inscribed_radius;owned_radius;anchor_radius` |
| `20211215ZhangLinFu` | `mpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20211215ZhangLinFu` | `lpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20211215ZhangLinFu` | `rpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20220919LiuGuoQing` | `mpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20220930JinJunTing` | `sv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20220930JinJunTing` | `smv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20221003JiZhangKui` | `mpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20221003JiZhangKui` | `smv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20221008LiHuaMin` | `lgv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;inscribed_radius;owned_radius;anchor_radius` |
| `20221116WangJiuCen#` | `lpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20221227JinJunTing#` | `mpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20221227JinJunTing#` | `lpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20221227JinJunTing#` | `tips` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20230222LuJun#` | `mpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20230316WangTianShun` | `rpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20230408LiuTongLi` | `smv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;inscribed_radius;owned_radius;anchor_radius` |
| `20230408LiuTongLi` | `rpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;inscribed_radius;owned_radius;anchor_radius` |
| `20230408LiuTongLi` | `pgv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;inscribed_radius;owned_radius;anchor_radius` |
| `20230428GaoChunFeng#@@@@` | `mpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20230428GaoChunFeng#@@@@` | `smv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20230506HuangYongFeng` | `smv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20230719LiuYanChang#` | `lpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;inscribed_radius;owned_radius;anchor_radius` |
| `20230719LiuYanChang#` | `rpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20230729GuanJunLian` | `mpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20230729GuanJunLian` | `lpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20230814XuYongXin#` | `smv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20230902ZhaoSuCai#` | `lpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;inscribed_radius;owned_radius;anchor_radius` |
| `20230902ZhaoSuCai#` | `rpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;inscribed_radius;owned_radius;anchor_radius` |
| `20240128HanYiXing#` | `mpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20240128HanYiXing#` | `smv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20240226YangLin#` | `mpv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20240226YangLin#` | `smv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20240226ZhangHuaiQing$seg` | `smv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |
| `20240305ZhangHuaiQing$seg` | `smv` | `area;raw_area;eq_diameter;raw_eq_diameter;hydraulic_diameter;owned_radius;anchor_radius` |

## 核心血管问题（MPV/SMV/SV）

| 样本 | 血管 | 无效剖面数组 | 无效汇总特征 |
|---|---|---|---|
| `0019864392GuChangChun` | `mpv` | `area;eq_diameter;perimeter;raw_area;raw_eq_diameter;raw_perimeter;anchor_radius;owned_radius;hydraulic_diameter;circularity;solidity;n_components;dA_ds_norm` | `mpv_mean_diameter;mpv_max_diameter;mpv_mean_area;mpv_area_cv;mpv_mean_circularity` |
| `20210331ZhangJun` | `mpv` | `area;eq_diameter;perimeter;raw_area;raw_eq_diameter;raw_perimeter;anchor_radius;owned_radius;hydraulic_diameter;circularity;solidity;n_components;dA_ds_norm` | `mpv_mean_diameter;mpv_max_diameter;mpv_mean_area;mpv_area_cv;mpv_mean_circularity` |
| `20210412FanYuYing@xueshuan` | `sv` | `area;eq_diameter;perimeter;raw_area;raw_eq_diameter;raw_perimeter;anchor_radius;owned_radius;hydraulic_diameter;circularity;solidity;r_insc_to_r_eq_ratio;n_components;dA_ds_norm;inscribed_radius` | `sv_mean_diameter;sv_max_diameter;sv_mean_area;sv_area_cv;sv_mean_circularity` |
| `20210603LiQin` | `sv` | `area;eq_diameter;perimeter;raw_area;raw_eq_diameter;raw_perimeter;anchor_radius;owned_radius;hydraulic_diameter;circularity;solidity;r_insc_to_r_eq_ratio;n_components;dA_ds_norm;inscribed_radius` | `sv_mean_diameter;sv_max_diameter;sv_mean_area;sv_area_cv;sv_mean_circularity` |
| `20211006ZhaoShuangYing@centerline` | `mpv` | `area;eq_diameter;perimeter;raw_area;raw_eq_diameter;raw_perimeter;anchor_radius;owned_radius;hydraulic_diameter;circularity;solidity;n_components;dA_ds_norm` | `mpv_mean_diameter;mpv_max_diameter;mpv_mean_area;mpv_area_cv;mpv_mean_circularity` |
| `20211208DuanXiuXia` | `smv` | `area;eq_diameter;perimeter;raw_area;raw_eq_diameter;raw_perimeter;anchor_radius;owned_radius;hydraulic_diameter;circularity;solidity;r_insc_to_r_eq_ratio;n_components;dA_ds_norm;inscribed_radius` | `smv_mean_diameter;smv_max_diameter;smv_mean_area;smv_area_cv;smv_mean_circularity` |
| `20211215ZhangLinFu` | `mpv` | `area;eq_diameter;perimeter;raw_area;raw_eq_diameter;raw_perimeter;anchor_radius;owned_radius;hydraulic_diameter;circularity;solidity;n_components;dA_ds_norm` | `mpv_mean_diameter;mpv_max_diameter;mpv_mean_area;mpv_area_cv;mpv_mean_circularity` |
| `20220919LiuGuoQing` | `mpv` | `area;eq_diameter;perimeter;raw_area;raw_eq_diameter;raw_perimeter;anchor_radius;owned_radius;hydraulic_diameter;circularity;solidity;n_components;dA_ds_norm` | `mpv_mean_diameter;mpv_max_diameter;mpv_mean_area;mpv_area_cv;mpv_mean_circularity` |
| `20220930JinJunTing` | `smv` | `area;eq_diameter;perimeter;raw_area;raw_eq_diameter;raw_perimeter;anchor_radius;owned_radius;hydraulic_diameter;circularity;solidity;n_components;dA_ds_norm` | `smv_mean_diameter;smv_max_diameter;smv_mean_area;smv_area_cv;smv_mean_circularity` |
| `20220930JinJunTing` | `sv` | `area;eq_diameter;perimeter;raw_area;raw_eq_diameter;raw_perimeter;anchor_radius;owned_radius;hydraulic_diameter;circularity;solidity;n_components;dA_ds_norm` | `sv_mean_diameter;sv_max_diameter;sv_mean_area;sv_area_cv;sv_mean_circularity` |
| `20221003JiZhangKui` | `mpv` | `area;eq_diameter;perimeter;raw_area;raw_eq_diameter;raw_perimeter;anchor_radius;owned_radius;hydraulic_diameter;circularity;solidity;n_components;dA_ds_norm` | `mpv_mean_diameter;mpv_max_diameter;mpv_mean_area;mpv_area_cv;mpv_mean_circularity` |
| `20221003JiZhangKui` | `smv` | `area;eq_diameter;perimeter;raw_area;raw_eq_diameter;raw_perimeter;anchor_radius;owned_radius;hydraulic_diameter;circularity;solidity;n_components;dA_ds_norm` | `smv_mean_diameter;smv_max_diameter;smv_mean_area;smv_area_cv;smv_mean_circularity` |
| `20221227JinJunTing#` | `mpv` | `area;eq_diameter;perimeter;raw_area;raw_eq_diameter;raw_perimeter;anchor_radius;owned_radius;hydraulic_diameter;circularity;solidity;n_components;dA_ds_norm` | `mpv_mean_diameter;mpv_max_diameter;mpv_mean_area;mpv_area_cv;mpv_mean_circularity` |
| `20230222LuJun#` | `mpv` | `area;eq_diameter;perimeter;raw_area;raw_eq_diameter;raw_perimeter;anchor_radius;owned_radius;hydraulic_diameter;circularity;solidity;n_components;dA_ds_norm` | `mpv_mean_diameter;mpv_max_diameter;mpv_mean_area;mpv_area_cv;mpv_mean_circularity` |
| `20230408LiuTongLi` | `smv` | `area;eq_diameter;perimeter;raw_area;raw_eq_diameter;raw_perimeter;anchor_radius;owned_radius;hydraulic_diameter;circularity;solidity;r_insc_to_r_eq_ratio;n_components;dA_ds_norm;inscribed_radius` | `smv_mean_diameter;smv_max_diameter;smv_mean_area;smv_area_cv;smv_mean_circularity` |
| `20230428GaoChunFeng#@@@@` | `mpv` | `area;eq_diameter;perimeter;raw_area;raw_eq_diameter;raw_perimeter;anchor_radius;owned_radius;hydraulic_diameter;circularity;solidity;n_components;dA_ds_norm` | `mpv_mean_diameter;mpv_max_diameter;mpv_mean_area;mpv_area_cv;mpv_mean_circularity` |
| `20230428GaoChunFeng#@@@@` | `smv` | `area;eq_diameter;perimeter;raw_area;raw_eq_diameter;raw_perimeter;anchor_radius;owned_radius;hydraulic_diameter;circularity;solidity;n_components;dA_ds_norm` | `smv_mean_diameter;smv_max_diameter;smv_mean_area;smv_area_cv;smv_mean_circularity` |
| `20230506HuangYongFeng` | `smv` | `area;eq_diameter;perimeter;raw_area;raw_eq_diameter;raw_perimeter;anchor_radius;owned_radius;hydraulic_diameter;circularity;solidity;n_components;dA_ds_norm` | `smv_mean_diameter;smv_max_diameter;smv_mean_area;smv_area_cv;smv_mean_circularity` |
| `20230729GuanJunLian` | `mpv` | `area;eq_diameter;perimeter;raw_area;raw_eq_diameter;raw_perimeter;anchor_radius;owned_radius;hydraulic_diameter;circularity;solidity;n_components;dA_ds_norm` | `mpv_mean_diameter;mpv_max_diameter;mpv_mean_area;mpv_area_cv;mpv_mean_circularity` |
| `20230814XuYongXin#` | `smv` | `area;eq_diameter;perimeter;raw_area;raw_eq_diameter;raw_perimeter;anchor_radius;owned_radius;hydraulic_diameter;circularity;solidity;n_components;dA_ds_norm` | `smv_mean_diameter;smv_max_diameter;smv_mean_area;smv_area_cv;smv_mean_circularity` |
| `20240128HanYiXing#` | `mpv` | `area;eq_diameter;perimeter;raw_area;raw_eq_diameter;raw_perimeter;anchor_radius;owned_radius;hydraulic_diameter;circularity;solidity;n_components;dA_ds_norm` | `mpv_mean_diameter;mpv_max_diameter;mpv_mean_area;mpv_area_cv;mpv_mean_circularity` |
| `20240128HanYiXing#` | `smv` | `area;eq_diameter;perimeter;raw_area;raw_eq_diameter;raw_perimeter;anchor_radius;owned_radius;hydraulic_diameter;circularity;solidity;n_components;dA_ds_norm` | `smv_mean_diameter;smv_max_diameter;smv_mean_area;smv_area_cv;smv_mean_circularity` |
| `20240226YangLin#` | `mpv` | `area;eq_diameter;perimeter;raw_area;raw_eq_diameter;raw_perimeter;anchor_radius;owned_radius;hydraulic_diameter;circularity;solidity;n_components;dA_ds_norm` | `mpv_mean_diameter;mpv_max_diameter;mpv_mean_area;mpv_area_cv;mpv_mean_circularity` |
| `20240226YangLin#` | `smv` | `area;eq_diameter;perimeter;raw_area;raw_eq_diameter;raw_perimeter;anchor_radius;owned_radius;hydraulic_diameter;circularity;solidity;n_components;dA_ds_norm` | `smv_mean_diameter;smv_max_diameter;smv_mean_area;smv_area_cv;smv_mean_circularity` |
| `20240226ZhangHuaiQing$seg` | `smv` | `area;eq_diameter;perimeter;raw_area;raw_eq_diameter;raw_perimeter;anchor_radius;owned_radius;hydraulic_diameter;circularity;solidity;n_components;dA_ds_norm` | `smv_mean_diameter;smv_max_diameter;smv_mean_area;smv_area_cv;smv_mean_circularity` |
| `20240305ZhangHuaiQing$seg` | `smv` | `area;eq_diameter;perimeter;raw_area;raw_eq_diameter;raw_perimeter;anchor_radius;owned_radius;hydraulic_diameter;circularity;solidity;n_components;dA_ds_norm` | `smv_mean_diameter;smv_max_diameter;smv_mean_area;smv_area_cv;smv_mean_circularity` |

## 有问题样本

| 样本 | 问题 |
|---|---|
| `0017989647TaQing` | `rpv_invalid_profile_arrays;rpv_bad_summary_stats` |
| `0019864392GuChangChun` | `mpv_invalid_profile_arrays;mpv_bad_summary_stats;rpv_invalid_profile_arrays;rpv_bad_summary_stats` |
| `0020022521HouZhengXu` | `lpv_invalid_profile_arrays;lpv_bad_summary_stats` |
| `20201207XuErMin@@@@` | `lpv_invalid_profile_arrays;lpv_bad_summary_stats` |
| `20210305XuErMin@@@@` | `lpv_invalid_profile_arrays;lpv_bad_summary_stats` |
| `20210331ZhangJun` | `mpv_invalid_profile_arrays;mpv_bad_summary_stats;lpv_invalid_profile_arrays;lpv_bad_summary_stats` |
| `20210412FanYuYing@xueshuan` | `sv_invalid_profile_arrays;sv_bad_summary_stats;lpv_invalid_profile_arrays;lpv_bad_summary_stats;rpv_invalid_profile_arrays;rpv_bad_summary_stats;lgv_invalid_profile_arrays;lgv_bad_summary_stats` |
| `20210504FanYuYing#@xueshuan` | `lgv_invalid_profile_arrays;lgv_bad_summary_stats` |
| `20210510HanShengLi#` | `lgv_invalid_profile_arrays;lgv_bad_summary_stats` |
| `20210603LiQin` | `sv_invalid_profile_arrays;sv_bad_summary_stats` |
| `20210616LiQin#` | `lgv_invalid_profile_arrays;lgv_bad_summary_stats` |
| `20210702WangTuanJie` | `lpv_invalid_profile_arrays;lpv_bad_summary_stats` |
| `20210909WuJinHeng` | `rpv_invalid_profile_arrays;rpv_bad_summary_stats` |
| `20210930XieFengE` | `lpv_invalid_profile_arrays;lpv_bad_summary_stats` |
| `20211002WangTuanJie#` | `lpv_invalid_profile_arrays;lpv_bad_summary_stats` |
| `20211006ZhaoShuangYing@centerline` | `mpv_invalid_profile_arrays;mpv_bad_summary_stats` |
| `20211208DuanXiuXia` | `smv_invalid_profile_arrays;smv_bad_summary_stats;lpv_invalid_profile_arrays;lpv_bad_summary_stats;rpv_invalid_profile_arrays;rpv_bad_summary_stats;tips_invalid_profile_arrays;tips_bad_summary_stats;lgv_invalid_profile_arrays;lgv_bad_summary_stats` |
| `20211215ZhangLinFu` | `mpv_invalid_profile_arrays;mpv_bad_summary_stats;lpv_invalid_profile_arrays;lpv_bad_summary_stats;rpv_invalid_profile_arrays;rpv_bad_summary_stats` |
| `20220919LiuGuoQing` | `mpv_invalid_profile_arrays;mpv_bad_summary_stats` |
| `20220930JinJunTing` | `sv_invalid_profile_arrays;sv_bad_summary_stats;smv_invalid_profile_arrays;smv_bad_summary_stats` |
| `20221003JiZhangKui` | `mpv_invalid_profile_arrays;mpv_bad_summary_stats;smv_invalid_profile_arrays;smv_bad_summary_stats` |
| `20221008LiHuaMin` | `lgv_invalid_profile_arrays;lgv_bad_summary_stats` |
| `20221116WangJiuCen#` | `lpv_invalid_profile_arrays;lpv_bad_summary_stats` |
| `20221227JinJunTing#` | `mpv_invalid_profile_arrays;mpv_bad_summary_stats;lpv_invalid_profile_arrays;lpv_bad_summary_stats;tips_invalid_profile_arrays;tips_bad_summary_stats` |
| `20230222LuJun#` | `mpv_invalid_profile_arrays;mpv_bad_summary_stats` |
| `20230316WangTianShun` | `rpv_invalid_profile_arrays;rpv_bad_summary_stats` |
| `20230408LiuTongLi` | `smv_invalid_profile_arrays;smv_bad_summary_stats;rpv_invalid_profile_arrays;rpv_bad_summary_stats;pgv_invalid_profile_arrays;pgv_bad_summary_stats` |
| `20230428GaoChunFeng#@@@@` | `mpv_invalid_profile_arrays;mpv_bad_summary_stats;smv_invalid_profile_arrays;smv_bad_summary_stats` |
| `20230506HuangYongFeng` | `smv_invalid_profile_arrays;smv_bad_summary_stats` |
| `20230719LiuYanChang#` | `lpv_invalid_profile_arrays;lpv_bad_summary_stats;rpv_invalid_profile_arrays;rpv_bad_summary_stats` |
| `20230729GuanJunLian` | `mpv_invalid_profile_arrays;mpv_bad_summary_stats;lpv_invalid_profile_arrays;lpv_bad_summary_stats` |
| `20230814XuYongXin#` | `smv_invalid_profile_arrays;smv_bad_summary_stats` |
| `20230902ZhaoSuCai#` | `lpv_invalid_profile_arrays;lpv_bad_summary_stats;rpv_invalid_profile_arrays;rpv_bad_summary_stats` |
| `20240128HanYiXing#` | `mpv_invalid_profile_arrays;mpv_bad_summary_stats;smv_invalid_profile_arrays;smv_bad_summary_stats` |
| `20240226YangLin#` | `mpv_invalid_profile_arrays;mpv_bad_summary_stats;smv_invalid_profile_arrays;smv_bad_summary_stats` |
| `20240226ZhangHuaiQing$seg` | `smv_invalid_profile_arrays;smv_bad_summary_stats` |
| `20240305ZhangHuaiQing$seg` | `smv_invalid_profile_arrays;smv_bad_summary_stats` |
