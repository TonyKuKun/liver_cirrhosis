"""
门静脉特征与临床指标相关性分析（v2 - 适配新特征 schema）
========================================================
新版变化:
  - 与 extract_features v3 字段对齐 (mpv_mean_area 替代 mpv_cross_section_area 等)
  - 支持新增段: TIPS / LGV / PGV / LPV / RPV 完整特征集
  - 自动跳过该患者缺失的段 (None 值)

文件夹结构:
    root_folder/
      patient_001/
        features/
          unified_features.json
        label/
          PVP.txt
          PCG.txt
"""

import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.stats import spearmanr, pearsonr, t as student_t
import warnings
warnings.filterwarnings('ignore')

from features_layout import (
    FEATURES_DIRNAME,
    UNIFIED_FEATURES_NAME,
)
from system_features import SYSTEM_FEATURE_NAMES, SYSTEM_FEATURE_LABELS_CN


# ============================================================
# 特征字段定义（与 extract_features.py v3 一致）
# ============================================================

ALL_SEG_NAMES = ['mpv', 'sv', 'smv', 'lpv', 'rpv', 'tips', 'lgv', 'pgv']

# 每段的 9 个特征
PER_SEG_FEATURES = [
    'length', 'tortuosity',
    'mean_curvature', 'max_curvature',
    'mean_diameter', 'max_diameter',
    'mean_area', 'area_cv',
    'mean_circularity',
]

# 全局特征
GLOBAL_FEATURES = [
    'total_centerline_length',
    'sv_smv_diameter_ratio',
    'sv_smv_angle',
    'has_lgv', 'has_pgv', 'has_compensation_vessel', 'has_tips',
]

# 系统/联合特征 (literature-driven, 见 system_features.py)
SYSTEM_FEATURES = list(SYSTEM_FEATURE_NAMES)

# TIPS 术后记录不参与侧支分析。侧支仅指自然形成的 LGV/PGV，绝不包含
# 人工分流道 TIPS。
COLLATERAL_FEATURES = {
    *(f"{seg}_{feat}" for seg in ('lgv', 'pgv') for feat in PER_SEG_FEATURES),
    'has_lgv', 'has_pgv', 'has_compensation_vessel',
    'lgv_mpv_diameter_ratio', 'pgv_mpv_diameter_ratio',
    'collateral_length_mpv_ratio', 'collateral_burden_score',
    'n_collaterals_detected', 'max_collateral_diameter_mm',
}

# 术前专门检验的侧支指标。排除与 has_lgv/has_pgv 完全冗余的
# has_compensation_vessel，保留 n_collaterals 仅供探索性核对。
PRETIPS_COLLATERAL_FEATURES = [
    'max_collateral_diameter_mm',
    'collateral_burden_score',
    'has_lgv', 'has_pgv',
    'n_collaterals_detected',
    'collateral_length_mpv_ratio',
    'lgv_mpv_diameter_ratio', 'pgv_mpv_diameter_ratio',
]


def _build_feature_names():
    """组合出所有特征字段名 (顺序: 段×特征 + 全局 + 系统)"""
    names = []
    for seg in ALL_SEG_NAMES:
        for feat in PER_SEG_FEATURES:
            names.append(f"{seg}_{feat}")
    names.extend(GLOBAL_FEATURES)
    names.extend(SYSTEM_FEATURES)
    return names


FEATURE_NAMES = _build_feature_names()


# ============================================================
# 标签 (中文显示)
# ============================================================

SEG_LABELS_CN = {
    'mpv': 'MPV', 'sv': 'SV', 'smv': 'SMV',
    'lpv': 'LPV', 'rpv': 'RPV', 'tips': 'TIPS',
    'lgv': 'LGV', 'pgv': 'PGV',
}

FEAT_LABELS_CN = {
    'length':           '长度',
    'tortuosity':       '曲折度',
    'mean_curvature':   '平均曲率',
    'max_curvature':    '最大曲率',
    'mean_diameter':    '平均直径',
    'max_diameter':     '最大直径',
    'mean_area':        '平均截面积',
    'area_cv':          '面积变异',
    'mean_circularity': '圆度',
}


def _build_feature_labels_cn():
    labels = {}
    for seg in ALL_SEG_NAMES:
        for feat in PER_SEG_FEATURES:
            key = f"{seg}_{feat}"
            labels[key] = f"{SEG_LABELS_CN[seg]}_{FEAT_LABELS_CN[feat]}"
    labels['total_centerline_length'] = '总中心线长'
    labels['sv_smv_diameter_ratio'] = 'SV/SMV直径比'
    labels['sv_smv_angle'] = 'SV-SMV夹角'
    labels['has_lgv'] = '存在LGV'
    labels['has_pgv'] = '存在PGV'
    labels['has_compensation_vessel'] = '存在代偿血管'
    labels['has_tips'] = '术后(TIPS)'
    # 系统特征
    for k, v in SYSTEM_FEATURE_LABELS_CN.items():
        labels.setdefault(k, v)
    return labels


FEATURE_LABELS_CN = _build_feature_labels_cn()


TARGET_LABELS = {
    'PVP': {'cn': '门静脉压力', 'unit': 'mmHg'},
    'PCG': {'cn': '门静脉压力梯度', 'unit': 'mmHg'},
}


# 按特征类型分组(用于配色和分组分析)
FEATURE_GROUPS = {
    '长度': [f"{seg}_length" for seg in ALL_SEG_NAMES] + ['total_centerline_length'],
    '曲折度': [f"{seg}_tortuosity" for seg in ALL_SEG_NAMES],
    '曲率': ([f"{seg}_mean_curvature" for seg in ALL_SEG_NAMES]
             + [f"{seg}_max_curvature" for seg in ALL_SEG_NAMES]),
    '直径/面积': ([f"{seg}_mean_diameter" for seg in ALL_SEG_NAMES]
                + [f"{seg}_max_diameter" for seg in ALL_SEG_NAMES]
                + [f"{seg}_mean_area" for seg in ALL_SEG_NAMES]
                + [f"{seg}_area_cv" for seg in ALL_SEG_NAMES]
                + ['sv_smv_diameter_ratio']),
    '圆度': [f"{seg}_mean_circularity" for seg in ALL_SEG_NAMES],
    '夹角': ['sv_smv_angle'],
    # ---- 系统/联合特征 ----
    '系统-角度': [n for n in SYSTEM_FEATURE_NAMES
                if n.startswith('angle_') or 'planarity' in n],
    '系统-Murray/比率': [
        'sv_smv_diameter_asymmetry', 'sv_mpv_diameter_ratio',
        'smv_mpv_diameter_ratio',
        'confluence_murray3_ratio', 'confluence_murray3_deviation',
        'confluence_area_ratio',
        'mpv_bifurc_murray3_ratio', 'mpv_bifurc_murray3_deviation',
        'mpv_bifurc_area_ratio',
        'lpv_rpv_diameter_asymmetry',
        'lgv_mpv_diameter_ratio', 'pgv_mpv_diameter_ratio',
        'splenic_dominance_index'],
    '系统-长度/弯曲': [
        'splenoportal_path_chord_ratio',
        'collateral_length_mpv_ratio',
        'diameter_weighted_tortuosity'],
    '系统-阻力': [
        'mpv_resistance_integral', 'sv_resistance_integral',
        'smv_resistance_integral', 'lpv_resistance_integral',
        'rpv_resistance_integral', 'tips_resistance_integral',
        'inflow_parallel_resistance', 'inflow_resistance_asymmetry',
        'mpv_effective_radius', 'tips_inflow_resistance_ratio'],
    '系统-拓扑': [
        'collateral_burden_score', 'n_collaterals_detected',
        'branchpoint_density_per_cm',
        'mpv_taper_coefficient',
        'mpv_proximal_diameter', 'mpv_distal_diameter',
        'mpv_min_max_diameter_ratio',
        'tree_area_conservation_mean_dev',
        'has_lgv', 'has_pgv', 'has_compensation_vessel', 'has_tips'],
    '系统-临床派生': [
        'sv_max_to_mpv_max_diam_ratio', 'mpv_trunk_length_mm',
        'max_tortuosity_index', 'mean_tortuosity_index',
        'max_collateral_diameter_mm',
        'area_conservation_bifurc_deviation',
        'tips_stent_diameter_mm', 'tips_stent_length_mm',
        'pvt_severity_grade', 'min_lumen_area_to_max_ratio_mpv',
        'cavernous_transformation_flag'],
}

GROUP_COLORS = {
    '长度': '#3b82f6', '曲折度': '#f59e0b', '曲率': '#8b5cf6',
    '直径/面积': '#ef4444', '圆度': '#ec4899', '夹角': '#10b981',
    '系统-角度': '#0ea5e9',
    '系统-Murray/比率': '#dc2626',
    '系统-长度/弯曲': '#f97316',
    '系统-阻力': '#7c3aed',
    '系统-拓扑': '#16a34a',
    '系统-临床派生': '#475569',
}


def _get_group(feat):
    for g, feats in FEATURE_GROUPS.items():
        if feat in feats:
            return g
    return '其他'


def _sig(p):
    if pd.isna(p): return ''
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    return ''


def _sig_text(value):
    return '' if pd.isna(value) else str(value)


def _fdr_bh(p_values):
    """Benjamini-Hochberg adjusted p-values, preserving NaN positions."""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid_idx = np.flatnonzero(np.isfinite(values))
    if len(valid_idx) == 0:
        return adjusted

    valid = values[valid_idx]
    order = np.argsort(valid)
    ranked = valid[order]
    n_tests = len(ranked)
    q_ranked = ranked * n_tests / np.arange(1, n_tests + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)

    q_valid = np.empty(n_tests, dtype=float)
    q_valid[order] = q_ranked
    adjusted[valid_idx] = q_valid
    return adjusted


# ============================================================
# 数据收集
# ============================================================

def _flatten_unified_to_features(unified):
    """从 unified_features.json 还原成相关性分析使用的 flat dict。"""
    out = {}
    # statistical: nested → flat
    statistical = unified.get('statistical') or {}
    for seg, block in statistical.items():
        if not isinstance(block, dict):
            continue
        for k, v in block.items():
            out[f"{seg}_{k}"] = v
    # global
    glob = unified.get('global') or {}
    for k, v in glob.items():
        out[k] = v
    # system: v1 是扁平 dict; v2 拆成 available / unavailable。
    system = unified.get('system') or {}
    if isinstance(system, dict) and (
            'available' in system or 'unavailable' in system):
        for k, v in (system.get('available') or {}).items():
            out[k] = v
        all_values = system.get('all_values') or {}
        for k in SYSTEM_FEATURES:
            if k not in out:
                out[k] = all_values.get(k)
        for k in (system.get('unavailable') or {}):
            out.setdefault(k, None)
    else:
        for k, v in system.items():
            out[k] = v
    return out


def _read_label_value(folder_path, target):
    """从 patient/label/{target}.txt 读数值。"""
    txt_path = os.path.join(folder_path, "label", f"{target}.txt")
    if not os.path.exists(txt_path):
        return None
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            for token in text.replace(',', ' ').split():
                try:
                    value = float(token)
                    return value if np.isfinite(value) else None
                except ValueError:
                    continue
    except Exception:
        pass
    return None


def _infer_subject_id(sample_name):
    """Collapse dated pre/post folders while preserving non-date hospital IDs."""
    value = str(sample_name).strip()
    if re.match(r'^(?:19|20)\d{6}', value):
        value = value[8:]
    value = re.split(r'[#@$]', value, maxsplit=1)[0].strip()
    return value or str(sample_name)


def _cluster_jackknife_inference(x, y, subject_ids, estimate, statistic):
    """Patient-cluster jackknife p-value and confidence interval."""
    subject_ids = np.asarray(subject_ids, dtype=object)
    unique_subjects = pd.unique(subject_ids)
    n_subjects = len(unique_subjects)
    if n_subjects == len(x) or n_subjects < 3:
        return np.nan, np.nan, np.nan, n_subjects

    leave_one_out = []
    for subject_id in unique_subjects:
        keep = subject_ids != subject_id
        if np.sum(keep) < 3:
            return np.nan, np.nan, np.nan, n_subjects
        try:
            value = float(statistic(x[keep], y[keep]))
        except Exception:
            return np.nan, np.nan, np.nan, n_subjects
        if not np.isfinite(value):
            return np.nan, np.nan, np.nan, n_subjects
        leave_one_out.append(value)

    leave_one_out = np.asarray(leave_one_out, dtype=float)
    pseudo_values = (n_subjects * estimate
                     - (n_subjects - 1) * leave_one_out)
    standard_error = float(
        np.std(pseudo_values, ddof=1) / np.sqrt(n_subjects))
    if standard_error <= 1e-15:
        p_value = 0.0 if abs(estimate) > 1e-15 else 1.0
        return p_value, estimate, estimate, n_subjects

    z_value = estimate / standard_error
    p_value = float(2 * student_t.sf(
        abs(z_value), df=n_subjects - 1))
    critical = float(student_t.ppf(0.975, df=n_subjects - 1))
    ci_low = max(-1.0, float(estimate - critical * standard_error))
    ci_high = min(1.0, float(estimate + critical * standard_error))
    return p_value, ci_low, ci_high, n_subjects


def collect_features(root_folder, target="PVP", output_txt=None,
                     drop_features_above_missing=0.5,
                     skip_folder_markers=('@',)):
    """
    遍历子文件夹, 读取 features/unified_features.json + label, 汇总为 TSV。

    参数:
        drop_features_above_missing: 缺失率超过此比例的特征不参与分析
                                     (例如 LGV 在大多数患者里都没有, 应剔除)
        skip_folder_markers: 文件夹名包含任一标记时跳过；默认跳过 ``@``。

    返回:
        output_txt: TSV 文件路径
        active_features: 实际保留的特征列表 (写在 TSV 头部)
    """
    target = target.upper()
    print(f"\n{'='*60}")
    print(f"收集特征 + {target}值: {root_folder}")
    print(f"{'='*60}")

    if output_txt is None:
        output_txt = os.path.join(root_folder, f"all_features_{target.lower()}.txt")

    subfolders = sorted(
        d for d in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, d)))

    rows = []  # 每行: (sample_name, {feat: value or None}, target_val)
    no_json, no_label, skipped_marked = [], [], []
    skip_folder_markers = tuple(skip_folder_markers or ())

    for folder in subfolders:
        if any(marker in folder for marker in skip_folder_markers):
            skipped_marked.append(folder)
            continue
        folder_path = os.path.join(root_folder, folder)
        unified_path = os.path.join(
            folder_path, FEATURES_DIRNAME, UNIFIED_FEATURES_NAME)
        if not os.path.exists(unified_path):
            no_json.append(folder)
            continue
        try:
            with open(unified_path, 'r', encoding='utf-8') as f:
                unified = json.load(f)
            features = _flatten_unified_to_features(unified)
        except Exception as e:
            print(f"  ERROR {folder}: unified JSON 失败 ({e})")
            no_json.append(folder)
            continue

        label_val = _read_label_value(folder_path, target)
        if label_val is None:
            no_label.append(folder)
            continue

        feat_dict = {}
        for fn in FEATURE_NAMES:
            v = features.get(fn)
            try:
                value = float(v) if v is not None else np.nan
            except (TypeError, ValueError, OverflowError):
                value = np.nan
            feat_dict[fn] = value if np.isfinite(value) else None

        if feat_dict.get('has_tips') == 1.0:
            for fn in COLLATERAL_FEATURES:
                if fn in feat_dict:
                    feat_dict[fn] = None

        rows.append((folder, feat_dict, label_val))
        print(f"  OK {folder}: {target}={label_val:.1f}")

    if not rows:
        print("  无可用数据!")
        return None, []

    # ----- 计算每个特征的缺失率, 剔除缺失率过高的 -----
    n_samples = len(rows)
    miss_rate = {}
    for fn in FEATURE_NAMES:
        miss = sum(1 for _, fd, _ in rows if fd[fn] is None)
        miss_rate[fn] = miss / n_samples

    active = [fn for fn in FEATURE_NAMES
              if miss_rate[fn] <= drop_features_above_missing]
    dropped = [fn for fn in FEATURE_NAMES
               if miss_rate[fn] > drop_features_above_missing]

    print(f"\n汇总: {n_samples} 个样本")
    if skipped_marked:
        print(f"  按文件夹标记跳过 ({len(skipped_marked)}): "
              f"{skipped_marked[:8]}"
              f"{'...' if len(skipped_marked) > 8 else ''}")
    if no_json:
        print(f"  无 JSON ({len(no_json)}): "
              f"{no_json[:5]}{'...' if len(no_json)>5 else ''}")
    if no_label:
        print(f"  无 {target} ({len(no_label)}): "
              f"{no_label[:5]}{'...' if len(no_label)>5 else ''}")
    print(f"  特征: 保留 {len(active)} 个, 剔除 {len(dropped)} 个 "
          f"(缺失率 > {100*drop_features_above_missing:.0f}%)")
    if dropped:
        print(f"  剔除特征 (缺失率): " +
              ", ".join(f"{fn}({100*miss_rate[fn]:.0f}%)" for fn in dropped[:8])
              + ("..." if len(dropped) > 8 else ""))

    # ----- 写 TSV (缺失值用 NaN, 不要用 0) -----
    header = ['sample'] + active + [target]
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write('\t'.join(header) + '\n')
        for name, fd, tval in rows:
            vals = []
            for fn in active:
                v = fd[fn]
                vals.append('NaN' if v is None else f'{v:.6f}')
            f.write(name + '\t' + '\t'.join(vals)
                    + '\t' + f'{tval:.6f}' + '\n')

    print(f"\n已保存: {output_txt}")
    return output_txt, active


# ============================================================
# 数据加载 (从已汇总 TSV)
# ============================================================

def load_data(filepath, target="PVP"):
    """加载汇总 TSV, 返回 (DataFrame, active_features)"""
    target = target.upper()
    print(f"加载数据: {filepath}")

    # 第一行是表头
    df = pd.read_csv(filepath, sep='\t', header=0, na_values=['NaN', 'nan'])

    if target not in df.columns:
        raise ValueError(f"TSV 中缺少 {target} 列")

    sample_col = df.columns[0]
    target_col = target
    feature_cols = [c for c in df.columns if c not in (sample_col, target_col)]

    print(f"  样本数: {len(df)}, 特征数: {len(feature_cols)}")
    unit = TARGET_LABELS.get(target, {}).get('unit', '')
    y = df[target_col].astype(float)
    print(f"  {target} 范围: {y.min():.1f} - {y.max():.1f} {unit}")

    return df, feature_cols


def _deduplicate_feature_columns(df, features):
    """Remove non-constant columns that contain exactly the same observations."""
    retained = []
    retained_values = {}
    aliases = {}
    for feature in features:
        values = pd.to_numeric(df[feature], errors='coerce').to_numpy(
            dtype=float)
        finite_values = values[np.isfinite(values)]
        is_variable = (
            finite_values.size >= 2
            and np.ptp(finite_values) > 1e-12
        )
        duplicate_of = None
        if is_variable:
            for retained_feature, retained_array in retained_values.items():
                if np.array_equal(values, retained_array, equal_nan=True):
                    duplicate_of = retained_feature
                    break
        if duplicate_of is None:
            retained.append(feature)
            if is_variable:
                retained_values[feature] = values
        else:
            aliases[feature] = duplicate_of
    return retained, aliases


# ============================================================
# 相关性计算
# ============================================================

def compute_correlations(df, active_features, target="PVP", min_samples=10):
    """相关性 + patient-cluster inference + BH-FDR."""
    target = target.upper()
    print(f"\n计算与 {target} 的相关性...")

    y = df[target].values
    sample_col = df.columns[0]
    subject_ids = np.asarray([
        _infer_subject_id(value) for value in df[sample_col].astype(str)
    ], dtype=object)
    results = []

    for feat in active_features:
        x = df[feat].values
        mask = np.isfinite(x) & np.isfinite(y)
        x_c, y_c = x[mask], y[mask]

        if (len(x_c) < min_samples or np.std(x_c) < 1e-10
                or np.std(y_c) < 1e-10):
            results.append({
                'feature': feat, 'label': FEATURE_LABELS_CN.get(feat, feat),
                'group': _get_group(feat),
                'pearson_r': np.nan, 'pearson_p': np.nan,
                'spearman_r': np.nan, 'spearman_p': np.nan,
                'pearson_p_naive': np.nan, 'spearman_p_naive': np.nan,
                'pearson_ci_low': np.nan, 'pearson_ci_high': np.nan,
                'spearman_ci_low': np.nan, 'spearman_ci_high': np.nan,
                'abs_pearson': np.nan, 'abs_spearman': np.nan,
                'n_samples': len(x_c),
                'n_subjects': len(pd.unique(subject_ids[mask])),
            })
            continue

        pr, pp_naive = pearsonr(x_c, y_c)
        sr, sp_naive = spearmanr(x_c, y_c)
        cluster_ids = subject_ids[mask]
        pp, pr_low, pr_high, n_subjects = _cluster_jackknife_inference(
            x_c, y_c, cluster_ids, pr,
            lambda a, b: pearsonr(a, b)[0])
        sp, sr_low, sr_high, _ = _cluster_jackknife_inference(
            x_c, y_c, cluster_ids, sr,
            lambda a, b: spearmanr(a, b)[0])
        if not np.isfinite(pp):
            pp, pr_low, pr_high = pp_naive, np.nan, np.nan
        if not np.isfinite(sp):
            sp, sr_low, sr_high = sp_naive, np.nan, np.nan

        results.append({
            'feature': feat, 'label': FEATURE_LABELS_CN.get(feat, feat),
            'group': _get_group(feat),
            'pearson_r': pr, 'pearson_p': pp,
            'spearman_r': sr, 'spearman_p': sp,
            'pearson_p_naive': pp_naive, 'spearman_p_naive': sp_naive,
            'pearson_ci_low': pr_low, 'pearson_ci_high': pr_high,
            'spearman_ci_low': sr_low, 'spearman_ci_high': sr_high,
            'abs_pearson': abs(pr), 'abs_spearman': abs(sr),
            'n_samples': len(x_c),
            'n_subjects': n_subjects,
        })

    corr_df = pd.DataFrame(results)
    corr_df = corr_df.sort_values('abs_spearman', ascending=False,
                                  na_position='last').reset_index(drop=True)
    corr_df['pearson_q'] = _fdr_bh(corr_df['pearson_p'].to_numpy())
    corr_df['spearman_q'] = _fdr_bh(corr_df['spearman_p'].to_numpy())
    corr_df['pearson_sig'] = corr_df['pearson_p'].apply(_sig)
    corr_df['spearman_sig'] = corr_df['spearman_p'].apply(_sig)
    corr_df['pearson_fdr_sig'] = corr_df['pearson_q'].apply(_sig)
    corr_df['spearman_fdr_sig'] = corr_df['spearman_q'].apply(_sig)

    print(f"\n  Top 10 (Spearman |ρ|):")
    for _, row in corr_df.head(10).iterrows():
        if pd.isna(row['spearman_r']):
            continue
        print(f"    {row['label']:>16s}: ρ={row['spearman_r']:+.3f} "
              f"(p={row['spearman_p']:.4f}, q={row['spearman_q']:.4f}) "
              f"{row['spearman_fdr_sig']}  "
              f"N={row['n_samples']}, subjects={row['n_subjects']}")

    return corr_df


def compute_pretips_collateral_correlations(df, target="PVP",
                                             min_samples=10):
    """Focused collateral analysis restricted to pre-TIPS observations."""
    if 'has_tips' not in df.columns:
        return df.iloc[0:0].copy(), pd.DataFrame()
    pre_tips = df[df['has_tips'] == 0].copy()
    features = [
        feature for feature in PRETIPS_COLLATERAL_FEATURES
        if feature in pre_tips.columns
    ]
    if len(pre_tips) < min_samples or not features:
        return pre_tips, pd.DataFrame()
    result = compute_correlations(
        pre_tips, features, target=target, min_samples=min_samples)
    result.insert(0, 'analysis_subset', 'pre_tips_only')
    return pre_tips, result


# ============================================================
# 可视化
# ============================================================

def _setup_matplotlib():
    plt.rcParams['font.sans-serif'] = [
        'SimHei', 'Microsoft YaHei', 'DejaVu Sans',
        'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False


def plot_heatmap(df, active_features, output_dir, target="PVP"):
    """所有特征 + target 的相关性矩阵热力图。特征过多时只画 Top-N。"""
    target = target.upper()
    target_cn = TARGET_LABELS.get(target, {}).get('cn', target)
    print("\n绘制热力图...")

    # 特征过多时, 只取与 target 相关性最强的 Top 30
    MAX_HEATMAP_FEATURES = 30
    if len(active_features) > MAX_HEATMAP_FEATURES:
        # 先算 |Spearman ρ| 选 top
        y = df[target].values
        scores = []
        for f in active_features:
            x = df[f].values
            m = np.isfinite(x) & np.isfinite(y)
            if np.sum(m) >= 3 and np.std(x[m]) > 1e-10:
                r, _ = spearmanr(x[m], y[m])
                scores.append((f, abs(r) if not np.isnan(r) else 0))
            else:
                scores.append((f, 0))
        scores.sort(key=lambda t: t[1], reverse=True)
        plot_features = [s[0] for s in scores[:MAX_HEATMAP_FEATURES]]
        print(f"  特征过多({len(active_features)}), 仅画 Top {MAX_HEATMAP_FEATURES}")
    else:
        plot_features = active_features

    cols = plot_features + [target]
    labels = [FEATURE_LABELS_CN.get(c, c) for c in plot_features] + [target]
    corr_matrix = df[cols].corr(method='spearman')

    n = len(labels)
    figsize = (max(14, n * 0.55), max(12, n * 0.5))
    fig, ax = plt.subplots(figsize=figsize)
    cmap = LinearSegmentedColormap.from_list('custom',
        ['#1e3a5f', '#2563eb', '#60a5fa', '#bfdbfe',
         '#ffffff',
         '#fecaca', '#f87171', '#dc2626', '#7f1d1d'], N=256)

    im = ax.imshow(corr_matrix.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = corr_matrix.values[i, j]
            if pd.isna(val):
                continue
            if i == n - 1 or j == n - 1 or abs(val) > 0.5:
                color = 'white' if abs(val) > 0.6 else 'black'
                weight = 'bold' if abs(val) > 0.5 else 'normal'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=6.5, color=color, fontweight=weight)

    ax.axhline(y=n - 1.5, color='#eab308', linewidth=2)
    ax.axvline(x=n - 1.5, color='#eab308', linewidth=2)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Spearman Correlation', fontsize=11)
    ax.set_title(f'Portal Vein Features — Correlation with {target}\n'
                 f'门静脉特征 vs {target_cn} 相关性矩阵',
                 fontsize=15, fontweight='bold', pad=20)

    plt.tight_layout()
    path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  保存: {path}")


def plot_scatter_matrix(df, corr_df, output_dir, target="PVP",
                         max_features=30):
    """各特征与 target 的散点图。特征多时只画 Top max_features 个。"""
    target = target.upper()
    target_cn = TARGET_LABELS.get(target, {}).get('cn', target)
    unit = TARGET_LABELS.get(target, {}).get('unit', '')
    print("\n绘制散点图...")

    # 选 Top max_features 个 (按 |Spearman ρ|)
    valid_corr = corr_df.dropna(subset=['spearman_r']).copy()
    plot_corr = valid_corr.head(max_features)
    n_features = len(plot_corr)

    if n_features == 0:
        print("  无有效特征, 跳过")
        return

    n_cols = 5
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, n_rows * 4.2))
    if n_rows == 1:
        axes = np.atleast_2d(axes)
    fig.suptitle(f'Top {n_features} Features vs {target}\n'
                 f'与{target_cn}最相关的特征散点图',
                 fontsize=18, fontweight='bold', y=1.01)

    y = df[target].values

    for idx, (_, row) in enumerate(plot_corr.iterrows()):
        feat = row['feature']
        r, c = idx // n_cols, idx % n_cols
        ax = axes[r, c]

        x = df[feat].values
        mask = np.isfinite(x) & np.isfinite(y)
        x_c, y_c = x[mask], y[mask]

        color = GROUP_COLORS.get(_get_group(feat), '#64748b')
        ax.scatter(x_c, y_c, c=color, alpha=0.6, s=40,
                   edgecolors='white', linewidths=0.5, zorder=3)

        if len(x_c) >= 3 and np.std(x_c) > 1e-10:
            slope, intercept, _, _, std_err = stats.linregress(x_c, y_c)
            x_line = np.linspace(x_c.min(), x_c.max(), 100)
            ax.plot(x_line, slope * x_line + intercept,
                    color=color, linewidth=2, alpha=0.8, zorder=2)

            text_color = '#dc2626' if row['spearman_q'] < 0.05 else '#64748b'
            ax.text(0.05, 0.95,
                    f"ρ={row['spearman_r']:.3f}{row['spearman_fdr_sig']}\n"
                    f"q={row['spearman_q']:.3g}\n"
                    f"N={int(row['n_samples'])}",
                    transform=ax.transAxes, fontsize=9, fontweight='bold',
                    color=text_color, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              alpha=0.85, edgecolor=text_color))

        ax.set_xlabel(FEATURE_LABELS_CN.get(feat, feat),
                      fontsize=9, fontweight='bold')
        ax.set_ylabel(f'{target} ({unit})', fontsize=9)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for idx in range(n_features, n_rows * n_cols):
        r, c = idx // n_cols, idx % n_cols
        axes[r, c].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, 'scatter_plots.png')
    plt.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  保存: {path}")


def plot_top_features(corr_df, output_dir, target="PVP", top_n=30):
    """排名条形图。特征多时只画 Top N。"""
    target = target.upper()
    print(f"\n绘制排名图 (Top {top_n})...")

    plot_corr = corr_df.dropna(subset=['spearman_r']).head(top_n)
    if len(plot_corr) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, max(10, top_n * 0.4)))
    for ax, r_col, p_col, title in [
        (ax1, 'spearman_r', 'spearman_q', 'Spearman ρ'),
        (ax2, 'pearson_r', 'pearson_q', 'Pearson r'),
    ]:
        sub = plot_corr.sort_values(r_col, ascending=True)
        labels = [FEATURE_LABELS_CN.get(f, f) for f in sub['feature']]
        values = sub[r_col].values
        p_values = sub[p_col].values
        colors = [GROUP_COLORS.get(g, '#64748b') for g in sub['group']]

        ax.barh(range(len(labels)), values, color=colors, alpha=0.85,
                edgecolor='white', linewidth=0.5, height=0.75)
        for i, (v, p) in enumerate(zip(values, p_values)):
            if not pd.isna(p) and p < 0.05:
                ax.text(v + (0.02 if v >= 0 else -0.04), i,
                        '★' if p < 0.01 else '☆',
                        fontsize=12, va='center', color='#dc2626')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(title, fontsize=12, fontweight='bold')
        ax.set_title(f'{title} with {target}', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=0.8)
        for t in [0.3, -0.3]:
            ax.axvline(x=t, color='#dc2626', linewidth=0.8,
                       linestyle='--', alpha=0.5)
        ax.grid(True, axis='x', alpha=0.2, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    from matplotlib.patches import Patch
    legend = [Patch(facecolor=c, label=g) for g, c in GROUP_COLORS.items()]
    legend.append(Patch(facecolor='white', edgecolor='#dc2626',
                        label='★ q<0.01  ☆ q<0.05 (BH-FDR)'))
    fig.legend(handles=legend, loc='lower center',
               ncol=len(GROUP_COLORS) + 1, fontsize=10,
               frameon=True, fancybox=True, shadow=True,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    path = os.path.join(output_dir, 'top_features_bar.png')
    plt.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  保存: {path}")


def plot_group_analysis(corr_df, output_dir, target="PVP"):
    target = target.upper()
    print("\n绘制分组分析图...")

    valid_corr = corr_df.dropna(subset=['spearman_r'])

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 分组平均 |ρ|
    ax1 = fig.add_subplot(gs[0, 0])
    group_means = {}
    for g, feats in FEATURE_GROUPS.items():
        sub = valid_corr[valid_corr['feature'].isin(feats)]
        if len(sub) > 0:
            group_means[g] = sub['abs_spearman'].mean()
    groups = list(group_means.keys())
    means = [group_means[g] for g in groups]
    colors = [GROUP_COLORS[g] for g in groups]

    bars = ax1.bar(groups, means, color=colors, alpha=0.85,
                    edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Mean |Spearman ρ|', fontsize=11)
    ax1.set_title('各特征组平均相关性', fontsize=13, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, axis='y', alpha=0.2, linestyle='--')

    # 显著特征数
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(groups))
    width = 0.25
    for offset, threshold, color, label in [
        (-width, 1.0, '#94a3b8', '总数'),
        (0, 0.05, '#f59e0b', 'q<0.05'),
        (width, 0.01, '#dc2626', 'q<0.01'),
    ]:
        counts = []
        for g in groups:
            sub = valid_corr[valid_corr['feature'].isin(FEATURE_GROUPS[g])]
            counts.append(len(sub) if threshold >= 1.0
                          else len(sub[sub['spearman_q'] < threshold]))
        ax2.bar(x + offset, counts, width, color=color, label=label,
                alpha=0.5 if threshold >= 1.0 else 0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(groups)
    ax2.set_ylabel('特征数量', fontsize=11)
    ax2.set_title('显著特征数量', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Pearson vs Spearman
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(valid_corr['pearson_r'], valid_corr['spearman_r'],
                c=[GROUP_COLORS.get(g, '#64748b') for g in valid_corr['group']],
                s=60, alpha=0.7, edgecolors='white', linewidths=0.5)
    if len(valid_corr) > 0:
        lim = max(valid_corr['pearson_r'].abs().max(),
                  valid_corr['spearman_r'].abs().max()) + 0.1
        ax3.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3)
    ax3.set_xlabel('Pearson r', fontsize=11)
    ax3.set_ylabel('Spearman ρ', fontsize=11)
    ax3.set_title('Pearson vs Spearman', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.2, linestyle='--')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_aspect('equal')

    # Top 10 表格
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')
    top10 = valid_corr.head(10)
    table_data = []
    for _, row in top10.iterrows():
        table_data.append([
            row['label'], row['group'],
            f"{row['spearman_r']:+.4f}",
            f"{row['spearman_q']:.4f}{row['spearman_fdr_sig']}",
            f"{row['pearson_r']:+.4f}",
            f"{row['pearson_q']:.4f}{row['pearson_fdr_sig']}",
            str(int(row['n_samples'])),
        ])
    if table_data:
        table = ax4.table(
            cellText=table_data,
            colLabels=['特征', '分组', 'Spearman ρ', 'FDR q',
                       'Pearson r', 'FDR q', 'N'],
            cellLoc='center', loc='center',
            colWidths=[0.20, 0.10, 0.12, 0.14, 0.12, 0.14, 0.06])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)
        for j in range(7):
            table[0, j].set_facecolor('#1e293b')
            table[0, j].set_text_props(color='white', fontweight='bold')
        for i in range(len(table_data)):
            sp = top10.iloc[i]['spearman_q']
            if pd.isna(sp):
                bg = 'white'
            elif sp < 0.01:
                bg = '#fef2f2'
            elif sp < 0.05:
                bg = '#fffbeb'
            else:
                bg = '#f8fafc' if i % 2 == 0 else 'white'
            for j in range(7):
                table[i + 1, j].set_facecolor(bg)
    ax4.set_title(f'Top 10 Features Correlated with {target}',
                   fontsize=14, fontweight='bold', pad=20)

    path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  保存: {path}")


# ============================================================
# 文字报告
# ============================================================

def generate_report(df, corr_df, active_features, output_dir, target="PVP",
                    skip_folder_markers=(), pretips_collateral_df=None,
                    duplicate_feature_aliases=None):
    target = target.upper()
    target_cn = TARGET_LABELS.get(target, {}).get('cn', target)
    unit = TARGET_LABELS.get(target, {}).get('unit', '')
    print("\n生成报告...")

    y = df[target]
    sample_col = df.columns[0]
    n_subjects = len({
        _infer_subject_id(value) for value in df[sample_col].astype(str)
    })
    valid_corr = corr_df.dropna(subset=['spearman_r'])
    sig5 = valid_corr[valid_corr['spearman_q'] < 0.05]
    sig1 = valid_corr[valid_corr['spearman_q'] < 0.01]

    lines = [
        "=" * 70,
        f"门静脉特征与{target_cn}({target})相关性分析报告",
        "=" * 70, "",
        f"样本数量: {len(df)}",
        f"推断患者簇数量: {n_subjects} (按日期前缀和术后标记归并)",
        f"文件夹过滤: 排除名称包含 {tuple(skip_folder_markers or ())} 的对象",
        f"特征数量: {len(active_features)} (去除侧支及完全重复列后)",
        f"{target}范围: {y.min():.1f} - {y.max():.1f} {unit}",
        f"{target}均值: {y.mean():.1f} ± {y.std():.1f} {unit}",
        f"{target}中位数: {y.median():.1f} {unit}", "",
        "-" * 70,
        f"显著相关特征 (Spearman, Benjamini-Hochberg FDR): "
        f"q<0.05: {len(sig5)}, q<0.01: {len(sig1)}", "",
    ]

    duplicate_feature_aliases = duplicate_feature_aliases or {}
    if duplicate_feature_aliases:
        lines[8:8] = [
            f"完全重复特征别名: {len(duplicate_feature_aliases)} 个已从统计检验中移除",
            *(
                f"  {alias} -> {retained}"
                for alias, retained in duplicate_feature_aliases.items()
            ),
        ]

    if len(sig5) > 0:
        lines.append("显著特征详情 (FDR q < 0.05, 按|ρ|降序):")
        lines.append(f"  {'特征':>18s}  {'Spearman ρ':>12s}  "
                     f"{'cluster p':>10s}  {'FDR q':>10s}  "
                     f"{'cluster 95% CI':>20s}  "
                     f"{'分组':>8s}  {'N':>4s}")
        lines.append("  " + "-" * 70)
        for _, r in sig5.iterrows():
            lines.append(
                f"  {r['label']:>18s}  {r['spearman_r']:>+12.4f}  "
                f"{r['spearman_p']:>10.4f}  {r['spearman_q']:>10.4f}  "
                f"[{r['spearman_ci_low']:+.3f}, {r['spearman_ci_high']:+.3f}]  "
                f"{r['group']:>8s}  {int(r['n_samples']):>4d}")
    else:
        lines.append("  FDR 校正后未发现显著相关特征")

    lines += ["", "-" * 70, "仅术前侧支相关性:"]
    if pretips_collateral_df is not None and len(pretips_collateral_df):
        pretips_valid = pretips_collateral_df.dropna(
            subset=['spearman_r']).copy()
        pretips_sig = pretips_valid[pretips_valid['spearman_q'] < 0.05]
        n_scans = (int(pretips_valid['n_samples'].max())
                   if len(pretips_valid) else 0)
        n_clusters = (int(pretips_valid['n_subjects'].max())
                      if len(pretips_valid) else 0)
        lines.append(
            f"  术前扫描 N={n_scans}, 患者簇={n_clusters}; "
            f"侧支指标内 BH-FDR q<0.05: {len(pretips_sig)}")
        lines.append(f"  {'特征':>18s}  {'Spearman ρ':>12s}  "
                     f"{'cluster p':>10s}  {'FDR q':>10s}  "
                     f"{'N':>4s}")
        for _, row in pretips_valid.iterrows():
            lines.append(
                f"  {row['label']:>18s}  {row['spearman_r']:>+12.4f}  "
                f"{row['spearman_p']:>10.4f}  "
                f"{row['spearman_q']:>10.4f}  "
                f"{int(row['n_samples']):>4d}")
    else:
        lines.append("  无足够的术前侧支数据。")

    lines += ["", "-" * 70, "分组平均相关性:"]
    for g, feats in FEATURE_GROUPS.items():
        sub = valid_corr[valid_corr['feature'].isin(feats)]
        if len(sub) > 0:
            mean_r = sub['abs_spearman'].mean()
            best = sub.loc[sub['abs_spearman'].idxmax()]
            lines.append(f"  {g:>10s}: mean|ρ|={mean_r:.4f}  "
                         f"最强: {best['label']} "
                         f"(ρ={best['spearman_r']:+.4f})")

    lines += ["", "-" * 70, "全部特征 (按|Spearman ρ|降序):"]
    lines.append(f"  {'#':>3s}  {'特征':>18s}  {'Spearman ρ':>12s}  "
                 f"{'cluster p':>10s}  {'FDR q':>10s}  {'sig':>4s}  "
                 f"{'Pearson r':>10s}  {'N':>4s}")
    lines.append("  " + "-" * 84)
    for i, (_, r) in enumerate(corr_df.iterrows()):
        if pd.isna(r['spearman_r']):
            lines.append(f"  {i+1:>3d}  {r['label']:>18s}  "
                         f"{'N/A':>12s}  {'N/A':>10s}  {'N/A':>10s}  "
                         f"{'':>4s}  {'N/A':>10s}  "
                         f"{int(r['n_samples']):>4d}")
        else:
            fdr_sig = _sig_text(r['spearman_fdr_sig'])
            lines.append(
                f"  {i+1:>3d}  {r['label']:>18s}  "
                f"{r['spearman_r']:>+12.4f}  {r['spearman_p']:>10.4f}  "
                f"{r['spearman_q']:>10.4f}  {fdr_sig:>4s}  "
                f"{r['pearson_r']:>+10.4f}  {int(r['n_samples']):>4d}")

    # ----- 自动相关性解读 (Top-K 解读 + 候选特征建议) -----
    interp = _auto_interpret_correlations(corr_df, df, target=target)
    lines += ["", "=" * 70, "自动相关性解读 (Top-K)", "=" * 70]
    lines += interp

    lines += [
        "", "=" * 70,
        "注: 星号按 BH-FDR q 值: *** q<0.001, ** q<0.01, * q<0.05",
        "    p 值和 95% CI 使用患者簇 jackknife；CSV 保留 naive p 供核对",
        "    该分析控制重复测量依赖，但未把 TIPS 状态作为混杂变量消除",
        f"    目标变量: {target} ({target_cn})",
        f"    标签来源: patient/label/{target}.txt",
        f"    特征来源: patient/features/unified_features.json",
        "=" * 70,
    ]

    report = '\n'.join(lines)
    path = os.path.join(output_dir, 'analysis_report.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  保存: {path}")
    print("\n" + report)


def _auto_interpret_correlations(corr_df, df, target="PVP",
                                  top_k=15, strong_threshold=0.4,
                                  moderate_threshold=0.25):
    """
    自动给出 Top-K 解读, 包含:
      - 强 / 中等相关阈值划分
      - 系统特征 vs 单段特征 的相关性对比
      - 推荐用于训练的候选特征列表 (相关性 + 缺失率综合)
    返回 list[str], 用于追加到报告.
    """
    out = []
    valid = corr_df.dropna(subset=['spearman_r']).copy()
    if len(valid) == 0:
        out.append("  (无有效特征, 跳过解读)")
        return out

    valid['abs_rho'] = valid['spearman_r'].abs()
    valid = valid.sort_values('abs_rho', ascending=False).reset_index(drop=True)

    n_strong = (valid['abs_rho'] >= strong_threshold).sum()
    n_moderate = ((valid['abs_rho'] >= moderate_threshold)
                  & (valid['abs_rho'] < strong_threshold)).sum()
    n_total = len(valid)

    out.append(f"  样本数 N={len(df)}, 有效特征 {n_total}")
    out.append(f"  |ρ| ≥ {strong_threshold} (强): {n_strong}, "
               f"{moderate_threshold} ≤ |ρ| < {strong_threshold} (中): "
               f"{n_moderate}")
    out.append("")

    # ---- Top-K ----
    out.append(f"  Top {top_k} 相关特征 (按 |Spearman ρ|):")
    out.append("  " + "-" * 90)
    out.append(f"  {'#':>3s}  {'特征':>22s}  {'分组':>14s}  "
               f"{'ρ':>8s}  {'FDR q':>10s}  {'方向':>6s}  {'强度':>6s}")
    for i, row in valid.head(top_k).iterrows():
        sign = '↑' if row['spearman_r'] > 0 else '↓'
        if row['abs_rho'] >= strong_threshold:
            level = '强'
        elif row['abs_rho'] >= moderate_threshold:
            level = '中'
        else:
            level = '弱'
        out.append(f"  {i+1:>3d}  {row['label']:>22s}  {row['group']:>14s}  "
                   f"{row['spearman_r']:>+8.3f}  {row['spearman_q']:>10.4f}  "
                   f"{sign:>6s}  {level:>6s}")
    out.append("")

    # ---- 单段 vs 系统特征 ----
    sys_groups = ['系统-角度', '系统-Murray/比率', '系统-长度/弯曲',
                  '系统-阻力', '系统-拓扑', '系统-临床派生']
    seg_groups = ['长度', '曲折度', '曲率', '直径/面积', '圆度', '夹角']
    sys_mask = valid['group'].isin(sys_groups)
    seg_mask = valid['group'].isin(seg_groups)
    if sys_mask.any() and seg_mask.any():
        sys_top = valid[sys_mask].head(5)
        seg_top = valid[seg_mask].head(5)
        out.append("  --- 系统特征 vs 单段特征 (Top 5 各) ---")
        out.append("  系统/联合特征 Top 5:")
        for _, r in sys_top.iterrows():
            out.append(f"    {r['label']:>22s}  ρ={r['spearman_r']:+.3f}  "
                       f"q={r['spearman_q']:.4f}"
                       f"{_sig_text(r['spearman_fdr_sig'])}")
        out.append("  单段特征 Top 5:")
        for _, r in seg_top.iterrows():
            out.append(f"    {r['label']:>22s}  ρ={r['spearman_r']:+.3f}  "
                       f"q={r['spearman_q']:.4f}"
                       f"{_sig_text(r['spearman_fdr_sig'])}")
        max_sys = sys_top['abs_rho'].max() if len(sys_top) else 0
        max_seg = seg_top['abs_rho'].max() if len(seg_top) else 0
        out.append(f"  最强 |ρ|: 系统={max_sys:.3f}, 单段={max_seg:.3f}; "
                   f"{'系统' if max_sys >= max_seg else '单段'}特征更优.")
        out.append("")

    # ---- 推荐用于建模的特征 ----
    # 标准: |ρ|、FDR q、覆盖率同时达标。
    n_samples_full = len(df)
    keep = valid[(valid['abs_rho'] >= moderate_threshold)
                 & (valid['spearman_q'] < 0.05)
                 & (valid['n_samples'] >= 0.7 * n_samples_full)].copy()
    out.append(f"  候选特征 (|ρ|≥{moderate_threshold} ∧ FDR q<0.05 ∧ "
               f"覆盖≥70%): {len(keep)} 个")
    if len(keep) > 0:
        for _, r in keep.head(20).iterrows():
            out.append(f"    [{r['group']:>10s}] {r['label']:>22s}  "
                       f"ρ={r['spearman_r']:+.3f} p={r['spearman_p']:.4f} "
                       f"q={r['spearman_q']:.4f} N={int(r['n_samples'])}")
        if len(keep) > 20:
            out.append(f"    ... (共 {len(keep)} 个, 仅显示前 20)")
    else:
        out.append("    (没有特征同时满足相关性 + 显著性 + 覆盖率, "
                   "建议放宽阈值或扩样本)")
    out.append("")

    # ---- 解读建议 ----
    out.append("  --- 模型/可解释性建议 ---")
    if n_strong >= 3:
        out.append(f"  · 已有 {n_strong} 个强相关特征, 单变量回归 / 树模型可作为基线.")
    elif n_strong + n_moderate >= 5:
        out.append("  · 强相关特征不足, 建议线性模型 + L1 正则做特征选择, "
                   "或用 GBDT/XGBoost 自动挖非线性.")
    else:
        out.append("  · 总体相关性偏弱, 优先排查: "
                   "(a) 样本量是否足够; "
                   "(b) 端点掩码 / 截面边界过滤是否过度; "
                   "(c) 是否需要非线性 / 多变量组合特征.")
    if sys_mask.any():
        out.append(f"  · 系统特征推荐重点关注: 汇合 Murray-3 偏离, "
                   f"入流阻力不对称, 侧支负担, MPV 锥度.")

    return out


# ============================================================
# 主流程
# ============================================================

def run_analysis(filepath, output_dir=None, target="PVP",
                 skip_folder_markers=()):
    """完整分析: 加载 → 相关性 → 可视化 → 报告"""
    target = target.upper()
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(output_dir, exist_ok=True)
    _setup_matplotlib()

    df, active_features = load_data(filepath, target)
    analysis_features = [
        feature for feature in active_features
        if feature not in COLLATERAL_FEATURES
    ]
    analysis_features, duplicate_feature_aliases = (
        _deduplicate_feature_columns(df, analysis_features))
    if duplicate_feature_aliases:
        print("  移除完全重复特征别名: " + ", ".join(
            f"{alias}->{retained}"
            for alias, retained in duplicate_feature_aliases.items()))
    corr_df = compute_correlations(df, analysis_features, target)
    _, pretips_collateral_df = compute_pretips_collateral_correlations(
        df, target=target)

    csv_path = os.path.join(output_dir, 'correlation_results.csv')
    corr_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n相关性 CSV: {csv_path}")

    pretips_path = os.path.join(
        output_dir, 'pretips_collateral_correlations.csv')
    pretips_collateral_df.to_csv(
        pretips_path, index=False, encoding='utf-8-sig')
    print(f"术前侧支相关性 CSV: {pretips_path}")

    duplicate_path = os.path.join(
        output_dir, 'duplicate_feature_aliases.csv')
    pd.DataFrame([
        {
            'excluded_feature': alias,
            'retained_feature': retained,
            'reason': 'exactly_equal_across_samples',
        }
        for alias, retained in duplicate_feature_aliases.items()
    ], columns=['excluded_feature', 'retained_feature', 'reason']).to_csv(
        duplicate_path, index=False, encoding='utf-8-sig')
    print(f"重复特征审计 CSV: {duplicate_path}")

    # 候选特征 CSV (|ρ|≥0.25 + BH-FDR q<0.05 + 覆盖≥70%)
    valid = corr_df.dropna(subset=['spearman_r']).copy()
    n_full = len(df)
    valid['abs_rho'] = valid['spearman_r'].abs()
    keep = valid[(valid['abs_rho'] >= 0.25)
                 & (valid['spearman_q'] < 0.05)
                 & (valid['n_samples'] >= 0.7 * n_full)].copy()
    keep = keep.sort_values('abs_rho', ascending=False)
    rec_path = os.path.join(output_dir, 'recommended_features.csv')
    keep.to_csv(rec_path, index=False, encoding='utf-8-sig')
    print(f"候选特征 CSV: {rec_path}  ({len(keep)} 个)")

    plot_heatmap(df, analysis_features, output_dir, target)
    plot_scatter_matrix(df, corr_df, output_dir, target)
    plot_top_features(corr_df, output_dir, target)
    plot_group_analysis(corr_df, output_dir, target)
    generate_report(
        df, corr_df, analysis_features, output_dir, target,
        skip_folder_markers=skip_folder_markers,
        pretips_collateral_df=pretips_collateral_df,
        duplicate_feature_aliases=duplicate_feature_aliases)

    print(f"\n{'='*60}")
    print(f"分析完成! 结果: {output_dir}")
    print(f"{'='*60}")
    return df, corr_df


def collect_and_analyze(root_folder, output_dir=None, target="PVP",
                        drop_features_above_missing=0.5,
                        skip_folder_markers=('@',)):
    """一键: 收集 → 汇总 → 分析"""
    target = target.upper()
    if output_dir is None:
        output_dir = os.path.join(
            root_folder, 'correlationship', 'scalar')
    os.makedirs(output_dir, exist_ok=True)

    txt_path = os.path.join(output_dir, f"all_features_{target.lower()}.txt")
    result, active = collect_features(
        root_folder, target, txt_path,
        drop_features_above_missing=drop_features_above_missing,
        skip_folder_markers=skip_folder_markers)
    if result is None:
        print("收集失败")
        return
    run_analysis(
        txt_path, output_dir, target,
        skip_folder_markers=skip_folder_markers)


# ============================================================
# 用户配置
# ============================================================

if __name__ == '__main__':

    TARGET = "PVP"
    MODE = "run"

    ROOT_FOLDER = r"F:\PCG data\dataset\zhengzhou_vkan_qian47"
    OUTPUT_DIR = r"F:\results\correlation"
    DATA_FILE = r"F:\results\all_features_pvp.txt"

    if MODE == "analyze":
        run_analysis(DATA_FILE, OUTPUT_DIR, TARGET)
    elif MODE == "collect":
        collect_features(ROOT_FOLDER, TARGET,
                         os.path.join(OUTPUT_DIR, f"all_features_{TARGET.lower()}.txt"))
    elif MODE == "run":
        collect_and_analyze(ROOT_FOLDER, OUTPUT_DIR, TARGET)
    else:
        print(f"未知模式: {MODE}")
