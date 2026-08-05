"""
中心线剖面特征与临床指标逐点相关性分析（v4 - cluster permutation）
============================================================
新版变化:
  - 优先读取 unified_features.json 内清洗、重采样后的 pointwise
  - 每个分支/特征固定完整病例队列，避免位置间样本集合变化
  - 连续区间置换检验 + 跨区间 BH-FDR，替代逐点原始 p 值
  - 支持的分支: MPV / SV / SMV / LPV / RPV / TIPS / LGV / PGV
    (任何在 JSON 中存在的非 None 段都会被分析)
  - 自动跳过该患者缺失的段

输出:
  1. pointwise_correlation.png  — 逐点相关性曲线 (显著区域高亮)
  2. profile_heatmap.png        — 剖面热力图 (按 target 排序)
  3. group_comparison.png       — 高/低 target 组剖面对比
  4. correlation_regions.csv    — 所有候选连续相关区间
  5. peak_correlations.csv      — 每个分支/特征摘要
  6. profile_report.txt         — 文字报告
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
from scipy.stats import rankdata, t as student_t
import warnings
warnings.filterwarnings('ignore')

from features_layout import (
    FEATURES_DIRNAME,
    POINTWISE_TEMP_NAME,
    UNIFIED_FEATURES_NAME,
)


# ============================================================
# 配置
# ============================================================

# 全部可能出现的分支 (按解剖顺序)
ALL_BRANCH_NAMES = ['mpv', 'sv', 'smv', 'lpv', 'rpv', 'tips', 'lgv', 'pgv']

BRANCH_LABELS = {
    'mpv':  'MPV (门静脉主干)',
    'sv':   'SV (脾静脉)',
    'smv':  'SMV (肠系膜上静脉)',
    'lpv':  'LPV (左肝静脉)',
    'rpv':  'RPV (右肝静脉)',
    'tips': 'TIPS (支架)',
    'lgv':  'LGV (胃左静脉)',
    'pgv':  'PGV (胃后静脉)',
}

BRANCH_COLORS = {
    'mpv':  '#ef4444',
    'sv':   '#3b82f6',
    'smv':  '#f59e0b',
    'lpv':  '#a855f7',
    'rpv':  '#10b981',
    'tips': '#06b6d4',
    'lgv':  '#eab308',
    'pgv':  '#ec4899',
}

FEATURE_KEYS = ['area', 'eq_diameter', 'circularity', 'curvature',
                'perimeter', 'inscribed_radius']

FEATURE_LABELS = {
    'area':             '真实截面积 (mm²)',
    'eq_diameter':      '等效直径 (mm)',
    'circularity':      '截面圆度',
    'curvature':        '曲率 (1/mm)',
    'perimeter':        '截面周长 (mm)',
    'inscribed_radius': '内切圆半径 (mm)',
}

TARGET_INFO = {
    'PVP': {'cn': '门静脉压力', 'unit': 'mmHg'},
    'PCG': {'cn': '门静脉压力梯度', 'unit': 'mmHg'},
}

PROFILE_FILENAME = POINTWISE_TEMP_NAME


# ============================================================
# 数据收集
# ============================================================

def _read_label(folder_path, target):
    txt = os.path.join(folder_path, "label", f"{target}.txt")
    if not os.path.exists(txt):
        return None
    try:
        with open(txt, 'r', encoding='utf-8') as f:
            for token in f.read().strip().replace(',', ' ').split():
                try:
                    value = float(token)
                    return value if np.isfinite(value) else None
                except ValueError:
                    continue
    except Exception:
        pass
    return None


def _candidate_feature_paths(folder_path, filename):
    return (
        os.path.join(folder_path, FEATURES_DIRNAME, filename),
        os.path.join(folder_path, filename),
    )


def _infer_subject_id(sample_name):
    value = str(sample_name).strip()
    if re.match(r'^(?:19|20)\d{6}', value):
        value = value[8:]
    value = re.split(r'[#@$]', value, maxsplit=1)[0].strip()
    return value or str(sample_name)


def _load_patient_profiles(folder_path):
    """Prefer cleaned profiles embedded in unified_features.json."""
    for path in _candidate_feature_paths(folder_path, UNIFIED_FEATURES_NAME):
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                unified = json.load(f)
            pointwise = unified.get('pointwise')
            if isinstance(pointwise, dict) and pointwise:
                profiles = dict(pointwise)
                meta = unified.get('pointwise_meta')
                if isinstance(meta, dict):
                    profiles['_meta'] = meta
                return profiles, path, 'unified'
        except Exception as exc:
            print(f"  跳过损坏的统一特征 {path}: {exc}")

    for path in _candidate_feature_paths(folder_path, PROFILE_FILENAME):
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                profiles = json.load(f)
            if isinstance(profiles, dict) and profiles:
                return profiles, path, 'pointwise'
        except Exception as exc:
            print(f"  跳过损坏的逐点特征 {path}: {exc}")
    return None, None, None


def collect_profiles(root_folder, target="PVP",
                     min_branch_coverage=0.3):
    """
    收集所有患者的剖面 + target 值。

    参数:
        min_branch_coverage: 一条分支需在至少这个比例的样本中存在
                             才参与分析 (例如 LGV 太罕见会被剔除)

    返回:
        data: list of dict
        n_points: int
        active_branches: list[str], 实际参与分析的分支
    """
    target = target.upper()
    print(f"\n收集剖面 + {target}...")

    subfolders = sorted(
        d for d in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, d)))

    data = []
    n_points = None

    for folder in subfolders:
        folder_path = os.path.join(root_folder, folder)
        label_val = _read_label(folder_path, target)
        if label_val is None:
            continue

        profiles, profile_path, profile_source = _load_patient_profiles(folder_path)
        if profiles is None:
            continue

        # 取每条剖面的点数 (用第一个非空段)
        if n_points is None:
            for branch in ALL_BRANCH_NAMES:
                p = profiles.get(branch)
                if p and 'position' in p:
                    n_points = len(p['position'])
                    break

        data.append({
            'name': folder,
            'subject_id': _infer_subject_id(folder),
            'is_post_tips': bool(
                (profiles.get('_meta') or {}).get('is_post_tips', '#' in folder)),
            'target_value': label_val,
            'profiles': profiles,
            'profile_path': profile_path,
            'profile_source': profile_source,
        })

    if not data:
        print("  没有可用样本!")
        return data, n_points, []

    # 选活跃分支: 出现率 ≥ min_branch_coverage
    n_total = len(data)
    branch_coverage = {}
    for branch in ALL_BRANCH_NAMES:
        n_present = sum(1 for d in data
                        if d['profiles'].get(branch) is not None)
        branch_coverage[branch] = n_present / n_total

    active_branches = [b for b in ALL_BRANCH_NAMES
                       if branch_coverage[b] >= min_branch_coverage]
    dropped = [b for b in ALL_BRANCH_NAMES
               if 0 < branch_coverage[b] < min_branch_coverage]

    print(f"  收集到 {n_total} 个样本, 每条剖面 {n_points} 个点")
    source_counts = {
        source: sum(d['profile_source'] == source for d in data)
        for source in ('unified', 'pointwise')
    }
    print(f"  剖面来源: unified={source_counts['unified']}, "
          f"standalone={source_counts['pointwise']}")
    print(f"  活跃分支 (覆盖率 ≥ {100*min_branch_coverage:.0f}%): "
          f"{[b.upper() for b in active_branches]}")
    if dropped:
        print(f"  剔除分支 (覆盖率不足): " +
              ", ".join(f"{b.upper()}({100*branch_coverage[b]:.0f}%)"
                        for b in dropped))

    return data, n_points, active_branches


# ============================================================
# 逐点相关性 (NaN-aware)
# ============================================================

def _fdr_bh(p_values):
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
    q_valid = np.empty(n_tests, dtype=float)
    q_valid[order] = np.clip(q_ranked, 0.0, 1.0)
    adjusted[valid_idx] = q_valid
    return adjusted


def _rank_unit_columns(values):
    ranks = rankdata(values, axis=0, method='average')
    centered = ranks - ranks.mean(axis=0, keepdims=True)
    norms = np.sqrt(np.sum(centered ** 2, axis=0))
    unit = np.divide(centered, norms, out=np.zeros_like(centered),
                     where=norms > 1e-12)
    return unit, norms > 1e-12


def _rho_p_values(rho, n_samples):
    p_values = np.full(np.asarray(rho).shape, np.nan, dtype=float)
    finite = np.isfinite(rho)
    if n_samples < 3 or not np.any(finite):
        return p_values
    clipped = np.clip(np.abs(np.asarray(rho)[finite]), 0, 1 - 1e-15)
    t_stat = clipped * np.sqrt((n_samples - 2) / (1 - clipped ** 2))
    p_values[finite] = 2 * student_t.sf(t_stat, df=n_samples - 2)
    return p_values


def _find_clusters(rho, threshold, min_cluster_points=3):
    """Find same-sign contiguous supra-threshold regions."""
    rho = np.asarray(rho, dtype=float)
    above = np.isfinite(rho) & (np.abs(rho) >= threshold)
    clusters = []
    start = None
    sign = None
    for idx in range(len(rho) + 1):
        current_sign = (np.sign(rho[idx])
                        if idx < len(rho) and above[idx] else None)
        if start is None and current_sign is not None:
            start, sign = idx, current_sign
        elif start is not None and current_sign != sign:
            end = idx - 1
            if end - start + 1 >= min_cluster_points:
                clusters.append({
                    'start_idx': int(start),
                    'end_idx': int(end),
                    'direction': 'positive' if sign > 0 else 'negative',
                    'mass': float(np.sum(np.abs(rho[start:end + 1]))),
                })
            start = idx if current_sign is not None else None
            sign = current_sign
    return clusters


def _complete_profile_matrix(data, branch, feat, n_points):
    """Use a fixed patient cohort across positions for valid permutations."""
    rows, targets, subject_ids, strata = [], [], [], []
    for item in data:
        profile = item['profiles'].get(branch)
        if not isinstance(profile, dict) or feat not in profile:
            continue
        try:
            values = np.asarray(profile[feat], dtype=float)
        except (TypeError, ValueError):
            continue
        if values.shape != (n_points,) or not np.all(np.isfinite(values)):
            continue
        target_value = float(item['target_value'])
        if not np.isfinite(target_value):
            continue
        rows.append(values)
        targets.append(target_value)
        subject_ids.append(item.get('subject_id') or item.get('name'))
        strata.append(bool(item.get('is_post_tips', False)))
    if not rows:
        return (np.empty((0, n_points)), np.empty(0),
                np.empty(0, dtype=object), np.empty(0, dtype=bool))
    return (np.vstack(rows), np.asarray(targets, dtype=float),
            np.asarray(subject_ids, dtype=object), np.asarray(strata, dtype=bool))


def _block_permutation_indices(subject_ids, n_permutations, rng, strata=None):
    """Exchange patient blocks with matching size and treatment pattern."""
    subject_ids = np.asarray(subject_ids, dtype=object)
    if strata is None:
        strata = np.zeros(len(subject_ids), dtype=bool)
    strata = np.asarray(strata)
    groups = {}
    for row_idx, subject_id in enumerate(subject_ids):
        groups.setdefault(subject_id, []).append(row_idx)
    compatible_groups = {}
    for indices in groups.values():
        indices = np.asarray(indices, dtype=int)
        key = (len(indices), tuple(strata[indices].tolist()))
        compatible_groups.setdefault(key, []).append(indices)

    permutations = np.tile(
        np.arange(len(subject_ids), dtype=int), (n_permutations, 1))
    for perm_idx in range(n_permutations):
        for clusters in compatible_groups.values():
            source_order = rng.permutation(len(clusters))
            for destination, source_idx in zip(clusters, source_order):
                permutations[perm_idx, destination] = clusters[source_idx]
    return permutations


def compute_pointwise_correlation(
        data, n_points, active_branches, target="PVP", min_samples=10,
        n_permutations=1000, cluster_alpha=0.05,
        min_cluster_points=3, random_state=20260805):
    """
    Spearman curves with cluster-permutation inference.

    Each branch/feature uses one complete-case cohort across all positions.
    Cluster p-values control family-wise error across positions in that curve;
    cluster q-values then control FDR across all detected regions.
    """
    target = target.upper()
    results = {}
    all_clusters = []
    combo_index = 0

    for branch in active_branches:
        results[branch] = {}
        for feat in FEATURE_KEYS:
            matrix, target_values, subject_ids, strata = _complete_profile_matrix(
                data, branch, feat, n_points)
            n_valid = len(target_values)
            n_subjects = len(pd.unique(subject_ids))
            rho = np.full(n_points, np.nan)
            pval = np.full(n_points, np.nan)
            clusters = []

            if (n_valid >= min_samples
                    and np.std(target_values) > 1e-10):
                x_unit, variable_cols = _rank_unit_columns(matrix)
                y_ranks = rankdata(target_values, method='average')
                y_centered = y_ranks - y_ranks.mean()
                y_norm = np.sqrt(np.sum(y_centered ** 2))

                if y_norm > 1e-12:
                    y_unit = y_centered / y_norm
                    rho_values = y_unit @ x_unit
                    rho[variable_cols] = rho_values[variable_cols]
                    pval = _rho_p_values(rho, n_valid)

                    t_critical = student_t.ppf(
                        1 - cluster_alpha / 2, df=n_valid - 2)
                    rho_threshold = float(np.sqrt(
                        t_critical ** 2
                        / (t_critical ** 2 + n_valid - 2)))
                    clusters = _find_clusters(
                        rho, rho_threshold, min_cluster_points)

                    if clusters and n_permutations > 0:
                        rng = np.random.default_rng(random_state + combo_index)
                        permutation_idx = _block_permutation_indices(
                            subject_ids, n_permutations, rng, strata=strata)
                        permuted_rho = y_unit[permutation_idx] @ x_unit
                        null_max_mass = np.zeros(n_permutations, dtype=float)
                        for perm_idx, perm_curve in enumerate(permuted_rho):
                            perm_clusters = _find_clusters(
                                perm_curve, rho_threshold,
                                min_cluster_points)
                            if perm_clusters:
                                null_max_mass[perm_idx] = max(
                                    item['mass'] for item in perm_clusters)

                        for cluster in clusters:
                            cluster['p_cluster_fwer'] = float(
                                (1 + np.sum(null_max_mass >= cluster['mass']))
                                / (n_permutations + 1))
                    else:
                        for cluster in clusters:
                            cluster['p_cluster_fwer'] = np.nan

            for cluster in clusters:
                cluster['branch'] = branch
                cluster['feature_key'] = feat
                cluster['q_cluster_fdr'] = np.nan
                all_clusters.append(cluster)

            results[branch][feat] = {
                'position': np.linspace(0, 1, n_points),
                'rho': rho,
                'p_value': pval,
                'n_valid': np.full(n_points, n_valid, dtype=int),
                'n_subjects': int(n_subjects),
                'clusters': clusters,
                'min_samples': int(min_samples),
                'n_permutations': int(n_permutations),
                'cluster_alpha': float(cluster_alpha),
                'min_cluster_points': int(min_cluster_points),
                'cohort_policy': 'complete profile per branch-feature',
            }
            combo_index += 1

    cluster_q = _fdr_bh([
        cluster['p_cluster_fwer'] for cluster in all_clusters
    ])
    for cluster, q_value in zip(all_clusters, cluster_q):
        cluster['q_cluster_fdr'] = float(q_value)
        cluster['significant'] = bool(q_value < 0.05)
    return results


def extract_peak_correlations(pw_results):
    """
    每个 (branch, feature) 的最可信连续区间及区间内峰值摘要。
    """
    rows = []
    for branch, feats in pw_results.items():
        for feat, res in feats.items():
            rho = res['rho']
            pval = res['p_value']
            n_valid = res['n_valid']
            valid_mask = np.isfinite(rho)
            if not np.any(valid_mask):
                continue

            abs_rho_full = np.where(valid_mask, np.abs(rho), -np.inf)
            peak_idx = int(np.argmax(abs_rho_full))
            clusters = res.get('clusters') or []
            selected_cluster = None
            if clusters:
                selected_cluster = min(
                    clusters,
                    key=lambda item: (
                        item.get('q_cluster_fdr', np.inf),
                        item.get('p_cluster_fwer', np.inf),
                        -item['mass']))
                region_slice = slice(
                    selected_cluster['start_idx'],
                    selected_cluster['end_idx'] + 1)
                local = np.abs(rho[region_slice])
                peak_idx = (selected_cluster['start_idx']
                            + int(np.nanargmax(local)))

            peak_pos = float(res['position'][peak_idx])
            peak_rho = float(rho[peak_idx])
            peak_p = float(pval[peak_idx])
            peak_n = int(n_valid[peak_idx])

            significant_clusters = [
                item for item in clusters if item.get('significant', False)]
            significant_points = sum(
                item['end_idx'] - item['start_idx'] + 1
                for item in significant_clusters)
            sig_frac = significant_points / len(rho)

            if selected_cluster is None:
                region_start = region_end = np.nan
                cluster_p = cluster_q = np.nan
                cluster_mass = np.nan
                significant = False
            else:
                region_start = float(
                    res['position'][selected_cluster['start_idx']])
                region_end = float(
                    res['position'][selected_cluster['end_idx']])
                cluster_p = selected_cluster.get('p_cluster_fwer', np.nan)
                cluster_q = selected_cluster.get('q_cluster_fdr', np.nan)
                cluster_mass = selected_cluster['mass']
                significant = selected_cluster.get('significant', False)

            rows.append({
                'branch': branch.upper(),
                'feature': FEATURE_LABELS.get(feat, feat),
                'feature_key': feat,
                'peak_position': round(peak_pos, 3),
                'peak_position_pct': f"{peak_pos*100:.1f}%",
                'peak_rho': round(peak_rho, 4),
                'peak_p_raw': round(peak_p, 5),
                'peak_n': peak_n,
                'n_subjects': int(res.get('n_subjects', peak_n)),
                'region_start_pct': (f"{region_start*100:.1f}%"
                                     if np.isfinite(region_start) else ''),
                'region_end_pct': (f"{region_end*100:.1f}%"
                                   if np.isfinite(region_end) else ''),
                'cluster_mass': cluster_mass,
                'cluster_p_fwer': cluster_p,
                'cluster_q_fdr': cluster_q,
                'significant_fraction': round(sig_frac, 3),
                'significant': bool(significant),
            })
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df['_sort_q'] = df['cluster_q_fdr'].fillna(np.inf)
        df = df.sort_values(
            ['significant', '_sort_q', 'peak_rho'],
            ascending=[False, True, False]).drop(columns='_sort_q').reset_index(drop=True)
    return df


def extract_correlation_regions(pw_results):
    rows = []
    for branch, feats in pw_results.items():
        for feat, result in feats.items():
            rho = result['rho']
            position = result['position']
            n_valid = int(result['n_valid'][0]) if len(result['n_valid']) else 0
            for cluster in result.get('clusters') or []:
                region = slice(cluster['start_idx'], cluster['end_idx'] + 1)
                peak_idx = (cluster['start_idx']
                            + int(np.nanargmax(np.abs(rho[region]))))
                rows.append({
                    'branch': branch.upper(),
                    'feature': FEATURE_LABELS.get(feat, feat),
                    'feature_key': feat,
                    'direction': cluster['direction'],
                    'start_position_pct': round(
                        float(position[cluster['start_idx']] * 100), 2),
                    'end_position_pct': round(
                        float(position[cluster['end_idx']] * 100), 2),
                    'width_points': (cluster['end_idx']
                                     - cluster['start_idx'] + 1),
                    'peak_position_pct': round(
                        float(position[peak_idx] * 100), 2),
                    'peak_rho': round(float(rho[peak_idx]), 4),
                    'cluster_mass': round(cluster['mass'], 4),
                    'cluster_p_fwer': cluster.get('p_cluster_fwer', np.nan),
                    'cluster_q_fdr': cluster.get('q_cluster_fdr', np.nan),
                    'n_samples': n_valid,
                    'n_subjects': int(result.get('n_subjects', n_valid)),
                    'significant': cluster.get('significant', False),
                })
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(
            ['significant', 'cluster_q_fdr', 'cluster_mass'],
            ascending=[False, True, False], na_position='last').reset_index(drop=True)
    return df


# ============================================================
# 可视化 1: 逐点相关性曲线
# ============================================================

def plot_pointwise_correlation(pw_results, active_branches,
                                output_dir, target="PVP"):
    target = target.upper()
    target_cn = TARGET_INFO.get(target, {}).get('cn', target)
    print("\n绘制逐点相关性曲线...")

    n_branches = len(active_branches)
    if n_branches == 0:
        return

    fig, axes = plt.subplots(len(FEATURE_KEYS), n_branches,
                              figsize=(5 * n_branches, 4 * len(FEATURE_KEYS)))
    if n_branches == 1:
        axes = axes[:, np.newaxis]
    if len(FEATURE_KEYS) == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(f'逐点 Spearman 相关性: 剖面特征 vs {target}\n'
                 f'Cluster-permutation inference vs {target_cn}',
                 fontsize=16, fontweight='bold', y=1.02)

    for fi, feat in enumerate(FEATURE_KEYS):
        for bi, branch in enumerate(active_branches):
            ax = axes[fi, bi]
            res = pw_results[branch][feat]
            pos, rho = res['position'], res['rho']
            color = BRANCH_COLORS.get(branch, '#64748b')

            ax.plot(pos * 100, rho, color=color, linewidth=2, zorder=3)

            significant_clusters = [
                item for item in res.get('clusters', [])
                if item.get('significant', False)]
            for cluster_idx, cluster in enumerate(significant_clusters):
                start = pos[cluster['start_idx']] * 100
                end = pos[cluster['end_idx']] * 100
                ax.axvspan(
                    start, end, alpha=0.18, color=color, zorder=1,
                    label=('cluster q < 0.05'
                           if cluster_idx == 0 else None))

            ax.axhline(y=0, color='black', linewidth=0.5)
            ax.axhline(y=0.3, color='gray', linewidth=0.5,
                       linestyle='--', alpha=0.5)
            ax.axhline(y=-0.3, color='gray', linewidth=0.5,
                       linestyle='--', alpha=0.5)

            if significant_clusters:
                strongest = min(
                    significant_clusters,
                    key=lambda item: item['q_cluster_fdr'])
                region = slice(strongest['start_idx'], strongest['end_idx'] + 1)
                peak_idx = (strongest['start_idx']
                            + int(np.nanargmax(np.abs(rho[region]))))
                ax.plot(pos[peak_idx] * 100, rho[peak_idx], 'o',
                        color='red', markersize=8, zorder=5)
                ax.annotate(f"ρ={rho[peak_idx]:.3f}\n"
                            f"q={strongest['q_cluster_fdr']:.3g}",
                            xy=(pos[peak_idx] * 100, rho[peak_idx]),
                            xytext=(10, 10), textcoords='offset points',
                            fontsize=8, fontweight='bold', color='red',
                            arrowprops=dict(arrowstyle='->', color='red', lw=1))

            ax.set_xlim(0, 100)
            ax.set_ylim(-1, 1)
            ax.grid(True, alpha=0.2, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if fi == 0:
                ax.set_title(BRANCH_LABELS.get(branch, branch),
                             fontsize=11, fontweight='bold', color=color)
            if bi == 0:
                ax.set_ylabel(f'{FEATURE_LABELS[feat]}\nSpearman ρ',
                              fontsize=10)
            if fi == len(FEATURE_KEYS) - 1:
                ax.set_xlabel('归一化位置 (%)', fontsize=10)
            if significant_clusters:
                ax.legend(fontsize=8, loc='lower right')

    plt.tight_layout()
    path = os.path.join(output_dir, 'pointwise_correlation.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  保存: {path}")


# ============================================================
# 可视化 2: 剖面热力图 (按 target 排序)
# ============================================================

def plot_profile_heatmaps(data, active_branches, output_dir, target="PVP"):
    target = target.upper()
    print("\n绘制剖面热力图...")

    sorted_data = sorted(data, key=lambda d: d['target_value'])
    n_branches = len(active_branches)
    if n_branches == 0:
        return

    feats_to_plot = ['area', 'eq_diameter']

    fig, axes = plt.subplots(len(feats_to_plot), n_branches,
                              figsize=(6 * n_branches, 5 * len(feats_to_plot)))
    if n_branches == 1:
        axes = axes[:, np.newaxis]
    if len(feats_to_plot) == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(f'剖面热力图 (按 {target} 排序)',
                 fontsize=16, fontweight='bold', y=1.02)

    for fi, feat in enumerate(feats_to_plot):
        for bi, branch in enumerate(active_branches):
            ax = axes[fi, bi]
            rows, valid_targets = [], []
            for d in sorted_data:
                prof = d['profiles'].get(branch)
                if prof and feat in prof:
                    rows.append(prof[feat])
                    valid_targets.append(d['target_value'])

            if not rows:
                ax.text(0.5, 0.5, '无数据', transform=ax.transAxes,
                        ha='center', va='center')
                continue

            # 转 float 数组, NaN 保留 (matplotlib 会以白色显示)
            matrix = np.array(rows, dtype=float)
            # 用 masked array 让 NaN 显示为透明 (避免 imshow 把 NaN 当 0)
            masked = np.ma.masked_invalid(matrix)
            cmap = plt.cm.YlOrRd.copy()
            cmap.set_bad(color='white', alpha=0.5)

            im = ax.imshow(masked, aspect='auto', cmap=cmap,
                           extent=[0, 100, len(rows), 0])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label=FEATURE_LABELS.get(feat, feat))

            n_ticks = min(8, len(valid_targets))
            tick_positions = np.linspace(0, len(valid_targets) - 1,
                                         n_ticks, dtype=int)
            ax.set_yticks(tick_positions + 0.5)
            ax.set_yticklabels([f'{valid_targets[i]:.0f}'
                                for i in tick_positions], fontsize=8)

            if fi == 0:
                ax.set_title(BRANCH_LABELS.get(branch, branch),
                             fontsize=11, fontweight='bold',
                             color=BRANCH_COLORS.get(branch, '#64748b'))
            if bi == 0:
                ax.set_ylabel(f'{FEATURE_LABELS[feat]}\n{target} →',
                              fontsize=10)
            if fi == len(feats_to_plot) - 1:
                ax.set_xlabel('归一化位置 (%)', fontsize=10)

    plt.tight_layout()
    path = os.path.join(output_dir, 'profile_heatmap.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  保存: {path}")


# ============================================================
# 可视化 3: 高/低 target 组对比 (NaN-aware)
# ============================================================

def plot_group_comparison(data, active_branches, output_dir, target="PVP"):
    target = target.upper()
    unit = TARGET_INFO.get(target, {}).get('unit', '')
    print("\n绘制高/低组对比...")

    target_vals = np.array([d['target_value'] for d in data])
    median_val = np.median(target_vals)
    low_data = [d for d, m in zip(data, target_vals <= median_val) if m]
    high_data = [d for d, m in zip(data, target_vals > median_val) if m]
    print(f"  低组: n={len(low_data)}, {target}≤{median_val:.1f}")
    print(f"  高组: n={len(high_data)}, {target}>{median_val:.1f}")

    feats_to_plot = ['area', 'eq_diameter', 'circularity', 'curvature']
    n_branches = len(active_branches)
    if n_branches == 0:
        return

    fig, axes = plt.subplots(len(feats_to_plot), n_branches,
                              figsize=(6 * n_branches, 4 * len(feats_to_plot)))
    if n_branches == 1:
        axes = axes[:, np.newaxis]
    if len(feats_to_plot) == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(f'高/低 {target} 组剖面对比 (中位数={median_val:.1f}{unit})',
                 fontsize=16, fontweight='bold', y=1.02)

    for fi, feat in enumerate(feats_to_plot):
        for bi, branch in enumerate(active_branches):
            ax = axes[fi, bi]

            for grp, label, color, alpha in [
                (low_data, f'低{target}', '#3b82f6', 0.2),
                (high_data, f'高{target}', '#ef4444', 0.2),
            ]:
                values = []
                for d in grp:
                    prof = d['profiles'].get(branch)
                    if prof and feat in prof:
                        values.append(prof[feat])
                if not values:
                    continue
                # 分组图只做描述性展示；使用中位数/IQR，避免正态假设。
                matrix = np.array(values, dtype=float)
                if not np.any(np.isfinite(matrix)):
                    continue
                median = np.nanmedian(matrix, axis=0)
                q25 = np.nanpercentile(matrix, 25, axis=0)
                q75 = np.nanpercentile(matrix, 75, axis=0)
                x = np.linspace(0, 100, len(median))
                ax.plot(x, median, color=color, linewidth=2, label=label)
                ax.fill_between(x, q25, q75,
                                color=color, alpha=alpha)

            ax.legend(fontsize=9, loc='upper right')
            ax.grid(True, alpha=0.2, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if fi == 0:
                ax.set_title(BRANCH_LABELS.get(branch, branch),
                             fontsize=11, fontweight='bold',
                             color=BRANCH_COLORS.get(branch, '#64748b'))
            if bi == 0:
                ax.set_ylabel(FEATURE_LABELS.get(feat, feat), fontsize=10)
            if fi == len(feats_to_plot) - 1:
                ax.set_xlabel('归一化位置 (%)', fontsize=10)

    plt.tight_layout()
    path = os.path.join(output_dir, 'group_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  保存: {path}")


# ============================================================
# 报告
# ============================================================

def generate_profile_report(regions_df, data, active_branches,
                             output_dir, target="PVP"):
    target = target.upper()
    target_cn = TARGET_INFO.get(target, {}).get('cn', target)
    unit = TARGET_INFO.get(target, {}).get('unit', '')
    print("\n生成报告...")

    target_vals = [d['target_value'] for d in data]
    n_subjects = len({d.get('subject_id') or d['name'] for d in data})

    # 端点掩码元信息 (从第一个有 _meta 的样本读)
    edge_pct = None
    edge_mm = None
    for d in data:
        meta = d['profiles'].get('_meta')
        if meta:
            edge_pct = meta.get('edge_margin_pct')
            edge_mm = meta.get('edge_margin_mm')
            break

    source_counts = {
        source: sum(d.get('profile_source') == source for d in data)
        for source in ('unified', 'pointwise')
    }
    significant_regions = (
        regions_df[regions_df['significant']]
        if len(regions_df) else regions_df)

    lines = [
        "=" * 70,
        f"中心线剖面特征与{target_cn}({target})逐点相关性报告",
        "=" * 70, "",
        f"样本数: {len(data)}",
        f"推断患者簇数: {n_subjects} (置换时保留簇内记录结构)",
        f"{target} 范围: {min(target_vals):.1f} - {max(target_vals):.1f} {unit}",
        f"{target} 均值: {np.mean(target_vals):.1f} ± {np.std(target_vals):.1f} {unit}",
        f"分析的分支: {[b.upper() for b in active_branches]}",
        f"剖面来源: unified={source_counts['unified']}, "
        f"standalone={source_counts['pointwise']}",
    ]
    if edge_pct is not None:
        lines.append(f"端点掩码: 比例 {edge_pct*100:.1f}%, 距离 {edge_mm:.1f} mm")
    lines += [
        "",
        "-" * 70,
        "校正后显著的连续相关区间:", "",
    ]

    if len(significant_regions) > 0:
        lines.append(
            f"  显著区间 (cluster FWER + 全局 FDR q<0.05): "
            f"{len(significant_regions)} / {len(regions_df)}")
        lines.append("")
        lines.append(f"  {'分支':>6s}  {'特征':>14s}  {'区间':>15s}  "
                     f"{'峰值ρ':>9s}  {'cluster p':>10s}  "
                     f"{'FDR q':>10s}  {'N':>4s}  {'簇':>4s}")
        lines.append("  " + "-" * 88)
        for _, row in significant_regions.iterrows():
            interval = (f"{row['start_position_pct']:.1f}-"
                        f"{row['end_position_pct']:.1f}%")
            lines.append(
                f"  {row['branch']:>6s}  {row['feature']:>14s}  "
                f"{interval:>15s}  {row['peak_rho']:>+9.4f}  "
                f"{row['cluster_p_fwer']:>10.4f}  "
                f"{row['cluster_q_fdr']:>10.4f}  "
                f"{int(row['n_samples']):>4d}  "
                f"{int(row['n_subjects']):>4d}")
    else:
        lines.append("  校正后未发现显著连续区间。")

    lines += [
        "", "-" * 70,
        "分析说明:", "",
        "  ・每条分支被归一化到 [0%, 100%], 0%=起点 bp, 100%=末端",
        "  ・统一剖面先删除无效截面，再仅在保留弧段内部重采样到固定点数",
        "  ・每个分支/特征固定使用完整剖面病例，所有位置 N 相同",
        "  ・逐点 Spearman 仅描述相关曲线，不用原始逐点 p 值判显著",
        "  ・连续同方向超阈值位置组成区间，1000 次患者簇置换控制曲线内 FWER",
        "  ・同一患者标签作为整组，仅在记录数和 TIPS 状态序列相同的患者簇间交换",
        "  ・所有候选区间的 cluster p 再做 BH-FDR，最终以 q<0.05 判显著",
        "  ・高/低组图仅展示中位数和四分位距，不进行二分后的重复检验",
        "  ・分析单位仍是一次扫描；同一患者术前/术后记录需结合研究设计解读",
        "", "=" * 70,
    ]

    report = '\n'.join(lines)
    path = os.path.join(output_dir, 'profile_report.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  保存: {path}")
    print("\n" + report)


# ============================================================
# 主流程
# ============================================================

def _setup_matplotlib():
    plt.rcParams['font.sans-serif'] = [
        'SimHei', 'Microsoft YaHei', 'DejaVu Sans',
        'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False


def run_profile_analysis(root_folder, output_dir=None, target="PVP",
                         min_branch_coverage=0.3, min_samples=10,
                         n_permutations=1000, random_state=20260805):
    """完整流程: 收集 → 逐点相关 → 可视化 → 报告。"""
    target = target.upper()
    if output_dir is None:
        output_dir = os.path.join(root_folder,
                                   f"profile_correlation_{target.lower()}")
    os.makedirs(output_dir, exist_ok=True)
    _setup_matplotlib()

    data, n_points, active_branches = collect_profiles(
        root_folder, target, min_branch_coverage)

    if len(data) < min_samples:
        print(f"样本数 ({len(data)}) 不足, 至少需要 {min_samples} 个")
        return
    if not active_branches:
        print(f"无活跃分支 (覆盖率均不足 {100*min_branch_coverage:.0f}%)")
        return

    print(f"\n计算逐点相关性 ({len(data)} 样本 × {n_points} 位置 × "
          f"{len(active_branches)} 分支)...")
    pw_results = compute_pointwise_correlation(
        data, n_points, active_branches, target,
        min_samples=min_samples,
        n_permutations=n_permutations,
        random_state=random_state)

    peak_df = extract_peak_correlations(pw_results)
    csv_path = os.path.join(output_dir, 'peak_correlations.csv')
    peak_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  分支/特征摘要 CSV: {csv_path}")

    regions_df = extract_correlation_regions(pw_results)
    regions_path = os.path.join(output_dir, 'correlation_regions.csv')
    regions_df.to_csv(regions_path, index=False, encoding='utf-8-sig')
    print(f"  连续相关区间 CSV: {regions_path}")

    plot_pointwise_correlation(pw_results, active_branches, output_dir, target)
    plot_profile_heatmaps(data, active_branches, output_dir, target)
    plot_group_comparison(data, active_branches, output_dir, target)
    generate_profile_report(regions_df, data, active_branches,
                            output_dir, target)

    print(f"\n{'='*60}")
    print(f"剖面分析完成! 结果: {output_dir}")
    print(f"{'='*60}")


# ============================================================
# 用户配置
# ============================================================

if __name__ == '__main__':
    TARGET = "PVP"
    ROOT_FOLDER = r"F:\PCG data\dataset\zhengzhou_vkan_qian47"
    OUTPUT_DIR = None  # None = 自动: root/profile_correlation_pvp/

    run_profile_analysis(ROOT_FOLDER, OUTPUT_DIR, TARGET)
