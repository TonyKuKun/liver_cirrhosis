import json
from pathlib import Path

import numpy as np
import pandas as pd

from correlation_analysis import (
    _deduplicate_feature_columns,
    _fdr_bh as scalar_fdr_bh,
    _auto_interpret_correlations,
    _infer_subject_id as infer_scalar_subject_id,
    collect_features,
    compute_correlations,
    compute_pretips_collateral_correlations,
)
from profile_correlation import (
    _block_permutation_indices,
    _load_patient_profiles,
    _profile_for_analysis,
    collect_profiles,
    compute_pointwise_correlation,
    extract_correlation_regions,
)


def _write_unified(path: Path, mpv_length: float = 12.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        'statistical': {'mpv': {'length': mpv_length}},
        'global': {},
        'system': {},
        'pointwise': {
            'mpv': {
                'position': [0.0, 1.0],
                'area': [1.0, 2.0],
            },
        },
        'pointwise_meta': {'unified_target_n_points': 2},
    }), encoding='utf-8')


def test_collect_features_prefers_canonical_features_directory(tmp_path):
    patient = tmp_path / 'patient_1'
    (patient / 'label').mkdir(parents=True)
    (patient / 'label' / 'PVP.txt').write_text('12.5', encoding='utf-8')
    _write_unified(patient / 'features' / 'unified_features.json', 17.0)
    _write_unified(patient / 'unified_features.json', 99.0)

    output = tmp_path / 'all.tsv'
    result, active = collect_features(str(tmp_path), output_txt=str(output))

    assert result == str(output)
    assert active == ['mpv_length']
    frame = pd.read_csv(output, sep='\t')
    assert frame.loc[0, 'mpv_length'] == 17.0


def test_profile_loader_prefers_unified_pointwise(tmp_path):
    patient = tmp_path / 'patient_1'
    _write_unified(patient / 'features' / 'unified_features.json')
    standalone = patient / 'features' / 'pointwise_profiles.json'
    standalone.write_text(json.dumps({'mpv': {'area': [9.0, 9.0]}}),
                          encoding='utf-8')

    profiles, path, source = _load_patient_profiles(str(patient))

    assert source == 'unified'
    assert path.endswith('features\\unified_features.json')
    assert profiles['mpv']['area'] == [1.0, 2.0]
    assert profiles['_meta']['unified_target_n_points'] == 2


def test_collectors_do_not_fall_back_to_legacy_feature_files(tmp_path):
    patient = tmp_path / 'legacy_only'
    (patient / 'features').mkdir(parents=True)
    (patient / 'label').mkdir(parents=True)
    (patient / 'label' / 'PVP.txt').write_text('25', encoding='utf-8')
    (patient / 'features' / 'portal_vein_features.json').write_text(
        json.dumps({'mpv_length': 99.0}), encoding='utf-8')
    (patient / 'features' / 'pointwise_profiles.json').write_text(
        json.dumps({'mpv': {'position': [0.0, 1.0],
                            'area': [9.0, 9.0]}}),
        encoding='utf-8')

    output = tmp_path / 'legacy.tsv'
    result, active = collect_features(str(tmp_path), output_txt=str(output))
    profiles, path, source = _load_patient_profiles(str(patient))

    assert result is None
    assert active == []
    assert (profiles, path, source) == (None, None, None)


def test_scalar_and_profile_collectors_skip_at_marked_folders(tmp_path):
    for name, length, label in (
            ('included_patient', 11.0, 20.0),
            ('excluded@special', 99.0, 40.0)):
        patient = tmp_path / name
        (patient / 'label').mkdir(parents=True)
        (patient / 'label' / 'PVP.txt').write_text(
            str(label), encoding='utf-8')
        _write_unified(
            patient / 'features' / 'unified_features.json', length)

    output = tmp_path / 'filtered.tsv'
    _, active = collect_features(str(tmp_path), output_txt=str(output))
    frame = pd.read_csv(output, sep='\t')
    profiles, n_points, _ = collect_profiles(
        str(tmp_path), min_branch_coverage=0.0)

    assert active == ['mpv_length']
    assert frame['sample'].tolist() == ['included_patient']
    assert [item['name'] for item in profiles] == ['included_patient']
    assert n_points == 2


def test_post_tips_collateral_values_are_masked_from_scalar_analysis(tmp_path):
    for name, has_tips, collateral_diameter in (
            ('pre_patient', 0, 8.0),
            ('post_patient#', 1, 20.0)):
        patient = tmp_path / name
        (patient / 'label').mkdir(parents=True)
        (patient / 'label' / 'PVP.txt').write_text('25', encoding='utf-8')
        unified = {
            'statistical': {},
            'global': {
                'has_tips': has_tips,
                'has_lgv': 1,
                'has_pgv': 0,
                'has_compensation_vessel': 1,
            },
            'system': {
                'max_collateral_diameter_mm': collateral_diameter,
                'collateral_burden_score': 0.3,
                'n_collaterals_detected': 1,
            },
        }
        path = patient / 'features' / 'unified_features.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(unified), encoding='utf-8')

    output = tmp_path / 'collateral.tsv'
    collect_features(str(tmp_path), output_txt=str(output))
    frame = pd.read_csv(output, sep='\t')

    pre_value = frame.loc[
        frame['sample'] == 'pre_patient', 'max_collateral_diameter_mm'].iloc[0]
    post_value = frame.loc[
        frame['sample'] == 'post_patient#', 'max_collateral_diameter_mm'].iloc[0]
    assert pre_value == 8.0
    assert np.isnan(post_value)
    assert np.isnan(frame.loc[
        frame['sample'] == 'post_patient#', 'has_lgv'].iloc[0])


def test_pretips_collateral_analysis_excludes_post_tips_rows():
    frame = pd.DataFrame({
        'sample': [f'pre_{idx}' for idx in range(10)]
                  + [f'post_{idx}#' for idx in range(10)],
        'has_tips': [0.0] * 10 + [1.0] * 10,
        'max_collateral_diameter_mm': list(range(10)) + [np.nan] * 10,
        'PVP': list(range(10)) + list(range(10, 20)),
    })

    subset, result = compute_pretips_collateral_correlations(frame)

    assert len(subset) == 10
    assert result.loc[0, 'feature'] == 'max_collateral_diameter_mm'
    assert result.loc[0, 'n_samples'] == 10


def test_collateral_profile_coverage_and_values_are_pretips_only(tmp_path):
    for name, post_tips in (('pre_patient', False), ('post_patient#', True)):
        patient = tmp_path / name
        (patient / 'label').mkdir(parents=True)
        (patient / 'label' / 'PVP.txt').write_text('25', encoding='utf-8')
        path = patient / 'features' / 'unified_features.json'
        _write_unified(path)
        unified = json.loads(path.read_text(encoding='utf-8'))
        unified['pointwise']['lgv'] = {
            'position': [0.0, 1.0],
            'area': [3.0, 4.0],
        }
        unified['pointwise_meta']['is_post_tips'] = post_tips
        path.write_text(json.dumps(unified), encoding='utf-8')

    data, _, active = collect_profiles(
        str(tmp_path), min_branch_coverage=0.75)
    post_item = next(item for item in data if item['is_post_tips'])

    assert 'lgv' in active
    assert _profile_for_analysis(post_item, 'lgv') is None


def test_bh_fdr_and_minimum_sample_rule():
    adjusted = scalar_fdr_bh([0.01, 0.04, 0.03, np.nan])
    np.testing.assert_allclose(adjusted[:3], [0.03, 0.04, 0.04])
    assert np.isnan(adjusted[3])

    frame = pd.DataFrame({
        'sample': [f's{i}' for i in range(9)],
        'feature': np.arange(9, dtype=float),
        'PVP': np.arange(9, dtype=float),
    })
    result = compute_correlations(frame, ['feature'], min_samples=10)
    assert np.isnan(result.loc[0, 'spearman_r'])
    assert result.loc[0, 'n_samples'] == 9


def test_exact_variable_feature_aliases_are_removed_before_fdr():
    frame = pd.DataFrame({
        'original': [1.0, 2.0, np.nan, 4.0],
        'alias': [1.0, 2.0, np.nan, 4.0],
        'different': [1.0, 2.0, np.nan, 5.0],
        'constant_a': [0.0] * 4,
        'constant_b': [0.0] * 4,
    })

    retained, aliases = _deduplicate_feature_columns(
        frame, list(frame.columns))

    assert retained == [
        'original', 'different', 'constant_a', 'constant_b']
    assert aliases == {'alias': 'original'}


def test_auto_interpretation_formats_fdr_results():
    frame = pd.DataFrame({
        'sample': [f's{i}' for i in range(12)],
        'mpv_length': np.arange(12, dtype=float),
        'PVP': np.arange(12, dtype=float),
    })
    result = compute_correlations(frame, ['mpv_length'], min_samples=10)

    lines = _auto_interpret_correlations(result, frame)

    assert any('FDR q' in line for line in lines)
    assert any('MPV_长度' in line for line in lines)


def test_repeated_scans_use_patient_cluster_inference():
    rng = np.random.default_rng(3)
    samples, feature, target = [], [], []
    for subject_idx in range(6):
        for day, suffix in ((1, ''), (2, '#')):
            samples.append(f'2020010{day}Patient{subject_idx}{suffix}')
            value = subject_idx + day * 0.2
            feature.append(value + rng.normal(0, 0.1))
            target.append(value + rng.normal(0, 0.2))
    frame = pd.DataFrame({
        'sample': samples,
        'feature': feature,
        'PVP': target,
    })

    result = compute_correlations(frame, ['feature'], min_samples=10)

    assert infer_scalar_subject_id(samples[0]) == 'Patient0'
    assert result.loc[0, 'n_samples'] == 12
    assert result.loc[0, 'n_subjects'] == 6
    assert np.isfinite(result.loc[0, 'spearman_p_naive'])
    assert np.isfinite(result.loc[0, 'spearman_p'])
    assert np.isfinite(result.loc[0, 'spearman_ci_low'])


def test_block_permutation_keeps_repeated_records_together():
    subject_ids = np.array(['A', 'A', 'B', 'B', 'C'], dtype=object)
    permutations = _block_permutation_indices(
        subject_ids, 20, np.random.default_rng(9))

    pair_blocks = ({0, 1}, {2, 3})
    for permutation in permutations:
        assert set(permutation[[0, 1]]) in pair_blocks
        assert set(permutation[[2, 3]]) in pair_blocks
        assert permutation[4] == 4


def test_cluster_permutation_detects_known_contiguous_region():
    rng = np.random.default_rng(42)
    n_samples = 24
    n_points = 40
    target = np.linspace(-2, 2, n_samples)
    data = []
    for idx, target_value in enumerate(target):
        area = rng.normal(0, 1, n_points)
        area[10:21] += 3.0 * target_value
        data.append({
            'name': f's{idx}',
            'target_value': float(target_value),
            'profiles': {'mpv': {'area': area.tolist()}},
        })

    results = compute_pointwise_correlation(
        data, n_points, ['mpv'], min_samples=10,
        n_permutations=199, random_state=7)
    regions = extract_correlation_regions(results)

    significant = regions[regions['significant']]
    assert len(significant) >= 1
    strongest = significant.iloc[0]
    assert strongest['feature_key'] == 'area'
    assert strongest['start_position_pct'] <= 10 / 39 * 100
    assert strongest['end_position_pct'] >= 20 / 39 * 100 - 0.01
    assert strongest['cluster_q_fdr'] < 0.05
    assert strongest['n_samples'] == n_samples
