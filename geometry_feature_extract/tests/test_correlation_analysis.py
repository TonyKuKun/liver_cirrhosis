import json
from pathlib import Path

import numpy as np
import pandas as pd

from correlation_analysis import (
    _fdr_bh as scalar_fdr_bh,
    _auto_interpret_correlations,
    _infer_subject_id as infer_scalar_subject_id,
    collect_features,
    compute_correlations,
)
from profile_correlation import (
    _block_permutation_indices,
    _load_patient_profiles,
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
