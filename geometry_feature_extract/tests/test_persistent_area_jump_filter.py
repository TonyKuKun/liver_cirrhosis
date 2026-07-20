import numpy as np

from extract_profiles import _apply_persistent_area_jump_filter


def _profile(area):
    n = len(area)
    arc = np.linspace(0.0, 20.0, n)
    diameter = np.sqrt(4.0 * np.asarray(area, dtype=float) / np.pi)
    return {
        'position': np.linspace(0.0, 1.0, n).tolist(),
        'arc_length_mm': arc.tolist(),
        'area': list(map(float, area)),
        'perimeter': (np.pi * diameter).tolist(),
        'eq_diameter': diameter.tolist(),
        'circularity': [1.0] * n,
        'hydraulic_diameter': diameter.tolist(),
        'solidity': [1.0] * n,
        'n_components': [1.0] * n,
        'junction_replaced': [0.0] * n,
    }


def test_persistent_high_area_at_endpoint_is_masked_not_interpolated():
    profile = _profile([36, 36, 36, 36, 10, 10, 10, 10, 10, 10, 10])

    out = _apply_persistent_area_jump_filter(
        profile, ratio_threshold=1.8, window_mm=4.0,
        min_persistence_mm=4.0, max_terminal_extension_mm=8.0)

    assert out['n_area_jump_terminal_masked'] == 4
    assert out['n_area_jump_interpolated'] == 0
    assert np.all(np.isnan(np.asarray(out['area'][:4], dtype=float)))
    assert out['area_jump_terminal_mask'][:4] == [1.0] * 4


def test_persistent_high_area_interior_is_interpolated_from_both_sides():
    profile = _profile([10, 10, 10, 10, 40, 40, 40, 40, 10, 10, 10])

    out = _apply_persistent_area_jump_filter(
        profile, ratio_threshold=1.8, window_mm=4.0,
        min_persistence_mm=4.0, max_terminal_extension_mm=8.0)

    changed = np.asarray(out['area_jump_interpolated'], dtype=bool)
    assert np.any(changed)
    assert np.allclose(np.asarray(out['area'], dtype=float)[changed], 10.0)
    assert out['n_area_jump_terminal_masked'] == 0


def test_persistent_area_drop_is_reported_but_unchanged():
    profile = _profile([10, 10, 10, 10, 2, 2, 2, 2, 10, 10, 10])

    out = _apply_persistent_area_jump_filter(
        profile, ratio_threshold=1.8, window_mm=4.0,
        min_persistence_mm=4.0, max_terminal_extension_mm=8.0)

    candidate = np.asarray(out['area_drop_candidate'], dtype=bool)
    assert np.any(candidate)
    assert np.allclose(np.asarray(out['area'], dtype=float)[candidate], 2.0)
    assert out['n_area_jump_interpolated'] == 0
