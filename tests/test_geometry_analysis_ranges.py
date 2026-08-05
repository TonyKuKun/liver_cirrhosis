import json
import sys
from pathlib import Path

import numpy as np

from web import geometry_backend as geometry_web


def _profile(n_points=200, leading_zero=0, trailing_zero=0):
    valid_count = n_points - leading_zero - trailing_zero
    valid = [0.0] * leading_zero + [1.0] * valid_count + [0.0] * trailing_zero
    profile = {
        "position": [index / n_points for index in range(n_points)],
        "arc_length_mm": [float(index) for index in range(n_points)],
        "area": [0.0 if not flag else float(index + 1) for index, flag in enumerate(valid)],
        "eq_diameter": [0.0 if not flag else 2.0 for flag in valid],
        "perimeter": [0.0 if not flag else 3.0 for flag in valid],
        "section_valid": valid,
        "centerline_x": [float(index) for index in range(n_points)],
        "centerline_y": [0.0] * n_points,
        "centerline_z": [0.0] * n_points,
    }
    for key in geometry_web.POINTWISE_ANALYSIS_ZERO_KEYS:
        profile.setdefault(key, [float(index + 1) for index in range(n_points)])
    return profile


def test_pointwise_endpoint_zeros_sync_to_sample_count_fraction():
    detected = geometry_web._pointwise_range_from_profile(
        _profile(leading_zero=20, trailing_zero=10))

    assert detected["start_fraction"] == 0.10
    assert detected["end_fraction"] == 0.95
    assert detected["leading_invalid_points"] == 20
    assert detected["trailing_invalid_points"] == 10


def test_endpoint_ranges_follow_each_profile_array_direction():
    pointwise = {
        "mpv": _profile(leading_zero=0, trailing_zero=16),
        "rpv": _profile(leading_zero=12, trailing_zero=0),
    }

    ranges = geometry_web._pointwise_analysis_ranges(pointwise)

    assert ranges["mpv"]["start_fraction"] == 0.0
    assert ranges["mpv"]["end_fraction"] == 0.92
    assert ranges["mpv"]["trailing_invalid_points"] == 16
    assert ranges["rpv"]["start_fraction"] == 0.06
    assert ranges["rpv"]["end_fraction"] == 1.0
    assert ranges["rpv"]["leading_invalid_points"] == 12


def test_manual_range_masks_pointwise_and_resamples_unified_to_200_points():
    profile = _profile()
    masked = geometry_web._mask_pointwise_profile(profile, 0.10, 0.90)

    app_root = str(geometry_web.APP_ROOT)
    if app_root not in sys.path:
        sys.path.insert(0, app_root)
    from extract_features import _clean_pointwise_profile_for_unified

    unified = _clean_pointwise_profile_for_unified(masked, target_n_points=200)

    assert sum(masked["section_valid"]) == 160
    assert masked["area"][:20] == [0.0] * 20
    assert masked["area"][180:] == [0.0] * 20
    assert len(unified["area"]) == 200
    assert unified["area"][0] == 21.0
    assert unified["area"][-1] == 180.0
    assert unified["section_valid"] == [1.0] * 200
    assert unified["_point_filter"]["dropped_invalid_n_points"] == 40


def test_manual_start_is_absolute_and_zeros_every_profile_feature_channel():
    profile = _profile(leading_zero=20)
    masked = geometry_web._mask_pointwise_profile(profile, 0.20, 1.0)

    assert sum(masked["section_valid"]) == 160
    for key in geometry_web.POINTWISE_ANALYSIS_ZERO_KEYS:
        assert masked[key][:40] == [0.0] * 40, key
        assert masked[key][40] != 0.0, key
    assert masked["position"] == profile["position"]
    assert masked["centerline_x"] == profile["centerline_x"]


def test_one_save_updates_multiple_vessels_and_rebuilds_unified(monkeypatch, tmp_path):
    patient = tmp_path / "patient"
    features = patient / "features"
    features.mkdir(parents=True)
    stl_path = patient / "vessel.stl"
    stl_path.write_text("solid vessel\nendsolid vessel\n", encoding="ascii")

    vessels = ("mpv", "rpv", "lpv")
    pointwise = {
        "_meta": {"n_points": 200},
        **{
            vessel: _profile(leading_zero=20)
            for vessel in vessels
        },
    }
    (features / geometry_web.POINTWISE_TEMP_NAME).write_text(
        json.dumps(pointwise), encoding="utf-8")
    (features / geometry_web.UNIFIED_FEATURES_NAME).write_text(
        json.dumps({"pointwise": {}}), encoding="utf-8")

    app_root = str(geometry_web.APP_ROOT)
    if app_root not in sys.path:
        sys.path.insert(0, app_root)
    import extract_features

    def rebuild_unified(stl, write_legacy=False):
        source_path = Path(stl).parent / "features" / geometry_web.POINTWISE_TEMP_NAME
        source = json.loads(source_path.read_text(encoding="utf-8"))
        rebuilt = {
            "pointwise": {
                vessel: extract_features._clean_pointwise_profile_for_unified(
                    source[vessel], target_n_points=200)
                for vessel in vessels
            }
        }
        output_path = Path(stl).parent / "features" / geometry_web.UNIFIED_FEATURES_NAME
        output_path.write_text(json.dumps(rebuilt), encoding="utf-8")
        return rebuilt

    monkeypatch.setattr(extract_features, "extract_all_features", rebuild_unified)

    result = geometry_web.save_analysis_ranges(stl_path, [
        {"vessel": "mpv", "start_fraction": 0.10, "end_fraction": 1.0},
        {"vessel": "rpv", "start_fraction": 0.20, "end_fraction": 1.0},
        {"vessel": "lpv", "start_fraction": 0.15, "end_fraction": 1.0},
    ])

    assert result["masked_points"] == {"mpv": 20, "rpv": 40, "lpv": 30}
    assert not (features / geometry_web.SEGMENT_ASSIGNMENTS_NAME).exists()
    saved_pointwise = json.loads(
        (features / geometry_web.POINTWISE_TEMP_NAME).read_text(encoding="utf-8"))
    assert saved_pointwise["rpv"]["area"][:40] == [0.0] * 40
    assert saved_pointwise["rpv"]["area"][40] == 41.0
    saved_unified = json.loads(
        (features / geometry_web.UNIFIED_FEATURES_NAME).read_text(encoding="utf-8"))
    for vessel in vessels:
        assert len(saved_unified["pointwise"][vessel]["area"]) == 200
    assert saved_unified["pointwise"]["rpv"]["area"][0] == 41.0
    assert saved_unified["pointwise"]["rpv"]["area"][-1] == 200.0
    assert geometry_web._pointwise_analysis_ranges(saved_pointwise) == result["ranges"]


def test_masked_pointwise_sections_start_at_manual_boundary(monkeypatch):
    masked = geometry_web._mask_pointwise_profile(
        _profile(leading_zero=20), 0.20, 1.0)
    monkeypatch.setattr(
        geometry_web,
        "_pointwise_surface_section_arrays",
        lambda *args, **kwargs: np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]),
    )

    layers = geometry_web._build_pointwise_layers(
        {"segments": {"rpv": {"path": [0, 1]}}},
        {0: {"x": 0.0, "y": 0.0, "z": 0.0}},
        {"rpv": masked},
        section_stride=10,
    )

    for layer_name in ("sampled_sections", "surface_sections"):
        positions = [
            value for value in layers[layer_name]["rpv"]["position"]
            if value is not None
        ]
        assert positions
        assert min(positions) >= 0.20
