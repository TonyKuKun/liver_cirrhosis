import numpy as np

from extract_profiles import (
    _build_clinical_junction_plan,
    _interpolate_profile_intervals,
    _trim_coords_by_distance,
)


def _nodes(coords):
    return {
        idx: {"x": float(x), "y": float(y), "z": float(z)}
        for idx, (x, y, z) in coords.items()
    }


def test_endpoint_tributary_is_trimmed_by_receiving_vessel_radius():
    nodes = _nodes({
        0: (0, 0, 0),
        1: (10, 0, 0),
        2: (-10, 0, 0),
        3: (20, 0, 0),
    })
    seg_data = {
        "segments": {
            "mpv": {"path": [0, 1]},
            "sv": {"path": [0, 2]},
            "lpv": {"path": [1, 3]},
        }
    }
    coords = {
        "mpv": np.array([[0, 0, 0], [10, 0, 0]], dtype=float),
        "sv": np.array([[0, 0, 0], [-10, 0, 0]], dtype=float),
        "lpv": np.array([[10, 0, 0], [20, 0, 0]], dtype=float),
    }

    plan = _build_clinical_junction_plan(
        seg_data, nodes, coords, {"mpv": 4.0, "sv": 2.0, "lpv": 2.0},
        endpoint_factor=1.25,
        side_branch_factor=1.25,
    )
    trimmed = _trim_coords_by_distance(
        coords["sv"], plan["sv"]["trim_start_mm"], plan["sv"]["trim_end_mm"])

    assert plan["sv"]["trim_start_mm"] == 5.0
    assert plan["mpv"]["trim_start_mm"] == 0.0
    assert plan["mpv"]["trim_end_mm"] == 0.0
    assert np.allclose(trimmed[0], [-5.0, 0.0, 0.0])
    assert np.allclose(trimmed[-1], [-10.0, 0.0, 0.0])


def test_internal_side_branch_interval_is_interpolated_on_main_vessel():
    profile = {
        "position": [0, 0.25, 0.5, 0.75, 1.0],
        "arc_length_mm": [0, 5, 10, 15, 20],
        "area": [10, 11, 40, 13, 14],
        "perimeter": [5, 5.5, 20, 6.5, 7],
        "eq_diameter": [3, 3.1, 8, 3.3, 3.4],
        "n_components": [1, 1, 2, 1, 1],
    }

    out = _interpolate_profile_intervals(profile, [(8, 12)])

    assert out["area"][2] == 12.0
    assert out["junction_replaced"] == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert out["n_junction_replaced"] == 1
