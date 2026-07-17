import numpy as np

from web_frontend import (
    _apply_manual_segment_smoothing,
    _build_segments,
)


def _nodes(coords):
    return {
        idx: {
            "id": idx,
            "x": float(coord[0]),
            "y": float(coord[1]),
            "z": float(coord[2]),
            "parent": -1,
            "left": -1,
            "right": -1,
        }
        for idx, coord in coords.items()
    }


def test_manual_segment_smoothing_uses_whole_assigned_vessel():
    nodes = _nodes({
        0: (0.0, 0.0, 0.0),
        1: (5.0, 0.0, 0.0),
        2: (9.0, 4.0, 0.0),
        3: (14.0, 5.0, 0.0),
    })
    output = {
        "segments": {
            "mpv": {
                "path": [0, 1, 2, 3],
                "n_points": 4,
                "length_mm": 0.0,
                "tortuosity": 0.0,
                "mean_curvature": 0.0,
            }
        }
    }

    smoothed = _apply_manual_segment_smoothing(output, nodes)
    info = output["segments"]["mpv"]
    coords = np.asarray(info["smoothed_coords"], dtype=float)

    assert smoothed == ["mpv"]
    assert len(coords) > len(info["path"])
    assert np.allclose(coords[0], [0.0, 0.0, 0.0])
    assert np.allclose(coords[-1], [14.0, 5.0, 0.0])
    assert info["topology_n_points"] == 4
    assert info["n_points"] == len(coords)
    assert info["smoothing"]["method"] == "whole_anatomical_segment_spline"


def test_workbench_segments_prefer_smoothed_coords():
    nodes = _nodes({
        0: (0.0, 0.0, 0.0),
        1: (10.0, 0.0, 0.0),
    })
    seg_data = {
        "segments": {
            "mpv": {
                "path": [0, 1],
                "smoothed_coords": [
                    [0.0, 0.0, 0.0],
                    [4.0, 1.0, 0.0],
                    [10.0, 0.0, 0.0],
                ],
                "length_mm": 10.5,
                "tortuosity": 0.1,
                "mean_curvature": 0.01,
            }
        }
    }

    segments = _build_segments(seg_data, nodes)

    assert segments["mpv"]["x"] == [0.0, 4.0, 10.0]
    assert segments["mpv"]["y"] == [0.0, 1.0, 0.0]
    assert segments["mpv"]["n_points"] == 3
