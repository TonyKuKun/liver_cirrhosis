import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from web.geometry_backend import (
    _apply_manual_segment_smoothing,
    _build_segments,
    _rebuild_smoothed_assignment_tree,
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


def test_manual_smoothing_rewrites_paths_to_the_new_tree():
    nodes = _nodes({
        0: (0.0, 0.0, 0.0),
        1: (5.0, 0.0, 0.0),
        2: (10.0, 0.0, 0.0),
        3: (7.0, 4.0, 0.0),
    })
    output = {
        "segments": {
            "mpv": {"path": [0, 1, 2]},
            "lpv": {"path": [1, 3]},
        }
    }

    _apply_manual_segment_smoothing(output, nodes)
    tree = _rebuild_smoothed_assignment_tree(output, nodes)

    tree_ids = {int(row[0]) for row in tree}
    mpv_path = output["segments"]["mpv"]["path"]
    lpv_path = output["segments"]["lpv"]["path"]
    assert len(mpv_path) > 3
    assert set(mpv_path) <= tree_ids
    assert set(lpv_path) <= tree_ids
    shared = set(mpv_path) & set(lpv_path)
    assert shared
    assert not shared & {mpv_path[0], mpv_path[-1]}
    assert shared & {lpv_path[0], lpv_path[-1]}
    assert output["manual_segment_smoothing"]["rewrote_centerline"] is True
