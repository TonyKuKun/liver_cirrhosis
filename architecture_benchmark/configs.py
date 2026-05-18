"""Default experiment matrix for architecture benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class BenchmarkExperiment:
    name: str
    dataset_mode: str
    model_name: str
    question: str


DEFAULT_EXPERIMENTS: List[BenchmarkExperiment] = [
    BenchmarkExperiment("numeric_mlp", "numeric_only", "numeric_mlp", "Tabular deep baseline from extracted numeric features."),
    BenchmarkExperiment("numeric_cnn", "numeric_only", "numeric_cnn", "Local 1D profile patterns without branch graph."),
    BenchmarkExperiment("numeric_transformer", "numeric_only", "numeric_transformer", "Token attention over segment-point numeric profiles."),
    BenchmarkExperiment("numeric_gnn", "numeric_only", "numeric_gnn", "Branch-level graph using numeric summary nodes."),
    BenchmarkExperiment("numeric_cnn_gnn", "numeric_only", "numeric_cnn_gnn", "Local profile CNN plus portal topology GNN."),
    BenchmarkExperiment("stl_pointnet", "stl_only", "stl_pointnet", "Direct 3D surface point clouds from portal vein and organs."),
    BenchmarkExperiment("stl_centerline_gnn", "stl_only", "stl_centerline_gnn", "3D centerline graph from pointwise positions."),
    BenchmarkExperiment("stl_pointnet_centerline_gnn", "stl_only", "stl_pointnet_centerline_gnn", "Surface point cloud plus centerline graph."),
    BenchmarkExperiment("fusion_numeric_stl", "stl_numeric", "fusion_numeric_stl", "Numeric CNN-GNN fused with 3D STL encoders."),
]


def experiment_lookup():
    return {e.name: e for e in DEFAULT_EXPERIMENTS}


def select_experiments(names=None):
    if not names:
        return list(DEFAULT_EXPERIMENTS)
    lookup = experiment_lookup()
    selected = []
    for name in names:
        if name not in lookup:
            raise ValueError(f"Unknown experiment '{name}'. Valid: {', '.join(sorted(lookup))}")
        selected.append(lookup[name])
    return selected

