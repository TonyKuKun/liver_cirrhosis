from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .config import discover_patients, require_existing_labels
    from .mesh_ops import stl_to_voxels
except ImportError:
    from config import discover_patients, require_existing_labels
    from mesh_ops import stl_to_voxels


class VesselSTLDataset(Dataset):
    """Pairs coarse pretrain STL with manually extracted vessel STL labels."""

    def __init__(self, data_root: str | Path, grid_size: int = 96, require_pretrain: bool = True) -> None:
        self.data_root = Path(data_root)
        self.grid_size = int(grid_size)
        cases = require_existing_labels(discover_patients(self.data_root))
        if require_pretrain:
            cases = [case for case in cases if case.pretrain_stl.exists()]
        self.cases = cases
        if not self.cases:
            raise RuntimeError("No usable patient cases found. Need pretrain.stl and vessel.stl.")

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> dict:
        case = self.cases[idx]
        pre, bounds = stl_to_voxels(case.pretrain_stl, grid_size=self.grid_size)
        label, _ = stl_to_voxels(case.label_stl, grid_size=self.grid_size, bounds=bounds)
        x = torch.from_numpy(pre[None]).float()
        y = torch.from_numpy(label[None]).float()
        return {
            "name": case.name,
            "input": x,
            "label": y,
            "bounds": torch.from_numpy(bounds).float(),
            "is_post_tips": torch.tensor(float(case.is_post_tips), dtype=torch.float32),
        }


def collate_fn(items: list[dict]) -> dict:
    return {
        "name": [item["name"] for item in items],
        "input": torch.stack([item["input"] for item in items]),
        "label": torch.stack([item["label"] for item in items]),
        "bounds": torch.stack([item["bounds"] for item in items]),
        "is_post_tips": torch.stack([item["is_post_tips"] for item in items]),
    }

