from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

try:
    from ..pretrain.preprocess import mask_label_nifti_path, pretrain_nifti_path
    from ..utils.common import discover_patients, load_nifti_volume, resize_mask_to_grid, volume_bounds_xyz
except ImportError:
    try:
        from VKAN_segementation.pretrain.preprocess import mask_label_nifti_path, pretrain_nifti_path
        from VKAN_segementation.utils.common import discover_patients, load_nifti_volume, resize_mask_to_grid, volume_bounds_xyz
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from pretrain.preprocess import mask_label_nifti_path, pretrain_nifti_path
        from utils.common import discover_patients, load_nifti_volume, resize_mask_to_grid, volume_bounds_xyz


class VesselNiftiDataset(Dataset):
    """Pairs coarse pretrain NIfTI masks with manual mask NIfTI labels."""

    def __init__(self, data_root: str | Path, grid_size: int = 96, require_pretrain: bool = True, include_review: bool = False) -> None:
        self.data_root = Path(data_root)
        self.grid_size = int(grid_size)
        cases = discover_patients(self.data_root)
        if require_pretrain:
            cases = [case for case in cases if pretrain_nifti_path(case).exists()]
        cases = [case for case in cases if mask_label_nifti_path(case).exists()]
        if not include_review:
            cases = [case for case in cases if _pretrain_quality(case) != "review"]
        self.cases = cases
        if not self.cases:
            raise RuntimeError("No usable patient cases found. Need pretrain.nii.gz and mask.nii.gz.")

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> dict:
        case = self.cases[idx]
        pre_vol = load_nifti_volume(pretrain_nifti_path(case))
        label_vol = load_nifti_volume(mask_label_nifti_path(case))
        pre = resize_mask_to_grid(pre_vol.volume_hu, self.grid_size)
        label = resize_mask_to_grid(label_vol.volume_hu, self.grid_size)
        bounds = volume_bounds_xyz(pre_vol.volume_hu.shape, pre_vol.spacing_zyx, pre_vol.origin_xyz)
        return {
            "name": case.name,
            "input": torch.from_numpy(pre[None]).float(),
            "label": torch.from_numpy(label[None]).float(),
            "bounds": torch.from_numpy(bounds).float(),
            "is_post_tips": torch.tensor(float(case.is_post_tips), dtype=torch.float32),
        }


VesselSTLDataset = VesselNiftiDataset


def _pretrain_quality(case) -> str:
    meta_path = case.path / "vkan_work" / "pretrain_meta.json"
    if not meta_path.exists():
        return "ok"
    try:
        return str(json.loads(meta_path.read_text(encoding="utf-8")).get("pretrain_quality", "ok"))
    except Exception:
        return "ok"


def collate_fn(items: list[dict]) -> dict:
    return {
        "name": [item["name"] for item in items],
        "input": torch.stack([item["input"] for item in items]),
        "label": torch.stack([item["label"] for item in items]),
        "bounds": torch.stack([item["bounds"] for item in items]),
        "is_post_tips": torch.stack([item["is_post_tips"] for item in items]),
    }
