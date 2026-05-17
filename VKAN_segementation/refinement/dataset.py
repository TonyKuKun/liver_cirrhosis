from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from ..utils.common import INVALID_MARKERS, discover_patients, stl_to_voxels
except (ImportError, ValueError):
    try:
        from VKAN_segementation.utils.common import INVALID_MARKERS, discover_patients, stl_to_voxels
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from utils.common import INVALID_MARKERS, discover_patients, stl_to_voxels


DEFAULT_LABEL_NAMES = ("mask_label.nii.gz", "mask_smooth.nii.gz")


@dataclass(frozen=True)
class NiiCase:
    name: str
    path: Path
    pretrain_nii: Path
    label_nii: Path
    predict_mask_nii: Path
    predict_stl: Path
    is_post_tips: bool


def _load_nii(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import nibabel as nib

    img = nib.load(str(path))
    return np.asarray(img.dataobj, dtype=np.float32), img.affine.copy()


def _resize_volume(volume: np.ndarray, size: int, mode: str) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(volume, dtype=np.float32))[None, None]
    kwargs = {"mode": mode}
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        kwargs["align_corners"] = False
    return F.interpolate(x, size=(size, size, size), **kwargs)[0, 0]


def _resample_to_shape(volume: np.ndarray, shape: tuple[int, int, int], order: int) -> np.ndarray:
    if tuple(volume.shape) == tuple(shape):
        return volume
    try:
        from scipy import ndimage as ndi
    except ImportError as exc:
        raise ImportError("scipy is required when NIfTI shapes differ.") from exc
    zoom = np.asarray(shape, dtype=np.float64) / np.asarray(volume.shape, dtype=np.float64)
    return ndi.zoom(volume, zoom, order=order)


def _foreground_bbox(mask: np.ndarray, margin: int) -> tuple[slice, slice, slice]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return tuple(slice(0, s) for s in mask.shape)  # type: ignore[return-value]
    lo = np.maximum(coords.min(axis=0) - int(margin), 0)
    hi = np.minimum(coords.max(axis=0) + int(margin) + 1, np.asarray(mask.shape))
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))  # type: ignore[return-value]


def _crop_slices_to_tensor(crop: tuple[slice, slice, slice]) -> torch.Tensor:
    return torch.tensor([[s.start or 0, s.stop or 0] for s in crop], dtype=torch.int64)


def _resolve_label(path: Path, label_name: str) -> Path | None:
    if label_name != "auto":
        label = path / label_name
        return label if label.exists() else None
    for name in DEFAULT_LABEL_NAMES:
        label = path / name
        if label.exists():
            return label
    return None


def discover_nii_cases(
    root: str | Path,
    pretrain_name: str = "pretrain.nii.gz",
    label_name: str = "auto",
    include_invalid: bool = False,
) -> list[NiiCase]:
    root = Path(root)
    patient_dirs = [root] if (root / pretrain_name).exists() else sorted(p for p in root.iterdir() if p.is_dir())
    cases: list[NiiCase] = []
    for path in patient_dirs:
        if "@" in path.name:
            continue
        if not include_invalid and any(marker in path.name for marker in INVALID_MARKERS):
            continue
        pretrain = path / pretrain_name
        label = _resolve_label(path, label_name)
        if not pretrain.exists() or label is None:
            continue
        cases.append(
            NiiCase(
                name=path.name,
                path=path,
                pretrain_nii=pretrain,
                label_nii=label,
                predict_mask_nii=path / "predict_mask.nii.gz",
                predict_stl=path / "predict.stl",
                is_post_tips="#" in path.name,
            )
        )
    return cases


class VesselNiiDataset(Dataset):
    """Pairs pretrain NIfTI masks with manual NIfTI labels using cropped ROIs."""

    def __init__(
        self,
        data_root: str | Path,
        grid_size: int = 96,
        pretrain_name: str = "pretrain.nii.gz",
        label_name: str = "mask.nii.gz",
        label_threshold: float = 0.5,
        roi_margin: int = 16,
        crop_source: str = "union",
        include_invalid: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.grid_size = int(grid_size)
        self.label_threshold = float(label_threshold)
        self.roi_margin = int(roi_margin)
        self.crop_source = crop_source
        self.cases = discover_nii_cases(
            self.data_root,
            pretrain_name=pretrain_name,
            label_name=label_name,
            include_invalid=include_invalid,
        )
        if not self.cases:
            raise RuntimeError("No usable NIfTI cases found. Need pretrain.nii.gz and a label NIfTI.")

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> dict:
        case = self.cases[idx]
        pre, affine = _load_nii(case.pretrain_nii)
        label, label_affine = _load_nii(case.label_nii)
        if label.shape != pre.shape:
            label = _resample_to_shape(label, tuple(pre.shape), order=0)
        pre_mask = pre > 0.5
        label_mask = label > self.label_threshold
        if self.crop_source == "pretrain":
            crop_mask = pre_mask
        elif self.crop_source == "label":
            crop_mask = label_mask
        else:
            crop_mask = pre_mask | label_mask
        crop = _foreground_bbox(crop_mask, self.roi_margin)
        pre_crop = pre_mask[crop].astype(np.float32)
        label_crop = label_mask[crop].astype(np.float32)
        return {
            "name": case.name,
            "input": _resize_volume(pre_crop, self.grid_size, "trilinear")[None],
            "label": _resize_volume(label_crop, self.grid_size, "nearest")[None],
            "crop_slices": _crop_slices_to_tensor(crop),
            "original_shape": torch.tensor(pre.shape, dtype=torch.int64),
            "affine": torch.from_numpy(affine).float(),
            "label_affine_matches": torch.tensor(bool(np.allclose(affine, label_affine))),
            "is_post_tips": torch.tensor(float(case.is_post_tips), dtype=torch.float32),
        }


class VesselSTLDataset(Dataset):
    """Pairs coarse pretrain STL candidates with manual vessel STL labels."""

    def __init__(self, data_root: str | Path, grid_size: int = 96, require_pretrain: bool = True, include_review: bool = False) -> None:
        self.data_root = Path(data_root)
        self.grid_size = int(grid_size)
        cases = discover_patients(self.data_root)
        if require_pretrain:
            cases = [case for case in cases if case.pretrain_stl.exists()]
        cases = [case for case in cases if case.label_stl.exists()]
        if not include_review:
            cases = [case for case in cases if _pretrain_quality(case) != "review"]
        self.cases = cases
        if not self.cases:
            raise RuntimeError("No usable patient cases found. Need pretrain.stl and vessel.stl.")

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> dict:
        case = self.cases[idx]
        pre, bounds = stl_to_voxels(case.pretrain_stl, grid_size=self.grid_size)
        label, _ = stl_to_voxels(case.label_stl, grid_size=self.grid_size, bounds=bounds)
        return {
            "name": case.name,
            "input": torch.from_numpy(pre[None]).float(),
            "label": torch.from_numpy(label[None]).float(),
            "bounds": torch.from_numpy(bounds).float(),
            "is_post_tips": torch.tensor(float(case.is_post_tips), dtype=torch.float32),
        }


def _pretrain_quality(case) -> str:
    meta_path = case.path / "vkan_work" / "pretrain_meta.json"
    if not meta_path.exists():
        return "ok"
    try:
        return str(json.loads(meta_path.read_text(encoding="utf-8")).get("pretrain_quality", "ok"))
    except Exception:
        return "ok"


def collate_fn(items: list[dict]) -> dict:
    batch = {
        "name": [item["name"] for item in items],
        "input": torch.stack([item["input"] for item in items]),
        "label": torch.stack([item["label"] for item in items]),
        "is_post_tips": torch.stack([item["is_post_tips"] for item in items]),
    }
    for key in ("bounds", "crop_slices", "original_shape", "affine", "label_affine_matches"):
        if key in items[0]:
            batch[key] = torch.stack([item[key] for item in items])
    return batch
