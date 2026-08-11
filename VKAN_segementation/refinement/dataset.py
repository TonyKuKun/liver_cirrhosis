from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from ..utils.common import discover_patients, stl_to_voxels
except (ImportError, ValueError):
    try:
        from VKAN_segementation.utils.common import discover_patients, stl_to_voxels
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from utils.common import discover_patients, stl_to_voxels


DEFAULT_LABEL_NAMES = ("mask_label.nii.gz", "mask_smooth.nii.gz")


@dataclass(frozen=True)
class NiiCase:
    name: str
    path: Path
    pretrain_nii: Path
    pretrain_stl: Path
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


def _resample_nii_to_reference(
    volume: np.ndarray,
    affine: np.ndarray,
    reference_shape: tuple[int, int, int],
    reference_affine: np.ndarray,
    order: int,
) -> np.ndarray:
    if tuple(volume.shape) == tuple(reference_shape) and np.allclose(affine, reference_affine):
        return volume
    try:
        import nibabel as nib
        from nibabel.processing import resample_from_to
    except ImportError as exc:
        raise ImportError("nibabel and scipy are required when NIfTI spaces differ.") from exc
    source = nib.Nifti1Image(np.asarray(volume, dtype=np.float32), affine)
    target = (tuple(int(s) for s in reference_shape), np.asarray(reference_affine, dtype=np.float64))
    aligned = resample_from_to(source, target, order=order)
    return np.asarray(aligned.dataobj, dtype=np.float32)


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
    pretrain_stl_name: str = "pretrain.stl",
    label_name: str = "auto",
    require_pretrain_stl: bool = False,
    include_invalid: bool = False,
) -> list[NiiCase]:
    root = Path(root)
    patient_dirs = [root] if (root / pretrain_name).exists() else sorted(p for p in root.iterdir() if p.is_dir())
    cases: list[NiiCase] = []
    for path in patient_dirs:
        if "$" in path.name:
            continue
        pretrain = path / pretrain_name
        pretrain_stl = path / pretrain_stl_name
        label = _resolve_label(path, label_name)
        if require_pretrain_stl and not pretrain_stl.exists():
            continue
        if not pretrain.exists() or label is None:
            continue
        cases.append(
            NiiCase(
                name=path.name,
                path=path,
                pretrain_nii=pretrain,
                pretrain_stl=pretrain_stl,
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
        pretrain_stl_name: str = "pretrain.stl",
        label_name: str = "mask.nii.gz",
        label_threshold: float = 0.5,
        roi_margin: int = 16,
        crop_source: str = "union",
        require_pretrain_stl: bool = False,
        include_invalid: bool = False,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.grid_size = int(grid_size)
        self.label_threshold = float(label_threshold)
        self.roi_margin = int(roi_margin)
        self.crop_source = crop_source
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cases = discover_nii_cases(
            self.data_root,
            pretrain_name=pretrain_name,
            pretrain_stl_name=pretrain_stl_name,
            label_name=label_name,
            require_pretrain_stl=require_pretrain_stl,
            include_invalid=include_invalid,
        )
        if not self.cases:
            raise RuntimeError("No usable NIfTI cases found. Need pretrain.nii.gz and a label NIfTI.")

    def __len__(self) -> int:
        return len(self.cases)

    def _cache_path(self, case: NiiCase) -> Path:
        """Use a digest so patient names cannot create unsafe or colliding filenames."""
        identity = f"{case.path.resolve()}\0{case.name}".encode("utf-8")
        return self.cache_dir / f"{hashlib.sha256(identity).hexdigest()[:20]}.pt"  # type: ignore[operator]

    @staticmethod
    def _file_signature(path: Path) -> dict[str, int | str]:
        stat = path.stat()
        return {"path": str(path.resolve()), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}

    def _cache_signature(self, case: NiiCase) -> dict[str, object]:
        return {
            "version": 1,
            "case": case.name,
            "pretrain": self._file_signature(case.pretrain_nii),
            "label": self._file_signature(case.label_nii),
            "grid_size": self.grid_size,
            "label_threshold": self.label_threshold,
            "roi_margin": self.roi_margin,
            "crop_source": self.crop_source,
        }

    def _build_item(self, case: NiiCase) -> dict:
        pre, affine = _load_nii(case.pretrain_nii)
        label, label_affine = _load_nii(case.label_nii)
        label_space_matches = bool(tuple(label.shape) == tuple(pre.shape) and np.allclose(affine, label_affine))
        if not label_space_matches:
            label = _resample_nii_to_reference(label, label_affine, tuple(pre.shape), affine, order=0)
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
            "label_affine_matches": torch.tensor(label_space_matches),
            "label_resampled_to_pretrain": torch.tensor(not label_space_matches),
            "is_post_tips": torch.tensor(float(case.is_post_tips), dtype=torch.float32),
        }

    def _save_cached_item(self, case: NiiCase, item: dict) -> None:
        if self.cache_dir is None:
            return
        cache_path = self._cache_path(case)
        temp_path = cache_path.with_name(f"{cache_path.name}.tmp")
        torch.save({"signature": self._cache_signature(case), "item": item}, temp_path)
        temp_path.replace(cache_path)

    def cache_case(self, idx: int, force: bool = False) -> bool:
        """Materialize one case and return whether a cache file was written."""
        if self.cache_dir is None:
            return False
        case = self.cases[idx]
        cache_path = self._cache_path(case)
        signature = self._cache_signature(case)
        if not force and cache_path.exists():
            try:
                cached = torch.load(cache_path, map_location="cpu", weights_only=False)
                if (
                    isinstance(cached, dict)
                    and cached.get("signature") == signature
                    and isinstance(cached.get("item"), dict)
                ):
                    return False
            except (OSError, RuntimeError, EOFError, ValueError, KeyError, AttributeError, TypeError):
                pass
        self._save_cached_item(case, self._build_item(case))
        return True

    def build_cache(self, force: bool = False) -> int:
        """Materialize all cases before training starts; return number of files written."""
        if self.cache_dir is None:
            return 0
        return sum(self.cache_case(index, force=force) for index in range(len(self)))

    def __getitem__(self, idx: int) -> dict:
        case = self.cases[idx]
        if self.cache_dir is not None:
            cache_path = self._cache_path(case)
            signature = self._cache_signature(case)
            if cache_path.exists():
                try:
                    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
                    if (
                        isinstance(cached, dict)
                        and cached.get("signature") == signature
                        and isinstance(cached.get("item"), dict)
                    ):
                        return cached["item"]
                except (OSError, RuntimeError, EOFError, ValueError, KeyError, AttributeError, TypeError):
                    pass
            item = self._build_item(case)
            self._save_cached_item(case, item)
            return item
        return self._build_item(case)


class VesselSTLDataset(Dataset):
    """Pairs coarse pretrain STL candidates with manual vessel STL labels."""

    def __init__(self, data_root: str | Path, grid_size: int = 96, require_pretrain: bool = True, include_review: bool = False) -> None:
        self.data_root = Path(data_root)
        self.grid_size = int(grid_size)
        cases = discover_patients(self.data_root)
        cases = [case for case in cases if "$" not in case.name]
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
    for key in ("bounds", "crop_slices", "original_shape", "affine", "label_affine_matches", "label_resampled_to_pretrain"):
        if key in items[0]:
            batch[key] = torch.stack([item[key] for item in items])
    return batch
