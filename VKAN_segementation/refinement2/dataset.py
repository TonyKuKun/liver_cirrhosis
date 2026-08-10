"""Datasets and shared preprocessing for CT-aware vessel refinement."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


CASE_EXCLUSION_MARKERS = ("$",)
DEFAULT_LABEL_NAMES = ("mask.nii.gz", "mask_label.nii.gz", "mask_smooth.nii.gz")


@dataclass(frozen=True)
class NiiCase:
    name: str
    path: Path
    orig_nii: Path
    pretrain_nii: Path
    label_nii: Path | None
    quality: str


@dataclass
class PreparedCase:
    name: str
    input: torch.Tensor
    crop: tuple[slice, slice, slice]
    affine: np.ndarray
    original_shape: tuple[int, int, int]
    label: torch.Tensor | None = None
    label_full: np.ndarray | None = None
    label_affine_matches: bool | None = None


def _contains_exclusion_marker(name: str) -> bool:
    return any(marker in name for marker in CASE_EXCLUSION_MARKERS)


def _pretrain_quality(path: Path) -> str:
    meta_path = path / "vkan_work" / "pretrain_meta.json"
    if not meta_path.exists():
        return "ok"
    try:
        return str(json.loads(meta_path.read_text(encoding="utf-8")).get("pretrain_quality", "ok"))
    except (OSError, json.JSONDecodeError):
        return "review"


def _resolve_label(path: Path, label_name: str) -> Path | None:
    names = DEFAULT_LABEL_NAMES if label_name == "auto" else (label_name,)
    for name in names:
        label = path / name
        if label.exists():
            return label
    return None


def _build_case(
    path: Path,
    orig_name: str,
    pretrain_name: str,
    label_name: str,
    require_label: bool,
) -> NiiCase | None:
    orig = path / orig_name
    pretrain = path / pretrain_name
    label = _resolve_label(path, label_name)
    if not orig.exists() or not pretrain.exists() or (require_label and label is None):
        return None
    return NiiCase(path.name, path, orig, pretrain, label, _pretrain_quality(path))


def discover_cases(
    data_root: str | Path,
    orig_name: str = "orig.nii.gz",
    pretrain_name: str = "pretrain.nii.gz",
    label_name: str = "mask.nii.gz",
) -> list[NiiCase]:
    """Discover valid supervised cases, excluding only $-marked folders."""
    root = Path(data_root)
    candidates = [root] if (root / orig_name).exists() else sorted(path for path in root.iterdir() if path.is_dir())
    cases: list[NiiCase] = []
    for path in candidates:
        if _contains_exclusion_marker(path.name):
            continue
        case = _build_case(path, orig_name, pretrain_name, label_name, require_label=True)
        if case is None:
            continue
        cases.append(case)
    return cases


def inference_case(
    patient_dir: str | Path,
    orig_name: str = "orig.nii.gz",
    pretrain_name: str = "pretrain.nii.gz",
    label_name: str = "mask.nii.gz",
) -> NiiCase:
    """Build a case for inference; a label is optional."""
    path = Path(patient_dir)
    case = _build_case(path, orig_name, pretrain_name, label_name, require_label=False)
    if case is None:
        raise FileNotFoundError(f"Need {orig_name} and {pretrain_name} in {path}")
    return case


def _load_nii(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import nibabel as nib

    image = nib.load(str(path))
    return np.asarray(image.dataobj, dtype=np.float32), np.asarray(image.affine, dtype=np.float64)


def _resample_to_reference(
    volume: np.ndarray,
    affine: np.ndarray,
    reference_shape: tuple[int, int, int],
    reference_affine: np.ndarray,
    order: int,
) -> np.ndarray:
    try:
        import nibabel as nib
        from nibabel.processing import resample_from_to
    except ImportError as exc:
        raise ImportError("nibabel and scipy are required to align NIfTI volumes.") from exc

    source = nib.Nifti1Image(np.asarray(volume, dtype=np.float32), affine)
    target = (reference_shape, np.asarray(reference_affine, dtype=np.float64))
    aligned = resample_from_to(source, target, order=order)
    return np.asarray(aligned.dataobj, dtype=np.float32)


def _align_to_reference(
    volume: np.ndarray,
    affine: np.ndarray,
    reference_shape: tuple[int, int, int],
    reference_affine: np.ndarray,
    order: int,
) -> tuple[np.ndarray, bool]:
    matches = bool(tuple(volume.shape) == reference_shape and np.allclose(affine, reference_affine))
    if matches:
        return volume, True
    return _resample_to_reference(volume, affine, reference_shape, reference_affine, order), False


def _foreground_bbox(mask: np.ndarray, margin: int) -> tuple[slice, slice, slice]:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        return tuple(slice(0, size) for size in mask.shape)  # type: ignore[return-value]
    lower = np.maximum(coordinates.min(axis=0) - int(margin), 0)
    upper = np.minimum(coordinates.max(axis=0) + int(margin) + 1, np.asarray(mask.shape))
    return tuple(slice(int(start), int(stop)) for start, stop in zip(lower, upper))  # type: ignore[return-value]


def _resize_volume(volume: np.ndarray, grid_size: int, mode: str) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(volume, dtype=np.float32))[None, None]
    kwargs: dict[str, object] = {"mode": mode}
    if mode == "trilinear":
        kwargs["align_corners"] = False
    return F.interpolate(tensor, size=(grid_size, grid_size, grid_size), **kwargs)[0, 0]


def normalize_hu(volume_hu: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    """Clip a CT window and map it to [0, 1] without changing spatial shape."""
    if hu_max <= hu_min:
        raise ValueError("hu_max must be greater than hu_min")
    clipped = np.clip(np.asarray(volume_hu, dtype=np.float32), hu_min, hu_max)
    return (clipped - float(hu_min)) / float(hu_max - hu_min)


def prepare_case(
    case: NiiCase,
    grid_size: int,
    roi_margin: int,
    hu_min: float,
    hu_max: float,
) -> PreparedCase:
    """Build the same two-channel input for both training and inference."""
    orig, affine = _load_nii(case.orig_nii)
    reference_shape = tuple(int(size) for size in orig.shape)
    pretrain, pretrain_affine = _load_nii(case.pretrain_nii)
    pretrain, _ = _align_to_reference(pretrain, pretrain_affine, reference_shape, affine, order=0)
    pretrain_mask = pretrain > 0.5
    crop = _foreground_bbox(pretrain_mask, roi_margin)

    ct_crop = normalize_hu(orig[crop], hu_min, hu_max)
    pretrain_crop = pretrain_mask[crop].astype(np.float32)
    model_input = torch.stack(
        [
            _resize_volume(ct_crop, grid_size, "trilinear"),
            _resize_volume(pretrain_crop, grid_size, "nearest"),
        ],
        dim=0,
    )
    prepared = PreparedCase(case.name, model_input, crop, affine, reference_shape)

    if case.label_nii is None:
        return prepared

    label, label_affine = _load_nii(case.label_nii)
    label, label_matches = _align_to_reference(label, label_affine, reference_shape, affine, order=0)
    label_full = label > 0.5
    prepared.label = _resize_volume(label_full[crop].astype(np.float32), grid_size, "nearest")[None]
    prepared.label_full = label_full
    prepared.label_affine_matches = label_matches
    return prepared


class CTHUVesselDataset(Dataset):
    """Supervised vessel dataset with normalized CT and coarse-mask channels."""

    def __init__(
        self,
        data_root: str | Path,
        grid_size: int = 160,
        roi_margin: int = 32,
        hu_min: float = -100.0,
        hu_max: float = 600.0,
        orig_name: str = "orig.nii.gz",
        pretrain_name: str = "pretrain.nii.gz",
        label_name: str = "mask.nii.gz",
        cache_dir: str | Path | None = None,
    ) -> None:
        self.grid_size = int(grid_size)
        self.roi_margin = int(roi_margin)
        self.hu_min = float(hu_min)
        self.hu_max = float(hu_max)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cases = discover_cases(
            data_root,
            orig_name=orig_name,
            pretrain_name=pretrain_name,
            label_name=label_name,
        )
        if not self.cases:
            raise RuntimeError("No usable CT/pretrain/label cases were found.")

    def __len__(self) -> int:
        return len(self.cases)

    def _cache_path(self, case: NiiCase) -> Path:
        digest = hashlib.sha256(case.name.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{digest}.pt"  # type: ignore[operator]

    def _cache_signature(self, case: NiiCase) -> dict[str, int | float]:
        if case.label_nii is None:
            raise RuntimeError(f"Missing label for supervised case {case.name}")
        return {
            "grid_size": self.grid_size,
            "roi_margin": self.roi_margin,
            "hu_min": self.hu_min,
            "hu_max": self.hu_max,
            "orig_mtime_ns": case.orig_nii.stat().st_mtime_ns,
            "pretrain_mtime_ns": case.pretrain_nii.stat().st_mtime_ns,
            "label_mtime_ns": case.label_nii.stat().st_mtime_ns,
            "orig_size": case.orig_nii.stat().st_size,
            "pretrain_size": case.pretrain_nii.stat().st_size,
            "label_size": case.label_nii.stat().st_size,
        }

    @staticmethod
    def _to_item(prepared: PreparedCase) -> dict[str, object]:
        if prepared.label is None:
            raise RuntimeError(f"Missing label for supervised case {prepared.name}")
        return {
            "name": prepared.name,
            "input": prepared.input,
            "label": prepared.label,
            "crop_slices": torch.tensor(
                [[item.start or 0, item.stop or 0] for item in prepared.crop], dtype=torch.int64
            ),
            "original_shape": torch.tensor(prepared.original_shape, dtype=torch.int64),
            "label_affine_matches": torch.tensor(bool(prepared.label_affine_matches)),
        }

    def __getitem__(self, index: int) -> dict[str, object]:
        case = self.cases[index]
        signature = self._cache_signature(case)
        cache_path = self._cache_path(case) if self.cache_dir is not None else None
        if cache_path is not None and cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            if cached.get("signature") == signature:
                return cached["item"]

        prepared = prepare_case(case, self.grid_size, self.roi_margin, self.hu_min, self.hu_max)
        item = self._to_item(prepared)
        if cache_path is not None:
            temp_path = cache_path.with_suffix(".tmp")
            torch.save({"signature": signature, "item": item}, temp_path)
            temp_path.replace(cache_path)
        return item

    def build_cache(self) -> None:
        """Materialize all preprocessed tensors before epoch timing begins."""
        if self.cache_dir is None:
            return
        for index in range(len(self)):
            self[index]


def collate_fn(items: list[dict[str, object]]) -> dict[str, object]:
    return {
        "name": [str(item["name"]) for item in items],
        "input": torch.stack([item["input"] for item in items]),  # type: ignore[list-item]
        "label": torch.stack([item["label"] for item in items]),  # type: ignore[list-item]
        "crop_slices": torch.stack([item["crop_slices"] for item in items]),  # type: ignore[list-item]
        "original_shape": torch.stack([item["original_shape"] for item in items]),  # type: ignore[list-item]
        "label_affine_matches": torch.stack([item["label_affine_matches"] for item in items]),  # type: ignore[list-item]
    }
