"""pretrain v5 — 基于 TotalSegmentator 的门静脉预分割。

流程（极简，确定性初始分割）：
    1. 加载 orig.nii.gz
    2. 从 bone 分割定位 Z 轴（膈肌到髂骨）
    3. 从 portal_vein 分割采样 HU → 得到该患者的精确阈值（±5 HU 边距）
    4. HU 阈值分割（在 Z 轴范围内）
    5. 减去 bone / spleen / liver / kidney / IVC / aorta
    6. 从 portal_vein 质心做区域生长 → pretrain.stl

依赖：
    patient/segmentation/ 下的 TotalSegmentator 结果（由 totalseg_integration.py 生成）
    patient/orig.nii.gz
"""
from __future__ import annotations

import argparse
import gc
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from scipy import ndimage as ndi
except ImportError:
    ndi = None

try:
    from ..utils.common import DicomVolume, discover_patients, stl_to_voxels, zyx_mask_to_stl
except (ImportError, ValueError):
    try:
        from VKAN_segementation.utils.common import DicomVolume, discover_patients, stl_to_voxels, zyx_mask_to_stl
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from utils.common import DicomVolume, discover_patients, stl_to_voxels, zyx_mask_to_stl

try:
    from .totalseg_integration import (
        BONE_LABELS, load_organ_mask, get_exclusion_mask, get_portal_seed,
        get_portal_vein_mask, get_liver_mask, get_z_range_from_bone,
    )
except ImportError:
    try:
        from totalseg import (
            BONE_LABELS, load_organ_mask, get_exclusion_mask, get_portal_seed,
            get_portal_vein_mask, get_liver_mask, get_z_range_from_bone,
        )
    except ImportError:
        try:
            from VKAN_segementation.pretrain.totalseg import (
                BONE_LABELS, load_organ_mask, get_exclusion_mask, get_portal_seed,
                get_portal_vein_mask, get_liver_mask, get_z_range_from_bone,
            )
        except ImportError:
            BONE_LABELS = {}  # type: ignore
            load_organ_mask = None  # type: ignore
            get_exclusion_mask = None  # type: ignore
            get_portal_seed = None  # type: ignore
            get_portal_vein_mask = None  # type: ignore
            get_liver_mask = None  # type: ignore
            get_z_range_from_bone = None  # type: ignore


PRETRAIN_ALGORITHM_VERSION = "2026-08-11-v24-supported-opening-restore8"
PRETRAIN_META_NAME = "pretrain_meta.json"
PRETRAIN_NII_NAME = "pretrain.nii.gz"
MAX_STL_BYTES = 20_000 * 1024
TARGET_VOXELS = 420_000
TARGET_VOXELS_TIPS = 330_000
REGION_GROW_BRIDGE_MM = 8.0
REGION_GROW_MAX_SEED_SNAP_MM = 30.0
PORTAL_REFERENCE_CLEANUP_RADIUS_MM = 25.0
PORTAL_REFERENCE_CLEANUP_RADIUS_TIPS_MM = 60.0
PORTAL_REFERENCE_CLEANUP_SEED_DILATE = 2
PORTAL_REFERENCE_MIN_P50_HU = 100.0
LIVER_SPLEEN_FALLBACK_MIN_HU = 150.0
LIVER_SPLEEN_FALLBACK_MAX_HU = 260.0
LIVER_SPLEEN_FALLBACK_EDGE_MARGIN_HU = 10.0
LIVER_SPLEEN_FALLBACK_EDGE_HIGH_MARGIN_HU = 20.0
PORTAL_BOUNDARY_P50_MARGIN_HU = 15.0
TIPS_LUMEN_FILL_RADIUS_MM = 5.0
TIPS_LUMEN_FILL_BIN_MM = 2.0
HU_MARGIN = 5.0  # 门静脉 HU 采样后上下各扩展的边距
HU_LOW_FLOOR = 75.0
OPENING_RESTORE_MIN_CORE_NEIGHBORS = 8
DEFAULT_HU_HIGH_CAP = 600.0
TIPS_HU_HIGH_CAP = 3071.0

DEFAULT_BONE_NAMES = [
    "vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2", "vertebrae_L1",
    "vertebrae_T12", "vertebrae_T11", "vertebrae_T10", "vertebrae_T9", "vertebrae_T8",
    "vertebrae_T7", "vertebrae_T6", "vertebrae_T5", "vertebrae_T4", "vertebrae_T3",
    "vertebrae_T2", "vertebrae_T1",
    "rib_left_1", "rib_left_2", "rib_left_3", "rib_left_4", "rib_left_5", "rib_left_6",
    "rib_left_7", "rib_left_8", "rib_left_9", "rib_left_10", "rib_left_11", "rib_left_12",
    "rib_right_1", "rib_right_2", "rib_right_3", "rib_right_4", "rib_right_5", "rib_right_6",
    "rib_right_7", "rib_right_8", "rib_right_9", "rib_right_10", "rib_right_11", "rib_right_12",
    "hip_left", "hip_right", "sacrum",
]
VERTEBRA_RANK_INFERIOR_TO_SUPERIOR = {
    "vertebrae_L5": 0, "vertebrae_L4": 1, "vertebrae_L3": 2, "vertebrae_L2": 3, "vertebrae_L1": 4,
    "vertebrae_T12": 5, "vertebrae_T11": 6, "vertebrae_T10": 7, "vertebrae_T9": 8, "vertebrae_T8": 9,
    "vertebrae_T7": 10, "vertebrae_T6": 11, "vertebrae_T5": 12, "vertebrae_T4": 13,
    "vertebrae_T3": 14, "vertebrae_T2": 15, "vertebrae_T1": 16,
    "vertebrae_C7": 17, "vertebrae_C6": 18, "vertebrae_C5": 19, "vertebrae_C4": 20,
    "vertebrae_C3": 21, "vertebrae_C2": 22, "vertebrae_C1": 23,
}
BONE_NAMES = list(BONE_LABELS.keys()) if BONE_LABELS else DEFAULT_BONE_NAMES
EXCLUSION_NAMES = ("bone_all", "spleen", "liver", "kidney_left", "kidney_right", "inferior_vena_cava", "aorta")
STRUCTURE_ALIASES = {
    "portal_vein": ("portal_vein", "portal_vein_and_splenic_vein"),
}
MaskCache = dict


@dataclass(frozen=True)
class PretrainResult:
    path: Path
    status: str


# =========================================================================
# NIfTI 加载
# =========================================================================

def _load_nifti(path: Path):
    """加载 NIfTI，返回 (data, affine, spacing_zyx, origin_xyz)。"""
    import nibabel as nib
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    affine = img.affine.copy()
    # 从 affine 提取 spacing 和 origin
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    # NIfTI arrays from nibabel are indexed as (x, y, z). The pipeline works in
    # DICOM-style (z, y, x), so transpose both images and masks consistently.
    if len(data.shape) == 3:
        data = np.transpose(data, (2, 1, 0))
        spacing_zyx = (float(spacing[2]), float(spacing[1]), float(spacing[0]))
    else:
        spacing_zyx = tuple(float(s) for s in spacing[:3])

    origin = affine[:3, 3]
    origin_xyz = (float(origin[0]), float(origin[1]), float(origin[2]))
    return data, affine, spacing_zyx, origin_xyz


def _load_nifti_as_dicomvolume(path: Path) -> DicomVolume:
    """加载 orig.nii.gz 为 DicomVolume。"""
    data, affine, spacing_zyx, origin_xyz = _load_nifti(path)
    return DicomVolume(volume_hu=data, spacing_zyx=spacing_zyx, origin_xyz=origin_xyz)


# =========================================================================
# Precomputed TotalSegmentator NIfTI helpers
# =========================================================================

def _pretrain_nii_path(case) -> Path:
    return case.path / PRETRAIN_NII_NAME


def _segmentation_nii_dirs(case) -> list[Path]:
    seg_dir = case.path / "segmentation"
    dirs = [seg_dir / "totalseg_output", seg_dir / "ts_raw", seg_dir]
    unique: list[Path] = []
    for path in dirs:
        if path not in unique:
            unique.append(path)
    return unique


def _resample_bool_mask(mask: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray | None:
    if mask.shape == target_shape:
        return mask.astype(bool, copy=False)
    if ndi is None:
        return None
    zoom = np.asarray(target_shape, dtype=np.float64) / np.asarray(mask.shape, dtype=np.float64)
    try:
        return ndi.zoom(mask.astype(np.float32), zoom, order=0) > 0.5
    except Exception:
        return None


def _reference_nii_for_segmentation_mask(path: Path) -> Path | None:
    """Find patient/orig.nii.gz for a mask path."""
    path = Path(path)
    for parent in path.parents:
        ref = parent / "orig.nii.gz"
        if ref.exists() and ref != path:
            return ref
    return None


def _load_mask_nii(
    path: Path,
    target_shape: tuple[int, int, int] | None = None,
    cache: MaskCache | None = None,
) -> np.ndarray | None:
    cache_key = ("path", str(path), target_shape, "orig_affine_resampled")
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    try:
        if target_shape is not None:
            import nibabel as nib

            img = nib.load(str(path))
            ref_path = _reference_nii_for_segmentation_mask(path)
            if ref_path is not None:
                ref = nib.load(str(ref_path))
                ref_shape_zyx = (int(ref.shape[2]), int(ref.shape[1]), int(ref.shape[0]))
                if ref_shape_zyx == tuple(int(v) for v in target_shape):
                    data = np.asarray(img.dataobj)
                    if img.shape != ref.shape or not np.allclose(img.affine, ref.affine, atol=1e-4):
                        try:
                            from nibabel.orientations import apply_orientation, io_orientation, ornt_transform

                            transform = ornt_transform(io_orientation(img.affine), io_orientation(ref.affine))
                            data = apply_orientation(data, transform)
                        except Exception:
                            data = None
                        if data is None or data.shape != ref.shape:
                            from nibabel.processing import resample_from_to

                            img = resample_from_to(img, (ref.shape, ref.affine), order=0)
                            data = np.asarray(img.dataobj)
                    mask = np.asarray(data) > 0
                    if mask.ndim == 3:
                        mask = np.transpose(mask, (2, 1, 0))
                    if cache is not None:
                        cache[cache_key] = mask
                    return mask

        data, _, _, _ = _load_nifti(path)
    except Exception:
        if cache is not None:
            cache[cache_key] = None
        return None
    mask = np.asarray(data) > 0
    if target_shape is not None:
        mask = _resample_bool_mask(mask, target_shape)
    if cache is not None:
        cache[cache_key] = mask
    return mask


def _structure_candidate_paths(case, name: str) -> list[Path]:
    aliases = STRUCTURE_ALIASES.get(name, (name,))
    paths: list[Path] = []
    for directory in _segmentation_nii_dirs(case):
        for alias in aliases:
            paths.append(directory / f"{alias}.nii.gz")
    return paths


def _load_precomputed_structure_mask(
    case,
    name: str,
    target_shape: tuple[int, int, int],
    cache: MaskCache | None = None,
) -> tuple[np.ndarray | None, dict]:
    """Load a structure mask directly from precomputed .nii.gz files."""
    info: dict = {"structure": name, "source": "precomputed_nii", "loaded": []}
    cache_key = ("structure", f"{case.path}:{name}", target_shape)
    if cache is not None and cache_key in cache:
        cached = cache[cache_key]
        status = "ok" if cached is not None else "missing"
        voxels = int(cached.sum()) if cached is not None else 0
        return cached, {"structure": name, "source": "cache", "status": status, "voxels": voxels, "loaded": []}

    if name == "bone_all":
        for path in _structure_candidate_paths(case, "bone_all"):
            if not path.exists():
                continue
            mask = _load_mask_nii(path, target_shape, cache)
            if mask is not None:
                info.update({"status": "ok", "loaded": [str(path)], "voxels": int(mask.sum())})
                if cache is not None:
                    cache[cache_key] = mask
                return mask, info

        combined: np.ndarray | None = None
        for directory in _segmentation_nii_dirs(case):
            loaded_here = []
            for bone_name in BONE_NAMES:
                path = directory / f"{bone_name}.nii.gz"
                if not path.exists():
                    continue
                part = _load_mask_nii(path, target_shape, cache)
                if part is None:
                    continue
                combined = part.copy() if combined is None else (combined | part)
                loaded_here.append(path.name)
            if loaded_here:
                info["loaded"].extend(str(directory / file_name) for file_name in loaded_here)
                info["status"] = "ok"
                info["voxels"] = int(combined.sum()) if combined is not None else 0
                if cache is not None:
                    cache[cache_key] = combined
                return combined, info

        if cache is not None:
            cache[cache_key] = None
        return None, {"status": "missing", "structure": name, "source": "precomputed_nii"}

    for path in _structure_candidate_paths(case, name):
        if not path.exists():
            continue
        mask = _load_mask_nii(path, target_shape, cache)
        if mask is not None:
            info.update({"status": "ok", "loaded": [str(path)], "voxels": int(mask.sum())})
            if cache is not None:
                cache[cache_key] = mask
            return mask, info

    if cache is not None:
        cache[cache_key] = None
    return None, {"status": "missing", "structure": name, "source": "precomputed_nii"}


def _get_portal_vein_mask_fast(
    case,
    vol_shape: tuple[int, int, int],
    cache: MaskCache | None = None,
) -> tuple[np.ndarray | None, dict]:
    return _load_precomputed_structure_mask(case, "portal_vein", vol_shape, cache)


def _get_portal_seed_fast(
    case,
    vol_shape: tuple[int, int, int],
    cache: MaskCache | None = None,
) -> tuple[tuple[float, float, float] | None, dict]:
    mask, info = _get_portal_vein_mask_fast(case, vol_shape, cache)
    if mask is None or mask.sum() == 0:
        return None, info
    coords = np.argwhere(mask)
    seed = tuple(float(v) for v in coords.mean(axis=0))
    info = dict(info)
    info.update({"status": "ok", "seed_zyx": [round(v, 1) for v in seed]})
    return seed, info


def _get_exclusion_mask_fast(
    case,
    vol_shape: tuple[int, int, int],
    dilate_bone: int = 3,
    dilate_organ: int = 2,
    cache: MaskCache | None = None,
) -> tuple[np.ndarray, dict]:
    exclusion = np.zeros(vol_shape, dtype=bool)
    info: dict = {"source": "precomputed_nii", "loaded": []}
    portal_protect: np.ndarray | None = None

    for name in EXCLUSION_NAMES:
        mask, mask_info = _load_precomputed_structure_mask(case, name, vol_shape, cache)
        if mask is None:
            continue
        dilate = dilate_bone if name == "bone_all" else dilate_organ
        if ndi is not None and dilate > 0:
            mask = ndi.binary_dilation(mask, iterations=dilate)
        if name == "liver":
            if portal_protect is None:
                portal_protect, portal_info = _get_portal_vein_mask_fast(case, vol_shape, cache)
                if portal_protect is not None and ndi is not None and dilate_organ > 0:
                    portal_protect = ndi.binary_dilation(portal_protect, iterations=dilate_organ)
                info["portal_protection"] = {
                    "status": "ok" if portal_protect is not None and portal_protect.any() else "missing",
                    "source": portal_info.get("source"),
                    "voxels": int(portal_protect.sum()) if portal_protect is not None else 0,
                }
            if portal_protect is not None:
                mask = mask & ~portal_protect
        exclusion |= mask
        info["loaded"].append({"name": name, "files": mask_info.get("loaded", [])})

    info["total_excluded_voxels"] = int(exclusion.sum())
    info["status"] = "ok" if info["loaded"] else "empty"
    return exclusion, info


def _z_extent(mask: np.ndarray) -> tuple[int, int, float] | None:
    z_values = np.flatnonzero(np.any(mask, axis=(1, 2)))
    if len(z_values) == 0:
        return None
    z_min = int(z_values[0])
    z_max = int(z_values[-1])
    return z_min, z_max, float((z_min + z_max) / 2.0)


def _infer_vertebra_z_direction(extents: dict[str, tuple[int, int, float]]) -> str | None:
    ranked = [
        (VERTEBRA_RANK_INFERIOR_TO_SUPERIOR[name], extent[2])
        for name, extent in extents.items()
        if name in VERTEBRA_RANK_INFERIOR_TO_SUPERIOR
    ]
    if len(ranked) < 2:
        return None
    ranks = np.asarray([item[0] for item in ranked], dtype=np.float64)
    centers = np.asarray([item[1] for item in ranked], dtype=np.float64)
    denom = float(np.sum((ranks - ranks.mean()) ** 2))
    if denom <= 0:
        return None
    slope = float(np.sum((ranks - ranks.mean()) * (centers - centers.mean())) / denom)
    if abs(slope) < 1e-3:
        return None
    return "z_up" if slope > 0 else "z_down"


def _z_range_from_totalseg_vertebrae_nii(
    case,
    vol_shape: tuple[int, int, int],
    spacing_zyx: tuple[float, float, float],
    margin_mm: float = 0.0,
    cache: MaskCache | None = None,
) -> tuple[int | None, int | None, dict]:
    """Use segmentation/totalseg_output vertebra masks for the z ROI.

    The lower anatomical bound is the inferior edge of L3. The upper bound is
    T8 when available, otherwise T9, otherwise the most superior loaded
    vertebra point.
    """
    nz = vol_shape[0]
    ts_output = case.path / "segmentation" / "totalseg_output"
    if not ts_output.is_dir():
        return None, None, {"status": "missing_totalseg_output", "source": "totalseg_vertebrae_nii"}

    extents: dict[str, tuple[int, int, float]] = {}
    loaded: list[str] = []
    for name in VERTEBRA_RANK_INFERIOR_TO_SUPERIOR:
        path = ts_output / f"{name}.nii.gz"
        if not path.exists():
            continue
        mask = _load_mask_nii(path, vol_shape, cache)
        if mask is None or not mask.any():
            continue
        extent = _z_extent(mask)
        if extent is None:
            continue
        extents[name] = extent
        loaded.append(str(path))

    if "vertebrae_L3" not in extents:
        return None, None, {
            "status": "missing_L3",
            "source": "totalseg_vertebrae_nii",
            "loaded_count": len(loaded),
            "loaded_examples": loaded[:5],
        }

    direction = _infer_vertebra_z_direction(extents)
    if direction is None:
        return None, None, {
            "status": "unknown_vertebra_z_direction",
            "source": "totalseg_vertebrae_nii",
            "loaded_count": len(loaded),
            "loaded_examples": loaded[:5],
        }

    upper_name = "vertebrae_T8" if "vertebrae_T8" in extents else ("vertebrae_T9" if "vertebrae_T9" in extents else None)
    upper_source = upper_name or "highest_loaded_vertebra_point"

    l3_min, l3_max, _ = extents["vertebrae_L3"]
    if direction == "z_down":
        lower_z = l3_max
        if upper_name is not None:
            upper_z = extents[upper_name][0]
        else:
            upper_z = min(extent[0] for extent in extents.values())
    else:
        lower_z = l3_min
        if upper_name is not None:
            upper_z = extents[upper_name][1]
        else:
            upper_z = max(extent[1] for extent in extents.values())

    dz = max(float(spacing_zyx[0]), 1e-3)
    margin = max(0, int(round(margin_mm / dz)))
    z_start = max(0, min(int(upper_z), int(lower_z)) - margin)
    z_end = min(nz - 1, max(int(upper_z), int(lower_z)) + margin)
    if z_end <= z_start:
        return None, None, {
            "status": "invalid_vertebra_z_range",
            "source": "totalseg_vertebrae_nii",
            "upper_z": int(upper_z),
            "lower_z": int(lower_z),
            "z_direction": direction,
        }

    return z_start, z_end, {
        "status": "ok",
        "source": "totalseg_vertebrae_nii",
        "z_direction": direction,
        "upper_source": upper_source,
        "lower_source": "vertebrae_L3_inferior_edge",
        "upper_z": int(upper_z),
        "lower_z": int(lower_z),
        "z_start": z_start,
        "z_end": z_end,
        "z_range_mm": round(float((z_end - z_start) * dz), 1),
        "margin_mm": float(margin_mm),
        "loaded_count": len(loaded),
        "loaded_examples": loaded[:5],
    }


def _z_range_from_precomputed_bone_nii(
    case,
    vol_shape: tuple[int, int, int],
    spacing_zyx: tuple[float, float, float],
    margin_mm: float,
    cache: MaskCache | None = None,
) -> tuple[int | None, int | None, dict]:
    """Use individual bone .nii.gz files to infer the z ROI without building STL."""
    nz = vol_shape[0]
    z_has = np.zeros(nz, dtype=bool)
    x_min = np.full(nz, vol_shape[2], dtype=np.int32)
    x_max = np.full(nz, -1, dtype=np.int32)
    loaded: list[str] = []

    for directory in _segmentation_nii_dirs(case):
        loaded_before = len(loaded)
        for bone_name in BONE_NAMES:
            path = directory / f"{bone_name}.nii.gz"
            if not path.exists():
                continue
            mask = _load_mask_nii(path, vol_shape, cache)
            if mask is None or not mask.any():
                continue
            loaded.append(str(path))
            z_has |= np.any(mask, axis=(1, 2))
            zx = np.any(mask, axis=1)
            z_idx, x_idx = np.where(zx)
            if len(z_idx):
                np.minimum.at(x_min, z_idx, x_idx)
                np.maximum.at(x_max, z_idx, x_idx)
        if len(loaded) > loaded_before:
            break

    if not loaded:
        combined, combined_info = _load_precomputed_structure_mask(case, "bone_all", vol_shape, cache)
        if combined is None or not combined.any():
            return None, None, {"status": "no_precomputed_bone_nii", **combined_info}
        loaded = list(combined_info.get("loaded", []))
        z_has = np.any(combined, axis=(1, 2))
        zx = np.any(combined, axis=1)
        z_idx, x_idx = np.where(zx)
        if len(z_idx):
            np.minimum.at(x_min, z_idx, x_idx)
            np.maximum.at(x_max, z_idx, x_idx)

    if not z_has.any():
        return None, None, {"status": "empty_bone", "loaded": loaded}

    dz = float(spacing_zyx[0])
    margin = max(1, int(round(margin_mm / max(0.1, dz))))
    bone_z = np.where(z_has)[0]
    z_start = max(0, int(bone_z.min()) - margin)
    z_end = min(nz - 1, int(bone_z.max()) + margin)

    for z in range(int(bone_z.max()), int(bone_z.min()), -1):
        if x_max[z] < x_min[z]:
            continue
        width_mm = float((x_max[z] - x_min[z]) * spacing_zyx[2])
        if width_mm > 90.0:
            z_end = max(z_start + 1, z - margin)
            break

    return z_start, z_end, {
        "status": "ok",
        "source": "precomputed_bone_nii",
        "loaded_count": len(loaded),
        "loaded_examples": loaded[:5],
        "z_start": z_start,
        "z_end": z_end,
        "z_range_mm": round(float((z_end - z_start) * dz), 1),
    }


def _z_range_from_portal_nii(
    case,
    vol_shape: tuple[int, int, int],
    spacing_zyx: tuple[float, float, float],
    margin_mm: float,
    cache: MaskCache | None = None,
) -> tuple[int | None, int | None, dict]:
    """Infer the z ROI from the portal/splenic vein mask before falling back to bone."""
    portal_mask, portal_info = _get_portal_vein_mask_fast(case, vol_shape, cache)
    info = dict(portal_info)
    info["source"] = "precomputed_portal_vein_nii"
    if portal_mask is None or not portal_mask.any():
        info["status"] = "no_portal_mask_for_z"
        return None, None, info

    z_values = np.flatnonzero(np.any(portal_mask, axis=(1, 2)))
    if len(z_values) == 0:
        info["status"] = "empty_portal_z"
        return None, None, info

    nz = vol_shape[0]
    dz = max(float(spacing_zyx[0]), 1e-3)
    margin_slices = int(np.ceil(margin_mm / dz))
    raw_start = int(z_values[0])
    raw_end = int(z_values[-1])
    z_start = max(0, raw_start - margin_slices)
    z_end = min(nz - 1, raw_end + margin_slices)
    info.update({
        "status": "ok",
        "z_start": z_start,
        "z_end": z_end,
        "z_range_mm": round(float((z_end - z_start) * dz), 1),
        "raw_portal_z_start": raw_start,
        "raw_portal_z_end": raw_end,
        "margin_mm": margin_mm,
        "margin_slices": margin_slices,
        "portal_z_slices": int(len(z_values)),
        "portal_voxels": int(portal_mask.sum()),
    })
    return z_start, z_end, info


def _precomputed_segmentation_mtime(case) -> float:
    latest = 0.0
    seg_dir = case.path / "segmentation"
    paths = list(seg_dir.glob("*.nii.gz")) if seg_dir.exists() else []
    for directory in (seg_dir / "totalseg_output", seg_dir / "ts_raw"):
        if directory.exists():
            paths.extend(directory.glob("*.nii.gz"))
    for path in paths:
        try:
            latest = max(latest, path.stat().st_mtime)
        except OSError:
            continue
    return latest


def _save_pretrain_nifti(mask: np.ndarray, reference_nii: Path, out_path: Path) -> Path:
    import nibabel as nib

    ref = nib.load(str(reference_nii))
    data = mask.astype(np.uint8)
    if data.shape != ref.shape and len(ref.shape) == 3:
        zyx_shape = (ref.shape[2], ref.shape[1], ref.shape[0])
        if data.shape == zyx_shape:
            data = np.transpose(data, (2, 1, 0))
    header = ref.header.copy()
    header.set_data_dtype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data, ref.affine, header)

    qform, qcode = ref.get_qform(coded=True)
    sform, scode = ref.get_sform(coded=True)
    img.set_qform(qform if qform is not None else ref.affine, int(qcode) if qform is not None else 1)
    img.set_sform(sform if sform is not None else ref.affine, int(scode) if sform is not None else 1)
    nib.save(img, str(out_path))
    return out_path


# =========================================================================
# Step 1: Z 轴标定（从 bone 分割）
# =========================================================================

def _standardize_z_from_bone(
    case,
    vol_shape: tuple[int, int, int],
    spacing_zyx: tuple[float, float, float],
    margin_mm: float = 20.0,
    cache: MaskCache | None = None,
) -> tuple[int, int, dict]:
    """用 TotalSegmentator 的 bone 分割做 Z 轴标定。

    上界：骨骼出现的最高层（膈肌附近）
    下界：髂骨上缘（骨质横向跨度突然增大处）

    如果 bone 分割不可用，退化为 volume 的 20%-82%。
    """
    z_start, z_end, info = _z_range_from_totalseg_vertebrae_nii(
        case, vol_shape, spacing_zyx, margin_mm=0.0, cache=cache
    )
    if z_start is not None and z_end is not None:
        return z_start, z_end, info

    portal_margin_mm = max(margin_mm, 45.0 if getattr(case, "is_post_tips", False) else 35.0)
    z_start, z_end, portal_info = _z_range_from_portal_nii(case, vol_shape, spacing_zyx, portal_margin_mm, cache)
    if z_start is not None and z_end is not None:
        portal_info["vertebra_z_fallback"] = info
        return z_start, z_end, portal_info

    z_start, z_end, info = _z_range_from_precomputed_bone_nii(case, vol_shape, spacing_zyx, margin_mm, cache)
    if z_start is not None and z_end is not None:
        return z_start, z_end, info

    if get_z_range_from_bone is not None:
        z_start, z_end, info = get_z_range_from_bone(case, vol_shape, spacing_zyx, margin_mm)
        if z_start is not None and z_end is not None:
            return z_start, z_end, info

    # fallback
    nz = vol_shape[0]
    z_start = int(nz * 0.20)
    z_end = int(nz * 0.82)
    return z_start, z_end, {"status": "fallback", "z_start": z_start, "z_end": z_end}


# =========================================================================
# Step 2: 从门静脉分割采样 HU → 精确阈值
# =========================================================================

def _sample_hu_from_portal_vein(
    vol: np.ndarray,
    case,
    vol_shape: tuple[int, int, int],
    hu_margin: float = HU_MARGIN,
    cache: MaskCache | None = None,
) -> tuple[float, float, dict]:
    """从 TotalSegmentator 的门静脉 mask 中采样 HU 值。

    策略：
    1. 加载 portal_vein mask
    2. 在 mask 内采样所有体素的 HU 值
    3. 取 P2 和 P98 作为门静脉的 HU 范围
    4. 上下各扩展 hu_margin

    这样得到的阈值是该患者特异性的，完美解决
    "不同 CT 灰度分布不同"的问题。
    """
    info: dict = {}

    pv_mask, pv_info = _get_portal_vein_mask_fast(case, vol_shape, cache)
    info.update(pv_info)
    if (pv_mask is None or pv_mask.sum() == 0) and get_portal_vein_mask is not None:
        pv_mask = get_portal_vein_mask(case, vol_shape)
        info["source"] = "totalseg_module"
    elif pv_mask is not None:
        info["source"] = "precomputed_nii"

    if pv_mask is None or pv_mask.sum() == 0:
        info["status"] = "no_portal_vein_mask"
        hu_high = TIPS_HU_HIGH_CAP if getattr(case, "is_post_tips", False) else 350.0
        info["hu_high_source"] = "tips_high_cap" if getattr(case, "is_post_tips", False) else "fallback"
        return 100.0, hu_high, info

    # 在门静脉 mask 内采样 HU
    pv_hu = vol[pv_mask]
    info["n_samples"] = int(len(pv_hu))

    if len(pv_hu) < 50:
        info["status"] = "too_few_samples"
        hu_high = TIPS_HU_HIGH_CAP if getattr(case, "is_post_tips", False) else 350.0
        info["hu_high_source"] = "tips_high_cap" if getattr(case, "is_post_tips", False) else "fallback"
        return 100.0, hu_high, info

    p2 = float(np.percentile(pv_hu, 2))
    p10 = float(np.percentile(pv_hu, 10))
    p20 = float(np.percentile(pv_hu, 20))
    p25 = float(np.percentile(pv_hu, 25))
    p50 = float(np.percentile(pv_hu, 50))
    p90 = float(np.percentile(pv_hu, 90))
    p98 = float(np.percentile(pv_hu, 98))

    hu_low = max(p20 - hu_margin, p50 - 25.0)
    hu_high = p98 + hu_margin

    # p2/p10 are too permissive for partial-volume edge voxels and weakly
    # enhanced liver tissue. Tie the lower bound to the portal-vein body.
    hu_low = max(HU_LOW_FLOOR, hu_low)
    if getattr(case, "is_post_tips", False):
        hu_high = max(hu_high, TIPS_HU_HIGH_CAP)
        hu_high_source = "tips_high_cap"
    else:
        # Non-TIPS cases should not pull in bone/metal-like high HU structures.
        hu_high = min(DEFAULT_HU_HIGH_CAP, hu_high)
        hu_high_source = "portal_p98_plus_margin"

    info.update({
        "status": "ok",
        "p2": round(p2, 1),
        "p10": round(p10, 1),
        "p20": round(p20, 1),
        "p25": round(p25, 1),
        "p50": round(p50, 1),
        "p90": round(p90, 1),
        "p98": round(p98, 1),
        "hu_low": round(hu_low, 1),
        "hu_high": round(hu_high, 1),
        "hu_margin": hu_margin,
        "hu_low_source": "max(HU_LOW_FLOOR, p20 - hu_margin, p50 - 25)",
        "hu_low_floor": HU_LOW_FLOOR,
        "hu_high_source": hu_high_source,
        "default_hu_high_cap": DEFAULT_HU_HIGH_CAP,
        "tips_hu_high_cap": TIPS_HU_HIGH_CAP,
    })
    return hu_low, hu_high, info


# =========================================================================
# Step 3: 阈值分割 + 器官减去 + 区域生长
# =========================================================================

def _threshold_segment(
    vol: np.ndarray,
    hu_low: float,
    hu_high: float,
    z_start: int,
    z_end: int,
) -> np.ndarray:
    """在 Z 轴范围内做 HU 阈值分割。"""
    mask = np.zeros(vol.shape, dtype=bool)
    z0 = max(0, z_start)
    z1 = min(vol.shape[0], z_end + 1)
    roi = vol[z0:z1]
    mask[z0:z1] = (roi >= hu_low) & (roi <= hu_high)
    return mask


def _subtract_organs(
    mask: np.ndarray,
    case,
    vol_shape: tuple[int, int, int],
    dilate_bone: int = 3,
    dilate_organ: int = 2,
    cache: MaskCache | None = None,
) -> tuple[np.ndarray, dict]:
    """减去 bone/spleen/liver/kidney/IVC/aorta 的区域。"""
    exclusion, excl_info = _get_exclusion_mask_fast(
        case, vol_shape,
        dilate_bone=dilate_bone,
        dilate_organ=dilate_organ,
        cache=cache,
    )
    if not excl_info.get("loaded") and get_exclusion_mask is not None:
        exclusion, excl_info = get_exclusion_mask(
            case, vol_shape,
            dilate_bone=dilate_bone,
            dilate_organ=dilate_organ,
        )
        excl_info["source"] = "totalseg_module"
    elif not excl_info.get("loaded"):
        return mask, {"status": "module_unavailable"}

    before = int(mask.sum())
    mask = mask & ~exclusion
    after = int(mask.sum())
    del exclusion
    gc.collect()

    excl_info["voxels_before"] = before
    excl_info["voxels_after"] = after
    excl_info["voxels_removed"] = before - after
    return mask, excl_info


def _get_tips_exclusion_mask_fast(
    case,
    vol_shape: tuple[int, int, int],
    cache: MaskCache | None = None,
) -> tuple[np.ndarray, dict]:
    """Build a conservative exclusion mask for high-HU TIPS recovery.

    Liver and IVC are intentionally not excluded here because a TIPS stent runs
    through liver parenchyma toward the hepatic venous outflow.
    """
    exclusion = np.zeros(vol_shape, dtype=bool)
    info: dict = {"source": "precomputed_nii", "loaded": []}
    for name, dilate in (("bone_all", 2), ("aorta", 1), ("kidney_left", 1), ("kidney_right", 1), ("spleen", 1)):
        mask, mask_info = _load_precomputed_structure_mask(case, name, vol_shape, cache)
        if mask is None:
            continue
        if ndi is not None and dilate > 0:
            mask = ndi.binary_dilation(mask, iterations=dilate)
        exclusion |= mask
        info["loaded"].append({"name": name, "files": mask_info.get("loaded", [])})
    info["total_excluded_voxels"] = int(exclusion.sum())
    info["status"] = "ok" if info["loaded"] else "empty"
    return exclusion, info


def _morphological_cleanup(mask: np.ndarray) -> np.ndarray:
    """Remove opening noise while restoring well-supported boundary voxels."""
    if ndi is None:
        return mask

    original = np.asarray(mask, dtype=bool)
    opened = ndi.binary_opening(original, iterations=1)
    core_support = ndi.convolve(
        opened.astype(np.uint8),
        np.ones((3, 3, 3), dtype=np.uint8),
        mode="constant",
        cval=0,
    )
    mask = opened | (
        original & (core_support >= OPENING_RESTORE_MIN_CORE_NEIGHBORS)
    )
    mask = ndi.binary_closing(mask, iterations=1)
    # 只填小洞
    filled = ndi.binary_fill_holes(mask)
    holes = filled & ~mask
    del filled
    if holes.sum() > 0:
        hl = np.empty(holes.shape, dtype=np.int32)
        nh = ndi.label(holes, output=hl)
        counts = np.bincount(hl.ravel(), minlength=nh + 1)
        fill_labels = np.flatnonzero((counts <= 500) & (np.arange(nh + 1) > 0))
        if len(fill_labels):
            mask = mask | np.isin(hl, fill_labels)
        del hl
    del holes
    return mask


def _keep_largest_connected_component(mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """Keep only the largest 26-connected foreground component."""
    mask = np.asarray(mask, dtype=bool)
    info = {"input_voxels": int(mask.sum())}
    if ndi is None or not mask.any():
        info.update({"components": 0, "output_voxels": int(mask.sum()), "removed_voxels": 0})
        return mask, info

    labels, components = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    counts = np.bincount(labels.ravel(), minlength=components + 1)
    counts[0] = 0
    largest_label = int(np.argmax(counts))
    result = labels == largest_label
    info.update({
        "components": int(components),
        "largest_label": largest_label,
        "output_voxels": int(result.sum()),
        "removed_voxels": int(mask.sum() - result.sum()),
    })
    return result, info


def _region_grow_from_seed(
    mask: np.ndarray,
    seed_zyx: tuple[float, float, float],
    spacing_zyx: tuple[float, float, float],
    bridge_mm: float = REGION_GROW_BRIDGE_MM,
) -> tuple[np.ndarray, dict]:
    """从 seed 做区域生长 + 桥接断裂分支。"""
    mask = np.asarray(mask, dtype=bool)
    info: dict = {"input_voxels": int(mask.sum())}

    if ndi is None or mask.sum() == 0:
        info["output_voxels"] = int(mask.sum())
        return mask, info

    labels = np.empty(mask.shape, dtype=np.int32)
    n = ndi.label(mask, output=labels)
    if n <= 1:
        info.update({"output_voxels": int(mask.sum()), "components": n})
        return mask, info

    # 找 seed 最近的前景体素
    coords = np.argwhere(mask)
    seed = np.asarray(seed_zyx, dtype=np.float32)
    sp = np.asarray(spacing_zyx, dtype=np.float32)
    delta_mm = (coords.astype(np.float32) - seed) * sp
    dist2 = np.sum(delta_mm ** 2, axis=1)
    ni = int(np.argmin(dist2))
    nearest_mm = float(np.sqrt(dist2[ni]))
    del delta_mm, dist2

    info["nearest_seed_distance_mm"] = round(nearest_mm, 2)
    if nearest_mm > REGION_GROW_MAX_SEED_SNAP_MM:
        info.update({"output_voxels": int(mask.sum()), "skipped": "seed_too_far"})
        return mask, info

    z, y, x = coords[ni]
    ml = int(labels[z, y, x])
    mc = (labels == ml)

    # 桥接
    bv = max(1, int(round(bridge_mm / float(np.min(sp)))))
    bz = ndi.binary_dilation(mc, iterations=bv)

    counts = np.bincount(labels.ravel(), minlength=n + 1)

    bridge_labels = np.unique(labels[bz])
    bridged = set(int(lb) for lb in bridge_labels if lb > 0 and counts[int(lb)] >= 32)
    bridged.add(ml)
    del bz

    result = np.isin(labels, list(bridged))
    del labels

    info.update({
        "output_voxels": int(result.sum()),
        "components_total": n,
        "bridged": len(bridged) - 1,
        "removed": n - len(bridged),
    })
    return result, info


def _region_grow_from_portal_mask(
    mask: np.ndarray,
    portal_mask: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    bridge_mm: float = REGION_GROW_BRIDGE_MM,
) -> tuple[np.ndarray, dict]:
    """用门静脉 mask（而非单点）做区域生长。

    比单点 seed 更稳健：portal_mask 覆盖整个门静脉主干，
    任何和它重叠的连通域都保留。
    """
    mask = np.asarray(mask, dtype=bool)
    info: dict = {"input_voxels": int(mask.sum())}

    if ndi is None or mask.sum() == 0 or portal_mask is None:
        info["output_voxels"] = int(mask.sum())
        return mask, info

    labels = np.empty(mask.shape, dtype=np.int32)
    n = ndi.label(mask, output=labels)
    if n <= 1:
        info.update({"output_voxels": int(mask.sum()), "components": n})
        return mask, info

    portal_labels = np.unique(labels[portal_mask])
    portal_labels_hit = set(int(v) for v in portal_labels if v > 0)

    counts = np.bincount(labels.ravel(), minlength=n + 1)
    if not portal_labels_hit:
        # portal_mask 没有和任何前景重叠 → 退化为保留最大连通域
        counts[0] = 0
        portal_labels_hit = {int(np.argmax(counts))}
        info["fallback"] = "largest_component"

    # 合并所有命中的 label 为主区域
    main_region = np.isin(labels, list(portal_labels_hit))

    # 桥接：膨胀主区域，检查哪些其他连通域能桥接到
    sp = np.asarray(spacing_zyx, dtype=np.float32)
    bv = max(1, int(round(bridge_mm / float(np.min(sp)))))
    bridge_zone = ndi.binary_dilation(main_region, iterations=bv)

    bridge_labels = np.unique(labels[bridge_zone])
    all_keep = {
        int(lb)
        for lb in bridge_labels
        if lb > 0 and (int(lb) in portal_labels_hit or counts[int(lb)] >= 32)
    }
    all_keep.update(portal_labels_hit)
    del bridge_zone

    result = np.isin(labels, list(all_keep))
    del labels

    info.update({
        "output_voxels": int(result.sum()),
        "portal_labels_hit": len(portal_labels_hit),
        "components_total": n,
        "bridged": len(all_keep) - len(portal_labels_hit),
        "removed": n - len(all_keep),
    })
    return result, info


def _limit_to_portal_reference_neighborhood(
    mask: np.ndarray,
    portal_mask: np.ndarray | None,
    spacing_zyx: tuple[float, float, float],
    radius_mm: float = PORTAL_REFERENCE_CLEANUP_RADIUS_MM,
    seed_dilate: int = PORTAL_REFERENCE_CLEANUP_SEED_DILATE,
) -> tuple[np.ndarray, dict]:
    """Clip a broad portal-connected component to the portal reference area."""
    mask = np.asarray(mask, dtype=bool)
    info: dict = {
        "enabled": portal_mask is not None,
        "radius_mm": float(radius_mm),
        "input_voxels": int(mask.sum()),
    }
    if ndi is None or portal_mask is None or not portal_mask.any() or not mask.any():
        info["status"] = "skipped"
        info["output_voxels"] = int(mask.sum())
        return mask, info

    portal = np.asarray(portal_mask, dtype=bool)
    distance_mm = ndi.distance_transform_edt(~portal, sampling=spacing_zyx)
    clipped = mask & (distance_mm <= float(radius_mm))
    info["after_distance_voxels"] = int(clipped.sum())
    del distance_mm

    if not clipped.any():
        info["status"] = "empty_after_distance_clip"
        info["output_voxels"] = 0
        return clipped, info

    labels = np.empty(clipped.shape, dtype=np.int32)
    n = ndi.label(clipped, output=labels)
    seed = portal
    if seed_dilate > 0:
        seed = ndi.binary_dilation(portal, iterations=seed_dilate)
    hit_labels = {int(v) for v in np.unique(labels[seed]) if int(v) > 0}
    if hit_labels:
        result = np.isin(labels, list(hit_labels))
    else:
        result = clipped
    del labels

    info.update({
        "status": "ok",
        "components_total": int(n),
        "portal_labels_hit": int(len(hit_labels)),
        "output_voxels": int(result.sum()),
        "removed_voxels": int(mask.sum() - result.sum()),
    })
    return result, info


def _portal_reference_quality(
    vol: np.ndarray,
    portal_mask: np.ndarray | None,
) -> dict:
    info: dict = {"available": portal_mask is not None and bool(portal_mask.any())}
    if portal_mask is None or not portal_mask.any():
        info["status"] = "missing"
        info["reliable"] = False
        return info
    vals = np.asarray(vol[portal_mask], dtype=np.float32)
    if vals.size == 0:
        info["status"] = "empty"
        info["reliable"] = False
        return info
    p10, p25, p50, p75, p90 = np.percentile(vals, [10, 25, 50, 75, 90])
    info.update({
        "status": "ok",
        "voxels": int(portal_mask.sum()),
        "p10": round(float(p10), 1),
        "p25": round(float(p25), 1),
        "p50": round(float(p50), 1),
        "p75": round(float(p75), 1),
        "p90": round(float(p90), 1),
        "reliable": bool(p50 >= PORTAL_REFERENCE_MIN_P50_HU),
        "min_p50_hu": PORTAL_REFERENCE_MIN_P50_HU,
    })
    return info


def _portal_boundary_hu_low(hu_low: float, portal_quality_info: dict) -> float:
    """Raise the low HU cut just enough to drop the low-density portal shell."""
    p50 = portal_quality_info.get("p50")
    if p50 is None:
        return float(hu_low)
    return max(float(hu_low), float(p50) - PORTAL_BOUNDARY_P50_MARGIN_HU)


def _liver_spleen_portal_fallback(
    mask: np.ndarray,
    vol: np.ndarray,
    case,
    vol_shape: tuple[int, int, int],
    spacing_zyx: tuple[float, float, float],
    hu_high: float,
    cache: MaskCache | None = None,
) -> tuple[np.ndarray | None, dict]:
    info: dict = {"enabled": True, "source": "liver_spleen_hu_component"}
    if ndi is None or not mask.any():
        info["status"] = "skipped"
        return None, info

    liver, liver_info = _load_precomputed_structure_mask(case, "liver", vol_shape, cache)
    spleen, spleen_info = _load_precomputed_structure_mask(case, "spleen", vol_shape, cache)
    info["liver"] = liver_info
    info["spleen"] = spleen_info
    if liver is None or not liver.any() or spleen is None or not spleen.any():
        info["status"] = "missing_liver_or_spleen"
        return None, info

    high_low = max(LIVER_SPLEEN_FALLBACK_MIN_HU, min(180.0, float(hu_high) - 25.0))
    high_high = min(float(hu_high), LIVER_SPLEEN_FALLBACK_MAX_HU)
    if high_high <= high_low:
        high_high = max(high_low + 20.0, float(hu_high))
    candidate = mask & (vol >= high_low) & (vol <= high_high)
    info.update({
        "hu_low": round(float(high_low), 1),
        "hu_high": round(float(high_high), 1),
        "candidate_voxels": int(candidate.sum()),
    })
    if not candidate.any():
        info["status"] = "empty_candidate"
        return None, info

    liver_dist = ndi.distance_transform_edt(~liver.astype(bool), sampling=spacing_zyx)
    spleen_dist = ndi.distance_transform_edt(~spleen.astype(bool), sampling=spacing_zyx)
    labels = np.empty(candidate.shape, dtype=np.int32)
    n = ndi.label(candidate, output=labels)
    counts = np.bincount(labels.ravel(), minlength=n + 1)
    counts[0] = 0

    best_label = 0
    best_score = -1e9
    scored: list[dict] = []
    for lb in np.flatnonzero(counts >= 1000):
        comp = labels == int(lb)
        dl = float(np.percentile(liver_dist[comp], 50))
        ds = float(np.percentile(spleen_dist[comp], 50))
        if not (25.0 <= dl <= 130.0 and 25.0 <= ds <= 130.0):
            continue
        hu50 = float(np.percentile(vol[comp], 50))
        score = (
            2.0 * float(np.log1p(counts[int(lb)]))
            - 0.03 * (dl + ds)
            - 0.05 * abs(dl - ds)
            + 0.01 * hu50
        )
        item = {
            "label": int(lb),
            "voxels": int(counts[int(lb)]),
            "score": round(float(score), 3),
            "dist_liver_p50_mm": round(dl, 1),
            "dist_spleen_p50_mm": round(ds, 1),
            "hu_p50": round(hu50, 1),
        }
        scored.append(item)
        if score > best_score:
            best_score = score
            best_label = int(lb)

    del liver_dist, spleen_dist
    scored.sort(key=lambda item: item["score"], reverse=True)
    info["top_components"] = scored[:8]
    if best_label <= 0:
        del labels
        info["status"] = "no_anatomic_component"
        return None, info

    selected = labels == best_label
    del labels
    info.update({
        "status": "ok",
        "components_total": int(n),
        "selected_label": int(best_label),
        "output_voxels": int(selected.sum()),
    })
    return selected, info


# =========================================================================
# TIPS 支架处理
# =========================================================================

def _add_tips_stent(
    mask: np.ndarray,
    vol: np.ndarray,
    z_start: int,
    z_end: int,
    tips_hu_low: float = 430.0,
    tips_hu_high: float = TIPS_HU_HIGH_CAP,
    exclusion_mask: np.ndarray | None = None,
    portal_mask: np.ndarray | None = None,
    spacing_zyx: tuple[float, float, float] | None = None,
    max_portal_distance_mm: float = PORTAL_REFERENCE_CLEANUP_RADIUS_TIPS_MM,
) -> tuple[np.ndarray, dict]:
    """单独处理 TIPS 支架（高 HU 通道）。"""
    info: dict = {}
    z0 = max(0, z_start)
    z1 = min(vol.shape[0], z_end + 1)

    tips_mask = np.zeros(vol.shape, dtype=bool)
    tips_mask[z0:z1] = (vol[z0:z1] >= tips_hu_low) & (vol[z0:z1] <= tips_hu_high)

    if exclusion_mask is not None:
        tips_mask = tips_mask & ~exclusion_mask

    if portal_mask is not None and spacing_zyx is not None and ndi is not None and portal_mask.any():
        distance_mm = ndi.distance_transform_edt(~portal_mask.astype(bool), sampling=spacing_zyx)
        tips_mask = tips_mask & (distance_mm <= float(max_portal_distance_mm))
        info["portal_distance_limit_mm"] = float(max_portal_distance_mm)
        info["tips_voxels_after_distance"] = int(tips_mask.sum())
        del distance_mm

    if ndi is not None:
        tips_mask = ndi.binary_closing(tips_mask, iterations=1)
        # 只保留较大的连通域
        lb = np.empty(tips_mask.shape, dtype=np.int32)
        n = ndi.label(tips_mask, output=lb)
        if n > 0:
            counts = np.bincount(lb.ravel(), minlength=n + 1)
            counts[0] = 0
            keep = [i for i in np.argsort(counts)[::-1][:4] if counts[i] >= 32]
            tips_clean = np.zeros(tips_mask.shape, dtype=bool)
            for i in keep:
                tips_clean |= (lb == i)
            tips_mask = tips_clean
            del lb, tips_clean

    tips_mask, lumen_info = _fill_local_tips_lumen(
        tips_mask, spacing_zyx, max_distance_mm=TIPS_LUMEN_FILL_RADIUS_MM,
    )
    info["lumen_fill"] = lumen_info
    tips_voxels = int(tips_mask.sum())
    info["tips_voxels"] = tips_voxels

    if tips_voxels > 0:
        combined = mask | tips_mask
        info["combined_voxels"] = int(combined.sum())
        return combined, info

    return mask, info


def _fill_local_tips_lumen(
    tips_mask: np.ndarray,
    spacing_zyx: tuple[float, float, float] | None,
    max_distance_mm: float = TIPS_LUMEN_FILL_RADIUS_MM,
) -> tuple[np.ndarray, dict]:
    """Fill hollow TIPS lumen only inside the localized high-HU stent region."""
    info: dict = {"enabled": bool(tips_mask.any()), "max_distance_mm": float(max_distance_mm)}
    if ndi is None or not tips_mask.any():
        info["status"] = "skipped"
        info["filled_voxels"] = 0
        return tips_mask, info

    fill_candidates = np.zeros(tips_mask.shape, dtype=bool)
    coords = np.argwhere(tips_mask)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    spacing = np.asarray(spacing_zyx or (1.0, 1.0, 1.0), dtype=np.float64)
    pad = np.ceil((float(max_distance_mm) + 2.0) / spacing).astype(np.int32)
    lo = np.maximum(0, lo - pad)
    hi = np.minimum(np.asarray(tips_mask.shape), hi + pad)
    local = tips_mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    local_fill = np.zeros_like(local, dtype=bool)

    # First catch genuinely closed small holes.
    for axis in range(3):
        moved = np.moveaxis(local, axis, 0)
        moved_fill = np.moveaxis(local_fill, axis, 0)
        for idx in range(moved.shape[0]):
            sl = moved[idx]
            if not sl.any():
                continue
            holes = ndi.binary_fill_holes(sl) & ~sl
            if holes.any():
                moved_fill[idx] |= holes

    # TIPS is often a non-closed metal lattice, so holes are not always closed
    # in 2D slices. Build a solid local tube from each stent component instead.
    labels = np.empty(local.shape, dtype=np.int32)
    n = ndi.label(local, output=labels)
    counts = np.bincount(labels.ravel(), minlength=n + 1)
    counts[0] = 0
    tube_fill = np.zeros_like(local, dtype=bool)
    components: list[dict] = []
    for label_id in np.flatnonzero(counts >= 32):
        component = labels == int(label_id)
        comp_coords = np.argwhere(component)
        if comp_coords.shape[0] < 32:
            continue

        phys = comp_coords.astype(np.float64) * spacing
        center = phys.mean(axis=0)
        centered = phys - center
        try:
            cov = np.cov(centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
        except Exception:
            continue
        axis = eigvecs[:, int(np.argmax(eigvals))]
        t = centered @ axis
        span = float(t.max() - t.min()) if t.size else 0.0
        if span < 6.0:
            continue

        n_bins = max(3, int(np.ceil(span / TIPS_LUMEN_FILL_BIN_MM)))
        edges = np.linspace(float(t.min()), float(t.max()), n_bins + 1)
        bin_t = (edges[:-1] + edges[1:]) / 2.0
        centers: list[np.ndarray | None] = []
        radii: list[float | None] = []
        for idx in range(n_bins):
            in_bin = (t >= edges[idx]) & (t < edges[idx + 1] if idx < n_bins - 1 else t <= edges[idx + 1])
            if int(in_bin.sum()) < 5:
                centers.append(None)
                radii.append(None)
                continue
            bin_phys = phys[in_bin]
            bin_center = bin_phys.mean(axis=0)
            rel = bin_phys - bin_center
            perp = rel - np.outer(rel @ axis, axis)
            radius = float(np.percentile(np.linalg.norm(perp, axis=1), 85) + 0.6)
            centers.append(bin_center)
            radii.append(float(np.clip(radius, 2.0, float(max_distance_mm))))

        valid = [idx for idx, value in enumerate(centers) if value is not None]
        if len(valid) < 2:
            continue

        valid_t = bin_t[valid]
        center_values = np.vstack([centers[idx] for idx in valid if centers[idx] is not None])
        radius_values = np.asarray([radii[idx] for idx in valid if radii[idx] is not None], dtype=np.float64)
        centerline = np.vstack([
            np.interp(bin_t, valid_t, center_values[:, dim])
            for dim in range(3)
        ]).T
        radius_by_bin = np.interp(bin_t, valid_t, radius_values)

        comp_pad = np.ceil((float(max_distance_mm) + 2.0) / spacing).astype(np.int32)
        comp_lo = np.maximum(0, comp_coords.min(axis=0) - comp_pad)
        comp_hi = np.minimum(np.asarray(local.shape), comp_coords.max(axis=0) + comp_pad + 1)
        zz, yy, xx = np.mgrid[
            comp_lo[0]:comp_hi[0],
            comp_lo[1]:comp_hi[1],
            comp_lo[2]:comp_hi[2],
        ]
        grid = np.stack([zz, yy, xx], axis=-1).reshape(-1, 3)
        grid_phys = grid.astype(np.float64) * spacing
        grid_t = (grid_phys - center) @ axis
        bin_idx = np.searchsorted(edges, grid_t, side="right") - 1
        valid_grid = (bin_idx >= 0) & (bin_idx < n_bins)
        tube_flat = np.zeros(grid.shape[0], dtype=bool)
        grid_ids = np.flatnonzero(valid_grid)
        rel = grid_phys[grid_ids] - centerline[bin_idx[grid_ids]]
        perp = rel - np.outer(rel @ axis, axis)
        distance = np.linalg.norm(perp, axis=1)
        tube_flat[grid_ids] = distance <= radius_by_bin[bin_idx[grid_ids]]
        tube = tube_flat.reshape(tuple((comp_hi - comp_lo).tolist()))
        tube_fill[
            comp_lo[0]:comp_hi[0],
            comp_lo[1]:comp_hi[1],
            comp_lo[2]:comp_hi[2],
        ] |= tube
        components.append({
            "label": int(label_id),
            "stent_voxels": int(counts[int(label_id)]),
            "span_mm": round(span, 1),
            "tube_voxels": int(tube.sum()),
            "median_radius_mm": round(float(np.median(radius_by_bin)), 2),
        })

    local_fill |= tube_fill & ~local
    fill_candidates[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = local_fill
    if fill_candidates.any() and spacing_zyx is not None:
        distance_mm = ndi.distance_transform_edt(~tips_mask, sampling=spacing_zyx)
        fill_candidates &= distance_mm <= float(max_distance_mm)
        del distance_mm

    filled = tips_mask | fill_candidates
    info.update({
        "status": "ok",
        "closed_hole_voxels": int((local_fill & ~tube_fill).sum()),
        "tube_candidate_voxels": int(tube_fill.sum()),
        "components": components[:8],
        "candidate_voxels": int(local_fill.sum()),
        "filled_voxels": int(fill_candidates.sum()),
        "output_voxels": int(filled.sum()),
    })
    return filled, info


# =========================================================================
# 质量评估
# =========================================================================

def _pretrain_quality(mask: np.ndarray, stl_bytes: int, max_voxels: int) -> tuple[str, list[str], dict]:
    issues: list[str] = []
    voxels = int(mask.sum())
    stats = {"voxels": voxels}
    if voxels == 0:
        issues.append("empty")
    if stl_bytes > MAX_STL_BYTES:
        issues.append("stl_over_20mb")
    if voxels > max_voxels:
        issues.append("too_many_voxels")
    if ndi is not None and voxels > 0:
        lb = np.empty(mask.shape, dtype=np.int32)
        n = ndi.label(mask, output=lb)
        counts = np.bincount(lb.ravel(), minlength=n + 1)
        cc = int(np.count_nonzero(counts[1:] >= 64))
        stats["components"] = cc
        if cc > 16:
            issues.append("too_many_components")
        del lb
    return ("review" if issues else "ok", issues, stats)


def _evaluate_against_label(case, grid_size=96):
    label_nii = case.path / "mask.nii.gz"
    pretrain_nii = _pretrain_nii_path(case)
    orig_nii = case.path / "orig.nii.gz"
    if label_nii.exists() and pretrain_nii.exists() and orig_nii.exists():
        try:
            dcm = _load_nifti_as_dicomvolume(orig_nii)
            pre = _load_mask_nii(pretrain_nii, dcm.volume_hu.shape).astype(bool)
            label = _load_mask_nii(label_nii, dcm.volume_hu.shape).astype(bool)
            inter = int(np.logical_and(pre, label).sum())
            pc, lc = int(pre.sum()), int(label.sum())
            d = pc + lc
            return {
                "source": "mask.nii.gz",
                "dice": round(float((2 * inter / d) if d else 1.0), 4),
                "precision": round(float((inter / pc) if pc else 0.0), 4),
                "recall": round(float((inter / lc) if lc else 0.0), 4),
                "pretrain_voxels": pc,
                "label_voxels": lc,
                "intersection_voxels": inter,
            }
        except Exception as e:
            nii_error = str(e)
    else:
        nii_error = None

    if not case.label_stl.exists() or not case.pretrain_stl.exists():
        return None
    try:
        pre, bounds = stl_to_voxels(case.pretrain_stl, grid_size=grid_size)
        label, _ = stl_to_voxels(case.label_stl, grid_size=grid_size, bounds=bounds)
    except Exception as e:
        return {"error": str(e)}
    pm, lm = pre > 0.5, label > 0.5
    inter = int(np.logical_and(pm, lm).sum())
    pc, lc = int(pm.sum()), int(lm.sum())
    d = pc + lc
    return {
        "source": "label_stl",
        "nii_error": nii_error,
        "dice": round(float((2 * inter / d) if d else 1.0), 4),
        "precision": round(float((inter / pc) if pc else 0.0), 4),
        "recall": round(float((inter / lc) if lc else 0.0), 4),
        "pretrain_voxels": pc,
        "label_voxels": lc,
    }


def _binary_stl_triangle_count(path):
    with Path(path).open("rb") as f:
        f.seek(80)
        raw = f.read(4)
    return int(struct.unpack("<I", raw)[0]) if len(raw) == 4 else 0


# =========================================================================
# 缓存
# =========================================================================

def _tree_mtime(p):
    p = Path(p)
    if not p.exists():
        return 0.0
    latest = p.stat().st_mtime
    if p.is_file():
        return latest
    for i in p.rglob("*"):
        if i.is_file():
            latest = max(latest, i.stat().st_mtime)
    return latest


def _should_rebuild(case, meta_path, input_mtime):
    meta_path = Path(meta_path)
    if not _pretrain_nii_path(case).exists():
        return True, "missing_nii"
    if not case.pretrain_stl.exists():
        return True, "missing_stl"
    if not meta_path.exists():
        return True, "missing_meta"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return True, "invalid_meta"
    if meta.get("algorithm_version") != PRETRAIN_ALGORITHM_VERSION:
        return True, "version"
    if abs(float(meta.get("input_mtime", -1)) - float(input_mtime)) > 1e-3:
        return True, "input_changed"
    return False, "up_to_date"


def _only_dollar_patients(cases):
    return [case for case in cases if "$" in case.name]


# =========================================================================
# 主流程
# =========================================================================

def pretrain_patient(case, force: bool = False) -> PretrainResult:
    work_dir = case.path / "vkan_work"
    meta_path = work_dir / PRETRAIN_META_NAME
    work_dir.mkdir(parents=True, exist_ok=True)

    # 加载 orig.nii.gz
    orig_path = case.path / "orig.nii.gz"
    if not orig_path.exists():
        raise FileNotFoundError(f"orig.nii.gz not found: {orig_path}")

    input_mtime = max(orig_path.stat().st_mtime, _precomputed_segmentation_mtime(case))
    if not force:
        ok, reason = _should_rebuild(case, meta_path, input_mtime)
        if not ok:
            return PretrainResult(case.pretrain_stl, "reused")
    else:
        reason = "forced"

    dcm = _load_nifti_as_dicomvolume(orig_path)
    vol = dcm.volume_hu
    vol_shape = vol.shape
    nz = vol_shape[0]
    mask_cache: MaskCache = {}

    print(f"    volume: {vol_shape}, spacing: {[round(s,2) for s in dcm.spacing_zyx]}mm")

    # ==============================================================
    # Step 1: Z 轴标定（从 bone）
    # ==============================================================
    z_start, z_end, z_info = _standardize_z_from_bone(case, vol_shape, dcm.spacing_zyx, cache=mask_cache)
    print(f"    Z range: [{z_start}, {z_end}] ({z_info.get('z_range_mm', '?')}mm)")

    # ==============================================================
    # Step 2: 从门静脉分割采样 HU
    # ==============================================================
    hu_low, hu_high, hu_info = _sample_hu_from_portal_vein(vol, case, vol_shape, cache=mask_cache)
    print(f"    HU from portal vein: [{hu_low:.1f}, {hu_high:.1f}] (p50={hu_info.get('p50', '?')})")

    # ==============================================================
    # Step 3: HU 阈值分割
    # ==============================================================
    mask = _threshold_segment(vol, hu_low, hu_high, z_start, z_end)
    print(f"    after threshold: {int(mask.sum())} voxels")

    # ==============================================================
    # Step 4: 减去 bone/spleen/liver/kidney/IVC/aorta
    # ==============================================================
    mask, excl_info = _subtract_organs(mask, case, vol_shape, cache=mask_cache)
    print(f"    after organ subtraction: {int(mask.sum())} voxels "
          f"(removed {excl_info.get('voxels_removed', 0)})")

    # ==============================================================
    # Step 5: 形态学清理
    # ==============================================================
    mask = _morphological_cleanup(mask)
    print(f"    after morphology: {int(mask.sum())} voxels")

    portal_mask, portal_mask_info = _get_portal_vein_mask_fast(case, vol_shape, mask_cache)
    if (portal_mask is None or portal_mask.sum() == 0) and get_portal_vein_mask:
        portal_mask = get_portal_vein_mask(case, vol_shape)
        portal_mask_info = {"source": "totalseg_module"}

    # ==============================================================
    # Step 6: TIPS 支架（如果是 post-TIPS）
    # ==============================================================
    tips_info: dict = {"is_post_tips": bool(case.is_post_tips)}
    if case.is_post_tips:
        # 加载排除 mask 用于 TIPS（避免骨骼被当成支架）
        excl_mask, tips_excl_info = _get_tips_exclusion_mask_fast(
            case, vol_shape, cache=mask_cache
        )
        mask, tips_info = _add_tips_stent(mask, vol, z_start, z_end,
                                           exclusion_mask=excl_mask,
                                           portal_mask=portal_mask,
                                           spacing_zyx=dcm.spacing_zyx)
        tips_info["exclusion"] = tips_excl_info
        del excl_mask
        print(f"    after TIPS: {int(mask.sum())} voxels (tips={tips_info.get('tips_voxels', 0)})")

    # ==============================================================
    # Step 7: 区域生长（从可信门静脉 mask、肝脾 fallback 或 seed 点）
    # ==============================================================
    portal_quality_info = _portal_reference_quality(vol, portal_mask)
    liver_spleen_info: dict = {"enabled": False, "status": "not_needed"}
    used_fallback_reference = False

    if not case.is_post_tips and not portal_quality_info.get("reliable", False):
        fallback_reference, liver_spleen_info = _liver_spleen_portal_fallback(
            mask, vol, case, vol_shape, dcm.spacing_zyx, hu_high, mask_cache,
        )
        if fallback_reference is not None and fallback_reference.any():
            before_fallback = int(mask.sum())
            edge_hu_low = max(
                LIVER_SPLEEN_FALLBACK_MIN_HU,
                float(liver_spleen_info.get("hu_low", LIVER_SPLEEN_FALLBACK_MIN_HU))
                - LIVER_SPLEEN_FALLBACK_EDGE_MARGIN_HU,
            )
            edge_hu_high = (
                float(liver_spleen_info.get("hu_high", hu_high))
                + LIVER_SPLEEN_FALLBACK_EDGE_HIGH_MARGIN_HU
            )
            fallback_edge = (
                ndi.binary_dilation(fallback_reference, iterations=1)
                & mask
                & ~fallback_reference
                & (vol >= edge_hu_low)
                & (vol <= edge_hu_high)
            )
            mask = fallback_reference | fallback_edge
            portal_cleanup_info = {
                "enabled": True,
                "reference": "liver_spleen_fallback",
                "radius_mm": None,
                "edge_hu_low": round(float(edge_hu_low), 1),
                "edge_hu_high": round(float(edge_hu_high), 1),
                "input_voxels": before_fallback,
                "reference_voxels": int(fallback_reference.sum()),
                "edge_voxels": int(fallback_edge.sum()),
                "output_voxels": int(mask.sum()),
                "removed_voxels": before_fallback - int(mask.sum()),
                "status": "ok",
            }
            grow_info = {
                "method": "liver_spleen_fallback",
                "input_voxels": before_fallback,
                "output_voxels": int(mask.sum()),
                "removed": before_fallback - int(mask.sum()),
                "portal_mask": portal_mask_info,
            }
            used_fallback_reference = True

    if not used_fallback_reference:
        if portal_mask is not None and portal_mask.sum() > 0 and portal_quality_info.get("reliable", False):
            # 优先用可信门静脉 mask 做区域生长（比单点更稳健）
            mask, grow_info = _region_grow_from_portal_mask(
                mask, portal_mask, dcm.spacing_zyx, REGION_GROW_BRIDGE_MM,
            )
            grow_info["method"] = "portal_mask"
            grow_info["portal_mask"] = portal_mask_info
        else:
            # 退化为单点 seed
            seed_zyx, seed_info = _get_portal_seed_fast(case, vol_shape, mask_cache)
            if seed_zyx is None and get_portal_seed is not None:
                seed_zyx, seed_info = get_portal_seed(case, dcm.spacing_zyx, dcm.origin_xyz)
            if seed_zyx is not None:
                mask, grow_info = _region_grow_from_seed(mask, seed_zyx, dcm.spacing_zyx)
                grow_info["method"] = "portal_seed"
                grow_info["seed"] = seed_info
            else:
                grow_info = {
                    "method": "none",
                    "reason": "no_reliable_portal_reference",
                    "input_voxels": int(mask.sum()),
                    "output_voxels": int(mask.sum()),
                }

        print(f"    after region grow: {int(mask.sum())} voxels "
              f"(removed {grow_info.get('input_voxels', 0) - grow_info.get('output_voxels', 0)})")

        cleanup_reference = portal_mask if portal_quality_info.get("reliable", False) else None
        mask, portal_cleanup_info = _limit_to_portal_reference_neighborhood(
            mask, cleanup_reference, dcm.spacing_zyx,
            radius_mm=PORTAL_REFERENCE_CLEANUP_RADIUS_TIPS_MM if case.is_post_tips else PORTAL_REFERENCE_CLEANUP_RADIUS_MM,
        )
        if cleanup_reference is not None and portal_quality_info.get("reliable", False):
            boundary_hu_low = _portal_boundary_hu_low(hu_low, portal_quality_info)
            before_boundary = int(mask.sum())
            mask = mask & (vol >= boundary_hu_low)
            portal_cleanup_info["boundary_hu_low"] = round(float(boundary_hu_low), 1)
            portal_cleanup_info["boundary_removed_voxels"] = before_boundary - int(mask.sum())
    else:
        print(f"    after liver/spleen fallback: {int(mask.sum())} voxels "
              f"(removed {grow_info.get('removed', 0)})")

    print(f"    after portal cleanup: {int(mask.sum())} voxels "
          f"(removed {portal_cleanup_info.get('removed_voxels', 0)})")

    portal_reliable_for_merge = bool(
        portal_mask is not None and portal_mask.any() and portal_quality_info.get("reliable", False)
    )
    portal_reference_info = {
        "enabled": portal_mask is not None,
        "reliable": bool(portal_quality_info.get("reliable", False)),
        "added_voxels": 0,
    }
    if portal_reliable_for_merge:
        before_portal_union = int(mask.sum())
        portal_merge_hu_low = _portal_boundary_hu_low(hu_low, portal_quality_info)
        portal_to_merge = portal_mask.astype(bool) & (vol >= portal_merge_hu_low)
        mask = mask | portal_to_merge
        portal_reference_info = {
            "enabled": True,
            "reliable": True,
            "portal_voxels": int(portal_mask.sum()),
            "merge_hu_low": round(float(portal_merge_hu_low), 1),
            "merge_voxels": int(portal_to_merge.sum()),
            "voxels_before": before_portal_union,
            "voxels_after": int(mask.sum()),
            "added_voxels": int(mask.sum()) - before_portal_union,
        }
        print(f"    after portal reference merge: {int(mask.sum())} voxels "
              f"(added {portal_reference_info['added_voxels']})")

    if case.is_post_tips:
        tips_excl_mask, tips_final_excl_info = _get_tips_exclusion_mask_fast(case, vol_shape, cache=mask_cache)
        mask, tips_final_info = _add_tips_stent(
            mask, vol, z_start, z_end,
            exclusion_mask=tips_excl_mask,
            portal_mask=portal_mask,
            spacing_zyx=dcm.spacing_zyx,
        )
        tips_info["final_recovery"] = tips_final_info
        tips_info["final_recovery_exclusion"] = tips_final_excl_info
        del tips_excl_mask
        print(f"    after final TIPS recovery: {int(mask.sum())} voxels "
              f"(tips={tips_final_info.get('tips_voxels', 0)})")
    del portal_mask

    mask, final_component_info = _keep_largest_connected_component(mask)
    print(f"    after largest component: {int(mask.sum())} voxels "
          f"(removed {final_component_info.get('removed_voxels', 0)})")

    # ==============================================================
    # Step 8: 输出
    # ==============================================================
    import nibabel as nib

    np.save(work_dir / "pretrain_mask.npy", mask.astype(np.uint8))
    nii_path = _save_pretrain_nifti(mask, orig_path, _pretrain_nii_path(case))
    out_path = zyx_mask_to_stl(mask, nib.load(str(nii_path)).affine, case.pretrain_stl, name="pretrain")
    stl_bytes = int(out_path.stat().st_size)
    target = TARGET_VOXELS_TIPS if case.is_post_tips else TARGET_VOXELS
    quality, issues, qstats = _pretrain_quality(mask, stl_bytes, target)
    eval_metrics = _evaluate_against_label(case, grid_size=96)

    meta = {
        "algorithm_version": PRETRAIN_ALGORITHM_VERSION,
        "status_reason": reason,
        "input": str(orig_path),
        "input_mtime": input_mtime,
        "is_post_tips": bool(case.is_post_tips),
        "pretrain_nii": str(nii_path),
        "pretrain_stl": str(out_path),
        "z_standardization": z_info,
        "hu_sampling": hu_info,
        "organ_subtraction": excl_info,
        "morphology_cleanup": {
            "method": "opening_supported_boundary_restore",
            "min_core_neighbors": OPENING_RESTORE_MIN_CORE_NEIGHBORS,
        },
        "tips": tips_info,
        "final_component_filter": final_component_info,
        "region_grow": grow_info,
        "portal_reference_quality": portal_quality_info,
        "portal_reference_cleanup": portal_cleanup_info,
        "portal_reference_merge": portal_reference_info,
        "liver_spleen_fallback": liver_spleen_info,
        "pretrain_quality": quality,
        "quality_issues": issues,
        "quality_stats": qstats,
        "pretrain_vessel_eval": eval_metrics,
        "volume_shape_zyx": list(vol_shape),
        "spacing_zyx": list(dcm.spacing_zyx),
        "origin_xyz": list(dcm.origin_xyz),
        "mask_voxels": int(mask.sum()),
        "stl_bytes": stl_bytes,
        "stl_triangles": _binary_stl_triangle_count(out_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"    → {quality} | {int(mask.sum())} voxels | {stl_bytes // 1024}KB")
    if eval_metrics and "dice" in eval_metrics:
        print(f"    → dice={eval_metrics['dice']:.3f} precision={eval_metrics['precision']:.3f} "
              f"recall={eval_metrics['recall']:.3f}")

    return PretrainResult(out_path, "review" if quality == "review" else "wrote")


def coarse_segment_patient(case, client=None, force=False):
    return pretrain_patient(case, force=force).path


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="precomputed NIfTI-driven portal vein extraction.",
    )
    parser.add_argument("--data_root", default=r"F:\PCG data\dataset\test4all_sample")
    parser.add_argument("--patient", default=None)
    parser.add_argument("--force", default=True)
    parser.add_argument("--skip_existing_pretrain", action="store_true")
    parser.add_argument(
        "--only_dollar_patients",
        action="store_true",
        help='Only preprocess patients whose folder name contains "$".',
    )
    args = parser.parse_args()

    if args.patient:
        patient_path = Path(args.patient)
        if patient_path.exists():
            cases = discover_patients(patient_path)
        else:
            cases = [c for c in discover_patients(args.data_root) if c.name == args.patient]
    else:
        cases = discover_patients(args.data_root)
    if args.patient:
        cases = [c for c in cases if c.name == Path(args.patient).name or Path(args.patient).exists()]
    if args.only_dollar_patients:
        cases = _only_dollar_patients(cases)
    print(f"[{PRETRAIN_ALGORITHM_VERSION}] {len(cases)} patients")
    for case in cases:
        print(f"[{PRETRAIN_ALGORITHM_VERSION}] {case.name}:")
        try:
            if args.skip_existing_pretrain and case.pretrain_stl.exists() and _pretrain_nii_path(case).exists():
                print("  result: skipped_existing")
                continue
            r = pretrain_patient(case, force=args.force and not args.skip_existing_pretrain)
            print(f"  result: {r.status}")
        except Exception as e:
            print(f"  FAILED: {e}")


if __name__ == "__main__":
    main()
