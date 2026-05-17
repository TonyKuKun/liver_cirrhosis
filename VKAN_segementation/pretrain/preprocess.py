"""pretrain v5 — 基于 TotalSegmentator 的门静脉预分割。

流程（极简，不依赖 LLM 做初始分割）：
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
    from ..utils.common import DicomVolume, GemmaClient, discover_patients, mask_to_stl, stl_to_voxels
except (ImportError, ValueError):
    try:
        from VKAN_segementation.utils.common import DicomVolume, GemmaClient, discover_patients, mask_to_stl, stl_to_voxels
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from utils.common import DicomVolume, GemmaClient, discover_patients, mask_to_stl, stl_to_voxels

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
            from VKAN_segementation.totalseg import (
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


PRETRAIN_ALGORITHM_VERSION = "2026-05-16-v6e-tips-high-hu"
PRETRAIN_META_NAME = "pretrain_meta.json"
PRETRAIN_NII_NAME = "pretrain.nii.gz"
MAX_STL_BYTES = 20_000 * 1024
TARGET_VOXELS = 420_000
TARGET_VOXELS_TIPS = 330_000
REGION_GROW_BRIDGE_MM = 8.0
REGION_GROW_MAX_SEED_SNAP_MM = 30.0
HU_MARGIN = 5.0  # 门静脉 HU 采样后上下各扩展的边距
HU_LOW_FLOOR = 75.0
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
BONE_NAMES = list(BONE_LABELS.keys()) if BONE_LABELS else DEFAULT_BONE_NAMES
EXCLUSION_NAMES = ("bone_all", "spleen", "kidney_left", "kidney_right", "inferior_vena_cava", "aorta")
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


def _load_mask_nii(
    path: Path,
    target_shape: tuple[int, int, int] | None = None,
    cache: MaskCache | None = None,
) -> np.ndarray | None:
    cache_key = ("path", str(path), target_shape)
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    try:
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

        for path in _structure_candidate_paths(case, "bone_all"):
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

    for name in EXCLUSION_NAMES:
        mask, mask_info = _load_precomputed_structure_mask(case, name, vol_shape, cache)
        if mask is None:
            continue
        dilate = dilate_bone if name == "bone_all" else dilate_organ
        if ndi is not None and dilate > 0:
            mask = ndi.binary_dilation(mask, iterations=dilate)
        exclusion |= mask
        info["loaded"].append({"name": name, "files": mask_info.get("loaded", [])})

    info["total_excluded_voxels"] = int(exclusion.sum())
    info["status"] = "ok" if info["loaded"] else "empty"
    return exclusion, info


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
    nib.save(nib.Nifti1Image(data, ref.affine, header), str(out_path))
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
    portal_margin_mm = max(margin_mm, 45.0 if getattr(case, "is_post_tips", False) else 35.0)
    z_start, z_end, info = _z_range_from_portal_nii(case, vol_shape, spacing_zyx, portal_margin_mm, cache)
    if z_start is not None and z_end is not None:
        return z_start, z_end, info

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


def _morphological_cleanup(mask: np.ndarray) -> np.ndarray:
    """形态学清理：opening 去噪 + closing 补小洞。"""
    if ndi is None:
        return mask
    mask = ndi.binary_opening(mask, iterations=1)
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
) -> tuple[np.ndarray, dict]:
    """单独处理 TIPS 支架（高 HU 通道）。"""
    info: dict = {}
    z0 = max(0, z_start)
    z1 = min(vol.shape[0], z_end + 1)

    tips_mask = np.zeros(vol.shape, dtype=bool)
    tips_mask[z0:z1] = (vol[z0:z1] >= tips_hu_low) & (vol[z0:z1] <= tips_hu_high)

    if exclusion_mask is not None:
        tips_mask = tips_mask & ~exclusion_mask

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

    tips_voxels = int(tips_mask.sum())
    info["tips_voxels"] = tips_voxels

    if tips_voxels > 0:
        combined = mask | tips_mask
        info["combined_voxels"] = int(combined.sum())
        return combined, info

    return mask, info


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


# =========================================================================
# 主流程
# =========================================================================

def pretrain_patient(case, client: GemmaClient | None = None, force: bool = False) -> PretrainResult:
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

    # ==============================================================
    # Step 6: TIPS 支架（如果是 post-TIPS）
    # ==============================================================
    tips_info: dict = {"is_post_tips": bool(case.is_post_tips)}
    if case.is_post_tips:
        # 加载排除 mask 用于 TIPS（避免骨骼被当成支架）
        excl_mask, tips_excl_info = _get_exclusion_mask_fast(
            case, vol_shape, dilate_bone=2, dilate_organ=0, cache=mask_cache
        )
        if not tips_excl_info.get("loaded") and get_exclusion_mask is not None:
            excl_mask, tips_excl_info = get_exclusion_mask(case, vol_shape, dilate_bone=2, dilate_organ=0)
            tips_excl_info["source"] = "totalseg_module"
        mask, tips_info = _add_tips_stent(mask, vol, z_start, z_end,
                                           exclusion_mask=excl_mask)
        tips_info["exclusion"] = tips_excl_info
        del excl_mask
        print(f"    after TIPS: {int(mask.sum())} voxels (tips={tips_info.get('tips_voxels', 0)})")

    # ==============================================================
    # Step 7: 区域生长（从门静脉 mask 或 seed 点）
    # ==============================================================
    portal_mask, portal_mask_info = _get_portal_vein_mask_fast(case, vol_shape, mask_cache)
    if (portal_mask is None or portal_mask.sum() == 0) and get_portal_vein_mask:
        portal_mask = get_portal_vein_mask(case, vol_shape)
        portal_mask_info = {"source": "totalseg_module"}

    if portal_mask is not None and portal_mask.sum() > 0:
        # 优先用门静脉 mask 做区域生长（比单点更稳健）
        mask, grow_info = _region_grow_from_portal_mask(
            mask, portal_mask, dcm.spacing_zyx, REGION_GROW_BRIDGE_MM,
        )
        grow_info["method"] = "portal_mask"
        grow_info["portal_mask"] = portal_mask_info
        del portal_mask
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
            grow_info = {"method": "none", "reason": "no_portal_reference"}

    print(f"    after region grow: {int(mask.sum())} voxels "
          f"(removed {grow_info.get('input_voxels', 0) - grow_info.get('output_voxels', 0)})")

    # ==============================================================
    # Step 8: 输出
    # ==============================================================
    np.save(work_dir / "pretrain_mask.npy", mask.astype(np.uint8))
    nii_path = _save_pretrain_nifti(mask, orig_path, _pretrain_nii_path(case))
    out_path = mask_to_stl(mask, dcm.spacing_zyx, case.pretrain_stl, origin_xyz=dcm.origin_xyz)
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
        "tips": tips_info,
        "region_grow": grow_info,
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
    return pretrain_patient(case, client=client, force=force).path


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="v6: precomputed NIfTI-driven portal vein extraction.",
    )
    parser.add_argument("--data_root", default=r"F:\PCG data\dataset\test4all_sample")
    parser.add_argument("--patient", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip_existing_pretrain", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--api_base_url", default=None)
    args = parser.parse_args()

    cases = discover_patients(args.data_root)
    if args.patient:
        cases = [c for c in cases if c.name == args.patient]
    print(f"[v6] {len(cases)} patients")
    for case in cases:
        print(f"[v6] {case.name}:")
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
