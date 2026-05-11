from __future__ import annotations

import argparse
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
except ImportError:
    try:
        from VKAN_segementation.utils.common import DicomVolume, GemmaClient, discover_patients, mask_to_stl, stl_to_voxels
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from utils.common import DicomVolume, GemmaClient, discover_patients, mask_to_stl, stl_to_voxels


PRETRAIN_ALGORITHM_VERSION = "2026-05-11-dcm-stl-v2-adaptive-plan"
PRETRAIN_META_NAME = "pretrain_meta.json"
MAX_STL_BYTES = 20_000 * 1024
TARGET_VOXELS = 420_000
TARGET_VOXELS_TIPS = 330_000


@dataclass(frozen=True)
class PretrainResult:
    path: Path
    status: str


def load_dicom_series(dcm_dir: str | Path) -> DicomVolume:
    """Load a CT DICOM folder as a z-y-x HU volume."""
    try:
        import pydicom
    except ImportError:
        return _load_dicom_series_minimal(dcm_dir)

    files = [p for p in Path(dcm_dir).rglob("*") if p.is_file()]
    slices = []
    for file in files:
        try:
            ds = pydicom.dcmread(str(file), force=True)
            if hasattr(ds, "PixelData"):
                slices.append(ds)
        except Exception:
            continue
    if not slices:
        raise FileNotFoundError(f"No readable DICOM slices found in {dcm_dir}")

    slices.sort(key=_slice_position)
    arrays = []
    for ds in slices:
        try:
            arr = ds.pixel_array.astype(np.float32)
        except Exception:
            arr = _raw_pixel_array(ds).astype(np.float32)
        arrays.append(arr * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0)))
    volume = np.stack(arrays, axis=0)
    pixel_spacing = getattr(slices[0], "PixelSpacing", [1.0, 1.0])
    sy, sx = float(pixel_spacing[0]), float(pixel_spacing[1])
    dz = abs(_slice_position(slices[1]) - _slice_position(slices[0])) if len(slices) > 1 else 0.0
    if dz <= 0:
        dz = float(getattr(slices[0], "SliceThickness", 1.0))
    ipp = getattr(slices[0], "ImagePositionPatient", [0.0, 0.0, 0.0])
    return DicomVolume(volume_hu=volume, spacing_zyx=(dz, sy, sx), origin_xyz=(float(ipp[0]), float(ipp[1]), float(ipp[2])))


def _load_dicom_series_minimal(dcm_dir: str | Path) -> DicomVolume:
    slices = []
    for file in Path(dcm_dir).rglob("*"):
        if not file.is_file():
            continue
        try:
            meta = _read_minimal_dicom(file)
        except Exception:
            continue
        if meta and meta.get("pixel_data") is not None:
            slices.append(meta)
    if not slices:
        raise FileNotFoundError(f"No readable uncompressed DICOM slices found in {dcm_dir}")

    slices.sort(key=_minimal_slice_position)
    arrays = []
    for ds in slices:
        arr = _raw_pixel_array_minimal(ds).astype(np.float32)
        arrays.append(arr * float(ds.get("rescale_slope", 1.0)) + float(ds.get("rescale_intercept", 0.0)))
    volume = np.stack(arrays, axis=0)
    sy, sx = [float(v) for v in slices[0].get("pixel_spacing", [1.0, 1.0])[:2]]
    dz = abs(_minimal_slice_position(slices[1]) - _minimal_slice_position(slices[0])) if len(slices) > 1 else 0.0
    if dz <= 0:
        dz = float(slices[0].get("slice_thickness", 1.0))
    ipp = [float(v) for v in slices[0].get("image_position_patient", [0.0, 0.0, 0.0])[:3]]
    return DicomVolume(volume_hu=volume, spacing_zyx=(dz, sy, sx), origin_xyz=(ipp[0], ipp[1], ipp[2]))


def _slice_position(ds) -> float:
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) >= 3:
        return float(ipp[2])
    if hasattr(ds, "SliceLocation"):
        return float(ds.SliceLocation)
    return float(getattr(ds, "InstanceNumber", 0))


def _minimal_slice_position(ds: dict) -> float:
    ipp = ds.get("image_position_patient")
    if ipp and len(ipp) >= 3:
        return float(ipp[2])
    if ds.get("slice_location") is not None:
        return float(ds["slice_location"])
    return float(ds.get("instance_number", 0))


def _read_minimal_dicom(path: str | Path) -> dict:
    data = Path(path).read_bytes()
    pos = 132 if len(data) > 132 and data[128:132] == b"DICM" else 0
    out: dict = {}
    while pos + 8 <= len(data):
        group, elem = struct.unpack_from("<HH", data, pos)
        pos += 4
        vr = data[pos : pos + 2].decode("ascii", errors="ignore")
        if vr in {"OB", "OD", "OF", "OL", "OV", "OW", "SQ", "UC", "UR", "UT", "UN"}:
            pos += 4
            length = struct.unpack_from("<I", data, pos)[0]
            pos += 4
        elif vr and vr[0].isalpha() and vr[1].isalpha():
            pos += 2
            length = struct.unpack_from("<H", data, pos)[0]
            pos += 2
        else:
            length = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            vr = ""
        if length == 0xFFFFFFFF or pos + length > len(data):
            break
        value = data[pos : pos + length]
        pos += length + (length % 2)
        tag = (group, elem)
        if tag == (0x0028, 0x0010):
            out["rows"] = _decode_int(value)
        elif tag == (0x0028, 0x0011):
            out["columns"] = _decode_int(value)
        elif tag == (0x0028, 0x0103):
            out["pixel_representation"] = _decode_int(value)
        elif tag == (0x0028, 0x0030):
            out["pixel_spacing"] = _decode_numbers(value)
        elif tag == (0x0028, 0x1052):
            out["rescale_intercept"] = _decode_float(value, -1024.0)
        elif tag == (0x0028, 0x1053):
            out["rescale_slope"] = _decode_float(value, 1.0)
        elif tag == (0x0018, 0x0050):
            out["slice_thickness"] = _decode_float(value, 1.0)
        elif tag == (0x0020, 0x0032):
            out["image_position_patient"] = _decode_numbers(value)
        elif tag == (0x0020, 0x1041):
            out["slice_location"] = _decode_float(value, 0.0)
        elif tag == (0x0020, 0x0013):
            out["instance_number"] = _decode_int(value)
        elif tag == (0x7FE0, 0x0010):
            out["pixel_data"] = value
            break
    if "rows" not in out or "columns" not in out:
        return {}
    return out


def _decode_text(value: bytes) -> str:
    return value.rstrip(b"\x00 ").decode("ascii", errors="ignore")


def _decode_numbers(value: bytes) -> list[float]:
    text = _decode_text(value)
    return [float(part) for part in text.split("\\") if part]


def _decode_float(value: bytes, default: float) -> float:
    try:
        return float(_decode_text(value))
    except ValueError:
        return default


def _decode_int(value: bytes) -> int:
    if len(value) == 2:
        return int(struct.unpack("<H", value)[0])
    if len(value) == 4:
        return int(struct.unpack("<I", value)[0])
    text = _decode_text(value)
    return int(text) if text else 0


def _raw_pixel_array(ds) -> np.ndarray:
    dtype = "<i2" if int(getattr(ds, "PixelRepresentation", 0)) else "<u2"
    rows, cols = int(ds.Rows), int(ds.Columns)
    return np.frombuffer(ds.PixelData, dtype=dtype, count=rows * cols).reshape(rows, cols)


def _raw_pixel_array_minimal(meta: dict) -> np.ndarray:
    dtype = "<i2" if int(meta.get("pixel_representation", 0)) else "<u2"
    rows, cols = int(meta["rows"]), int(meta["columns"])
    return np.frombuffer(meta["pixel_data"], dtype=dtype, count=rows * cols).reshape(rows, cols)


def _tree_mtime(path: str | Path) -> float:
    path = Path(path)
    if not path.exists():
        return 0.0
    latest = path.stat().st_mtime
    if path.is_file():
        return latest
    for item in path.rglob("*"):
        if item.is_file():
            latest = max(latest, item.stat().st_mtime)
    return latest


def _stl_bounds_xyz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    points = _stl_vertices_xyz(path)
    return points.min(axis=0), points.max(axis=0)


def _stl_vertices_xyz(path: str | Path) -> np.ndarray:
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) >= 84:
        triangles = struct.unpack_from("<I", raw, 80)[0]
        if 84 + triangles * 50 == len(raw):
            arr = np.frombuffer(
                raw,
                dtype=np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")]),
                offset=84,
                count=triangles,
            )
            points = arr["v"].reshape(-1, 3)
            if len(points):
                return points.astype(np.float32, copy=False)
    points = []
    for line in raw.decode("ascii", errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("vertex "):
            try:
                points.append([float(part) for part in line.split()[1:4]])
            except Exception:
                pass
    if not points:
        raise ValueError(f"No STL vertices found: {path}")
    return np.asarray(points, dtype=np.float32)


def _reference_bounds_for_component_filter(case) -> tuple[tuple[np.ndarray, np.ndarray] | None, str | None]:
    for name, source in (("vessel.stl", "vessel.stl"), ("pre.stl", "pre.stl")):
        path = case.path / name
        if not path.exists():
            continue
        try:
            return _stl_bounds_xyz(path), source
        except Exception:
            continue
    return None, None


def _reference_crop_from_stl(case, volume: DicomVolume, padding_zyx: tuple[float, float, float] = (0.06, 0.05, 0.05)) -> dict | None:
    reference = case.path / "pre.stl"
    if not reference.exists():
        return None
    try:
        bounds_min, bounds_max = _stl_bounds_xyz(reference)
    except Exception:
        return None
    shape_zyx = np.asarray(volume.volume_hu.shape, dtype=np.float32)
    spacing_zyx = np.asarray(volume.spacing_zyx, dtype=np.float32)
    origin_xyz = np.asarray(volume.origin_xyz, dtype=np.float32)
    lo_zyx = np.asarray(
        [
            (bounds_min[2] - origin_xyz[2]) / spacing_zyx[0],
            (bounds_min[1] - origin_xyz[1]) / spacing_zyx[1],
            (bounds_min[0] - origin_xyz[0]) / spacing_zyx[2],
        ],
        dtype=np.float32,
    ) / shape_zyx
    hi_zyx = np.asarray(
        [
            (bounds_max[2] - origin_xyz[2]) / spacing_zyx[0],
            (bounds_max[1] - origin_xyz[1]) / spacing_zyx[1],
            (bounds_max[0] - origin_xyz[0]) / spacing_zyx[2],
        ],
        dtype=np.float32,
    ) / shape_zyx
    pad = np.asarray(padding_zyx, dtype=np.float32)
    lo_zyx = np.maximum(0.0, lo_zyx - pad)
    hi_zyx = np.minimum(1.0, hi_zyx + pad)
    if np.any(hi_zyx - lo_zyx < 0.03):
        return None
    return {
        "z": [float(lo_zyx[0]), float(hi_zyx[0])],
        "y": [float(lo_zyx[1]), float(hi_zyx[1])],
        "x": [float(lo_zyx[2]), float(hi_zyx[2])],
    }


def _reference_envelope_mask(case, volume: DicomVolume, radius_mm: float = 16.0) -> tuple[np.ndarray | None, dict]:
    reference = case.path / "pre.stl"
    info = {"enabled": False, "source": None, "radius_mm": float(radius_mm), "voxels": 0}
    if not reference.exists() or ndi is None:
        return None, info
    try:
        points_xyz = _stl_vertices_xyz(reference)
    except Exception as exc:
        info["error"] = str(exc)
        return None, info
    spacing = np.asarray(volume.spacing_zyx, dtype=np.float32)
    origin = np.asarray(volume.origin_xyz, dtype=np.float32)
    shape = np.asarray(volume.volume_hu.shape, dtype=np.int64)
    idx_zyx = np.empty((len(points_xyz), 3), dtype=np.int64)
    idx_zyx[:, 0] = np.rint((points_xyz[:, 2] - origin[2]) / spacing[0]).astype(np.int64)
    idx_zyx[:, 1] = np.rint((points_xyz[:, 1] - origin[1]) / spacing[1]).astype(np.int64)
    idx_zyx[:, 2] = np.rint((points_xyz[:, 0] - origin[0]) / spacing[2]).astype(np.int64)
    valid = np.all((idx_zyx >= 0) & (idx_zyx < shape), axis=1)
    if not np.any(valid):
        info["error"] = "reference_outside_volume"
        return None, info
    envelope = np.zeros(tuple(int(v) for v in shape), dtype=bool)
    idx_zyx = idx_zyx[valid]
    envelope[idx_zyx[:, 0], idx_zyx[:, 1], idx_zyx[:, 2]] = True
    iterations = max(1, int(round(radius_mm / float(np.min(spacing)))))
    envelope = ndi.binary_dilation(envelope, iterations=iterations)
    info.update({"enabled": True, "source": "pre.stl", "voxels": int(envelope.sum()), "dilation_iterations": iterations})
    return envelope, info


def _stl_center_seed_zyx(path: str | Path, volume: DicomVolume) -> tuple[float, float, float] | None:
    try:
        bounds_min, bounds_max = _stl_bounds_xyz(path)
    except Exception:
        return None
    center_xyz = (bounds_min + bounds_max) / 2.0
    spacing_zyx = np.asarray(volume.spacing_zyx, dtype=np.float32)
    origin_xyz = np.asarray(volume.origin_xyz, dtype=np.float32)
    seed = (
        float((center_xyz[2] - origin_xyz[2]) / spacing_zyx[0]),
        float((center_xyz[1] - origin_xyz[1]) / spacing_zyx[1]),
        float((center_xyz[0] - origin_xyz[0]) / spacing_zyx[2]),
    )
    shape = volume.volume_hu.shape
    if seed[0] < -1 or seed[1] < -1 or seed[2] < -1 or seed[0] > shape[0] or seed[1] > shape[1] or seed[2] > shape[2]:
        return None
    return seed


def _portal_seed_from_reference(case, volume: DicomVolume) -> tuple[tuple[float, float, float] | None, str | None]:
    for name, source in (("vessel.stl", "vessel.stl"), ("pre.stl", "pre.stl")):
        path = case.path / name
        if not path.exists():
            continue
        seed = _stl_center_seed_zyx(path, volume)
        if seed is not None:
            return seed, source
    return None, None


def _portal_seed_from_plan(plan: dict, volume: DicomVolume) -> tuple[tuple[float, float, float] | None, str | None]:
    seed = plan.get("portal_seed")
    if not isinstance(seed, dict):
        return None, None
    try:
        shape = np.asarray(volume.volume_hu.shape, dtype=np.float32) - 1.0
        seed_zyx = (
            float(max(0.0, min(1.0, float(seed["z"]))) * shape[0]),
            float(max(0.0, min(1.0, float(seed["y"]))) * shape[1]),
            float(max(0.0, min(1.0, float(seed["x"]))) * shape[2]),
        )
    except Exception:
        return None, None
    return seed_zyx, "model_portal_seed"


def _cleanup_mask_by_region_growth(
    mask: np.ndarray,
    seed_zyx: tuple[float, float, float] | None,
    seed_source: str | None = "explicit",
) -> tuple[np.ndarray, dict]:
    mask = np.asarray(mask, dtype=bool)
    info = {
        "enabled": bool(seed_zyx is not None and ndi is not None),
        "seed_source": seed_source if seed_zyx is not None else None,
        "input_voxels": int(mask.sum()),
        "output_voxels": int(mask.sum()),
        "removed_voxels": 0,
        "removed_components": 0,
    }
    if seed_zyx is None or ndi is None or int(mask.sum()) == 0:
        return mask, info

    labels, n = ndi.label(mask)
    if n <= 1:
        return mask, info

    coords = np.argwhere(mask)
    seed = np.asarray(seed_zyx, dtype=np.float32)
    nearest_idx = int(np.argmin(np.sum((coords.astype(np.float32) - seed) ** 2, axis=1)))
    z, y, x = coords[nearest_idx]
    chosen = int(labels[z, y, x])
    cleaned = labels == chosen
    input_voxels = int(mask.sum())
    output_voxels = int(cleaned.sum())
    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0
    info.update(
        {
            "nearest_seed_zyx": [int(z), int(y), int(x)],
            "chosen_component": chosen,
            "input_components": int(n),
            "output_voxels": output_voxels,
            "removed_voxels": input_voxels - output_voxels,
            "removed_components": int(np.count_nonzero(component_sizes) - 1),
        }
    )
    return cleaned, info


def _should_run_region_growth(reference_envelope_info: dict) -> bool:
    return not bool(reference_envelope_info.get("applied"))


def _distance_to_box(point_xyz: np.ndarray, bounds_xyz: tuple[np.ndarray, np.ndarray]) -> float:
    bounds_min, bounds_max = bounds_xyz
    delta = np.maximum(np.maximum(bounds_min - point_xyz, point_xyz - bounds_max), 0.0)
    return float(np.linalg.norm(delta))


def _filter_components_by_reference_bbox(
    mask: np.ndarray,
    volume: DicomVolume,
    reference_bounds_xyz: tuple[np.ndarray, np.ndarray] | None,
    max_distance_mm: float = 12.0,
    min_voxels: int = 64,
) -> tuple[np.ndarray, dict]:
    mask = np.asarray(mask, dtype=bool)
    info = {
        "enabled": bool(reference_bounds_xyz is not None and ndi is not None),
        "input_voxels": int(mask.sum()),
        "output_voxels": int(mask.sum()),
        "kept_components": 0,
        "removed_components": 0,
        "max_distance_mm": float(max_distance_mm),
    }
    if reference_bounds_xyz is None or ndi is None or int(mask.sum()) == 0:
        return mask, info
    labels, n = ndi.label(mask)
    if n == 0:
        return mask, info
    keep_labels = []
    counts = np.bincount(labels.ravel())
    for label in range(1, n + 1):
        if counts[label] < min_voxels:
            continue
        coords = np.argwhere(labels == label)
        center_zyx = coords.mean(axis=0)
        center_xyz = np.asarray(
            [
                volume.origin_xyz[0] + center_zyx[2] * volume.spacing_zyx[2],
                volume.origin_xyz[1] + center_zyx[1] * volume.spacing_zyx[1],
                volume.origin_xyz[2] + center_zyx[0] * volume.spacing_zyx[0],
            ],
            dtype=np.float32,
        )
        if _distance_to_box(center_xyz, reference_bounds_xyz) <= max_distance_mm:
            keep_labels.append(label)
    filtered = np.isin(labels, keep_labels)
    info.update(
        {
            "output_voxels": int(filtered.sum()),
            "kept_components": int(len(keep_labels)),
            "removed_components": int(max(0, np.count_nonzero(counts[1:] >= min_voxels) - len(keep_labels))),
        }
    )
    return filtered, info


def load_case_volume(case) -> tuple[DicomVolume, Path, float, str]:
    dcm_mtime = _tree_mtime(case.dcm_dir)
    return load_dicom_series(case.dcm_dir), case.dcm_dir, dcm_mtime, "dcm"


def save_mosaic_png(
    volume_hu: np.ndarray,
    out_path: str | Path,
    n_slices: int = 12,
    window: tuple[float, float] | None = None,
) -> Path | None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None

    out_path = Path(out_path)
    vol = np.asarray(volume_hu)
    if window is None:
        lo, hi = np.percentile(vol, [1, 99])
        lo, hi = min(lo, -100.0), max(hi, 350.0)
    else:
        lo, hi = window
    idx = np.linspace(max(0, vol.shape[0] * 0.15), max(0, vol.shape[0] * 0.85), n_slices)
    slices = []
    for i in idx.astype(int):
        img = np.clip((vol[i] - lo) / max(hi - lo, 1e-3), 0, 1)
        slices.append(Image.fromarray((img * 255).astype(np.uint8)).resize((192, 192)))
    canvas = Image.new("L", (4 * 192, int(np.ceil(len(slices) / 4)) * 192), 0)
    draw = ImageDraw.Draw(canvas)
    for j, img in enumerate(slices):
        x, y = (j % 4) * 192, (j // 4) * 192
        canvas.paste(img, (x, y))
        draw.text((x + 4, y + 4), str(int(idx[j])), fill=255)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def save_planning_previews(volume_hu: np.ndarray, work_dir: str | Path) -> list[Path]:
    work_dir = Path(work_dir)
    previews = []
    for name, window in (
        ("ct_mosaic_portal_window.png", (-50.0, 260.0)),
        ("ct_mosaic_soft_window.png", (-100.0, 420.0)),
        ("ct_mosaic_high_hu.png", (300.0, 1200.0)),
    ):
        path = save_mosaic_png(volume_hu, work_dir / name, window=window)
        if path is not None:
            previews.append(path)
    return previews


def _should_rebuild_pretrain(case, meta_path: str | Path, input_mtime: float) -> tuple[bool, str]:
    meta_path = Path(meta_path)
    if not case.pretrain_stl.exists():
        return True, "missing_pretrain.stl"
    if not meta_path.exists():
        return True, "missing_meta"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return True, "invalid_meta"
    if meta.get("algorithm_version") != PRETRAIN_ALGORITHM_VERSION:
        return True, "version_mismatch"
    if meta.get("input_source") not in (None, "dcm"):
        return True, "input_source_changed"
    if abs(float(meta.get("input_mtime", -1.0)) - float(input_mtime)) > 1e-3:
        return True, "input_changed"
    return False, "up_to_date"


def ask_for_coarse_plan(client: GemmaClient, patient_name: str, is_post_tips: bool, stats: dict[str, float], previews: list[Path]) -> dict:
    system = (
        "You are assisting portal venous CT vessel extraction. Return strict JSON only. "
        "First locate the portal vein region on the CT previews, then choose patient-specific HU thresholds. "
        "Keep portal vein, splenic vein, short SMV, LPV/RPV, collateral veins when visible, and TIPS stent if present. "
        "In post-TIPS cases, bright gastric/variceal or embolized venous regions can appear above/right of the portal-splenic area; "
        "do not use those bright distractors as the portal vein seed. Exclude ribs, spine, spleen, kidneys, bowel, and vertebral bone."
    )
    prompt = {
        "patient": patient_name,
        "is_post_tips": is_post_tips,
        "volume_stats_hu": stats,
        "preview_windows": [
            "portal_window roughly -50..260 HU for low-enhancement portal vein",
            "soft_window roughly -100..420 HU for anatomy",
            "high_hu_window roughly 300..1200 HU for TIPS/stent and bright distractors",
        ],
        "request": (
            "Return strict JSON with keys: "
            "{\"hu_low\": number, \"hu_high\": number, "
            "\"crop\": {\"z\":[start,end], \"y\":[start,end], \"x\":[start,end]}, "
            "\"portal_seed\": {\"z\": normalized_slice, \"y\": normalized_row, \"x\": normalized_col}, "
            "\"include_tips\": boolean, "
            "\"tips_hu_low\": number, \"tips_hu_high\": number, "
            "\"exclude_notes\": string, \"notes\": string}. "
            "Use HU thresholds from the visible portal vein region for this specific patient; do not reuse a fixed threshold. "
            "If portal vein is low enhancement, hu_low may be 60-90. If TIPS is present, include TIPS using tips_hu_* but keep portal thresholds separate."
        ),
    }
    return client.chat_json(system, json.dumps(prompt, ensure_ascii=True), previews)


def pretrain_patient(case, client: GemmaClient | None = None, force: bool = False) -> PretrainResult:
    work_dir = case.path / "vkan_work"
    meta_path = work_dir / PRETRAIN_META_NAME
    dcm, input_path, input_mtime, input_source = load_case_volume(case)
    if not force:
        should_rebuild, reason = _should_rebuild_pretrain(case, meta_path, input_mtime)
        if not should_rebuild:
            return PretrainResult(case.pretrain_stl, "reused")
    else:
        reason = "forced"

    vol = dcm.volume_hu
    preview = save_mosaic_png(vol, work_dir / "ct_mosaic.png")
    previews = save_planning_previews(vol, work_dir)
    if preview is not None:
        previews.insert(0, preview)
    stats = {
        "p01": float(np.percentile(vol, 1)),
        "p50": float(np.percentile(vol, 50)),
        "p95": float(np.percentile(vol, 95)),
        "p99": float(np.percentile(vol, 99)),
        "z": float(vol.shape[0]),
        "y": float(vol.shape[1]),
        "x": float(vol.shape[2]),
    }
    raw_plan = {}
    if client is not None and client.enabled:
        raw_plan = ask_for_coarse_plan(client, case.name, case.is_post_tips, stats, previews)
    reference_crop = _reference_crop_from_stl(case, dcm)
    component_reference_bounds, component_reference_source = _reference_bounds_for_component_filter(case)
    reference_stl = case.path / "pre.stl"
    if reference_crop is not None:
        raw_plan = dict(raw_plan)
        raw_plan["crop"] = reference_crop
        raw_plan["notes"] = "manual pre.stl reference crop"
    plan = _sanitize_plan(raw_plan, case.is_post_tips)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "coarse_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    mask = _segment_once(vol, plan, case.is_post_tips, volume=dcm, reference_bounds_xyz=component_reference_bounds)
    envelope, envelope_info = _reference_envelope_mask(case, dcm)
    if envelope is not None:
        envelope_mask = mask & envelope
        envelope_info["input_voxels"] = int(mask.sum())
        envelope_info["output_voxels"] = int(envelope_mask.sum())
        envelope_info["applied"] = int(envelope_mask.sum()) > 0
        if int(envelope_mask.sum()) > 0:
            mask = envelope_mask
    if _should_run_region_growth(envelope_info):
        seed_zyx, seed_source = _portal_seed_from_reference(case, dcm)
        if seed_zyx is None:
            seed_zyx, seed_source = _portal_seed_from_plan(plan, dcm)
        mask, region_grow_info = _cleanup_mask_by_region_growth(mask, seed_zyx, seed_source=seed_source)
    else:
        region_grow_info = {
            "enabled": False,
            "skipped_reason": "pre_stl_envelope_applied",
            "input_voxels": int(mask.sum()),
            "output_voxels": int(mask.sum()),
            "removed_voxels": 0,
            "removed_components": 0,
        }
    np.save(work_dir / "pretrain_mask.npy", mask.astype(np.uint8))
    out_path = mask_to_stl(mask, dcm.spacing_zyx, case.pretrain_stl, origin_xyz=dcm.origin_xyz)
    stl_bytes = int(out_path.stat().st_size)
    stl_triangles = _binary_stl_triangle_count(out_path)
    quality, quality_issues, quality_stats = _pretrain_quality_details(mask, stl_bytes, TARGET_VOXELS_TIPS if case.is_post_tips else TARGET_VOXELS)
    eval_metrics = _evaluate_pretrain_against_label(case, grid_size=96)
    meta = {
        "algorithm_version": PRETRAIN_ALGORITHM_VERSION,
        "status_reason": reason,
        "input_source": input_source,
        "input_dcm": str(input_path),
        "input_mtime": input_mtime,
        "is_post_tips": bool(case.is_post_tips),
        "pretrain_stl": str(out_path),
        "reference_stl": str(reference_stl) if reference_crop is not None else None,
        "reference_crop": reference_crop,
        "component_reference_source": component_reference_source,
        "raw_model_plan": raw_plan,
        "plan": plan,
        "volume_stats_hu": stats,
        "pretrain_quality": quality,
        "quality_issues": quality_issues,
        "quality_stats": quality_stats,
        "reference_envelope": envelope_info,
        "region_grow": region_grow_info,
        "pretrain_vessel_eval": eval_metrics,
        "volume_shape_zyx": list(vol.shape),
        "spacing_zyx": list(dcm.spacing_zyx),
        "origin_xyz": list(dcm.origin_xyz),
        "mask_voxels": int(mask.sum()),
        "stl_bytes": stl_bytes,
        "stl_triangles": stl_triangles,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return PretrainResult(out_path, "review" if quality == "review" else "wrote")


def coarse_segment_patient(case, client: GemmaClient | None = None, force: bool = False) -> Path:
    return pretrain_patient(case, client=client, force=force).path


def _pretrain_quality(mask: np.ndarray, stl_bytes: int, max_voxels: int = TARGET_VOXELS) -> tuple[str, list[str]]:
    quality, issues, _ = _pretrain_quality_details(mask, stl_bytes, max_voxels)
    return quality, issues


def _pretrain_quality_details(mask: np.ndarray, stl_bytes: int, max_voxels: int = TARGET_VOXELS) -> tuple[str, list[str], dict]:
    issues = []
    mask = np.asarray(mask, dtype=bool)
    voxels = int(mask.sum())
    stats: dict[str, int | float] = {"voxels": voxels}
    if voxels == 0:
        issues.append("empty_pretrain_mask")
    if stl_bytes > MAX_STL_BYTES:
        issues.append("stl_over_20000kb")
    if voxels > max_voxels:
        issues.append("too_many_candidate_voxels")
    if ndi is not None and voxels > 0:
        labels, n = ndi.label(mask)
        counts = np.bincount(labels.ravel())
        component_count = int(np.count_nonzero(counts[1:] >= 64))
        stats["component_count"] = component_count
        if component_count > 16:
            issues.append("too_many_components")
        coords = np.argwhere(mask)
        extent = coords.max(axis=0) - coords.min(axis=0) + 1
        stats["bbox_z"] = int(extent[0])
        stats["bbox_y"] = int(extent[1])
        stats["bbox_x"] = int(extent[2])
        min_extent = max(int(extent.min()), 1)
        elongation = float(extent.max() / min_extent)
        stats["bbox_elongation"] = elongation
        if elongation > 12.0:
            issues.append("elongated_candidate")
    else:
        stats["component_count"] = 0 if voxels == 0 else 1
    return ("review" if issues else "ok", issues, stats)


def _evaluate_pretrain_against_label(case, grid_size: int = 96) -> dict | None:
    if not case.label_stl.exists() or not case.pretrain_stl.exists():
        return None
    try:
        pre, bounds = stl_to_voxels(case.pretrain_stl, grid_size=grid_size)
        label, _ = stl_to_voxels(case.label_stl, grid_size=grid_size, bounds=bounds)
    except Exception as exc:
        return {"error": str(exc)}
    pre_mask = pre > 0.5
    label_mask = label > 0.5
    intersection = int(np.logical_and(pre_mask, label_mask).sum())
    pre_count = int(pre_mask.sum())
    label_count = int(label_mask.sum())
    denom = pre_count + label_count
    return {
        "grid_size": int(grid_size),
        "dice": float((2 * intersection / denom) if denom else 1.0),
        "precision": float((intersection / pre_count) if pre_count else 0.0),
        "recall": float((intersection / label_count) if label_count else 0.0),
        "pretrain_voxels": pre_count,
        "label_voxels": label_count,
        "volume_ratio": float((pre_count / label_count) if label_count else 0.0),
    }


def _binary_stl_triangle_count(path: str | Path) -> int:
    path = Path(path)
    with path.open("rb") as fp:
        fp.seek(80)
        raw = fp.read(4)
    return int(struct.unpack("<I", raw)[0]) if len(raw) == 4 else 0


def _segment_once(
    vol: np.ndarray,
    plan: dict,
    is_post_tips: bool,
    volume: DicomVolume | None = None,
    reference_bounds_xyz: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    target = TARGET_VOXELS_TIPS if is_post_tips else TARGET_VOXELS
    if is_post_tips:
        portal_plan = dict(plan)
        portal_plan["hu_low"] = float(plan["hu_low"])
        portal_plan["hu_high"] = min(430.0, float(plan["hu_high"]))
        portal_plan["crop"] = _intersect_crop(plan["crop"], {"z": [0.30, 0.84], "y": [0.26, 0.76], "x": [0.16, 0.86]})
        portal = _threshold_components(vol, portal_plan, keep=6, max_voxels=target)
        include_tips = bool(plan.get("include_tips", True))
        if not include_tips:
            return portal
        tips_plan = {
            "hu_low": float(plan.get("tips_hu_low", 430.0)),
            "hu_high": float(plan.get("tips_hu_high", 3071.0)),
            "crop": _intersect_crop(plan["crop"], {"z": [0.28, 0.88], "y": [0.28, 0.70], "x": [0.24, 0.76]}),
        }
        tips = _threshold_components(vol, tips_plan, keep=4, close_iterations=1, max_voxels=target // 2)
        if volume is not None and reference_bounds_xyz is not None:
            tips, _ = _filter_components_by_reference_bbox(tips, volume, reference_bounds_xyz, max_distance_mm=12.0)
        return _largest_components(portal | tips, keep=8, min_voxels=64)
    return _threshold_components(vol, plan, keep=6, max_voxels=target)


def _threshold_components(vol: np.ndarray, plan: dict, keep: int, close_iterations: int = 2, max_voxels: int = TARGET_VOXELS) -> np.ndarray:
    crop_slices = _crop_slices(vol.shape, plan["crop"])
    roi = vol[crop_slices]
    raw = (roi >= plan["hu_low"]) & (roi <= plan["hu_high"])
    mask_roi = raw
    if ndi is not None:
        filtered = ndi.binary_opening(raw, iterations=1)
        filtered = ndi.binary_closing(filtered, iterations=close_iterations)
        filtered = ndi.binary_fill_holes(filtered)
        filtered = _largest_components(filtered, keep=keep, min_voxels=64)
        mask_roi = filtered if int(filtered.sum()) > 0 else _select_component_candidate(raw, keep=keep, max_voxels=max_voxels, close_iterations=close_iterations)
    else:
        mask_roi = _largest_components(mask_roi, keep=keep, min_voxels=64)
    mask = np.zeros(vol.shape, dtype=bool)
    mask[crop_slices] = mask_roi
    return mask


def _select_component_candidate(mask: np.ndarray, keep: int, max_voxels: int, close_iterations: int = 2) -> np.ndarray:
    candidates = [np.asarray(mask, dtype=bool)]
    if ndi is not None:
        for iterations in range(1, max(1, close_iterations) + 1):
            candidates.append(ndi.binary_closing(mask, iterations=iterations))
    kept = [_largest_components(candidate, keep=keep, min_voxels=64) for candidate in candidates]
    non_empty = [candidate for candidate in kept if int(candidate.sum()) > 0]
    if not non_empty:
        return np.asarray(mask, dtype=bool)
    under_limit = [candidate for candidate in non_empty if int(candidate.sum()) <= max_voxels]
    if under_limit:
        return max(under_limit, key=lambda candidate: int(candidate.sum()))
    return min(non_empty, key=lambda candidate: int(candidate.sum()))


def _default_plan(is_post_tips: bool) -> dict:
    if is_post_tips:
        return {"hu_low": 90.0, "hu_high": 680.0, "crop": {"z": [0.28, 0.88], "y": [0.24, 0.78], "x": [0.12, 0.88]}, "notes": "tight portal ROI with TIPS allowance"}
    return {"hu_low": 140.0, "hu_high": 420.0, "crop": {"z": [0.36, 0.80], "y": [0.32, 0.70], "x": [0.20, 0.82]}, "notes": "tight portal ROI fallback"}


def _sanitize_plan(plan: dict, is_post_tips: bool) -> dict:
    default = _default_plan(is_post_tips)
    out = dict(default)
    hu_bounds = (50.0, 720.0) if is_post_tips else (90.0, 420.0)
    for key in ("hu_low", "hu_high"):
        try:
            out[key] = float(plan.get(key, default[key]))
        except Exception:
            out[key] = default[key]
    out["hu_low"] = max(hu_bounds[0], min(out["hu_low"], hu_bounds[1] - 20.0))
    out["hu_high"] = max(out["hu_low"] + 20.0, min(out["hu_high"], hu_bounds[1]))
    out["include_tips"] = bool(plan.get("include_tips", is_post_tips))
    if is_post_tips:
        try:
            out["tips_hu_low"] = max(250.0, min(float(plan.get("tips_hu_low", 430.0)), 1200.0))
        except Exception:
            out["tips_hu_low"] = 430.0
        try:
            out["tips_hu_high"] = max(out["tips_hu_low"] + 20.0, min(float(plan.get("tips_hu_high", 3071.0)), 3071.0))
        except Exception:
            out["tips_hu_high"] = 3071.0
    seed = plan.get("portal_seed")
    if isinstance(seed, dict):
        try:
            out["portal_seed"] = {axis: float(max(0.0, min(float(seed[axis]), 1.0))) for axis in ("z", "y", "x")}
        except Exception:
            pass
    crop = default["crop"].copy()
    for axis in ("z", "y", "x"):
        vals = (plan.get("crop", {}) or {}).get(axis, crop[axis])
        try:
            a, b = max(0.0, min(float(vals[0]), 1.0)), max(0.0, min(float(vals[1]), 1.0))
            if b - a >= 0.05:
                crop[axis] = [a, b]
        except Exception:
            pass
    out["crop"] = _limit_crop_span(crop, {"z": 0.57, "y": 0.47, "x": 0.69} if is_post_tips else {"z": 0.46, "y": 0.40, "x": 0.66})
    out["notes"] = str(plan.get("notes", default["notes"]))
    if "exclude_notes" in plan:
        out["exclude_notes"] = str(plan.get("exclude_notes", ""))
    return out


def _limit_crop_span(crop: dict, max_span: dict[str, float]) -> dict:
    out = {}
    for axis in ("z", "y", "x"):
        a, b = crop[axis]
        if b - a > max_span[axis]:
            center = (a + b) / 2.0
            a, b = center - max_span[axis] / 2.0, center + max_span[axis] / 2.0
            if a < 0.0:
                b -= a
                a = 0.0
            if b > 1.0:
                a -= b - 1.0
                b = 1.0
        out[axis] = [round(float(max(0.0, a)), 4), round(float(min(1.0, b)), 4)]
    return out


def _intersect_crop(a: dict, b: dict) -> dict:
    out = {}
    for axis in ("z", "y", "x"):
        out[axis] = [max(float(a[axis][0]), float(b[axis][0])), min(float(a[axis][1]), float(b[axis][1]))]
        if out[axis][1] - out[axis][0] < 0.05:
            out[axis] = list(b[axis])
    return out


def _crop_slices(shape: tuple[int, int, int], crop: dict) -> tuple[slice, slice, slice]:
    spans = []
    for axis, n in zip(("z", "y", "x"), shape):
        a, b = crop[axis]
        start = max(0, min(n, int(round(a * n))))
        stop = max(start + 1, min(n, int(round(b * n))))
        spans.append(slice(start, stop))
    return tuple(spans)  # type: ignore[return-value]


def _largest_components(mask: np.ndarray, keep: int = 6, min_voxels: int = 64) -> np.ndarray:
    if ndi is None:
        return mask
    labels, n = ndi.label(mask)
    if n == 0:
        return mask
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    chosen = [i for i in np.argsort(counts)[::-1][:keep] if counts[i] >= min_voxels]
    return np.isin(labels, chosen)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pretrain.stl from patient/dcm DICOM slices.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--api_base_url", default=None)
    parser.add_argument("--model", default="gemma-4-31b-it")
    parser.add_argument("--patient", default=None, help="Process one patient folder name.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    client = GemmaClient(api_key=args.api_key, model=args.model, base_url=args.api_base_url)
    cases = discover_patients(args.data_root)
    if args.patient:
        cases = [case for case in cases if case.name == args.patient]
    print(f"[pretrain] found {len(cases)} patient folders")
    for case in cases:
        try:
            result = pretrain_patient(case, client=client, force=args.force)
            print(f"[pretrain] {case.name}: {result.status} {result.path}")
        except Exception as exc:
            print(f"[pretrain] {case.name}: failed: {exc}")


if __name__ == "__main__":
    main()
