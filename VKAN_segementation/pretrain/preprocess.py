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
    from ..utils.common import DicomVolume, GemmaClient, discover_patients, load_nifti_volume, mask_to_stl, save_nifti_volume
except ImportError:
    try:
        from VKAN_segementation.utils.common import DicomVolume, GemmaClient, discover_patients, load_nifti_volume, mask_to_stl, save_nifti_volume
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from utils.common import DicomVolume, GemmaClient, discover_patients, load_nifti_volume, mask_to_stl, save_nifti_volume


PRETRAIN_ALGORITHM_VERSION = "2026-05-10-nifti-training-v4"
PRETRAIN_META_NAME = "pretrain_meta.json"
TARGET_VOXELS = 420_000
TARGET_VOXELS_TIPS = 330_000
MAX_STL_BYTES = 20_000 * 1024
MAX_ADAPTIVE_STEPS = 14


@dataclass
class PretrainResult:
    path: Path
    status: str


def load_dicom_series(dcm_dir: str | Path) -> DicomVolume:
    """Load a CT DICOM folder as a z-y-x HU volume."""
    try:
        import pydicom
    except ImportError as exc:
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
        arr = ds.pixel_array.astype(np.float32)
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
    """Read uncompressed little-endian CT DICOM slices when pydicom is unavailable."""
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

    slices.sort(key=lambda ds: _minimal_slice_position(ds))
    arrays = []
    for ds in slices:
        dtype = np.int16 if int(ds.get("pixel_representation", 0)) else np.uint16
        arr = np.frombuffer(ds["pixel_data"], dtype="<i2" if dtype is np.int16 else "<u2")
        rows, cols = int(ds["rows"]), int(ds["columns"])
        arr = arr[: rows * cols].reshape(rows, cols).astype(np.float32)
        arrays.append(arr * float(ds.get("rescale_slope", 1.0)) + float(ds.get("rescale_intercept", 0.0)))
    volume = np.stack(arrays, axis=0)
    sy, sx = [float(v) for v in slices[0].get("pixel_spacing", [1.0, 1.0])[:2]]
    dz = abs(_minimal_slice_position(slices[1]) - _minimal_slice_position(slices[0])) if len(slices) > 1 else 0.0
    if dz <= 0:
        dz = float(slices[0].get("slice_thickness", 1.0))
    ipp = [float(v) for v in slices[0].get("image_position_patient", [0.0, 0.0, 0.0])[:3]]
    return DicomVolume(volume_hu=volume, spacing_zyx=(dz, sy, sx), origin_xyz=(ipp[0], ipp[1], ipp[2]))


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


def _slice_position(ds) -> float:
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) >= 3:
        return float(ipp[2])
    if hasattr(ds, "SliceLocation"):
        return float(ds.SliceLocation)
    return float(getattr(ds, "InstanceNumber", 0))


def save_mosaic_png(volume_hu: np.ndarray, out_path: str | Path, n_slices: int = 12) -> Path | None:
    """Save a compact CT preview for optional multimodal API prompts."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None

    out_path = Path(out_path)
    vol = np.asarray(volume_hu)
    lo, hi = np.percentile(vol, [1, 99])
    lo, hi = min(lo, -100.0), max(hi, 350.0)
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


def save_mask_projection(mask: np.ndarray, out_path: str | Path) -> Path | None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.asarray(mask, dtype=bool)
    views = [("axial z", mask.max(axis=0)), ("coronal y", mask.max(axis=1)), ("sagittal x", mask.max(axis=2))]
    panels = []
    for label, arr in views:
        img = Image.fromarray((arr.astype(np.uint8) * 255)).convert("L")
        img.thumbnail((360, 360))
        panel = Image.new("L", (360, 390), 0)
        panel.paste(img, ((360 - img.width) // 2, 25))
        ImageDraw.Draw(panel).text((8, 6), label, fill=255)
        panels.append(panel)
    canvas = Image.new("L", (1080, 390), 0)
    for i, panel in enumerate(panels):
        canvas.paste(panel, (i * 360, 0))
    canvas.save(out_path)
    return out_path


def case_nifti_path(case) -> Path:
    return case.path / f"{case.name}.nii.gz"


def pretrain_nifti_path(case) -> Path:
    return case.path / "pretrain.nii.gz"


def mask_label_nifti_path(case) -> Path:
    return case.path / "mask.nii.gz"


def mask_stl_path(case) -> Path:
    return case.path / "mask.stl"


def _tree_mtime(path: str | Path) -> float:
    path = Path(path)
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_mtime
    latest = path.stat().st_mtime
    for item in path.rglob("*"):
        if item.is_file():
            latest = max(latest, item.stat().st_mtime)
    return latest


def _case_input_mtime(case) -> tuple[Path, float, bool]:
    nii_path = case_nifti_path(case)
    dcm_mtime = _tree_mtime(case.dcm_dir)
    if nii_path.exists() and nii_path.stat().st_mtime >= dcm_mtime:
        return nii_path, nii_path.stat().st_mtime, True
    return nii_path, dcm_mtime, False


def load_case_volume(case) -> tuple[DicomVolume, Path, float, str]:
    nii_path, _, use_cache = _case_input_mtime(case)
    if use_cache:
        return load_nifti_volume(nii_path), nii_path, nii_path.stat().st_mtime, "nii"
    dcm = load_dicom_series(case.dcm_dir)
    save_nifti_volume(dcm, nii_path)
    return dcm, nii_path, nii_path.stat().st_mtime, "dcm"


def convert_mask_folder_to_nifti(dcm_dir: str | Path, mask_dir: str | Path, out_path: str | Path, min_voxels: int = 8) -> Path:
    dcm_dir = Path(dcm_dir)
    mask_dir = Path(mask_dir)
    dcm_files = {p.name: p for p in dcm_dir.rglob("*") if p.is_file()}
    mask_files = [p for p in mask_dir.rglob("*") if p.is_file() and p.name in dcm_files]
    if not mask_files:
        raise FileNotFoundError(f"No matching DICOM/mask slices found in {dcm_dir} and {mask_dir}")
    slices = []
    for mask_file in mask_files:
        dcm_meta = _read_minimal_dicom(dcm_files[mask_file.name])
        mask_meta = _read_minimal_dicom(mask_file)
        if not dcm_meta or not mask_meta:
            continue
        dcm_arr = _raw_pixel_array(dcm_meta)
        mask_arr = _raw_pixel_array(mask_meta)
        if dcm_arr.shape != mask_arr.shape:
            continue
        label = mask_arr != dcm_arr
        slices.append((_minimal_slice_position(mask_meta), label, mask_meta))
    if not slices:
        raise FileNotFoundError(f"No readable matching DICOM/mask slices found in {dcm_dir} and {mask_dir}")
    slices.sort(key=lambda item: item[0])
    label_vol = np.stack([item[1] for item in slices], axis=0)
    label_vol = _largest_components(label_vol, keep=8, min_voxels=min_voxels)
    first = slices[0][2]
    sy, sx = [float(v) for v in first.get("pixel_spacing", [1.0, 1.0])[:2]]
    dz = abs(slices[1][0] - slices[0][0]) if len(slices) > 1 else 0.0
    if dz <= 0:
        dz = float(first.get("slice_thickness", 1.0))
    ipp = [float(v) for v in first.get("image_position_patient", [0.0, 0.0, 0.0])[:3]]
    return save_nifti_volume(DicomVolume(label_vol.astype(np.uint8), (dz, sy, sx), (ipp[0], ipp[1], ipp[2])), out_path)


def _raw_pixel_array(meta: dict) -> np.ndarray:
    dtype = "<i2" if int(meta.get("pixel_representation", 0)) else "<u2"
    rows, cols = int(meta["rows"]), int(meta["columns"])
    return np.frombuffer(meta["pixel_data"], dtype=dtype, count=rows * cols).reshape(rows, cols)


def _should_rebuild_pretrain(
    stl_path: str | Path,
    meta_path: str | Path,
    nii_path: str | Path,
    input_mtime: float,
    required_outputs: list[str | Path] | None = None,
) -> tuple[bool, str]:
    stl_path = Path(stl_path)
    meta_path = Path(meta_path)
    nii_path = Path(nii_path)
    if not stl_path.exists():
        return True, "missing_stl"
    if not nii_path.exists():
        return True, "missing_nii"
    for output in required_outputs or []:
        if not Path(output).exists():
            return True, f"missing_{Path(output).name}"
    if not meta_path.exists():
        return True, "missing_meta"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return True, "invalid_meta"
    if meta.get("algorithm_version") != PRETRAIN_ALGORITHM_VERSION:
        return True, "version_mismatch"
    if abs(float(meta.get("input_mtime", -1.0)) - float(input_mtime)) > 1e-3:
        return True, "input_changed"
    return False, "up_to_date"


def ask_for_coarse_plan(client: GemmaClient, patient_name: str, is_post_tips: bool, stats: dict[str, float], previews: list[Path]) -> dict:
    system = (
        "You are assisting portal venous CT vessel extraction. Return strict JSON only. "
        "First choose a tight rectangular abdominal ROI, then threshold/grow only inside it. "
        "Keep portal vein, splenic vein, a short SMV segment, LPV/RPV near the hepatic hilum, "
        "left gastric or posterior gastric collateral veins when visible, and TIPS stent if present. "
        "Exclude ribs, spine, spleen, kidneys, liver parenchyma, bowel wall, and long lower abdominal veins."
    )
    prompt = {
        "patient": patient_name,
        "is_post_tips": is_post_tips,
        "volume_stats_hu": stats,
        "request": (
            "Choose HU threshold [low, high] and a tight normalized crop box keys z,y,x each [start,end]. "
            "The crop should cover the portal venous confluence and hepatic hilum while cutting away most ribs, spleen, kidneys, and pelvis. "
            "For non-TIPS cases prefer crop spans no wider than z 0.46, y 0.40, x 0.66 unless anatomy clearly requires it. "
            "Return {\"hu_low\": number, \"hu_high\": number, \"crop\": {\"z\":[a,b], \"y\":[a,b], \"x\":[a,b]}, \"notes\": string}."
        ),
    }
    return client.chat_json(system, json.dumps(prompt, ensure_ascii=True), previews)


def pretrain_patient(case, client: GemmaClient | None = None, force: bool = False) -> PretrainResult:
    work_dir = case.path / "vkan_work"
    meta_path = work_dir / PRETRAIN_META_NAME
    old_stl_exists = case.pretrain_stl.exists()
    nii_path, input_mtime, cache_ready = _case_input_mtime(case)
    if not force:
        required_outputs = [pretrain_nifti_path(case)]
        if (case.path / "mask").exists():
            required_outputs.append(mask_label_nifti_path(case))
        should_rebuild, reason = _should_rebuild_pretrain(
            case.pretrain_stl,
            meta_path,
            nii_path,
            input_mtime,
            required_outputs=required_outputs,
        )
        if not should_rebuild:
            return PretrainResult(case.pretrain_stl, "reused")
    else:
        reason = "forced"

    dcm, nii_path, input_mtime, input_source = load_case_volume(case)
    vol = dcm.volume_hu
    preview = save_mosaic_png(vol, work_dir / "ct_mosaic.png")
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
        raw_plan = ask_for_coarse_plan(client, case.name, case.is_post_tips, stats, [preview] if preview else [])
    plan = _sanitize_plan(raw_plan, case.is_post_tips)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "coarse_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    mask, final_plan, adaptive = _adaptive_segment(vol, plan, case.is_post_tips)
    np.save(work_dir / "pretrain_mask.npy", mask.astype(np.uint8))
    pretrain_nii = save_nifti_volume(DicomVolume(mask.astype(np.uint8), dcm.spacing_zyx, dcm.origin_xyz), pretrain_nifti_path(case))
    label_nii = None
    if (case.path / "mask").exists():
        try:
            label_nii = convert_mask_folder_to_nifti(case.dcm_dir, case.path / "mask", mask_label_nifti_path(case))
        except Exception:
            label_nii = None
    save_mask_projection(mask, work_dir / "pretrain_mask_projection.png")
    out_path = mask_to_stl(mask, dcm.spacing_zyx, case.pretrain_stl)
    quality, quality_issues = _pretrain_quality(mask, out_path.stat().st_size, TARGET_VOXELS_TIPS if case.is_post_tips else TARGET_VOXELS)
    meta = {
        "algorithm_version": PRETRAIN_ALGORITHM_VERSION,
        "status_reason": reason,
        "input_source": input_source,
        "input_nii": str(nii_path),
        "input_mtime": input_mtime,
        "is_post_tips": bool(case.is_post_tips),
        "tips_strategy": "portal_roi_plus_high_hu_tips_roi" if case.is_post_tips else "portal_roi_only",
        "pretrain_nifti": str(pretrain_nii),
        "mask_nifti": str(label_nii) if label_nii else None,
        "plan": final_plan,
        "adaptive": adaptive,
        "pretrain_quality": quality,
        "quality_issues": quality_issues,
        "volume_shape_zyx": list(vol.shape),
        "spacing_zyx": list(dcm.spacing_zyx),
        "mask_voxels": int(mask.sum()),
        "stl_bytes": int(out_path.stat().st_size),
        "stl_triangles": _binary_stl_triangle_count(out_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return PretrainResult(out_path, _write_status(old_stl_exists))


def coarse_segment_patient(case, client: GemmaClient | None = None, force: bool = False) -> Path:
    return pretrain_patient(case, client=client, force=force).path


def _write_status(old_stl_exists: bool) -> str:
    return "regenerated" if old_stl_exists else "wrote"


def _pretrain_quality(mask: np.ndarray, stl_bytes: int, max_voxels: int = TARGET_VOXELS_TIPS) -> tuple[str, list[str]]:
    issues = []
    if stl_bytes > MAX_STL_BYTES:
        issues.append("stl_over_20000kb")
    z_counts = np.asarray(mask, dtype=bool).sum(axis=(1, 2))
    if len(z_counts):
        slab_slices = int((z_counts > max(8_000, mask.shape[1] * mask.shape[2] * 0.025)).sum())
        if slab_slices >= 3:
            issues.append("large_slab_like_slices")
    if int(mask.sum()) > max_voxels:
        issues.append("too_many_candidate_voxels")
    return ("review" if issues else "ok", issues)


def _binary_stl_triangle_count(path: str | Path) -> int:
    path = Path(path)
    with path.open("rb") as fp:
        fp.seek(80)
        raw = fp.read(4)
    return int(struct.unpack("<I", raw)[0]) if len(raw) == 4 else 0


def _adaptive_segment(vol: np.ndarray, plan: dict, is_post_tips: bool) -> tuple[np.ndarray, dict, dict]:
    target = TARGET_VOXELS_TIPS if is_post_tips else TARGET_VOXELS
    final_plan = json.loads(json.dumps(plan))
    history = []
    mask = _segment_once(vol, final_plan, is_post_tips)
    history.append({"step": 0, "hu_low": final_plan["hu_low"], "crop": final_plan["crop"], "voxels": int(mask.sum())})
    for step in range(1, MAX_ADAPTIVE_STEPS + 1):
        if int(mask.sum()) <= target:
            break
        final_plan = _tighten_plan(final_plan, is_post_tips)
        next_mask = _segment_once(vol, final_plan, is_post_tips)
        history.append({"step": step, "hu_low": final_plan["hu_low"], "crop": final_plan["crop"], "voxels": int(next_mask.sum())})
        if next_mask.sum() == 0:
            break
        mask = next_mask
    return mask, final_plan, {"target_voxels": target, "history": history}


def _segment_once(vol: np.ndarray, plan: dict, is_post_tips: bool) -> np.ndarray:
    if is_post_tips:
        portal_plan = dict(plan)
        portal_plan["hu_low"] = max(130.0, float(plan["hu_low"]))
        portal_plan["hu_high"] = min(430.0, float(plan["hu_high"]))
        portal_plan["crop"] = _intersect_crop(plan["crop"], _tips_portal_crop())
        portal = _threshold_components(vol, portal_plan, keep=6, seed_crop=_seed_crop(False))
        tips_plan = {
            "hu_low": 430.0,
            "hu_high": 3071.0,
            "crop": _intersect_crop(plan["crop"], _tips_stent_crop()),
        }
        tips = _threshold_components(vol, tips_plan, keep=4, seed_crop=_tips_stent_seed_crop(), close_iterations=1, fill_holes=False)
        combined = portal | tips
        return _largest_components(combined, keep=8, seed_mask=_crop_mask(vol.shape, _tips_combined_seed_crop()), min_voxels=64)
    return _threshold_components(vol, plan, keep=6, seed_crop=_seed_crop(False))


def _threshold_components(
    vol: np.ndarray,
    plan: dict,
    keep: int,
    seed_crop: dict,
    close_iterations: int = 2,
    fill_holes: bool = True,
) -> np.ndarray:
    crop_slices = _crop_slices(vol.shape, plan["crop"])
    roi = vol[crop_slices]
    mask_roi = (roi >= plan["hu_low"]) & (roi <= plan["hu_high"])
    if ndi is not None:
        mask_roi = ndi.binary_opening(mask_roi, iterations=1)
        mask_roi = ndi.binary_closing(mask_roi, iterations=close_iterations)
        if fill_holes:
            mask_roi = ndi.binary_fill_holes(mask_roi)
    seed_mask = _crop_mask(vol.shape, seed_crop)[crop_slices]
    mask_roi = _largest_components(mask_roi, keep=keep, seed_mask=seed_mask)
    mask = np.zeros(vol.shape, dtype=bool)
    mask[crop_slices] = mask_roi
    return mask


def _tighten_plan(plan: dict, is_post_tips: bool) -> dict:
    out = json.loads(json.dumps(plan))
    out["hu_low"] = min(float(out["hu_low"]) + (10.0 if is_post_tips else 20.0), 220.0 if is_post_tips else 260.0)
    out["crop"] = _shrink_crop(out["crop"], z_delta=0.015 if is_post_tips else 0.02, y_delta=0.02, x_delta=0.01)
    return out


def _default_plan(is_post_tips: bool) -> dict:
    if is_post_tips:
        return {
            "hu_low": 90.0,
            "hu_high": 680.0,
            "crop": {"z": [0.28, 0.88], "y": [0.24, 0.78], "x": [0.12, 0.88]},
            "notes": "tight portal venous ROI fallback with TIPS allowance",
        }
    return {
        "hu_low": 140.0,
        "hu_high": 420.0,
        "crop": {"z": [0.36, 0.80], "y": [0.32, 0.70], "x": [0.20, 0.82]},
        "notes": "tight portal venous ROI fallback",
    }


def _sanitize_plan(plan: dict, is_post_tips: bool) -> dict:
    default = _default_plan(is_post_tips)
    out = dict(default)
    hu_bounds = (80.0, 720.0) if is_post_tips else (90.0, 420.0)
    for key in ("hu_low", "hu_high"):
        try:
            out[key] = float(plan.get(key, default[key]))
        except Exception:
            out[key] = default[key]
    out["hu_low"] = max(hu_bounds[0], min(out["hu_low"], hu_bounds[1] - 20.0))
    out["hu_high"] = max(out["hu_low"] + 20.0, min(out["hu_high"], hu_bounds[1]))
    if out["hu_high"] <= out["hu_low"]:
        out["hu_low"], out["hu_high"] = default["hu_low"], default["hu_high"]
    crop = default["crop"].copy()
    for axis in ("z", "y", "x"):
        vals = (plan.get("crop", {}) or {}).get(axis, crop[axis])
        try:
            a, b = max(0.0, min(float(vals[0]), 1.0)), max(0.0, min(float(vals[1]), 1.0))
            if b - a >= 0.05:
                crop[axis] = [a, b]
        except Exception:
            pass
    crop = _limit_crop_span(crop, _max_crop_span(is_post_tips))
    out["crop"] = crop
    out["notes"] = str(plan.get("notes", default["notes"]))
    return out


def _max_crop_span(is_post_tips: bool) -> dict[str, float]:
    if is_post_tips:
        return {"z": 0.57, "y": 0.47, "x": 0.69}
    return {"z": 0.46, "y": 0.40, "x": 0.66}


def _limit_crop_span(crop: dict, max_span: dict[str, float]) -> dict:
    limited = {}
    for axis in ("z", "y", "x"):
        a, b = crop[axis]
        span = b - a
        if span > max_span[axis]:
            center = (a + b) / 2.0
            half = max_span[axis] / 2.0
            a, b = center - half, center + half
            if a < 0.0:
                b -= a
                a = 0.0
            if b > 1.0:
                a -= b - 1.0
                b = 1.0
        limited[axis] = [round(float(max(0.0, a)), 4), round(float(min(1.0, b)), 4)]
    return limited


def _seed_crop(is_post_tips: bool) -> dict:
    if is_post_tips:
        return _tips_combined_seed_crop()
    return {"z": [0.42, 0.74], "y": [0.36, 0.66], "x": [0.28, 0.72]}


def _tips_portal_crop() -> dict:
    return {"z": [0.34, 0.82], "y": [0.30, 0.72], "x": [0.18, 0.84]}


def _tips_stent_crop() -> dict:
    return {"z": [0.30, 0.86], "y": [0.30, 0.68], "x": [0.28, 0.72]}


def _tips_stent_seed_crop() -> dict:
    return {"z": [0.34, 0.84], "y": [0.34, 0.64], "x": [0.32, 0.68]}


def _tips_combined_seed_crop() -> dict:
    return {"z": [0.36, 0.82], "y": [0.30, 0.70], "x": [0.24, 0.76]}


def _intersect_crop(a: dict, b: dict) -> dict:
    crop = {}
    for axis in ("z", "y", "x"):
        crop[axis] = [max(float(a[axis][0]), float(b[axis][0])), min(float(a[axis][1]), float(b[axis][1]))]
        if crop[axis][1] - crop[axis][0] < 0.05:
            crop[axis] = list(b[axis])
    return crop


def _shrink_crop(crop: dict, z_delta: float, y_delta: float, x_delta: float) -> dict:
    out = json.loads(json.dumps(crop))
    for axis, delta in (("z", z_delta), ("y", y_delta), ("x", x_delta)):
        a, b = float(out[axis][0]), float(out[axis][1])
        if b - a > 0.12:
            out[axis] = [round(a + delta, 4), round(b - delta, 4)]
    return out


def _crop_slices(shape: tuple[int, int, int], crop: dict) -> tuple[slice, slice, slice]:
    spans = []
    for axis, n in zip(("z", "y", "x"), shape):
        a, b = crop[axis]
        start = max(0, min(n, int(round(a * n))))
        stop = max(start + 1, min(n, int(round(b * n))))
        spans.append(slice(start, stop))
    return tuple(spans)  # type: ignore[return-value]


def _crop_mask(shape: tuple[int, int, int], crop: dict) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    spans = _crop_slices(shape, crop)
    mask[spans] = True
    return mask


def _largest_components(mask: np.ndarray, keep: int = 6, min_voxels: int = 64, seed_mask: np.ndarray | None = None) -> np.ndarray:
    if ndi is None:
        return mask
    labels, n = ndi.label(mask)
    if n == 0:
        return mask
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    if seed_mask is not None and seed_mask.shape == mask.shape:
        touched = np.unique(labels[seed_mask & (labels > 0)])
        touched = [int(i) for i in touched if counts[int(i)] >= min_voxels]
        chosen = sorted(touched, key=lambda i: counts[i], reverse=True)[:keep]
        if chosen:
            return np.isin(labels, chosen)
    chosen = [i for i in np.argsort(counts)[::-1][:keep] if counts[i] >= min_voxels]
    return np.isin(labels, chosen)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pretrain.stl from patient DICOM folders.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--api_base_url", default=None)
    parser.add_argument("--model", default="gemma-4-31b-it")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    client = GemmaClient(api_key=args.api_key, model=args.model, base_url=args.api_base_url)
    cases = discover_patients(args.data_root)
    print(f"[pretrain] found {len(cases)} patient folders")
    for case in cases:
        try:
            result = pretrain_patient(case, client=client, force=args.force)
            print(f"[pretrain] {case.name}: {result.status} {result.path}")
        except Exception as exc:
            print(f"[pretrain] {case.name}: failed: {exc}")


if __name__ == "__main__":
    main()

