from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from scipy import ndimage as ndi
except ImportError:
    ndi = None

try:
    from ..utils.common import DicomVolume, GemmaClient, discover_patients, mask_to_stl, save_nifti_volume
except ImportError:
    try:
        from VKAN_segementation.utils.common import DicomVolume, GemmaClient, discover_patients, mask_to_stl, save_nifti_volume
    except ImportError:
        from utils.common import DicomVolume, GemmaClient, discover_patients, mask_to_stl, save_nifti_volume


PRETRAIN_ALGORITHM_VERSION = "2026-05-11-nifti-training-v1"
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
    except ImportError as exc:
        raise ImportError("pydicom is required to read DICOM files.") from exc

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


def _slice_position(ds) -> float:
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) >= 3:
        return float(ipp[2])
    if hasattr(ds, "SliceLocation"):
        return float(ds.SliceLocation)
    return float(getattr(ds, "InstanceNumber", 0))


def save_mosaic_png(volume_hu: np.ndarray, out_path: str | Path, n_slices: int = 12) -> Path | None:
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


def case_nifti_path(case) -> Path:
    return case.path / f"{case.name}.nii.gz"


def pretrain_nifti_path(case) -> Path:
    return case.path / "pretrain.nii.gz"


def mask_label_nifti_path(case) -> Path:
    return case.path / "mask.nii.gz"


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


def load_case_volume(case) -> tuple[DicomVolume, Path, float, str]:
    nii_path = case_nifti_path(case)
    dcm_mtime = _tree_mtime(case.dcm_dir)
    if nii_path.exists() and nii_path.stat().st_mtime >= dcm_mtime:
        try:
            from ..utils.common import load_nifti_volume
        except ImportError:
            try:
                from VKAN_segementation.utils.common import load_nifti_volume
            except ImportError:
                from utils.common import load_nifti_volume
        return load_nifti_volume(nii_path), nii_path, nii_path.stat().st_mtime, "nii"
    dcm = load_dicom_series(case.dcm_dir)
    save_nifti_volume(dcm, nii_path)
    return dcm, nii_path, nii_path.stat().st_mtime, "dcm"


def convert_mask_folder_to_nifti(dcm_dir: str | Path, mask_dir: str | Path, out_path: str | Path, min_voxels: int = 8) -> Path:
    """Convert manual DICOM drawings in patient/mask to a binary mask.nii.gz."""
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError("pydicom is required to convert mask DICOM files.") from exc

    dcm_dir = Path(dcm_dir)
    mask_dir = Path(mask_dir)
    dcm_files = {p.name: p for p in dcm_dir.rglob("*") if p.is_file()}
    mask_files = [p for p in mask_dir.rglob("*") if p.is_file() and p.name in dcm_files]
    if not mask_files:
        raise FileNotFoundError(f"No matching DICOM/mask slices found in {dcm_dir} and {mask_dir}")

    slices = []
    for mask_file in mask_files:
        try:
            dcm_ds = pydicom.dcmread(str(dcm_files[mask_file.name]), force=True)
            mask_ds = pydicom.dcmread(str(mask_file), force=True)
            dcm_arr = _raw_pixel_array(dcm_ds)
            mask_arr = _raw_pixel_array(mask_ds)
        except Exception:
            continue
        if dcm_arr.shape != mask_arr.shape:
            continue
        label = mask_arr != dcm_arr
        slices.append((_slice_position(mask_ds), label, mask_ds))
    if not slices:
        raise FileNotFoundError(f"No readable matching DICOM/mask slices found in {dcm_dir} and {mask_dir}")

    slices.sort(key=lambda item: item[0])
    label_vol = np.stack([item[1] for item in slices], axis=0)
    label_vol = _largest_components(label_vol, keep=8, min_voxels=min_voxels)
    first = slices[0][2]
    pixel_spacing = getattr(first, "PixelSpacing", [1.0, 1.0])
    sy, sx = float(pixel_spacing[0]), float(pixel_spacing[1])
    dz = abs(slices[1][0] - slices[0][0]) if len(slices) > 1 else 0.0
    if dz <= 0:
        dz = float(getattr(first, "SliceThickness", 1.0))
    ipp = getattr(first, "ImagePositionPatient", [0.0, 0.0, 0.0])
    return save_nifti_volume(DicomVolume(label_vol.astype(np.uint8), (dz, sy, sx), (float(ipp[0]), float(ipp[1]), float(ipp[2]))), out_path)


def _raw_pixel_array(ds) -> np.ndarray:
    dtype = "<i2" if int(getattr(ds, "PixelRepresentation", 0)) else "<u2"
    rows, cols = int(ds.Rows), int(ds.Columns)
    return np.frombuffer(ds.PixelData, dtype=dtype, count=rows * cols).reshape(rows, cols)


def _should_rebuild_pretrain(case, meta_path: str | Path, input_mtime: float) -> tuple[bool, str]:
    meta_path = Path(meta_path)
    required = [pretrain_nifti_path(case)]
    if (case.path / "mask").exists():
        required.append(mask_label_nifti_path(case))
    for output in required:
        if not output.exists():
            return True, f"missing_{output.name}"
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
    mask_mtime = float(meta.get("mask_mtime", 0.0))
    if (case.path / "mask").exists() and abs(mask_mtime - _tree_mtime(case.path / "mask")) > 1e-3:
        return True, "mask_changed"
    return False, "up_to_date"


def ask_for_coarse_plan(client: GemmaClient, patient_name: str, is_post_tips: bool, stats: dict[str, float], previews: list[Path]) -> dict:
    system = (
        "You are assisting portal venous CT vessel extraction. Return strict JSON only. "
        "Choose a tight abdominal ROI. Keep portal vein, splenic vein, short SMV, LPV/RPV, "
        "collateral veins when visible, and TIPS stent if present. Exclude ribs, spine, spleen, kidneys, and bowel."
    )
    prompt = {
        "patient": patient_name,
        "is_post_tips": is_post_tips,
        "volume_stats_hu": stats,
        "request": (
            "Choose HU threshold [low, high] and a tight normalized crop box keys z,y,x each [start,end]. "
            "Return {\"hu_low\": number, \"hu_high\": number, \"crop\": {\"z\":[a,b], \"y\":[a,b], \"x\":[a,b]}, \"notes\": string}."
        ),
    }
    return client.chat_json(system, json.dumps(prompt, ensure_ascii=True), previews)


def pretrain_patient(case, client: GemmaClient | None = None, force: bool = False) -> PretrainResult:
    work_dir = case.path / "vkan_work"
    meta_path = work_dir / PRETRAIN_META_NAME
    dcm, nii_path, input_mtime, input_source = load_case_volume(case)
    if not force:
        should_rebuild, reason = _should_rebuild_pretrain(case, meta_path, input_mtime)
        if not should_rebuild:
            return PretrainResult(pretrain_nifti_path(case), "reused")
    else:
        reason = "forced"

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

    mask = _segment_once(vol, plan, case.is_post_tips)
    np.save(work_dir / "pretrain_mask.npy", mask.astype(np.uint8))
    pretrain_nii = save_nifti_volume(DicomVolume(mask.astype(np.uint8), dcm.spacing_zyx, dcm.origin_xyz), pretrain_nifti_path(case))
    label_nii = None
    label_error = None
    if (case.path / "mask").exists():
        try:
            label_nii = convert_mask_folder_to_nifti(case.dcm_dir, case.path / "mask", mask_label_nifti_path(case))
        except Exception as exc:
            label_error = str(exc)

    out_path = mask_to_stl(mask, dcm.spacing_zyx, case.pretrain_stl)
    stl_bytes = int(out_path.stat().st_size)
    stl_triangles = _binary_stl_triangle_count(out_path)
    quality, quality_issues = _pretrain_quality(mask, stl_bytes, TARGET_VOXELS_TIPS if case.is_post_tips else TARGET_VOXELS)
    if label_error:
        quality = "review"
        quality_issues.append("mask_nifti_failed")
    meta = {
        "algorithm_version": PRETRAIN_ALGORITHM_VERSION,
        "status_reason": reason,
        "input_source": input_source,
        "input_nii": str(nii_path),
        "input_mtime": input_mtime,
        "mask_mtime": _tree_mtime(case.path / "mask"),
        "is_post_tips": bool(case.is_post_tips),
        "pretrain_nifti": str(pretrain_nii),
        "mask_nifti": str(label_nii) if label_nii else None,
        "mask_nifti_error": label_error,
        "pretrain_stl": str(out_path),
        "plan": plan,
        "pretrain_quality": quality,
        "quality_issues": quality_issues,
        "volume_shape_zyx": list(vol.shape),
        "spacing_zyx": list(dcm.spacing_zyx),
        "mask_voxels": int(mask.sum()),
        "stl_bytes": stl_bytes,
        "stl_triangles": stl_triangles,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return PretrainResult(pretrain_nii, "review" if quality == "review" else "wrote")


def coarse_segment_patient(case, client: GemmaClient | None = None, force: bool = False) -> Path:
    return pretrain_patient(case, client=client, force=force).path


def _pretrain_quality(mask: np.ndarray, stl_bytes: int, max_voxels: int = TARGET_VOXELS) -> tuple[str, list[str]]:
    issues = []
    voxels = int(mask.sum())
    if voxels == 0:
        issues.append("empty_pretrain_mask")
    if stl_bytes > MAX_STL_BYTES:
        issues.append("stl_over_20000kb")
    if voxels > max_voxels:
        issues.append("too_many_candidate_voxels")
    return ("review" if issues else "ok", issues)


def _binary_stl_triangle_count(path: str | Path) -> int:
    path = Path(path)
    with path.open("rb") as fp:
        fp.seek(80)
        raw = fp.read(4)
    return int(struct.unpack("<I", raw)[0]) if len(raw) == 4 else 0


def _segment_once(vol: np.ndarray, plan: dict, is_post_tips: bool) -> np.ndarray:
    if is_post_tips:
        portal_plan = dict(plan)
        portal_plan["hu_low"] = max(130.0, float(plan["hu_low"]))
        portal_plan["hu_high"] = min(430.0, float(plan["hu_high"]))
        portal_plan["crop"] = _intersect_crop(plan["crop"], {"z": [0.30, 0.84], "y": [0.26, 0.76], "x": [0.16, 0.86]})
        portal = _threshold_components(vol, portal_plan, keep=6)
        tips_plan = {"hu_low": 430.0, "hu_high": 3071.0, "crop": _intersect_crop(plan["crop"], {"z": [0.28, 0.88], "y": [0.28, 0.70], "x": [0.24, 0.76]})}
        tips = _threshold_components(vol, tips_plan, keep=4, close_iterations=1, fill_holes=False)
        return _largest_components(portal | tips, keep=8, min_voxels=64)
    return _threshold_components(vol, plan, keep=6)


def _threshold_components(vol: np.ndarray, plan: dict, keep: int, close_iterations: int = 2, fill_holes: bool = True) -> np.ndarray:
    crop_slices = _crop_slices(vol.shape, plan["crop"])
    roi = vol[crop_slices]
    mask_roi = (roi >= plan["hu_low"]) & (roi <= plan["hu_high"])
    if ndi is not None:
        mask_roi = ndi.binary_opening(mask_roi, iterations=1)
        mask_roi = ndi.binary_closing(mask_roi, iterations=close_iterations)
        if fill_holes:
            mask_roi = ndi.binary_fill_holes(mask_roi)
    mask_roi = _largest_components(mask_roi, keep=keep)
    mask = np.zeros(vol.shape, dtype=bool)
    mask[crop_slices] = mask_roi
    return mask


def _default_plan(is_post_tips: bool) -> dict:
    if is_post_tips:
        return {"hu_low": 90.0, "hu_high": 680.0, "crop": {"z": [0.28, 0.88], "y": [0.24, 0.78], "x": [0.12, 0.88]}, "notes": "tight portal ROI with TIPS allowance"}
    return {"hu_low": 140.0, "hu_high": 420.0, "crop": {"z": [0.36, 0.80], "y": [0.32, 0.70], "x": [0.20, 0.82]}, "notes": "tight portal ROI fallback"}


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
    parser = argparse.ArgumentParser(description="Generate pretrain.nii.gz, mask.nii.gz, and inspection pretrain.stl.")
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
