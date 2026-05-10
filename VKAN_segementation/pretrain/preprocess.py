from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from scipy import ndimage as ndi
except ImportError:
    ndi = None

try:
    from ..utils.common import GemmaClient, discover_patients, mask_to_stl
except ImportError:
    from VKAN_segementation.utils.common import GemmaClient, discover_patients, mask_to_stl


@dataclass
class DicomVolume:
    volume_hu: np.ndarray
    spacing_zyx: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]


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


def ask_for_coarse_plan(client: GemmaClient, patient_name: str, is_post_tips: bool, stats: dict[str, float], previews: list[Path]) -> dict:
    system = (
        "You are assisting portal venous CT vessel extraction. Return strict JSON only. "
        "Favor high recall: keep portal vein, splenic vein, short SMV, LPV/RPV, "
        "compensation veins, and TIPS stent if present."
    )
    prompt = {
        "patient": patient_name,
        "is_post_tips": is_post_tips,
        "volume_stats_hu": stats,
        "request": (
            "Choose HU threshold [low, high] and normalized crop box keys z,y,x each [start,end]. "
            "Return {\"hu_low\": number, \"hu_high\": number, \"crop\": {\"z\":[a,b], \"y\":[a,b], \"x\":[a,b]}, \"notes\": string}."
        ),
    }
    return client.chat_json(system, json.dumps(prompt, ensure_ascii=True), previews)


def coarse_segment_patient(case, client: GemmaClient | None = None, force: bool = False) -> Path:
    if case.pretrain_stl.exists() and not force:
        return case.pretrain_stl
    dcm = load_dicom_series(case.dcm_dir)
    vol = dcm.volume_hu
    work_dir = case.path / "vkan_work"
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

    mask = (vol >= plan["hu_low"]) & (vol <= plan["hu_high"]) & _crop_mask(vol.shape, plan["crop"])
    if ndi is not None:
        mask = ndi.binary_opening(mask, iterations=1)
        mask = ndi.binary_closing(mask, iterations=2)
        mask = ndi.binary_fill_holes(mask)
    mask = _largest_components(mask, keep=8 if case.is_post_tips else 6)
    np.save(work_dir / "pretrain_mask.npy", mask.astype(np.uint8))
    return mask_to_stl(mask, dcm.spacing_zyx, case.pretrain_stl)


def _default_plan(is_post_tips: bool) -> dict:
    return {
        "hu_low": 80.0,
        "hu_high": 650.0 if is_post_tips else 320.0,
        "crop": {"z": [0.18, 0.86], "y": [0.18, 0.86], "x": [0.08, 0.92]},
        "notes": "heuristic fallback",
    }


def _sanitize_plan(plan: dict, is_post_tips: bool) -> dict:
    default = _default_plan(is_post_tips)
    out = dict(default)
    for key in ("hu_low", "hu_high"):
        try:
            out[key] = float(plan.get(key, default[key]))
        except Exception:
            out[key] = default[key]
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
    out["crop"] = crop
    out["notes"] = str(plan.get("notes", default["notes"]))
    return out


def _crop_mask(shape: tuple[int, int, int], crop: dict) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    spans = []
    for axis, n in zip(("z", "y", "x"), shape):
        a, b = crop[axis]
        spans.append(slice(max(0, int(round(a * n))), min(n, int(round(b * n)))))
    mask[spans[0], spans[1], spans[2]] = True
    return mask


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
            out = coarse_segment_patient(case, client=client, force=args.force)
            print(f"[pretrain] {case.name}: wrote {out}")
        except Exception as exc:
            print(f"[pretrain] {case.name}: failed: {exc}")


if __name__ == "__main__":
    main()

