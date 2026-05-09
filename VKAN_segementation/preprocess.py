from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from scipy import ndimage as ndi
except ImportError:
    ndi = None

try:
    from .config import discover_patients
    from .dicom_io import load_dicom_series, save_mosaic_png
    from .llm_client import GemmaClient, ask_for_coarse_plan
    from .mesh_ops import mask_to_stl
except ImportError:
    from config import discover_patients
    from dicom_io import load_dicom_series, save_mosaic_png
    from llm_client import GemmaClient, ask_for_coarse_plan
    from mesh_ops import mask_to_stl


def _default_plan(is_post_tips: bool) -> dict:
    return {
        "hu_low": 80.0,
        "hu_high": 650.0 if is_post_tips else 320.0,
        "crop": {
            "z": [0.18, 0.86],
            "y": [0.18, 0.86],
            "x": [0.08, 0.92],
        },
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
            a, b = float(vals[0]), float(vals[1])
            a, b = max(0.0, min(a, 1.0)), max(0.0, min(b, 1.0))
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
        lo = int(round(a * n))
        hi = int(round(b * n))
        spans.append(slice(max(0, lo), min(n, hi)))
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
    order = np.argsort(counts)[::-1]
    chosen = [i for i in order[:keep] if counts[i] >= min_voxels]
    return np.isin(labels, chosen)


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
    print(f"[preprocess] found {len(cases)} patient folders")
    for case in cases:
        try:
            out = coarse_segment_patient(case, client=client, force=args.force)
            print(f"[preprocess] {case.name}: wrote {out}")
        except Exception as exc:
            print(f"[preprocess] {case.name}: failed: {exc}")


if __name__ == "__main__":
    main()

