"""Standardize patient NIfTI files to a single RAS+ patient-space grid.

For each patient folder, ``orig.nii.gz`` is converted to closest-canonical
RAS+. Every other ``*.nii.gz`` under the same patient is then resampled onto
that canonical ``orig`` shape and affine.

Default mode is a dry run. Use ``--apply`` to write files; originals are copied
under ``patient/.orientation_backup`` before replacement.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Iterable

import numpy as np

TARGET_AXCODES = ("R", "A", "S")
TARGET_ORIENTATION = "".join(TARGET_AXCODES) + "+"
BACKUP_DIR_NAME = ".orientation_backup"
REPORT_NAME = "nifti_orientation_report.json"


def _load_nibabel():
    try:
        import nibabel as nib
        from nibabel.processing import resample_from_to
    except ImportError as exc:  # pragma: no cover - runtime dependency check
        raise ImportError("nibabel and scipy are required for NIfTI orientation standardization.") from exc
    return nib, resample_from_to


def _discover_patients(root: Path, patient: str | None = None) -> list[Path]:
    root = Path(root)
    if (root / "orig.nii.gz").exists():
        patients = [root]
    else:
        patients = sorted(p for p in root.iterdir() if p.is_dir() and (p / "orig.nii.gz").exists())
    if patient:
        patients = [p for p in patients if p.name == patient or str(p) == patient]
    return patients


def _iter_nii_files(patient_dir: Path) -> Iterable[Path]:
    for path in sorted(patient_dir.rglob("*.nii.gz")):
        if BACKUP_DIR_NAME in path.parts:
            continue
        yield path


def _closest_ras(img):
    nib, _ = _load_nibabel()
    try:
        return nib.as_closest_canonical(img, enforce_diag=False)
    except TypeError:  # older nibabel
        return nib.as_closest_canonical(img)


def _axcodes(img) -> str:
    nib, _ = _load_nibabel()
    return "".join(code if code is not None else "?" for code in nib.aff2axcodes(img.affine))


def _same_grid(a, b, atol: float = 1e-4) -> bool:
    return tuple(a.shape[:3]) == tuple(b.shape[:3]) and np.allclose(a.affine, b.affine, atol=atol)


def _is_mask_like(path: Path) -> bool:
    name = path.name.lower()
    parts = {part.lower() for part in path.parts}
    if "segmentation" in parts or "totalseg_output" in parts or "ts_raw" in parts:
        return True
    mask_tokens = (
        "mask",
        "label",
        "pretrain",
        "predict_mask",
        "portal_vein",
        "spleen",
        "liver",
        "kidney",
        "aorta",
        "vena_cava",
        "vertebrae",
        "bone",
    )
    return any(token in name for token in mask_tokens) and name not in {"orig.nii.gz", "origm.nii.gz"}


def _copy_backup(path: Path, patient_dir: Path, backup_root: Path) -> Path:
    rel = path.relative_to(patient_dir)
    backup = backup_root / rel
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def _make_img_like(data: np.ndarray, affine: np.ndarray, source_img, dtype: np.dtype | None = None):
    nib, _ = _load_nibabel()
    if dtype is not None:
        data = np.asarray(data).astype(dtype, copy=False)
    header = source_img.header.copy()
    header.set_data_dtype(np.asarray(data).dtype)
    out = nib.Nifti1Image(data, affine, header)
    out.set_qform(affine, code=1)
    out.set_sform(affine, code=1)
    return out


def _resample_to_reference(img, reference_img, path: Path):
    _, resample_from_to = _load_nibabel()
    mask_like = _is_mask_like(path)
    order = 0 if mask_like else 1
    note = None
    if _same_grid(img, reference_img):
        out = img
    else:
        try:
            out = resample_from_to(img, (reference_img.shape[:3], reference_img.affine), order=order)
        except Exception as exc:
            if tuple(img.shape[:3]) != tuple(reference_img.shape[:3]):
                raise
            # Some old masks have singular or missing affines but already live on
            # the patient voxel grid. Keep their array and stamp the canonical
            # orig affine so downstream code can use one coordinate frame.
            out = img
            note = f"resample_failed_same_shape_used_array: {exc}"

    data = np.asarray(out.dataobj)
    if mask_like:
        data = np.rint(data).astype(np.uint8, copy=False)
    else:
        data = data.astype(np.asarray(img.dataobj).dtype, copy=False)
    return _make_img_like(data, reference_img.affine, img), order, note


def standardize_patient(patient_dir: Path, apply: bool = False, backup_dir_name: str = BACKUP_DIR_NAME) -> dict:
    nib, _ = _load_nibabel()
    patient_dir = Path(patient_dir)
    orig_path = patient_dir / "orig.nii.gz"
    if not orig_path.exists():
        return {"patient": patient_dir.name, "status": "missing_orig"}

    orig_img = nib.load(str(orig_path))
    orig_ras = _closest_ras(orig_img)
    backup_root = patient_dir / backup_dir_name

    patient_report: dict = {
        "patient": patient_dir.name,
        "patient_dir": str(patient_dir),
        "target_axcodes": TARGET_AXCODES,
        "target_orientation": TARGET_ORIENTATION,
        "orig_before_axcodes": _axcodes(orig_img),
        "orig_after_axcodes": _axcodes(orig_ras),
        "orig_shape": list(orig_ras.shape[:3]),
        "orig_affine": np.asarray(orig_ras.affine, dtype=float).round(6).tolist(),
        "apply": bool(apply),
        "files": [],
    }

    if _axcodes(orig_ras) != "".join(TARGET_AXCODES):
        patient_report["status"] = "error"
        patient_report["error"] = f"Could not make orig.nii.gz {TARGET_ORIENTATION}; got {_axcodes(orig_ras)}"
        return patient_report

    for path in _iter_nii_files(patient_dir):
        try:
            img = nib.load(str(path))
            before_axcodes = _axcodes(img)
            if path == orig_path:
                out_img = _make_img_like(np.asarray(orig_ras.dataobj), orig_ras.affine, orig_ras)
                order = None
                note = None
            else:
                out_img, order, note = _resample_to_reference(img, orig_ras, path)
            changed = not _same_grid(img, out_img) or before_axcodes != _axcodes(out_img)
            item = {
                "path": str(path),
                "relative_path": str(path.relative_to(patient_dir)),
                "before_axcodes": before_axcodes,
                "after_axcodes": _axcodes(out_img),
                "before_shape": list(img.shape[:3]),
                "after_shape": list(out_img.shape[:3]),
                "mask_like": bool(_is_mask_like(path)),
                "interpolation_order": order,
                "note": note,
                "changed": bool(changed),
                "status": "would_write" if changed and not apply else ("wrote" if changed else "aligned"),
            }
            if changed and apply:
                backup = _copy_backup(path, patient_dir, backup_root)
                item["backup"] = str(backup)
                nib.save(out_img, str(path))
        except Exception as exc:
            item = {
                "path": str(path),
                "relative_path": str(path.relative_to(patient_dir)),
                "changed": False,
                "status": "error",
                "error": str(exc),
            }
        patient_report["files"].append(item)

    patient_report["status"] = "ok"
    patient_report["changed_files"] = int(sum(1 for item in patient_report["files"] if item["changed"]))
    return patient_report


def standardize_root(root: Path, patient: str | None = None, apply: bool = False) -> dict:
    root = Path(root)
    patients = _discover_patients(root, patient)
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(root),
        "target_axcodes": TARGET_AXCODES,
        "target_orientation": TARGET_ORIENTATION,
        "apply": bool(apply),
        "patients": [standardize_patient(p, apply=apply) for p in patients],
    }
    report["patient_count"] = len(report["patients"])
    report["changed_file_count"] = int(sum(p.get("changed_files", 0) for p in report["patients"]))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Standardize patient NIfTI files to RAS+ orig.nii.gz space.")
    parser.add_argument("--data_root", required=True, help="Dataset root or a single patient directory.")
    parser.add_argument("--patient", default=None, help="Optional patient folder name to process.")
    parser.add_argument("--apply", action="store_true", help="Write files in place. Default is dry run.")
    parser.add_argument("--report", default=None, help=f"Report path. Default: data_root/{REPORT_NAME}")
    args = parser.parse_args()

    root = Path(args.data_root)
    report = standardize_root(root, patient=args.patient, apply=args.apply)
    report_path = Path(args.report) if args.report else root / REPORT_NAME
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] target orientation: {TARGET_ORIENTATION}")
    print(f"patients: {report['patient_count']} | changed files: {report['changed_file_count']}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
