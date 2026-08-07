"""Derive binary manual masks from an ``origm`` overlay series.

Some manually edited DICOM series retain a different RescaleIntercept from
the source CT.  The overlay and source must use the same physical intensity
scale before their difference is thresholded into ``mask.nii.gz``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ConvertResult:
    patient: Path
    status: str
    mask_voxels: int = 0
    message: str = ""


@dataclass(frozen=True)
class RescaleParameters:
    slope: float
    intercept: float


def _load_nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - runtime dependency check
        raise ImportError("nibabel is required to derive overlay masks.") from exc
    return nib


def _backup_file(path: Path) -> Path:
    """Preserve an existing data file without overwriting an older backup."""
    candidate = path.with_name(path.name + ".bak")
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(path.name + f".bak{suffix}")
        suffix += 1
    shutil.copy2(path, candidate)
    return candidate


def _calibration_sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".calibration.json")


def _calibration_matches(path: Path, source: RescaleParameters, overlay: RescaleParameters) -> bool:
    sidecar = _calibration_sidecar(path)
    if not sidecar.exists():
        return False
    try:
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        return (
            int(record["corrected_mtime_ns"]) == path.stat().st_mtime_ns
            and np.isclose(float(record["source_slope"]), source.slope)
            and np.isclose(float(record["source_intercept"]), source.intercept)
            and np.isclose(float(record["overlay_slope"]), overlay.slope)
            and np.isclose(float(record["overlay_intercept"]), overlay.intercept)
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _write_calibration_sidecar(path: Path, source: RescaleParameters, overlay: RescaleParameters, backup: Path) -> None:
    _calibration_sidecar(path).write_text(
        json.dumps(
            {
                "method": "recalibrate_existing_nifti_from_paired_dicom_rescale",
                "source_slope": source.slope,
                "source_intercept": source.intercept,
                "overlay_slope": overlay.slope,
                "overlay_intercept": overlay.intercept,
                "backup": backup.name,
                "corrected_mtime_ns": path.stat().st_mtime_ns,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _matching_calibrated_backup(
    path: Path, source: RescaleParameters, overlay: RescaleParameters
) -> Path | None:
    """Recognize a repair performed before the calibration sidecar existed."""
    nib = _load_nibabel()
    current_img = nib.load(str(path))
    current = np.asarray(current_img.dataobj, dtype=np.float32)
    backups = sorted(path.parent.glob(path.name + ".bak*"), key=lambda item: item.stat().st_mtime, reverse=True)
    for backup in backups:
        # Backup suffixes such as ``.nii.gz.bak`` are not recognized by
        # nib.load, despite containing a valid gzip NIfTI payload.
        backup_img = nib.Nifti1Image.from_bytes(gzip.decompress(backup.read_bytes()))
        if tuple(backup_img.shape[:3]) != tuple(current_img.shape[:3]):
            continue
        if not np.allclose(backup_img.affine, current_img.affine, atol=1e-4):
            continue
        old = np.asarray(backup_img.dataobj, dtype=np.float32)
        expected = (old - overlay.intercept) / overlay.slope * source.slope + source.intercept
        if np.allclose(current, expected, rtol=0.0, atol=1e-4):
            return backup
    return None


def _assert_same_grid(reference, other, reference_path: Path, other_path: Path) -> None:
    if tuple(reference.shape[:3]) != tuple(other.shape[:3]):
        raise ValueError(
            f"Grid shape mismatch: {reference_path}={reference.shape[:3]}, "
            f"{other_path}={other.shape[:3]}"
        )
    if not np.allclose(reference.affine, other.affine, atol=1e-4):
        raise ValueError(f"Affine mismatch: {reference_path} vs {other_path}")


def _first_dicom_parameters(dicom_dir: Path) -> RescaleParameters:
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - runtime dependency check
        raise ImportError("pydicom is required to repair DICOM intensity scaling.") from exc

    for path in sorted(candidate for candidate in dicom_dir.rglob("*") if candidate.is_file()):
        try:
            dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        if not hasattr(dataset, "PixelData") and "Rows" not in dataset:
            continue
        return RescaleParameters(
            slope=float(getattr(dataset, "RescaleSlope", 1.0)),
            intercept=float(getattr(dataset, "RescaleIntercept", 0.0)),
        )
    raise FileNotFoundError(f"No readable DICOM image found in {dicom_dir}")


def repair_orig_m_intercept(patient_dir: str | Path, dry_run: bool = False) -> ConvertResult:
    """Recalibrate an existing ``origm.nii.gz`` from paired DICOM metadata.

    The NIfTI voxel ordering and geometry are already known to match ``orig``.
    This reconstructs the physical values from the overlay series' raw scale,
    then applies the source CT scale.  It avoids a second DICOM-to-NIfTI
    geometry conversion while correcting the erroneous intensity calibration.
    """
    patient = Path(patient_dir)
    origm_path = patient / "origm.nii.gz"
    source_dicom = patient / "dcm"
    overlay_dicom = patient / "origm"
    if not origm_path.exists():
        return ConvertResult(patient, "missing_origm_nifti", message=str(origm_path))
    if not source_dicom.is_dir() or not overlay_dicom.is_dir():
        return ConvertResult(patient, "missing_dicom", message="Need both dcm/ and origm/.")

    source_scale = _first_dicom_parameters(source_dicom)
    overlay_scale = _first_dicom_parameters(overlay_dicom)
    if _calibration_matches(origm_path, source_scale, overlay_scale):
        return ConvertResult(patient, "already_calibrated")
    if np.isclose(source_scale.slope, overlay_scale.slope) and np.isclose(
        source_scale.intercept, overlay_scale.intercept
    ):
        return ConvertResult(patient, "already_calibrated")
    if np.isclose(overlay_scale.slope, 0.0):
        return ConvertResult(patient, "invalid_overlay_slope", message=str(overlay_scale.slope))

    recovered_backup = _matching_calibrated_backup(origm_path, source_scale, overlay_scale)
    if recovered_backup is not None:
        if dry_run:
            return ConvertResult(patient, "would_register", message=f"backup={recovered_backup.name}")
        _write_calibration_sidecar(origm_path, source_scale, overlay_scale, recovered_backup)
        return ConvertResult(patient, "already_calibrated", message=f"registered backup={recovered_backup.name}")

    if dry_run:
        return ConvertResult(
            patient,
            "would_recalibrate",
            message=(
                f"overlay slope/intercept={overlay_scale.slope}/{overlay_scale.intercept}; "
                f"source slope/intercept={source_scale.slope}/{source_scale.intercept}"
            ),
        )

    nib = _load_nibabel()
    image = nib.load(str(origm_path))
    current = np.asarray(image.dataobj, dtype=np.float32)
    raw_pixels = (current - overlay_scale.intercept) / overlay_scale.slope
    corrected = raw_pixels * source_scale.slope + source_scale.intercept

    backup = _backup_file(origm_path)
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    corrected_image = nib.Nifti1Image(corrected.astype(np.float32), image.affine, header)
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    corrected_image.set_qform(qform if qform is not None else image.affine, int(qcode) if qform is not None else 1)
    corrected_image.set_sform(sform if sform is not None else image.affine, int(scode) if sform is not None else 1)
    nib.save(corrected_image, str(origm_path))
    _write_calibration_sidecar(origm_path, source_scale, overlay_scale, backup)
    return ConvertResult(
        patient,
        "recalibrated",
        message=(
            f"overlay slope/intercept={overlay_scale.slope}/{overlay_scale.intercept}; "
            f"source slope/intercept={source_scale.slope}/{source_scale.intercept}"
        ),
    )


def convert_patient(patient_dir: str | Path, threshold: float = 0.5, force: bool = False) -> ConvertResult:
    """Write ``mask.nii.gz`` from ``origm.nii.gz - orig.nii.gz``.

    For compatibility with older data, an existing ``mask.nii.gz`` is treated
    as an overlay only when ``origm.nii.gz`` is absent; it is moved to
    ``origm.nii.gz`` before the binary mask is written.
    """
    patient = Path(patient_dir)
    orig_path = patient / "orig.nii.gz"
    overlay_path = patient / "origm.nii.gz"
    mask_path = patient / "mask.nii.gz"
    if not orig_path.exists():
        return ConvertResult(patient, "missing_orig", message=str(orig_path))
    if not overlay_path.exists():
        if not mask_path.exists():
            return ConvertResult(patient, "missing_overlay", message="Need origm.nii.gz or overlay mask.nii.gz.")
        mask_path.replace(overlay_path)

    if mask_path.exists() and not force:
        return ConvertResult(patient, "skipped_existing", message=str(mask_path))

    nib = _load_nibabel()
    orig = nib.load(str(orig_path))
    overlay = nib.load(str(overlay_path))
    _assert_same_grid(orig, overlay, orig_path, overlay_path)
    difference = np.asarray(overlay.dataobj, dtype=np.float32) - np.asarray(orig.dataobj, dtype=np.float32)
    mask = (difference > float(threshold)).astype(np.uint8)

    if mask_path.exists():
        _backup_file(mask_path)
    header = orig.header.copy()
    header.set_data_dtype(np.uint8)
    output = nib.Nifti1Image(mask, orig.affine, header)
    qform, qcode = orig.get_qform(coded=True)
    sform, scode = orig.get_sform(coded=True)
    output.set_qform(qform if qform is not None else orig.affine, int(qcode) if qform is not None else 1)
    output.set_sform(sform if sform is not None else orig.affine, int(scode) if sform is not None else 1)
    nib.save(output, str(mask_path))
    return ConvertResult(patient, "wrote", mask_voxels=int(mask.sum()))


def mask_metrics(patient_dir: str | Path) -> dict[str, float | int] | None:
    """Return conservative mask checks without treating ``@`` as invalid."""
    patient = Path(patient_dir)
    pretrain_path = patient / "pretrain.nii.gz"
    mask_path = patient / "mask.nii.gz"
    if not pretrain_path.exists() or not mask_path.exists():
        return None
    nib = _load_nibabel()
    pretrain = nib.load(str(pretrain_path))
    mask = nib.load(str(mask_path))
    _assert_same_grid(pretrain, mask, pretrain_path, mask_path)
    pre = np.asarray(pretrain.dataobj) > 0.5
    label = np.asarray(mask.dataobj) > 0.5
    pre_voxels = int(pre.sum())
    mask_voxels = int(label.sum())
    intersection = int((pre & label).sum())
    return {
        "pretrain_voxels": pre_voxels,
        "mask_voxels": mask_voxels,
        "mask_ratio": float(mask_voxels / max(label.size, 1)),
        "mask_to_pretrain_ratio": float(mask_voxels / max(pre_voxels, 1)),
        "pretrain_dice": float((2 * intersection + 1) / (pre_voxels + mask_voxels + 1)),
    }


def append_mask_marker(patient_dir: str | Path) -> Path:
    """Append ``$mask`` once; any existing ``$`` already skips training."""
    patient = Path(patient_dir)
    if "$" in patient.name:
        return patient
    marked = patient.with_name(patient.name + "$mask")
    if marked.exists():
        raise FileExistsError(f"Cannot mark {patient}: {marked} already exists")
    patient.rename(marked)
    return marked


def _iter_patients(root: Path, selected: list[str]) -> list[Path]:
    if selected:
        return [root / name for name in selected]
    return sorted(path for path in root.iterdir() if path.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair origm DICOM calibration and derive binary overlay masks.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--patient", action="append", default=[], help="Patient folder name; repeat for multiple cases.")
    parser.add_argument("--repair_dicom_intercept", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing mask.nii.gz after creating a backup.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mark_abnormal", action="store_true")
    parser.add_argument("--max_mask_ratio", type=float, default=0.10)
    parser.add_argument("--min_mask_to_pretrain_ratio", type=float, default=0.10)
    parser.add_argument("--max_mask_to_pretrain_ratio", type=float, default=10.0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    root = Path(args.data_root)
    for patient in _iter_patients(root, args.patient):
        if not patient.is_dir():
            print(f"[missing] {patient}")
            continue
        if args.repair_dicom_intercept:
            result = repair_orig_m_intercept(patient, dry_run=args.dry_run)
            print(f"[{result.status}] {patient.name}: {result.message}")
        if args.force and not args.dry_run:
            result = convert_patient(patient, threshold=args.threshold, force=True)
            print(f"[{result.status}] {patient.name}: mask_voxels={result.mask_voxels}")
        metrics = mask_metrics(patient)
        if metrics is None:
            continue
        abnormal = (
            metrics["mask_voxels"] == 0
            or metrics["mask_ratio"] > args.max_mask_ratio
            or metrics["mask_to_pretrain_ratio"] < args.min_mask_to_pretrain_ratio
            or metrics["mask_to_pretrain_ratio"] > args.max_mask_to_pretrain_ratio
        )
        print(f"[audit] {patient.name}: {metrics}")
        if args.mark_abnormal and abnormal:
            if args.dry_run:
                print(f"[would_mark] {patient.name}$mask")
            else:
                marked = append_mask_marker(patient)
                print(f"[marked] {marked.name}")


if __name__ == "__main__":
    main()
