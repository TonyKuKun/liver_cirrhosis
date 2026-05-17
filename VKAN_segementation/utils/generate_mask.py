#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate orig.nii.gz and origm.nii.gz from DICOM folders, then derive binary
mask.nii.gz as (origm - orig) > threshold.

Folder structure expected:
    patient/
        dcm/         (or DCM/)   -> original DICOM series
        mask/        (or MASK/)  -> overlay DICOM series (annotations)

After processing:
    patient/
        orig.nii.gz
        origm.nii.gz
        mask.nii.gz
        origm/       (renamed from mask/, original DICOM backup)
"""

import argparse
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


# ----------------------------- DICOM -> NIfTI -----------------------------
def dicom_to_nifti(dicom_dir: Path, output_nifti: Path) -> bool:
    """
    Convert a DICOM series to a single NIfTI file.
    Returns True if successful, False otherwise.
    """
    try:
        import dicom2nifti
    except ImportError:
        raise ImportError(
            "dicom2nifti is required. Install via: pip install dicom2nifti"
        )

    try:
        # Convert the whole directory (assumes single series)
        dicom2nifti.convert_directory(str(dicom_dir), str(output_nifti.parent))
        # The above function writes a file named after the directory; we need to rename.
        # More reliable: use dicom2nifti.dicom_series_to_nifti
        # Let's use the single-series function:
        # dicom2nifti.dicom_series_to_nifti(str(dicom_dir), str(output_nifti))
        # However, convert_directory is simpler. We'll implement a robust version:
    except Exception:
        # Fallback: use pydicom + nibabel if dicom2nifti fails
        return _dicom_to_nifti_fallback(dicom_dir, output_nifti)

    # If convert_directory worked, we need to find the generated file
    # It usually creates a file named after the first DICOM or the folder.
    # Better to use the specific function:
    try:
        # Force using the explicit function
        import dicom2nifti
        dicom2nifti.dicom_series_to_nifti(str(dicom_dir), str(output_nifti))
        return True
    except Exception as e:
        print(f"dicom2nifti conversion failed: {e}, trying fallback...")
        return _dicom_to_nifti_fallback(dicom_dir, output_nifti)


def _dicom_to_nifti_fallback(dicom_dir: Path, output_nifti: Path) -> bool:
    """
    Fallback conversion using pydicom and nibabel.
    This is a minimal implementation; may not handle all DICOM variations.
    """
    try:
        import pydicom
        from pydicom.errors import InvalidDicomError
    except ImportError:
        raise ImportError("pydicom required for fallback DICOM conversion")

    # Collect all DICOM files
    dcm_files = sorted(dicom_dir.glob("*"))
    if not dcm_files:
        return False

    # Read the first file to get dimensions and sorting key
    slices = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(f, force=True)
            if hasattr(ds, 'ImagePositionPatient'):
                slices.append((ds, f))
        except InvalidDicomError:
            continue

    if not slices:
        return False

    # Sort by ImagePositionPatient (z-coordinate)
    slices.sort(key=lambda x: float(x[0].ImagePositionPatient[2]) if hasattr(x[0], 'ImagePositionPatient') else 0)

    # Build 3D volume
    first = slices[0][0]
    shape = (len(slices), int(first.Rows), int(first.Columns))
    data = np.zeros(shape, dtype=np.float32)
    for i, (ds, _) in enumerate(slices):
        data[i, :, :] = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept if hasattr(ds, 'RescaleSlope') else ds.pixel_array

    # Get affine from DICOM
    # For simplicity, create a simple affine (assuming isotropic spacing)
    # In practice you should compute proper affine.
    spacing = (float(first.PixelSpacing[0]), float(first.PixelSpacing[1]), 1.0)
    if len(slices) > 1:
        spacing = (spacing[0], spacing[1], abs(float(slices[1][0].ImagePositionPatient[2]) - float(slices[0][0].ImagePositionPatient[2])))
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])

    img = nib.Nifti1Image(data, affine)
    nib.save(img, output_nifti)
    return True


# ----------------------------- Mask derivation -----------------------------
def compute_mask_from_nifti(
    orig_path: Path,
    origm_path: Path,
    mask_path: Path,
    threshold: float = 0.5,
    use_absolute_diff: bool = False,
    force: bool = False,
    dry_run: bool = False,
):
    """
    Compute binary mask = (origm - orig) > threshold (or abs diff).
    Returns (status, voxels, message).
    """
    if not orig_path.exists():
        return "skipped", 0, "missing orig.nii.gz"
    if not origm_path.exists():
        return "skipped", 0, "missing origm.nii.gz"
    if mask_path.exists() and not force:
        return "skipped", 0, "mask.nii.gz exists; use --force to overwrite"

    if dry_run:
        return "dry_run", 0, "would compute mask"

    try:
        orig_img = nib.load(str(orig_path))
        origm_img = nib.load(str(origm_path))
    except Exception as e:
        return "error", 0, f"load failed: {e}"

    orig = np.asarray(orig_img.dataobj, dtype=np.float32)
    origm = np.asarray(origm_img.dataobj, dtype=np.float32)

    if orig.shape != origm.shape:
        return "skipped", 0, f"shape mismatch: {orig.shape} vs {origm.shape}"

    diff = origm - orig
    if use_absolute_diff:
        mask = np.abs(diff) > threshold
    else:
        mask = diff > threshold

    out = mask.astype(np.uint8)
    header = orig_img.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(out, orig_img.affine, header), str(mask_path))

    affine_match = np.allclose(orig_img.affine, origm_img.affine)
    msg = "" if affine_match else "affine mismatch; used orig affine"
    return "wrote", int(out.sum()), msg


# ----------------------------- Patient processing -----------------------------
def process_patient(
    patient_dir: Path,
    dcm_name: str = "dcm",
    mask_name: str = "mask",
    threshold: float = 0.5,
    use_absolute_diff: bool = False,
    force: bool = False,
    dry_run: bool = False,
    skip_invalid: bool = False,
) -> dict:
    """
    Process a single patient:
        - Convert DICOM series in dcm/ -> orig.nii.gz
        - Convert DICOM series in mask/ -> origm.nii.gz
        - Rename mask/ -> origm/
        - Compute mask.nii.gz from the two NIfTI files.
    Returns a dict with status and info.
    """
    result = {
        "patient": patient_dir.name,
        "status": "ok",
        "steps": [],
        "mask_voxels": 0,
        "message": "",
    }

    # Skip invalid marker check
    if skip_invalid and any(marker in patient_dir.name for marker in ("@", "!", "&")):
        result["status"] = "skipped"
        result["message"] = "invalid marker in name"
        return result

    # Locate folders (case-insensitive)
    dcm_folder = None
    mask_folder = None
    for child in patient_dir.iterdir():
        if child.is_dir():
            if child.name.lower() == dcm_name.lower():
                dcm_folder = child
            elif child.name.lower() == mask_name.lower():
                mask_folder = child

    if not dcm_folder:
        result["status"] = "skipped"
        result["message"] = f"missing {dcm_name}/ folder"
        return result
    if not mask_folder:
        result["status"] = "skipped"
        result["message"] = f"missing {mask_name}/ folder"
        return result

    # Define output paths
    orig_nii = patient_dir / "orig.nii.gz"
    origm_nii = patient_dir / "origm.nii.gz"
    mask_nii = patient_dir / "mask.nii.gz"

    # Step 1: convert dcm -> orig.nii.gz
    if not dry_run:
        try:
            if not dicom_to_nifti(dcm_folder, orig_nii):
                result["status"] = "error"
                result["message"] = "DICOM to NIfTI conversion failed for dcm"
                return result
        except Exception as e:
            result["status"] = "error"
            result["message"] = f"dcm conversion error: {e}"
            return result
    else:
        result["steps"].append("would convert dcm -> orig.nii.gz")

    # Step 2: convert mask -> origm.nii.gz
    if not dry_run:
        try:
            if not dicom_to_nifti(mask_folder, origm_nii):
                result["status"] = "error"
                result["message"] = "DICOM to NIfTI conversion failed for mask"
                return result
        except Exception as e:
            result["status"] = "error"
            result["message"] = f"mask conversion error: {e}"
            return result
    else:
        result["steps"].append("would convert mask -> origm.nii.gz")

    # Step 3: rename mask folder to origm (backup)
    origm_folder = patient_dir / "origm"
    if not dry_run:
        if origm_folder.exists():
            # avoid overwriting by appending .bak
            bak = patient_dir / "origm.bak"
            idx = 1
            while bak.exists():
                bak = patient_dir / f"origm.bak{idx}"
                idx += 1
            shutil.move(str(origm_folder), str(bak))
            result["steps"].append(f"existing origm/ moved to {bak.name}")
        shutil.move(str(mask_folder), str(origm_folder))
        result["steps"].append(f"renamed {mask_folder.name} -> origm/")
    else:
        result["steps"].append(f"would rename {mask_folder.name} -> origm/")

    # Step 4: compute mask from orig.nii.gz and origm.nii.gz
    status, voxels, msg = compute_mask_from_nifti(
        orig_nii, origm_nii, mask_nii,
        threshold=threshold,
        use_absolute_diff=use_absolute_diff,
        force=force,
        dry_run=dry_run,
    )
    result["status"] = status
    result["mask_voxels"] = voxels
    result["message"] = msg
    if status == "wrote":
        result["steps"].append(f"mask computed: {voxels} voxels")
    elif status == "dry_run":
        result["steps"].append("would compute mask")
    return result


# ----------------------------- Main -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate orig.nii.gz, origm.nii.gz and mask.nii.gz from DICOM folders."
    )
    parser.add_argument("--root", type=str, default=r"F:\PCG data\dataset\test4all_sample",
                        help="Root directory containing patient folders")
    parser.add_argument("--patient", type=str, default=None,
                        help="Single patient folder name (optional)")
    parser.add_argument("--dcm_folder", type=str, default="dcm",
                        help="Name of the DICOM folder (case-insensitive, default 'dcm')")
    parser.add_argument("--mask_folder", type=str, default="mask",
                        help="Name of the mask DICOM folder (case-insensitive, default 'mask')")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for binary mask (default 0.5)")
    parser.add_argument("--absolute_diff", action="store_true",
                        help="Use absolute difference instead of positive difference")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing mask.nii.gz")
    parser.add_argument("--dry_run", action="store_true",
                        help="Simulate without writing files")
    parser.add_argument("--skip_invalid", default=False,
                        help="Skip folders containing @, !, &")
    args = parser.parse_args()

    root = Path(args.root)
    if args.patient:
        patient_dirs = [root / args.patient]
    else:
        # Gather all subdirectories (skip files)
        patient_dirs = [p for p in root.iterdir() if p.is_dir()]

    if not patient_dirs:
        raise RuntimeError("No patient folders found.")

    summary = {}
    for patient in tqdm(patient_dirs):
        res = process_patient(
            patient,
            dcm_name=args.dcm_folder,
            mask_name=args.mask_folder,
            threshold=args.threshold,
            use_absolute_diff=args.absolute_diff,
            force=args.force,
            dry_run=args.dry_run,
            skip_invalid=args.skip_invalid,
        )
        summary[res["status"]] = summary.get(res["status"], 0) + 1
        suffix = f" voxels={res['mask_voxels']}" if res["mask_voxels"] else ""
        print(f"[generate-mask] {res['patient']}: {res['status']}{suffix} ({res['message']})")
        if res["steps"] and not args.dry_run:
            for step in res["steps"]:
                print(f"    -> {step}")

    print("[generate-mask] summary " + " ".join(f"{k}={v}" for k, v in sorted(summary.items())))


if __name__ == "__main__":
    main()