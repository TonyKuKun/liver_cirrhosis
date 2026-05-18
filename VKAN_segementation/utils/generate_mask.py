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

force=False (default): incremental mode, skip steps if outputs exist.
force=True: redo everything (overwrite).
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
        # Use explicit single-series conversion
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

    first = slices[0][0]
    shape = (len(slices), int(first.Rows), int(first.Columns))
    data = np.zeros(shape, dtype=np.float32)
    for i, (ds, _) in enumerate(slices):
        data[i, :, :] = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept if hasattr(ds, 'RescaleSlope') else ds.pixel_array

    # Simple affine (assume isotropic spacing)
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
        - Convert DICOM series in dcm/ -> orig.nii.gz (skip if exists and not force)
        - Convert DICOM series to origm.nii.gz: prefer origm/ folder, else rename mask/ -> origm/
        - Rename mask/ -> origm/ only if origm/ does not exist or force=True
        - Compute mask.nii.gz from the two NIfTI files (skip if exists and not force)
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
    origm_folder = patient_dir / "origm"   # target folder name after renaming

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

    # For origm source: either existing origm/ or mask/ (if origm/ missing)
    origm_source = None
    if origm_folder.exists():
        origm_source = origm_folder
    elif mask_folder:
        origm_source = mask_folder
    else:
        result["status"] = "skipped"
        result["message"] = f"neither {mask_name}/ nor origm/ folder found"
        return result

    # Define output paths
    orig_nii = patient_dir / "orig.nii.gz"
    origm_nii = patient_dir / "origm.nii.gz"
    mask_nii = patient_dir / "mask.nii.gz"

    # Step 1: convert dcm -> orig.nii.gz
    if orig_nii.exists() and not force:
        result["steps"].append(f"orig.nii.gz exists, skipping (use --force to redo)")
    else:
        if dry_run:
            result["steps"].append("would convert dcm -> orig.nii.gz")
        else:
            try:
                if not dicom_to_nifti(dcm_folder, orig_nii):
                    result["status"] = "error"
                    result["message"] = "DICOM to NIfTI conversion failed for dcm"
                    return result
                result["steps"].append("converted dcm -> orig.nii.gz")
            except Exception as e:
                result["status"] = "error"
                result["message"] = f"dcm conversion error: {e}"
                return result

    # Step 2: ensure origm/ folder exists (rename mask/ if needed) and convert to origm.nii.gz
    # First, handle folder existence/renaming
    if origm_folder.exists():
        if not force:
            result["steps"].append(f"origm/ already exists, using it as source")
        else:
            # force=True: if origm/ exists but we may want to re-copy from mask/? Let's define:
            # With force=True, we still use origm/ as source (no deletion), but we will re-convert
            # the NIfTI from it. Or we could delete origm/ and re-rename? For safety, we keep folder.
            result["steps"].append(f"origm/ already exists, will re-convert origm.nii.gz (force=True)")
    else:
        # origm/ does not exist; need to create it from mask_folder (if available)
        if mask_folder is None:
            result["status"] = "error"
            result["message"] = f"cannot create origm/: no mask/ folder and origm/ missing"
            return result
        if dry_run:
            result["steps"].append(f"would rename {mask_folder.name} -> origm/")
        else:
            # If force=False (normal case) and origm/ missing, we rename mask/ -> origm/
            # If force=True, also rename (overwrite any existing? but we already checked not exists)
            shutil.move(str(mask_folder), str(origm_folder))
            result["steps"].append(f"renamed {mask_folder.name} -> origm/")
            # Update origm_source after rename
            origm_source = origm_folder

    # Now convert origm.nii.gz from the source folder (origm/ or mask/ if not yet renamed)
    # Determine actual source folder (might be origm/ after potential rename)
    if origm_folder.exists():
        source_for_conversion = origm_folder
    elif mask_folder and not origm_folder.exists():
        # fallback: if we haven't renamed due to dry_run, use mask_folder
        source_for_conversion = mask_folder
    else:
        source_for_conversion = None

    if source_for_conversion is None:
        result["status"] = "error"
        result["message"] = "no source folder for origm conversion"
        return result

    if origm_nii.exists() and not force:
        result["steps"].append(f"origm.nii.gz exists, skipping (use --force to redo)")
    else:
        if dry_run:
            result["steps"].append("would convert source -> origm.nii.gz")
        else:
            try:
                if not dicom_to_nifti(source_for_conversion, origm_nii):
                    result["status"] = "error"
                    result["message"] = "DICOM to NIfTI conversion failed for origm"
                    return result
                result["steps"].append(f"converted {source_for_conversion.name} -> origm.nii.gz")
            except Exception as e:
                result["status"] = "error"
                result["message"] = f"origm conversion error: {e}"
                return result

    # Step 3: compute mask from orig.nii.gz and origm.nii.gz
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
    parser.add_argument("--force", action="store_true",default=False,
                        help="Force redo all steps (overwrite existing files and folders)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Simulate without writing files")
    parser.add_argument("--skip_invalid", action="store_true", default=False,
                        help="Skip folders containing @, !, &")
    args = parser.parse_args()

    root = Path(args.root)
    if args.patient:
        patient_dirs = [root / args.patient]
    else:
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
        if res["steps"]:
            for step in res["steps"]:
                print(f"    -> {step}")

    print("[generate-mask] summary " + " ".join(f"{k}={v}" for k, v in sorted(summary.items())))


if __name__ == "__main__":
    main()