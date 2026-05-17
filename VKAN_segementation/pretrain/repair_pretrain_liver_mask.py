"""
Repair pretrain vessel masks when segmentation/liver.nii.gz is missing.

Some patient folders contain segmentation/liver_left.nii.gz and
segmentation/liver_right.nii.gz, but no segmentation/liver.nii.gz. In that case
the liver-removal step can leave broad liver remnants in pretrain/pretrain.nii.gz
and pretrain/pretrain.stl.

This script:
  1. Creates segmentation/liver.nii.gz from liver_left + liver_right if needed.
  2. Removes the liver mask from pretrain.nii.gz.
  3. Rebuilds pretrain.stl from the repaired NIfTI mask.

Example:
    python repair_pretrain_liver_mask.py --patient "F:/PCG data/dataset/test4all_sample/20210208ZhangXiaoNi#"

Batch mode:
    python repair_pretrain_liver_mask.py --root "F:/PCG data/dataset/test4all_sample"
"""

import argparse
import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage import measure


def load_mask(path):
    img = nib.load(os.fspath(path))
    data = np.asanyarray(img.dataobj)
    return img, data > 0


def assert_same_grid(reference_img, other_img, other_path):
    if reference_img.shape != other_img.shape:
        raise ValueError(
            f"Shape mismatch for {other_path}: {other_img.shape} != {reference_img.shape}"
        )
    if not np.allclose(reference_img.affine, other_img.affine, atol=1e-4):
        raise ValueError(f"Affine mismatch for {other_path}")


def backup_file(path):
    path = Path(path)
    if not path.exists():
        return None
    backup = path.with_name(path.name + ".bak")
    if backup.exists():
        stem_backup = path.with_name(path.name + ".bak1")
        idx = 1
        while stem_backup.exists():
            idx += 1
            stem_backup = path.with_name(path.name + f".bak{idx}")
        backup = stem_backup
    shutil.copy2(path, backup)
    return backup


def ensure_liver_mask(segmentation_dir, overwrite=False):
    segmentation_dir = Path(segmentation_dir)
    liver_path = segmentation_dir / "liver.nii.gz"

    if liver_path.exists() and not overwrite:
        return liver_path, False
    liver_img, liver = merge_liver_parts(segmentation_dir)
    if liver_path.exists():
        backup_file(liver_path)
    nib.save(nib.Nifti1Image(liver.astype(np.uint8), liver_img.affine, liver_img.header), liver_path)
    return liver_path, True


def merge_liver_parts(segmentation_dir):
    segmentation_dir = Path(segmentation_dir)
    left_path = segmentation_dir / "liver_left.nii.gz"
    right_path = segmentation_dir / "liver_right.nii.gz"

    if not left_path.exists() or not right_path.exists():
        raise FileNotFoundError(
            "Need segmentation/liver.nii.gz, or both liver_left.nii.gz and liver_right.nii.gz"
        )

    left_img, left = load_mask(left_path)
    right_img, right = load_mask(right_path)
    assert_same_grid(left_img, right_img, right_path)

    liver = (left | right).astype(np.uint8)
    if liver.sum() == 0:
        raise ValueError("Merged liver mask is empty")

    return left_img, liver > 0


def save_mask_like(mask, reference_img, out_path):
    data = mask.astype(np.uint8)
    out = nib.Nifti1Image(data, reference_img.affine, reference_img.header)
    out.set_data_dtype(np.uint8)
    nib.save(out, os.fspath(out_path))


def resolve_pretrain_paths(patient_dir):
    patient_dir = Path(patient_dir)
    candidates = [
        (patient_dir / "pretrain.nii.gz", patient_dir / "pretrain.stl"),
        (patient_dir / "pretrain" / "pretrain.nii.gz", patient_dir / "pretrain" / "pretrain.stl"),
    ]
    for nii_path, stl_path in candidates:
        if nii_path.exists():
            return nii_path, stl_path
    return candidates[0]


def filter_small_components(mask, min_voxels=0):
    min_voxels = int(min_voxels)
    if min_voxels <= 0:
        return mask, 0
    labeled, n_labels = ndimage.label(mask)
    if n_labels == 0:
        return mask, 0
    counts = np.bincount(labeled.ravel())
    keep_labels = np.where(counts >= min_voxels)[0]
    keep_labels = keep_labels[keep_labels != 0]
    filtered = np.isin(labeled, keep_labels)
    removed = int(mask.sum() - filtered.sum())
    return filtered, removed


def repair_pretrain_nii(patient_dir, liver_path=None, liver_img=None, liver_mask=None,
                        liver_dilate=0, min_component_voxels=0, dry_run=False):
    pretrain_nii, _pretrain_stl = resolve_pretrain_paths(patient_dir)
    if not pretrain_nii.exists():
        raise FileNotFoundError(f"Missing {pretrain_nii}")

    pre_img, pre_mask = load_mask(pretrain_nii)
    if liver_img is None or liver_mask is None:
        liver_img, liver_mask = load_mask(liver_path)
    assert_same_grid(pre_img, liver_img, liver_path)

    if liver_dilate > 0:
        liver_mask = ndimage.binary_dilation(liver_mask, iterations=int(liver_dilate))

    repaired = pre_mask & ~liver_mask
    removed_by_liver = int(pre_mask.sum() - repaired.sum())
    repaired, removed_small = filter_small_components(repaired, min_voxels=min_component_voxels)
    if repaired.sum() == 0:
        raise ValueError(
            "Repair would remove the entire pretrain mask. Check that liver/pretrain are aligned."
        )

    if not dry_run:
        backup_file(pretrain_nii)
        save_mask_like(repaired, pre_img, pretrain_nii)
    return pretrain_nii, repaired, pre_img, removed_by_liver, removed_small


def voxel_to_world_vertices(vertices_zyx, affine):
    vertices_xyz = vertices_zyx[:, [2, 1, 0]]
    return nib.affines.apply_affine(affine, vertices_xyz)


def write_binary_stl(path, vertices, faces, name="pretrain_repaired"):
    triangles = vertices[faces].astype(np.float32, copy=False)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 0
    normals[valid] /= lengths[valid, None]
    normals[~valid] = 0

    header = name.encode("ascii", errors="ignore")[:80].ljust(80, b" ")
    record_dtype = np.dtype([
        ("normal", "<f4", (3,)),
        ("vertices", "<f4", (3, 3)),
        ("attribute", "<u2"),
    ])
    records = np.empty(len(faces), dtype=record_dtype)
    records["normal"] = normals.astype(np.float32, copy=False)
    records["vertices"] = triangles
    records["attribute"] = 0

    with open(path, "wb") as f:
        f.write(header)
        f.write(np.uint32(len(faces)).tobytes())
        records.tofile(f)


def rebuild_pretrain_stl(patient_dir, repaired_mask, reference_img, step_size=1, dry_run=False):
    _pretrain_nii, stl_path = resolve_pretrain_paths(patient_dir)
    if repaired_mask.sum() == 0:
        raise ValueError("Cannot build STL from an empty mask")

    padded = np.pad(repaired_mask.astype(np.float32), 1, mode="constant")
    verts, faces, _normals, _values = measure.marching_cubes(
        padded,
        level=0.5,
        spacing=(1.0, 1.0, 1.0),
        step_size=max(1, int(step_size)),
        allow_degenerate=False,
    )
    verts -= 1.0
    world_vertices = voxel_to_world_vertices(verts, reference_img.affine)

    if not dry_run:
        backup_file(stl_path)
        write_binary_stl(stl_path, world_vertices, faces)
    return stl_path, len(world_vertices), len(faces)


def repair_patient(patient_dir, liver_dilate=0, overwrite_liver=False, stl_step_size=1,
                   min_component_voxels=0, dry_run=False):
    patient_dir = Path(patient_dir)
    segmentation_dir = patient_dir / "segmentation"
    if not segmentation_dir.exists():
        raise FileNotFoundError(f"Missing segmentation folder: {segmentation_dir}")
    pretrain_nii, pretrain_stl = resolve_pretrain_paths(patient_dir)
    if not pretrain_nii.exists():
        raise FileNotFoundError(f"Missing pretrain NIfTI: {pretrain_nii}")

    liver_img = None
    liver_mask = None
    if dry_run:
        liver_path = segmentation_dir / "liver.nii.gz"
        if liver_path.exists():
            liver_created = False
            liver_img, liver_mask = load_mask(liver_path)
        else:
            liver_img, liver_mask = merge_liver_parts(segmentation_dir)
            liver_created = True
    else:
        liver_path, liver_created = ensure_liver_mask(segmentation_dir, overwrite=overwrite_liver)

    pretrain_nii, repaired_mask, reference_img, removed_by_liver, removed_small = repair_pretrain_nii(
        patient_dir,
        liver_path,
        liver_img=liver_img,
        liver_mask=liver_mask,
        liver_dilate=liver_dilate,
        min_component_voxels=min_component_voxels,
        dry_run=dry_run,
    )
    stl_path, n_vertices, n_faces = rebuild_pretrain_stl(
        patient_dir, repaired_mask, reference_img, step_size=stl_step_size, dry_run=dry_run
    )

    return {
        "patient": str(patient_dir),
        "liver_path": str(liver_path),
        "liver_created": liver_created,
        "pretrain_nii": str(pretrain_nii),
        "pretrain_stl": str(stl_path),
        "removed_voxels": removed_by_liver,
        "removed_small_component_voxels": removed_small,
        "remaining_voxels": int(repaired_mask.sum()),
        "stl_vertices": int(n_vertices),
        "stl_faces": int(n_faces),
    }


def iter_patient_dirs(root):
    root = Path(root)
    for child in sorted(root.iterdir()):
        has_root_pretrain = (child / "pretrain.nii.gz").exists()
        has_subdir_pretrain = (child / "pretrain" / "pretrain.nii.gz").exists()
        if child.is_dir() and (has_root_pretrain or has_subdir_pretrain) and (child / "segmentation").is_dir():
            yield child


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", type=str, default=None,
                        help="Single patient folder to repair")
    parser.add_argument("--root", type=str, default=r'F:\PCG data\dataset\test4all_sample',
                        help="Dataset root; repairs every child with pretrain/ and segmentation/")
    parser.add_argument("--liver_dilate", type=int, default=0,
                        help="Dilate liver mask by N voxels before subtraction")
    parser.add_argument("--overwrite_liver", action="store_true",
                        help="Regenerate liver.nii.gz even if it already exists")
    parser.add_argument("--stl_step_size", type=int, default=1,
                        help="Marching-cubes step size; larger is faster but coarser")
    parser.add_argument("--min_component_voxels", type=int, default=0,
                        help="Remove connected components smaller than this after liver subtraction")
    parser.add_argument("--dry_run", action="store_true",
                        help="Inspect and compute counts without writing repaired files")
    args = parser.parse_args()

    if not args.patient and not args.root:
        parser.error("Provide --patient or --root")

    patient_dirs = [Path(args.patient)] if args.patient else list(iter_patient_dirs(args.root))
    if not patient_dirs:
        raise RuntimeError("No patient folders found")

    failures = []
    for patient_dir in patient_dirs:
        try:
            result = repair_patient(
                patient_dir,
                liver_dilate=args.liver_dilate,
                overwrite_liver=args.overwrite_liver,
                stl_step_size=args.stl_step_size,
                min_component_voxels=args.min_component_voxels,
                dry_run=args.dry_run,
            )
            print(
                f"[{'DRY' if args.dry_run else 'OK'}] {Path(result['patient']).name}: "
                f"removed={result['removed_voxels']} remaining={result['remaining_voxels']} "
                f"small_removed={result['removed_small_component_voxels']} "
                f"stl_faces={result['stl_faces']} liver_created={result['liver_created']}"
            )
        except Exception as exc:
            failures.append((patient_dir, exc))
            print(f"[FAIL] {patient_dir}: {exc}")

    if failures:
        raise SystemExit(f"{len(failures)} patient(s) failed")


if __name__ == "__main__":
    main()
