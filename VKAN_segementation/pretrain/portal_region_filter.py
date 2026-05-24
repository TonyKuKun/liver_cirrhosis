"""Filter pretrain masks by growing from the portal-vein segmentation.

This is a standalone repair/debug tool. It uses the portal-vein NIfTI in
``patient/segmentation`` as the anatomical seed, then keeps only foreground
components in the candidate pretrain mask that overlap or bridge to that seed.

Examples:
    python pretrain/portal_region_filter.py --patient "F:\\PCG data\\dataset\\test4all_sample\\20201224WangMingLian#"
    python pretrain/portal_region_filter.py --patient "...\\20201224WangMingLian#" --preview
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from scipy import ndimage as ndi
except ImportError:  # pragma: no cover - handled at runtime
    ndi = None

try:
    from ..utils.common import zyx_mask_to_stl
except (ImportError, ValueError):
    try:
        from VKAN_segementation.utils.common import zyx_mask_to_stl
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from utils.common import zyx_mask_to_stl


FILTER_VERSION = "2026-05-16-portal-region-filter-v2-liver-first"
INVALID_MARKERS = ("@", "!", "&")
PORTAL_MASK_NAMES = (
    "portal_vein.nii.gz",
    "portal_vein_and_splenic_vein.nii.gz",
    "门静脉.nii.gz",
)
LIVER_MASK_NAMES = ("liver.nii.gz", "肝脏.nii.gz")


def load_nifti_zyx(path: Path):
    import nibabel as nib

    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    affine = img.affine.copy()
    spacing_xyz = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI: {path}")
    data = np.transpose(data, (2, 1, 0))
    spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    origin = affine[:3, 3]
    origin_xyz = (float(origin[0]), float(origin[1]), float(origin[2]))
    return data, img, spacing_zyx, origin_xyz


def load_mask_zyx_like_reference(path: Path, reference_img) -> np.ndarray:
    import nibabel as nib

    img = nib.load(str(path))
    if tuple(img.shape[:3]) != tuple(reference_img.shape[:3]) or not np.allclose(img.affine, reference_img.affine, atol=1e-4):
        from nibabel.processing import resample_from_to

        img = resample_from_to(img, (reference_img.shape[:3], reference_img.affine), order=0)
    data = np.asarray(img.dataobj) > 0
    return np.transpose(data, (2, 1, 0))


def save_mask_like(mask_zyx: np.ndarray, reference_img, out_path: Path) -> Path:
    import nibabel as nib

    data = np.asarray(mask_zyx, dtype=np.uint8)
    ref_shape = tuple(int(v) for v in reference_img.shape[:3])
    if data.shape == (ref_shape[2], ref_shape[1], ref_shape[0]):
        data = np.transpose(data, (2, 1, 0))
    elif data.shape != ref_shape:
        raise ValueError(f"Mask shape {data.shape} cannot be saved like reference {ref_shape}")

    header = reference_img.header.copy()
    header.set_data_dtype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, reference_img.affine, header), str(out_path))
    return out_path


def resample_bool_mask(mask: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape == target_shape:
        return mask
    if ndi is None:
        raise RuntimeError("scipy is required to resample masks with different shapes")
    zoom = np.asarray(target_shape, dtype=np.float64) / np.asarray(mask.shape, dtype=np.float64)
    return ndi.zoom(mask.astype(np.float32), zoom, order=0) > 0.5


def find_structure_mask_path(patient_dir: Path, names: tuple[str, ...], label: str) -> Path:
    search_dirs = (
        patient_dir / "segmentation" / "totalseg_output",
        patient_dir / "segmentation" / "ts_raw",
        patient_dir / "segmentation",
    )
    for directory in search_dirs:
        for name in names:
            path = directory / name
            if path.exists():
                return path
    raise FileNotFoundError(f"Missing {label} NIfTI under {patient_dir / 'segmentation'}")


def find_portal_mask_path(patient_dir: Path) -> Path:
    return find_structure_mask_path(patient_dir, PORTAL_MASK_NAMES, "portal-vein")


def find_liver_mask_path(patient_dir: Path) -> Path:
    return find_structure_mask_path(patient_dir, LIVER_MASK_NAMES, "liver")


def load_candidate_mask(patient_dir: Path, candidate_path: Optional[Path] = None, reference_img=None) -> tuple[np.ndarray, str]:
    if candidate_path is None:
        npy_path = patient_dir / "vkan_work" / "pretrain_mask.npy"
        nii_path = patient_dir / "pretrain.nii.gz"
        if npy_path.exists():
            candidate_path = npy_path
        elif nii_path.exists():
            candidate_path = nii_path
        else:
            raise FileNotFoundError(f"Missing pretrain candidate mask in {patient_dir}")

    if candidate_path.suffix.lower() == ".npy":
        return np.load(candidate_path).astype(bool), str(candidate_path)
    if candidate_path.name.endswith(".nii.gz") or candidate_path.suffix.lower() == ".nii":
        if reference_img is not None:
            return load_mask_zyx_like_reference(candidate_path, reference_img), str(candidate_path)
        data, _img, _spacing, _origin = load_nifti_zyx(candidate_path)
        return (data > 0), str(candidate_path)
    raise ValueError(f"Unsupported candidate mask format: {candidate_path}")


def _iterations_from_mm(mm: float, spacing_zyx: tuple[float, float, float]) -> int:
    return max(1, int(round(float(mm) / max(float(min(spacing_zyx)), 1e-3))))


def filter_by_portal_region(
    candidate_mask: np.ndarray,
    portal_mask: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    bridge_mm: float = 0.0,
    portal_dilate_mm: float = 2.0,
    min_bridge_voxels: int = 32,
    max_seed_snap_mm: float = 30.0,
) -> tuple[np.ndarray, dict]:
    """Keep candidate components connected to the portal-vein seed region."""
    if ndi is None:
        raise RuntimeError("scipy is required for portal region filtering")

    candidate = np.asarray(candidate_mask, dtype=bool)
    portal = np.asarray(portal_mask, dtype=bool)
    info: dict = {
        "version": FILTER_VERSION,
        "input_voxels": int(candidate.sum()),
        "portal_voxels": int(portal.sum()),
        "bridge_mm": bridge_mm,
        "portal_dilate_mm": portal_dilate_mm,
        "min_bridge_voxels": min_bridge_voxels,
    }
    if candidate.sum() == 0 or portal.sum() == 0:
        info["status"] = "empty_candidate_or_portal"
        info["output_voxels"] = int(candidate.sum())
        return candidate, info

    labels = np.empty(candidate.shape, dtype=np.int32)
    n_components = ndi.label(candidate, output=labels)
    counts = np.bincount(labels.ravel(), minlength=n_components + 1)
    counts[0] = 0
    info["components_total"] = int(n_components)

    seed = portal
    if portal_dilate_mm > 0:
        seed = ndi.binary_dilation(portal, iterations=_iterations_from_mm(portal_dilate_mm, spacing_zyx))

    hit_labels = set(int(v) for v in np.unique(labels[seed]) if v > 0)
    if not hit_labels:
        coords = np.argwhere(candidate)
        portal_center = np.argwhere(portal).mean(axis=0)
        delta_mm = (coords.astype(np.float32) - portal_center.astype(np.float32)) * np.asarray(spacing_zyx, dtype=np.float32)
        dist2 = np.sum(delta_mm ** 2, axis=1)
        nearest = int(np.argmin(dist2))
        nearest_mm = float(np.sqrt(dist2[nearest]))
        info["nearest_seed_distance_mm"] = round(nearest_mm, 2)
        if nearest_mm <= max_seed_snap_mm:
            z, y, x = coords[nearest]
            hit_labels = {int(labels[z, y, x])}
            info["fallback"] = "nearest_candidate_to_portal"
        else:
            hit_labels = {int(np.argmax(counts))}
            info["fallback"] = "largest_component"

    if bridge_mm > 0:
        main_region = np.isin(labels, list(hit_labels))
        bridge_zone = ndi.binary_dilation(main_region, iterations=_iterations_from_mm(bridge_mm, spacing_zyx))
        bridge_labels = set(int(v) for v in np.unique(labels[bridge_zone]) if v > 0)
        keep_labels = {
            label
            for label in bridge_labels
            if label in hit_labels or int(counts[label]) >= min_bridge_voxels
        }
        keep_labels.update(hit_labels)
    else:
        keep_labels = set(hit_labels)

    filtered = np.isin(labels, list(keep_labels))
    info.update({
        "status": "ok",
        "portal_labels_hit": int(len(hit_labels)),
        "labels_kept": int(len(keep_labels)),
        "labels_removed": int(max(0, n_components - len(keep_labels))),
        "output_voxels": int(filtered.sum()),
        "removed_voxels": int(candidate.sum() - filtered.sum()),
    })
    return filtered, info


def filter_patient(
    patient_dir: Path,
    candidate_path: Optional[Path] = None,
    overwrite: bool = True,
    subtract_liver: bool = True,
    bridge_mm: float = 0.0,
    portal_dilate_mm: float = 2.0,
    min_bridge_voxels: int = 32,
) -> dict:
    patient_dir = Path(patient_dir)
    orig_path = patient_dir / "orig.nii.gz"
    if not orig_path.exists():
        raise FileNotFoundError(f"Missing orig.nii.gz: {orig_path}")

    _orig_data, orig_img, spacing_zyx, origin_xyz = load_nifti_zyx(orig_path)
    target_shape = tuple(int(v) for v in _orig_data.shape)
    candidate, candidate_source = load_candidate_mask(patient_dir, candidate_path, reference_img=orig_img)
    candidate = resample_bool_mask(candidate, target_shape)

    liver_info: dict = {"enabled": bool(subtract_liver), "status": "skipped"}
    if subtract_liver:
        liver_path = find_liver_mask_path(patient_dir)
        liver_mask = resample_bool_mask(load_mask_zyx_like_reference(liver_path, orig_img), target_shape)
        before_liver = int(candidate.sum())
        candidate = candidate & ~liver_mask
        liver_info = {
            "enabled": True,
            "status": "ok",
            "path": str(liver_path),
            "liver_voxels": int(liver_mask.sum()),
            "candidate_voxels_before": before_liver,
            "candidate_voxels_after": int(candidate.sum()),
            "removed_voxels": int(before_liver - int(candidate.sum())),
        }

    portal_path = find_portal_mask_path(patient_dir)
    portal_mask = resample_bool_mask(load_mask_zyx_like_reference(portal_path, orig_img), target_shape)

    filtered, filter_info = filter_by_portal_region(
        candidate,
        portal_mask,
        spacing_zyx,
        bridge_mm=bridge_mm,
        portal_dilate_mm=portal_dilate_mm,
        min_bridge_voxels=min_bridge_voxels,
    )

    if overwrite:
        nii_out = patient_dir / "pretrain.nii.gz"
        stl_out = patient_dir / "pretrain.stl"
        npy_out = patient_dir / "vkan_work" / "pretrain_mask.npy"
        meta_out = patient_dir / "vkan_work" / "portal_region_filter_meta.json"
        cleanup_preview_outputs(patient_dir)
    else:
        nii_out = patient_dir / "pretrain_portal_filtered.nii.gz"
        stl_out = patient_dir / "pretrain_portal_filtered.stl"
        npy_out = patient_dir / "vkan_work" / "pretrain_mask_portal_filtered.npy"
        meta_out = patient_dir / "vkan_work" / "portal_region_filter_preview_meta.json"

    npy_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_out, filtered.astype(np.uint8))
    save_mask_like(filtered, orig_img, nii_out)
    zyx_mask_to_stl(filtered, orig_img.affine, stl_out, name="pretrain_portal_filtered")

    meta = {
        "patient": patient_dir.name,
        "patient_dir": str(patient_dir),
        "overwrite": overwrite,
        "orig": str(orig_path),
        "candidate_source": candidate_source,
        "liver_subtraction": liver_info,
        "portal_mask": str(portal_path),
        "output_npy": str(npy_out),
        "output_nii": str(nii_out),
        "output_stl": str(stl_out),
        "shape_zyx": list(target_shape),
        "spacing_zyx": list(spacing_zyx),
        "filter": filter_info,
    }
    meta_out.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def cleanup_preview_outputs(patient_dir: Path) -> None:
    for path in (
        patient_dir / "pretrain_portal_filtered.nii.gz",
        patient_dir / "pretrain_portal_filtered.stl",
        patient_dir / "vkan_work" / "pretrain_mask_portal_filtered.npy",
        patient_dir / "vkan_work" / "portal_region_filter_preview_meta.json",
    ):
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass


def iter_patient_dirs(data_root: Path, patient: Optional[str] = None) -> list[Path]:
    data_root = Path(data_root)
    if patient:
        p = Path(patient)
        if p.exists():
            return [p]
        return [data_root / patient]
    if (data_root / "orig.nii.gz").exists():
        return [data_root]
    return [
        p for p in sorted(data_root.iterdir())
        if p.is_dir() and (p / "orig.nii.gz").exists()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter pretrain mask by portal-vein NIfTI region growing.")
    parser.add_argument("--data_root", default=r"F:\PCG data\dataset\test4all_sample")
    parser.add_argument("--patient", default=None, help="Patient name or full patient directory path.")
    parser.add_argument("--candidate", default=None, help="Optional candidate mask: .npy or .nii.gz.")
    parser.add_argument("--preview", action="store_true", help="Write pretrain_portal_filtered.* instead of overwriting pretrain.*.")
    parser.add_argument("--keep_liver", action="store_true", help="Do not subtract segmentation/liver.nii.gz before portal filtering.")
    parser.add_argument("--bridge_mm", type=float, default=0.0)
    parser.add_argument("--portal_dilate_mm", type=float, default=2.0)
    parser.add_argument("--min_bridge_voxels", type=int, default=32)
    args = parser.parse_args()

    candidate = Path(args.candidate) if args.candidate else None
    for patient_dir in iter_patient_dirs(Path(args.data_root), args.patient):
        print(f"[portal-filter] {patient_dir.name}")
        try:
            meta = filter_patient(
                patient_dir,
                candidate_path=candidate,
                overwrite=not args.preview,
                subtract_liver=not args.keep_liver,
                bridge_mm=args.bridge_mm,
                portal_dilate_mm=args.portal_dilate_mm,
                min_bridge_voxels=args.min_bridge_voxels,
            )
            f = meta["filter"]
            print(
                f"  {f['input_voxels']} -> {f['output_voxels']} voxels, "
                f"kept {f.get('labels_kept', '?')}/{f.get('components_total', '?')} components"
            )
            print(f"  stl: {meta['output_stl']}")
        except Exception as exc:
            print(f"  FAILED: {exc}")


if __name__ == "__main__":
    main()
