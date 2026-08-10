r"""Generate mask.stl and mask_smooth.stl directly from mask.nii.gz.

Run every case containing mask.nii.gz:
    python tests/generate_mask_stls.py

Run selected cases:
    python tests/generate_mask_stls.py --patient "20230108TangXiuQin@centerline"
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from postprocess.check_and_smooth import _mesh_summary
from utils.common import nifti_mask_to_stl


DEFAULT_DATA_ROOT = Path(r"F:\PCG data\dataset\test4all_sample")


def discover_cases(data_root: Path, patient_names: list[str]) -> list[Path]:
    if patient_names:
        cases = [data_root / name for name in patient_names]
        missing = [path for path in cases if not (path / "mask.nii.gz").exists()]
        if missing:
            preview = ", ".join(str(path) for path in missing[:8])
            raise FileNotFoundError(f"mask.nii.gz is missing for: {preview}")
    elif (data_root / "mask.nii.gz").exists():
        cases = [data_root]
    else:
        cases = sorted(
            path
            for path in data_root.iterdir()
            if path.is_dir() and (path / "mask.nii.gz").exists()
        )
    return cases


def _needs_update(output_path: Path, source_path: Path, force: bool) -> bool:
    return force or not output_path.exists() or output_path.stat().st_mtime_ns < source_path.stat().st_mtime_ns


def _count_mask_components(mask: np.ndarray) -> int:
    from scipy import ndimage

    indices = np.argwhere(mask)
    lower = indices.min(axis=0)
    upper = indices.max(axis=0) + 1
    cropped = mask[tuple(slice(int(lower[i]), int(upper[i])) for i in range(3))]
    _, count = ndimage.label(cropped, structure=np.ones((3, 3, 3), dtype=np.uint8))
    return int(count)


def smooth_connected_mask_to_stl(
    mask: np.ndarray,
    affine: np.ndarray,
    reference_path: Path,
    output_path: Path,
    *,
    sigma_mm: float = 1.0,
    upsample_factor: int = 2,
) -> dict[str, object]:
    """Smooth a connected mask as a scalar volume before extracting its surface."""
    import trimesh
    from nibabel.affines import apply_affine
    from scipy import ndimage
    from skimage import measure

    sigma_mm = max(0.2, min(float(sigma_mm), 3.0))
    upsample_factor = max(1, min(int(upsample_factor), 4))
    affine = np.asarray(affine, dtype=np.float64)
    spacing_mm = np.linalg.norm(affine[:3, :3], axis=0)
    if not np.all(np.isfinite(spacing_mm)) or np.any(spacing_mm <= 0):
        raise ValueError(f"Invalid NIfTI spacing: {spacing_mm.tolist()}")

    sigma_voxels = sigma_mm / spacing_mm
    indices = np.argwhere(mask)
    margin = np.ceil(4.0 * sigma_voxels).astype(np.int64) + 2
    lower = np.maximum(indices.min(axis=0) - margin, 0)
    upper = np.minimum(indices.max(axis=0) + margin + 1, mask.shape)
    crop_slices = tuple(slice(int(lower[i]), int(upper[i])) for i in range(3))
    cropped = mask[crop_slices].astype(np.float32)

    scalar_field = ndimage.gaussian_filter(
        cropped,
        sigma=sigma_voxels,
        mode="constant",
        cval=0.0,
    )
    if upsample_factor > 1:
        scalar_field = ndimage.zoom(
            scalar_field,
            upsample_factor,
            order=3,
            mode="grid-constant",
            cval=0.0,
            prefilter=True,
            grid_mode=True,
        )

    vertices, faces, _, _ = measure.marching_cubes(
        scalar_field,
        level=0.5,
        step_size=upsample_factor,
        allow_degenerate=False,
    )
    if upsample_factor > 1:
        vertices = (vertices + 0.5) / upsample_factor - 0.5
    vertices += lower
    vertices_world = apply_affine(affine, vertices)
    mesh = trimesh.Trimesh(vertices=vertices_world, faces=faces, process=True)
    nondegenerate = getattr(mesh, "nondegenerate_faces", None)
    if callable(nondegenerate):
        mesh.update_faces(nondegenerate())
        mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    if mesh.is_empty or len(mesh.faces) == 0:
        raise RuntimeError("Volume smoothing produced an empty mesh")

    components = mesh.split(only_watertight=False)
    if len(components) != 1:
        raise RuntimeError(f"Volume smoothing changed one component into {len(components)} components")

    reference = trimesh.load_mesh(str(reference_path), process=True)
    if not isinstance(reference, trimesh.Trimesh) or reference.is_empty:
        raise RuntimeError(f"Invalid reference mesh: {reference_path}")

    extent_ratio = np.divide(
        mesh.extents,
        reference.extents,
        out=np.ones(3, dtype=np.float64),
        where=reference.extents > 0,
    )
    center_shift = float(np.linalg.norm(mesh.bounds.mean(axis=0) - reference.bounds.mean(axis=0)))
    area_ratio = float(mesh.area / reference.area) if reference.area > 0 else 1.0
    volume_ratio = (
        float(abs(mesh.volume) / abs(reference.volume))
        if mesh.is_watertight and reference.is_watertight and abs(reference.volume) > 0
        else None
    )
    invalid_volume = volume_ratio is not None and not 0.80 <= volume_ratio <= 1.10
    if (
        not np.all(np.isfinite(extent_ratio))
        or np.any(extent_ratio < 0.80)
        or np.any(extent_ratio > 1.20)
        or not np.isfinite(area_ratio)
        or not 0.70 <= area_ratio <= 1.10
        or center_shift > max(2.0, 2.0 * sigma_mm)
        or invalid_volume
    ):
        raise RuntimeError(
            "Volume smoothing changed mesh geometry too much: "
            f"extent_ratio={extent_ratio.tolist()}, area_ratio={area_ratio:.3f}, "
            f"volume_ratio={volume_ratio}, center_shift={center_shift:.3f} mm"
        )

    temporary_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    mesh.export(str(temporary_path), file_type="stl")
    temporary_path.replace(output_path)
    return {
        "method": "isotropic-gaussian-resample-marching-cubes",
        "sigma_mm": sigma_mm,
        "sigma_voxels": sigma_voxels.tolist(),
        "upsample_factor": upsample_factor,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "extent_ratio": extent_ratio.tolist(),
        "area_ratio": area_ratio,
        "volume_ratio": volume_ratio,
        "center_shift_mm": center_shift,
    }


def smooth_fragmented_mask_stl(
    input_path: Path,
    output_path: Path,
    *,
    iterations: int,
    strength: float,
    min_component_faces: int = 64,
) -> dict[str, object]:
    """Smooth a mask mesh without turning disconnected fragments into points.

    Binary STL stores three vertex records per triangle.  Loading it with
    ``process=False`` leaves every triangle disconnected, so a vertex filter
    moves each triangle independently and collapses it toward its centroid.
    We explicitly merge the STL vertices first, then smooth only components
    large enough to have a meaningful surface.  Small components are copied
    unchanged so this diagnostic output does not hide label fragmentation.
    """
    import trimesh
    from trimesh.smoothing import filter_humphrey, filter_taubin

    # ``process=True`` is important here: the mask STL writer emits triangle
    # records with duplicated vertices, while smoothing requires shared
    # topology to construct a neighbourhood for each vertex.
    mesh = trimesh.load_mesh(str(input_path), process=True)
    if hasattr(mesh, "geometry"):
        meshes = [part for part in mesh.geometry.values() if isinstance(part, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError(f"No mesh geometry found in {input_path}")
        mesh = trimesh.util.concatenate(meshes)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise RuntimeError(f"Invalid mesh: {input_path}")

    mesh.merge_vertices()
    components = mesh.split(only_watertight=False)
    if not components:
        raise RuntimeError(f"Mesh has no connected components: {input_path}")

    iterations = max(1, min(int(iterations), 80))
    strength = max(0.05, min(float(strength), 0.45))
    min_component_faces = max(8, int(min_component_faces))
    smoothable = [part for part in components if len(part.faces) >= min_component_faces]
    preserved_count = len(components) - len(smoothable)

    # A fragmented label is evidence that the input needs inspection.  Do not
    # manufacture a cleaner-looking label by deleting its small components.
    # If nothing is large enough to smooth, keep the faithful raw surface.
    if not smoothable:
        temporary_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
        shutil.copy2(input_path, temporary_path)
        temporary_path.replace(output_path)
        return {
            "method": "copy-fragmented-mask",
            "components": int(len(components)),
            "smoothed_components": 0,
            "preserved_components": int(preserved_count),
            "min_component_faces": min_component_faces,
        }

    original_area = float(mesh.area)
    original_center = mesh.bounds.mean(axis=0)
    original_extents = mesh.extents.copy()

    smoothed_parts = []
    smoothed_count = 0
    fallback_count = 0
    displacements: list[float] = []
    max_displacements: list[float] = []

    for part in components:
        if len(part.faces) < min_component_faces:
            smoothed_parts.append(part)
            continue

        original_part = part.copy()
        part_center = part.bounds.mean(axis=0)
        part_extents = part.extents.copy()
        part_vertices = part.vertices.copy()
        part_area = float(part.area)

        # Fragmented masks are intentionally smoothed conservatively.  A few
        # iterations remove voxel stair steps without erasing thin branches.
        part_iterations = min(iterations, 8 if len(components) > 1 else iterations)
        # trimesh applies the expansion sign internally.  ``nu`` must be a
        # little larger than ``lamb`` for Taubin's shrinkage compensation.
        taubin_nu = strength / (1.0 - 0.1 * strength)
        filter_taubin(part, lamb=strength, nu=taubin_nu, iterations=part_iterations)
        filter_humphrey(part, alpha=0.08, beta=0.55, iterations=max(1, part_iterations // 3))
        part.fix_normals()
        part.apply_translation(part_center - part.bounds.mean(axis=0))

        extent_ratio = np.divide(
            part.extents,
            part_extents,
            out=np.ones(3, dtype=np.float64),
            where=part_extents > 0,
        )
        area_ratio = float(part.area / part_area) if part_area > 0 else 1.0
        if (
            not np.all(np.isfinite(extent_ratio))
            or np.any(extent_ratio < 0.70)
            or np.any(extent_ratio > 1.30)
            or not np.isfinite(area_ratio)
            or area_ratio < 0.70
            or area_ratio > 1.30
        ):
            # Preserve the original component if this particular surface is
            # too thin for the requested smoothing strength.
            smoothed_parts.append(original_part)
            fallback_count += 1
            continue

        displacement = np.linalg.norm(part.vertices - part_vertices, axis=1)
        displacements.append(float(np.mean(displacement)))
        max_displacements.append(float(np.max(displacement)))
        smoothed_parts.append(part)
        smoothed_count += 1

    mesh = trimesh.util.concatenate(smoothed_parts)
    mesh.fix_normals()
    mesh.apply_translation(original_center - mesh.bounds.mean(axis=0))

    extent_ratio = np.divide(
        mesh.extents,
        original_extents,
        out=np.ones(3, dtype=np.float64),
        where=original_extents > 0,
    )
    global_area_ratio = float(mesh.area / original_area) if original_area > 0 else 1.0
    if (
        not np.all(np.isfinite(extent_ratio))
        or np.any(extent_ratio < 0.70)
        or np.any(extent_ratio > 1.30)
        or not np.isfinite(global_area_ratio)
        or global_area_ratio < 0.70
        or global_area_ratio > 1.30
    ):
        raise RuntimeError(
            "Smoothing changed mesh geometry too much: "
            f"extent_ratio={extent_ratio.tolist()}, area_ratio={global_area_ratio:.3f}"
        )

    center_shift = float(np.linalg.norm(mesh.bounds.mean(axis=0) - original_center))
    if center_shift > 1e-3:
        raise RuntimeError(f"Smoothing shifted mesh center by {center_shift:.6f} mm")

    temporary_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    mesh.export(str(temporary_path), file_type="stl")
    temporary_path.replace(output_path)
    return {
        "method": "merged-component-wise-taubin+humphrey",
        "iterations": iterations,
        "strength": strength,
        "components": int(len(components)),
        "smoothed_components": int(smoothed_count),
        "preserved_components": int(preserved_count + fallback_count),
        "min_component_faces": min_component_faces,
        "extent_ratio": extent_ratio.tolist(),
        "area_ratio": global_area_ratio,
        "center_shift_mm": center_shift,
        "mean_vertex_displacement": float(np.mean(displacements)) if displacements else 0.0,
        "max_vertex_displacement": float(np.max(max_displacements)) if max_displacements else 0.0,
    }


def generate_case(
    case_dir: Path,
    *,
    iterations: int = 20,
    strength: float = 0.25,
    min_component_faces: int = 64,
    smooth_sigma_mm: float = 1.0,
    smooth_upsample_factor: int = 2,
    force: bool = False,
) -> dict[str, object]:
    mask_path = case_dir / "mask.nii.gz"
    mask_stl_path = case_dir / "mask.stl"
    smooth_stl_path = case_dir / "mask_smooth.stl"

    image = nib.load(str(mask_path))
    mask = np.asarray(image.dataobj) > 0.5
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape={mask.shape}: {mask_path}")
    if not mask.any():
        raise ValueError(f"Mask is empty: {mask_path}")

    if _needs_update(mask_stl_path, mask_path, force):
        nifti_mask_to_stl(mask, image.affine, mask_stl_path, name="mask")
        mask_action = "wrote"
    else:
        mask_action = "kept"

    if _needs_update(smooth_stl_path, mask_stl_path, force):
        try:
            mask_components = _count_mask_components(mask)
            if mask_components == 1:
                smoothing = smooth_connected_mask_to_stl(
                    mask,
                    image.affine,
                    mask_stl_path,
                    smooth_stl_path,
                    sigma_mm=smooth_sigma_mm,
                    upsample_factor=smooth_upsample_factor,
                )
            else:
                smoothing = smooth_fragmented_mask_stl(
                    mask_stl_path,
                    smooth_stl_path,
                    iterations=iterations,
                    strength=strength,
                    min_component_faces=min_component_faces,
                )
            smoothing["mask_components"] = mask_components
            smooth_action = "wrote"
        except Exception as exc:
            # Never leave a stale point-like output when smoothing is not
            # geometrically safe.  The raw STL remains the source of truth.
            shutil.copy2(mask_stl_path, smooth_stl_path)
            smoothing = {"method": "copy-fallback", "error": str(exc)}
            smooth_action = "wrote-copy-fallback"
            print(f"[mask-stl] {case_dir.name}: smoothing fallback to raw mask: {exc}", flush=True)
    else:
        smoothing = {"method": "existing-file"}
        smooth_action = "kept"

    pretrain_path = case_dir / "pretrain.stl"
    return {
        "case": case_dir.name,
        "mask_voxels": int(mask.sum()),
        "mask_action": mask_action,
        "smooth_action": smooth_action,
        "mask": _mesh_summary(mask_stl_path),
        "mask_smooth": _mesh_summary(smooth_stl_path),
        "pretrain": _mesh_summary(pretrain_path) if pretrain_path.exists() else None,
        "smoothing": smoothing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--patient", action="append", default=[], help="Patient folder name; repeat as needed.")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--min_component_faces", type=int, default=64)
    parser.add_argument("--smooth_sigma_mm", type=float, default=1.0)
    parser.add_argument("--smooth_upsample_factor", type=int, default=2)
    parser.add_argument("--force", action="store_true", help="Overwrite current mask STL outputs.")
    args = parser.parse_args()

    cases = discover_cases(args.data_root, args.patient)
    print(f"[mask-stl] cases={len(cases)} data_root={args.data_root}", flush=True)
    failures = 0
    for case_dir in cases:
        try:
            result = generate_case(
                case_dir,
                iterations=args.iterations,
                strength=args.strength,
                min_component_faces=args.min_component_faces,
                smooth_sigma_mm=args.smooth_sigma_mm,
                smooth_upsample_factor=args.smooth_upsample_factor,
                force=args.force,
            )
            mask_bounds = result["mask"].get("bounds")  # type: ignore[union-attr]
            smooth_bounds = result["mask_smooth"].get("bounds")  # type: ignore[union-attr]
            pretrain = result["pretrain"]
            pretrain_bounds = pretrain.get("bounds") if isinstance(pretrain, dict) else None
            print(
                f"[mask-stl] {result['case']}: voxels={result['mask_voxels']} "
                f"mask={result['mask_action']} smooth={result['smooth_action']}\n"
                f"  mask_bounds={mask_bounds}\n"
                f"  smooth_bounds={smooth_bounds}\n"
                f"  pretrain_bounds={pretrain_bounds}\n"
                f"  smoothing={result['smoothing']}",
                flush=True,
            )
        except Exception as exc:
            failures += 1
            print(f"[mask-stl] {case_dir.name}: FAILED: {exc}", flush=True)

    if failures:
        raise SystemExit(f"{failures} case(s) failed")


if __name__ == "__main__":
    main()
