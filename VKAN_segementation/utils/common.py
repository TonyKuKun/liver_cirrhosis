from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


INVALID_MARKERS = ("@", "!", "&")


@dataclass(frozen=True)
class PatientCase:
    name: str
    path: Path
    dcm_dir: Path
    label_stl: Path
    pretrain_stl: Path
    predict_stl: Path
    is_post_tips: bool


@dataclass(frozen=True)
class DicomVolume:
    volume_hu: np.ndarray
    spacing_zyx: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]


def discover_patients(root: str | Path) -> list[PatientCase]:
    """Find patient folders and derive standard input/output paths."""
    root = Path(root)
    cases: list[PatientCase] = []
    patient_dirs = [root] if (root / "dcm").is_dir() else sorted(p for p in root.iterdir() if p.is_dir())
    for path in patient_dirs:
        dcm_dir = path / "dcm"
        if not dcm_dir.is_dir():
            continue
        cases.append(
            PatientCase(
                name=path.name,
                path=path,
                dcm_dir=dcm_dir,
                label_stl=path / "vessel.stl",
                pretrain_stl=path / "pretrain.stl",
                predict_stl=path / "predict.stl",
                is_post_tips="#" in path.name,
            )
        )
    return cases


def require_existing_labels(cases: Iterable[PatientCase]) -> list[PatientCase]:
    return [case for case in cases if case.label_stl.exists()]


def mask_to_stl(
    mask: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    out_path: str | Path,
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Path:
    """Convert a binary z-y-x mask to STL in x-y-z patient coordinates."""
    try:
        from skimage import measure
    except ImportError as exc:
        raise ImportError("scikit-image is required for marching cubes.") from exc

    out_path = Path(out_path)
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.sum() == 0:
        return write_binary_stl(out_path, np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64))
    padded = np.pad(mask, 1, mode="constant")
    verts_zyx, faces, _, _ = measure.marching_cubes(padded, level=0.5, spacing=spacing_zyx)
    verts_zyx -= np.asarray(spacing_zyx, dtype=np.float32)
    verts_xyz = verts_zyx[:, [2, 1, 0]] + np.asarray(origin_xyz, dtype=np.float32)
    return write_binary_stl(out_path, verts_xyz, faces)


def nifti_mask_to_stl(mask_xyz: np.ndarray, affine: np.ndarray, out_path: str | Path, name: str = "vessel") -> Path:
    """Convert a binary NIfTI-order x-y-z mask to STL using the NIfTI affine."""
    try:
        from nibabel.affines import apply_affine
        from skimage import measure
    except ImportError as exc:
        raise ImportError("nibabel and scikit-image are required for NIfTI STL export.") from exc

    out_path = Path(out_path)
    mask = np.asarray(mask_xyz, dtype=np.uint8)
    if mask.sum() == 0:
        return write_binary_stl(out_path, np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64))
    verts_ijk, faces, _, _ = measure.marching_cubes(np.pad(mask, 1), level=0.5)
    verts_ijk -= 1.0
    verts_xyz = apply_affine(np.asarray(affine, dtype=np.float64), verts_ijk)
    return write_binary_stl(out_path, np.asarray(verts_xyz, dtype=np.float32), faces, name)


def zyx_mask_to_stl(mask_zyx: np.ndarray, affine: np.ndarray, out_path: str | Path, name: str = "vessel") -> Path:
    """Convert an internal z-y-x mask to STL using the matching NIfTI affine."""
    mask_xyz = np.transpose(np.asarray(mask_zyx), (2, 1, 0))
    return nifti_mask_to_stl(mask_xyz, affine, out_path, name=name)


def write_binary_stl(out_path: str | Path, vertices_xyz: np.ndarray, faces: np.ndarray, name: str = "vessel") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    v = np.asarray(vertices_xyz, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)
    header = f"{name} binary stl".encode("ascii", errors="ignore")[:80].ljust(80, b" ")
    with out_path.open("wb") as fp:
        fp.write(header)
        fp.write(struct.pack("<I", len(f)))
        for tri in f:
            p0, p1, p2 = v[tri[0]], v[tri[1]], v[tri[2]]
            normal = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            fp.write(struct.pack("<3f", float(normal[0]), float(normal[1]), float(normal[2])))
            for p in (p0, p1, p2):
                fp.write(struct.pack("<3f", float(p[0]), float(p[1]), float(p[2])))
            fp.write(struct.pack("<H", 0))
    return out_path


def write_ascii_stl(out_path: str | Path, vertices_xyz: np.ndarray, faces: np.ndarray, name: str = "vessel") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    v = np.asarray(vertices_xyz, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    with out_path.open("w", encoding="ascii") as fp:
        fp.write(f"solid {name}\n")
        for tri in f:
            p0, p1, p2 = v[tri[0]], v[tri[1]], v[tri[2]]
            normal = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            fp.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
            fp.write("    outer loop\n")
            for p in (p0, p1, p2):
                fp.write(f"      vertex {p[0]:.6e} {p[1]:.6e} {p[2]:.6e}\n")
            fp.write("    endloop\n  endfacet\n")
        fp.write(f"endsolid {name}\n")
    return out_path


def smooth_stl(in_path: str | Path, out_path: str | Path, iterations: int = 8) -> Path:
    """Smooth an STL if trimesh is available, otherwise copy it unchanged."""
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import trimesh
        from trimesh.smoothing import filter_laplacian
    except ImportError:
        out_path.write_bytes(in_path.read_bytes())
        return out_path
    mesh = trimesh.load_mesh(str(in_path), process=False)
    if hasattr(mesh, "geometry"):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    original_center = np.asarray(mesh.bounds, dtype=np.float64).mean(axis=0)
    try:
        filter_laplacian(mesh, lamb=0.45, iterations=iterations, volume_constraint=False)
    except TypeError:
        filter_laplacian(mesh, lamb=0.45, iterations=iterations)
    smoothed_center = np.asarray(mesh.bounds, dtype=np.float64).mean(axis=0)
    if np.all(np.isfinite(original_center)) and np.all(np.isfinite(smoothed_center)):
        mesh.apply_translation(original_center - smoothed_center)
    mesh.export(str(out_path))
    return out_path


def stl_to_voxels(path: str | Path, grid_size: int = 96, bounds: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Voxelize STL to a dense z-y-x grid. Returns occupancy and xyz bounds."""
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError("trimesh is required for STL voxelization.") from exc

    mesh = trimesh.load_mesh(str(path), process=True)
    if hasattr(mesh, "geometry"):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if bounds is None:
        bounds = np.asarray(mesh.bounds, dtype=np.float32)
        center = bounds.mean(axis=0)
        extent = float(np.max(bounds[1] - bounds[0]))
        pad = extent * 0.08 + 1e-3
        bounds = np.stack([center - extent / 2 - pad, center + extent / 2 + pad], axis=0)

    pitch = float(np.max(bounds[1] - bounds[0]) / grid_size)
    vox = mesh.voxelized(pitch).fill()
    idx = np.floor((vox.points - bounds[0]) / pitch).astype(np.int64)
    valid = np.all((idx >= 0) & (idx < grid_size), axis=1)
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    idx = idx[valid]
    if len(idx):
        grid[idx[:, 2], idx[:, 1], idx[:, 0]] = 1.0
    return grid, bounds.astype(np.float32)


def voxels_to_stl(grid: np.ndarray, bounds: np.ndarray, out_path: str | Path, threshold: float = 0.5) -> Path:
    try:
        from skimage import measure
    except ImportError as exc:
        raise ImportError("scikit-image is required for marching cubes.") from exc

    grid = np.asarray(grid)
    mask = grid >= threshold
    if mask.sum() == 0:
        raise ValueError("Prediction is empty.")
    spacing_xyz = (bounds[1] - bounds[0]) / np.asarray([grid.shape[2], grid.shape[1], grid.shape[0]], dtype=np.float32)
    verts_zyx, faces, _, _ = measure.marching_cubes(np.pad(mask.astype(np.uint8), 1), level=0.5)
    verts_zyx -= 1.0
    verts_xyz = np.empty_like(verts_zyx)
    verts_xyz[:, 0] = bounds[0, 0] + verts_zyx[:, 2] * spacing_xyz[0]
    verts_xyz[:, 1] = bounds[0, 1] + verts_zyx[:, 1] * spacing_xyz[1]
    verts_xyz[:, 2] = bounds[0, 2] + verts_zyx[:, 0] * spacing_xyz[2]
    return write_ascii_stl(out_path, verts_xyz, faces, "prediction")

