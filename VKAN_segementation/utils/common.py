from __future__ import annotations

import base64
import gzip
import json
import os
import re
import struct
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


INVALID_MARKERS = ("@", "!", "&")


@dataclass
class DicomVolume:
    volume_hu: np.ndarray
    spacing_zyx: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class PatientCase:
    name: str
    path: Path
    dcm_dir: Path
    label_stl: Path
    pretrain_stl: Path
    predict_stl: Path
    is_post_tips: bool


def discover_patients(root: str | Path) -> list[PatientCase]:
    """Find valid patient folders and derive standard input/output paths."""
    root = Path(root)
    if (root / "dcm").exists() and not any(marker in root.name for marker in INVALID_MARKERS):
        return [
            PatientCase(
                name=root.name,
                path=root,
                dcm_dir=root / "dcm",
                label_stl=root / "vessel.stl",
                pretrain_stl=root / "pretrain.stl",
                predict_stl=root / "predict.stl",
                is_post_tips="#" in root.name,
            )
        ]

    cases: list[PatientCase] = []
    for path in sorted(p for p in root.iterdir() if p.is_dir()):
        if any(marker in path.name for marker in INVALID_MARKERS):
            continue
        dcm_dir = path / "dcm"
        if not dcm_dir.exists():
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


def save_nifti_volume(volume: DicomVolume, out_path: str | Path) -> Path:
    """Write a compact NIfTI-1 .nii.gz cache for z-y-x volumes."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.rint(volume.volume_hu).clip(np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype("<i2", copy=False)
    header = bytearray(348)
    struct.pack_into("<i", header, 0, 348)
    dim = [3, int(data.shape[0]), int(data.shape[1]), int(data.shape[2]), 1, 1, 1, 1]
    struct.pack_into("<8h", header, 40, *dim)
    struct.pack_into("<h", header, 70, 4)
    struct.pack_into("<h", header, 72, 16)
    pixdim = [0.0, float(volume.spacing_zyx[0]), float(volume.spacing_zyx[1]), float(volume.spacing_zyx[2]), 0.0, 0.0, 0.0, 0.0]
    struct.pack_into("<8f", header, 76, *pixdim)
    struct.pack_into("<f", header, 108, 352.0)
    struct.pack_into("<f", header, 112, 1.0)
    struct.pack_into("<f", header, 116, 0.0)
    struct.pack_into("<h", header, 252, 1)
    struct.pack_into("<h", header, 254, 1)
    struct.pack_into("<3f", header, 280, float(volume.origin_xyz[2]), float(volume.origin_xyz[1]), float(volume.origin_xyz[0]))
    header[344:348] = b"n+1\0"
    with gzip.open(out_path, "wb") as fp:
        fp.write(header)
        fp.write(b"\0\0\0\0")
        fp.write(data.tobytes(order="C"))
    return out_path


def load_nifti_volume(path: str | Path) -> DicomVolume:
    """Read a NIfTI-1 .nii.gz cache written by save_nifti_volume."""
    path = Path(path)
    try:
        import nibabel as nib
    except ImportError:
        nib = None
    if nib is not None:
        img = nib.load(str(path))
        data = np.asarray(img.dataobj, dtype=np.float32)
        spacing = tuple(round(float(v), 6) for v in img.header.get_zooms()[:3])
        affine = np.asarray(img.affine)
        origin = tuple(round(float(v), 6) for v in (affine[0, 3], affine[1, 3], affine[2, 3]))
        return DicomVolume(data, spacing, origin)

    with gzip.open(path, "rb") as fp:
        raw = fp.read()
    if len(raw) < 352 or struct.unpack_from("<i", raw, 0)[0] != 348:
        raise ValueError(f"Invalid NIfTI cache: {path}")
    dim = struct.unpack_from("<8h", raw, 40)
    shape = (int(dim[1]), int(dim[2]), int(dim[3]))
    datatype = struct.unpack_from("<h", raw, 70)[0]
    bitpix = struct.unpack_from("<h", raw, 72)[0]
    if datatype != 4 or bitpix != 16:
        raise ValueError(f"Unsupported NIfTI datatype in {path}: {datatype}/{bitpix}")
    pixdim = struct.unpack_from("<8f", raw, 76)
    offset = int(round(struct.unpack_from("<f", raw, 108)[0]))
    slope = float(struct.unpack_from("<f", raw, 112)[0]) or 1.0
    inter = float(struct.unpack_from("<f", raw, 116)[0])
    qoffset = struct.unpack_from("<3f", raw, 280)
    n_voxels = int(np.prod(shape))
    data = np.frombuffer(raw, dtype="<i2", count=n_voxels, offset=offset).reshape(shape).astype(np.float32)
    data = data * slope + inter
    spacing = (round(float(pixdim[1]), 6), round(float(pixdim[2]), 6), round(float(pixdim[3]), 6))
    origin = (round(float(qoffset[2]), 6), round(float(qoffset[1]), 6), round(float(qoffset[0]), 6))
    return DicomVolume(volume_hu=data, spacing_zyx=spacing, origin_xyz=origin)


def volume_bounds_xyz(shape_zyx: tuple[int, int, int], spacing_zyx: tuple[float, float, float], origin_xyz: tuple[float, float, float]) -> np.ndarray:
    extent_xyz = np.asarray(
        [shape_zyx[2] * spacing_zyx[2], shape_zyx[1] * spacing_zyx[1], shape_zyx[0] * spacing_zyx[0]],
        dtype=np.float32,
    )
    start = np.asarray(origin_xyz, dtype=np.float32)
    return np.stack([start, start + extent_xyz], axis=0)


def resize_mask_to_grid(mask: np.ndarray, grid_size: int) -> np.ndarray:
    mask = np.asarray(mask) > 0
    if mask.shape == (grid_size, grid_size, grid_size):
        return mask.astype(np.float32)
    try:
        from scipy import ndimage as ndi
    except ImportError as exc:
        raise ImportError("scipy is required to resize NIfTI masks for training.") from exc
    zoom = [grid_size / max(n, 1) for n in mask.shape]
    return (ndi.zoom(mask.astype(np.uint8), zoom=zoom, order=0) > 0).astype(np.float32)


class GemmaClient:
    """Small OpenAI-compatible chat client used by coarse planning and review."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemma-4-31b-it",
        base_url: str | None = None,
        timeout: int = 90,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMMA_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = (base_url or os.getenv("GEMMA_API_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "").rstrip("/")
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.base_url)

    def chat_json(self, system: str, prompt: str, image_paths: list[str | Path] | None = None) -> dict[str, Any]:
        if not self.enabled:
            return {}
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in image_paths or []:
            path = Path(path)
            if not path.exists():
                continue
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            b64 = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return {}
        message = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        if isinstance(message, list):
            message = "".join(part.get("text", "") for part in message if isinstance(part, dict))
        return _extract_json(str(message))


def _extract_json(text: str) -> dict[str, Any]:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return {}
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            return {}


def mask_to_stl(mask: np.ndarray, spacing_zyx: tuple[float, float, float], out_path: str | Path) -> Path:
    """Convert a binary z-y-x mask to STL in x-y-z physical coordinates."""
    try:
        from skimage import measure
    except ImportError as exc:
        raise ImportError("scikit-image is required for marching cubes.") from exc

    out_path = Path(out_path)
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.sum() == 0:
        raise ValueError("Cannot write STL for an empty mask.")
    padded = np.pad(mask, 1, mode="constant")
    verts_zyx, faces, _, _ = measure.marching_cubes(padded, level=0.5, spacing=spacing_zyx)
    verts_zyx -= np.asarray(spacing_zyx, dtype=np.float32)
    write_binary_stl(out_path, verts_zyx[:, [2, 1, 0]], faces)
    return out_path


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
            fp.write(struct.pack("<12fH", *(normal.tolist() + p0.tolist() + p1.tolist() + p2.tolist()), 0))
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
    mesh = trimesh.load_mesh(str(in_path), process=True)
    if hasattr(mesh, "geometry"):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    filter_laplacian(mesh, lamb=0.45, iterations=iterations)
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
    return write_binary_stl(out_path, verts_xyz, faces, "prediction")

