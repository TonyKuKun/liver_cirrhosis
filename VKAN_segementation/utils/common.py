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
    """Find valid patient folders and derive standard input/output paths."""
    root = Path(root)
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
        return write_binary_stl(out_path, np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64))
    padded = np.pad(mask, 1, mode="constant")
    verts_zyx, faces, _, _ = measure.marching_cubes(padded, level=0.5, spacing=spacing_zyx)
    verts_zyx -= np.asarray(spacing_zyx, dtype=np.float32)
    return write_binary_stl(out_path, verts_zyx[:, [2, 1, 0]], faces)


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


def save_nifti_volume(volume: DicomVolume, out_path: str | Path) -> Path:
    """Write a simple NIfTI-1 .nii.gz volume in z-y-x array order."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(volume.volume_hu)
    if data.dtype == np.bool_:
        data = data.astype(np.uint8)
    if data.dtype == np.uint8:
        out = data.astype(np.uint8, copy=False)
        datatype, bitpix = 2, 8
    elif np.issubdtype(data.dtype, np.floating):
        out = data.astype("<f4", copy=False)
        datatype, bitpix = 16, 32
    else:
        out = data.astype("<i2", copy=False)
        datatype, bitpix = 4, 16
    header = bytearray(348)
    struct.pack_into("<I", header, 0, 348)
    struct.pack_into("<8h", header, 40, 3, int(out.shape[2]), int(out.shape[1]), int(out.shape[0]), 1, 1, 1, 1)
    struct.pack_into("<h", header, 70, datatype)
    struct.pack_into("<h", header, 72, bitpix)
    struct.pack_into(
        "<8f",
        header,
        76,
        0.0,
        float(volume.spacing_zyx[2]),
        float(volume.spacing_zyx[1]),
        float(volume.spacing_zyx[0]),
        1.0,
        1.0,
        1.0,
        1.0,
    )
    struct.pack_into("<f", header, 108, 352.0)
    struct.pack_into("<f", header, 112, 1.0)
    struct.pack_into("<f", header, 116, 0.0)
    struct.pack_into("<h", header, 252, 1)
    struct.pack_into("<h", header, 254, 1)
    struct.pack_into("<4f", header, 280, 1.0, 0.0, 0.0, float(volume.origin_xyz[0]))
    struct.pack_into("<4f", header, 296, 0.0, 1.0, 0.0, float(volume.origin_xyz[1]))
    struct.pack_into("<4f", header, 312, 0.0, 0.0, 1.0, float(volume.origin_xyz[2]))
    header[344:348] = b"n+1\0"
    with gzip.open(out_path, "wb") as fp:
        fp.write(header)
        fp.write(b"\0\0\0\0")
        fp.write(np.transpose(out, (2, 1, 0)).tobytes(order="C"))
    return out_path


def load_nifti_volume(path: str | Path) -> DicomVolume:
    path = Path(path)
    opener = gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")
    with opener as fp:
        raw = fp.read()
    if len(raw) < 352 or struct.unpack_from("<I", raw, 0)[0] != 348:
        raise ValueError(f"Not a NIfTI-1 file: {path}")
    dim = struct.unpack_from("<8h", raw, 40)
    nx, ny, nz = int(dim[1]), int(dim[2]), int(dim[3])
    datatype = struct.unpack_from("<h", raw, 70)[0]
    pixdim = struct.unpack_from("<8f", raw, 76)
    offset = int(round(struct.unpack_from("<f", raw, 108)[0]))
    scl_slope = struct.unpack_from("<f", raw, 112)[0] or 1.0
    scl_inter = struct.unpack_from("<f", raw, 116)[0]
    dtype_map = {2: np.uint8, 4: "<i2", 8: "<i4", 16: "<f4", 64: "<f8"}
    if datatype not in dtype_map:
        raise ValueError(f"Unsupported NIfTI datatype {datatype}: {path}")
    data = np.frombuffer(raw, dtype=dtype_map[datatype], count=nx * ny * nz, offset=offset).reshape((nx, ny, nz))
    volume = np.transpose(data, (2, 1, 0)).astype(np.float32) * float(scl_slope) + float(scl_inter)
    sx, sy, sz = float(pixdim[1] or 1.0), float(pixdim[2] or 1.0), float(pixdim[3] or 1.0)
    origin = (
        float(struct.unpack_from("<f", raw, 292)[0]),
        float(struct.unpack_from("<f", raw, 308)[0]),
        float(struct.unpack_from("<f", raw, 324)[0]),
    )
    return DicomVolume(volume_hu=volume, spacing_zyx=(sz, sy, sx), origin_xyz=origin)


def resize_mask_to_grid(mask: np.ndarray, grid_size: int) -> np.ndarray:
    mask = np.asarray(mask) > 0.5
    if mask.shape == (grid_size, grid_size, grid_size):
        return mask.astype(np.float32)
    try:
        from scipy import ndimage as ndi
    except ImportError as exc:
        raise ImportError("scipy is required to resize NIfTI masks.") from exc
    zoom = [grid_size / max(int(n), 1) for n in mask.shape]
    return (ndi.zoom(mask.astype(np.float32), zoom, order=0) > 0.5).astype(np.float32)


def volume_bounds_xyz(shape_zyx: tuple[int, int, int], spacing_zyx: tuple[float, float, float], origin_xyz: tuple[float, float, float]) -> np.ndarray:
    size_xyz = np.asarray([shape_zyx[2] * spacing_zyx[2], shape_zyx[1] * spacing_zyx[1], shape_zyx[0] * spacing_zyx[0]], dtype=np.float32)
    origin = np.asarray(origin_xyz, dtype=np.float32)
    return np.stack([origin, origin + size_xyz], axis=0)


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
    return write_ascii_stl(out_path, verts_xyz, faces, "prediction")

