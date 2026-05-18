"""Datasets and STL preprocessing for architecture benchmarks."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import (
    AUX_FLAG_INDICES,
    AUX_KEYS,
    DEFAULT_N_POINTS,
    N_AUX,
    N_PROFILE_FEAT,
    N_SEGMENTS,
    PROFILE_KEYS,
    SEGMENTS,
    PortalVeinDataset,
)


DATASET_MODES = ("numeric_only", "stl_only", "stl_numeric")
STL_GLOBAL_KEYS = (
    "vessel_valid", "vessel_volume_ml", "vessel_surface_area_mm2", "vessel_bbox_x", "vessel_bbox_y",
    "vessel_bbox_z", "vessel_pca_elongation", "vessel_pca_flatness",
    "spleen_valid", "spleen_volume_ml", "spleen_surface_area_mm2", "spleen_bbox_x", "spleen_bbox_y",
    "spleen_bbox_z", "spleen_pca_elongation", "spleen_pca_flatness",
    "liver_valid", "liver_volume_ml", "liver_surface_area_mm2", "liver_bbox_x", "liver_bbox_y",
    "liver_bbox_z", "liver_pca_elongation", "liver_pca_flatness",
)
N_STL_GLOBAL = len(STL_GLOBAL_KEYS)


def subject_id_from_name(name: str) -> str:
    core = re.sub(r"^\d+", "", str(name))
    return core.split("#", 1)[0]


def stable_seed(text: str, offset: int = 0) -> int:
    value = 2166136261 + int(offset)
    for ch in str(text):
        value ^= ord(ch)
        value = (value * 16777619) & 0xFFFFFFFF
    return int(value)


def find_stl_path(patient_dir: str | os.PathLike, candidates: Sequence[str]) -> str | None:
    seg_dir = Path(patient_dir) / "segmentation"
    for name in candidates:
        direct = seg_dir / name
        if direct.exists():
            return str(direct)
    lower = {p.name.lower(): p for p in seg_dir.glob("*.stl")} if seg_dir.exists() else {}
    for name in candidates:
        p = lower.get(name.lower())
        if p is not None:
            return str(p)
    return None


def read_stl_mesh(stl_path: str | os.PathLike) -> Tuple[np.ndarray, np.ndarray]:
    """Read binary or ASCII STL and return vertices and triangular faces."""
    path = os.fspath(stl_path)
    with open(path, "rb") as f:
        header = f.read(80)
        count_bytes = f.read(4)
    is_ascii = header[:5].lower() == b"solid" and b"\x00" not in header
    if not is_ascii and len(count_bytes) == 4:
        try:
            return _read_binary_stl(path)
        except Exception:
            return _read_ascii_stl(path)
    return _read_ascii_stl(path)


def _read_ascii_stl(path: str) -> Tuple[np.ndarray, np.ndarray]:
    verts = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4 and parts[0].lower() == "vertex":
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    vertices = np.asarray(verts, dtype=np.float32)
    if len(vertices) < 3:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64)
    n_tri = len(vertices) // 3
    vertices = vertices[: n_tri * 3]
    faces = np.arange(n_tri * 3, dtype=np.int64).reshape(n_tri, 3)
    return vertices, faces


def _read_binary_stl(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        f.read(80)
        n_tri = int(np.frombuffer(f.read(4), dtype="<u4")[0])
        data = f.read(n_tri * 50)
    if len(data) < n_tri * 50:
        raise ValueError(f"Binary STL is truncated: {path}")
    dt = np.dtype([("normal", "<f4", 3), ("verts", "<f4", (3, 3)), ("attr", "<u2")])
    facets = np.frombuffer(data, dtype=dt, count=n_tri)
    tri = facets["verts"].astype(np.float32)
    vertices = tri.reshape(-1, 3)
    faces = np.arange(n_tri * 3, dtype=np.int64).reshape(n_tri, 3)
    return vertices, faces


def mesh_surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    if len(vertices) == 0 or len(faces) == 0:
        return np.nan
    tri = vertices[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    return float(np.nansum(area))


def mesh_volume_ml(vertices: np.ndarray, faces: np.ndarray) -> float:
    if len(vertices) == 0 or len(faces) == 0:
        return np.nan
    tri = vertices[faces].astype(np.float64)
    signed = np.sum(tri[:, 0] * np.cross(tri[:, 1], tri[:, 2]), axis=1) / 6.0
    volume_mm3 = abs(float(np.nansum(signed)))
    return volume_mm3 / 1000.0 if volume_mm3 > 0 else np.nan


def mesh_global_features(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    out = np.zeros(8, dtype=np.float32)
    if len(vertices) == 0:
        return out
    bbox = np.nanmax(vertices, axis=0) - np.nanmin(vertices, axis=0)
    centered = vertices - np.nanmean(vertices, axis=0, keepdims=True)
    if len(vertices) >= 3:
        cov = np.cov(centered.T)
        eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
        elong = eig[0] / max(eig[1], 1e-6)
        flat = eig[1] / max(eig[2], 1e-6)
    else:
        elong = 0.0
        flat = 0.0
    out[:] = [
        1.0,
        mesh_volume_ml(vertices, faces),
        mesh_surface_area(vertices, faces),
        bbox[0],
        bbox[1],
        bbox[2],
        elong,
        flat,
    ]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def sample_mesh_points(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_points: int,
    seed: int,
) -> Tuple[np.ndarray, float]:
    """Sample a fixed-size surface point cloud. Returns points and valid flag."""
    n_points = int(n_points)
    if len(vertices) == 0 or n_points <= 0:
        return np.zeros((max(n_points, 0), 3), dtype=np.float32), 0.0
    rng = np.random.RandomState(seed)
    if len(faces) == 0:
        idx = rng.choice(len(vertices), size=n_points, replace=len(vertices) < n_points)
        return vertices[idx].astype(np.float32), 1.0

    tri = vertices[faces].astype(np.float64)
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    areas = np.nan_to_num(areas, nan=0.0, posinf=0.0, neginf=0.0)
    if areas.sum() <= 0:
        idx = rng.choice(len(vertices), size=n_points, replace=len(vertices) < n_points)
        return vertices[idx].astype(np.float32), 1.0
    face_idx = rng.choice(len(faces), size=n_points, replace=True, p=areas / areas.sum())
    chosen = tri[face_idx]
    u = rng.rand(n_points, 1)
    v = rng.rand(n_points, 1)
    sqrt_u = np.sqrt(u)
    pts = (1.0 - sqrt_u) * chosen[:, 0] + sqrt_u * (1.0 - v) * chosen[:, 1] + sqrt_u * v * chosen[:, 2]
    return pts.astype(np.float32), 1.0


def _resample_array(values: np.ndarray, n_target: int) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    out = np.zeros((n_target, values.shape[-1] if values.ndim == 2 else 1), dtype=np.float32)
    valid = np.zeros(n_target, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if len(values) == 0:
        return out, valid
    finite = np.isfinite(values).all(axis=1)
    values = values[finite]
    if len(values) == 0:
        return out, valid
    if len(values) == 1:
        out[:] = values[0]
        valid[:] = 1.0
        return out, valid
    xp = np.linspace(0.0, 1.0, len(values))
    xnew = np.linspace(0.0, 1.0, n_target)
    for j in range(values.shape[1]):
        out[:, j] = np.interp(xnew, xp, values[:, j])
    valid[:] = 1.0
    return out, valid


def load_centerline_positions(unified_file: str | os.PathLike, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    out = np.zeros((N_SEGMENTS, n_points, 3), dtype=np.float32)
    valid = np.zeros((N_SEGMENTS, n_points), dtype=np.float32)
    if not os.path.exists(unified_file):
        return out, valid
    with open(unified_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    pointwise = data.get("pointwise", {}) or {}
    for si, seg in enumerate(SEGMENTS):
        seg_data = pointwise.get(seg, {}) or {}
        pos = seg_data.get("position", None)
        if pos is None:
            continue
        arr = np.asarray(pos, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3:
            continue
        out[si], valid[si] = _resample_array(arr, n_points)
    return out, valid


@dataclass
class FoldNormalizer:
    profile_mean: np.ndarray
    profile_std: np.ndarray
    aux_mean: np.ndarray
    aux_std: np.ndarray
    stl_global_mean: np.ndarray
    stl_global_std: np.ndarray
    label_mean: float
    label_std: float


def compute_normalizer(records: Sequence[Mapping[str, object]], indices: Sequence[int]) -> FoldNormalizer:
    profile_vals = [[] for _ in range(N_PROFILE_FEAT)]
    aux_vals = [[] for _ in range(N_AUX)]
    stl_vals = [[] for _ in range(N_STL_GLOBAL)]
    labels = []
    for idx in indices:
        rec = records[int(idx)]
        labels.append(float(rec["label"]))
        profiles = rec["profiles"]
        valid = rec["point_valid"] * rec["segment_mask"][:, None]
        for fi in range(N_PROFILE_FEAT):
            vals = profiles[:, :, fi][valid > 0.5]
            if len(vals):
                profile_vals[fi].append(vals)
        aux = rec["aux_scalars"]
        aux_mask = rec["aux_mask"]
        for ai in range(N_AUX):
            if ai in AUX_FLAG_INDICES:
                continue
            vals = aux[ai:ai + 1][aux_mask[ai:ai + 1] > 0.5]
            if len(vals):
                aux_vals[ai].append(vals)
        sg = rec["stl_global"]
        sm = rec["stl_global_mask"]
        for si in range(N_STL_GLOBAL):
            vals = sg[si:si + 1][sm[si:si + 1] > 0.5]
            if len(vals):
                stl_vals[si].append(vals)

    def mean_std(groups, n):
        mean = np.zeros(n, dtype=np.float32)
        std = np.ones(n, dtype=np.float32)
        for i, parts in enumerate(groups):
            if parts:
                arr = np.concatenate(parts).astype(np.float32)
                mean[i] = float(arr.mean())
                std[i] = float(max(arr.std(), 1e-6))
        return mean, std

    profile_mean, profile_std = mean_std(profile_vals, N_PROFILE_FEAT)
    aux_mean, aux_std = mean_std(aux_vals, N_AUX)
    for ai in AUX_FLAG_INDICES:
        if ai < len(aux_mean):
            aux_mean[ai] = 0.0
            aux_std[ai] = 1.0
    stl_mean, stl_std = mean_std(stl_vals, N_STL_GLOBAL)
    labels_arr = np.asarray(labels, dtype=np.float32)
    return FoldNormalizer(
        profile_mean=profile_mean,
        profile_std=profile_std,
        aux_mean=aux_mean,
        aux_std=aux_std,
        stl_global_mean=stl_mean,
        stl_global_std=stl_std,
        label_mean=float(labels_arr.mean()),
        label_std=float(max(labels_arr.std(), 1e-6)),
    )


def _normalize_points(points: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    if points.size == 0:
        return points.astype(np.float32)
    return ((points - center[None, :]) / max(float(scale), 1e-6)).astype(np.float32)


class ArchitectureDataset(Dataset):
    """Patient dataset for numeric, STL, and fused benchmark inputs."""

    def __init__(
        self,
        root_dir: str,
        n_profile_points: int = DEFAULT_N_POINTS,
        vessel_points: int = 4096,
        organ_points: int = 2048,
        centerline_points: int = 64,
        label_key: str = "PVP",
        verbose: bool = True,
    ):
        self.root_dir = root_dir
        self.vessel_points = int(vessel_points)
        self.organ_points = int(organ_points)
        self.centerline_points = int(centerline_points)
        self.base = PortalVeinDataset(root_dir, n_points=n_profile_points, label_key=label_key, verbose=verbose)
        patient_dirs = {p["name"]: p["dir"] for p in self.base.patients}
        self.records = [self._build_record(d, patient_dirs[str(d["name"])]) for d in self.base.data]

    def _load_mesh_cloud(self, patient_dir: str, candidates: Sequence[str], n_points: int, seed: int):
        path = find_stl_path(patient_dir, candidates)
        if path is None:
            return (
                np.zeros((n_points, 3), dtype=np.float32),
                0.0,
                np.zeros(8, dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
            )
        try:
            vertices, faces = read_stl_mesh(path)
            cloud, valid = sample_mesh_points(vertices, faces, n_points=n_points, seed=seed)
            feats = mesh_global_features(vertices, faces)
            return cloud, valid, feats, vertices.astype(np.float32)
        except Exception:
            return (
                np.zeros((n_points, 3), dtype=np.float32),
                0.0,
                np.zeros(8, dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
            )

    def _build_record(self, item: Mapping[str, object], patient_dir: str) -> Dict[str, object]:
        name = str(item["name"])
        vessel_cloud, vessel_valid, vessel_feats, vessel_vertices = self._load_mesh_cloud(
            patient_dir,
            ("portal_vein.stl", "vessel.stl"),
            self.vessel_points,
            stable_seed(name, 11),
        )
        spleen_cloud, spleen_valid, spleen_feats, spleen_vertices = self._load_mesh_cloud(
            patient_dir, ("spleen.stl",), self.organ_points, stable_seed(name, 23)
        )
        liver_cloud, liver_valid, liver_feats, liver_vertices = self._load_mesh_cloud(
            patient_dir, ("liver.stl",), self.organ_points, stable_seed(name, 37)
        )
        centerline_pos, centerline_valid = load_centerline_positions(
            os.path.join(patient_dir, "unified_features.json"), self.centerline_points
        )

        raw_groups = [vessel_vertices, spleen_vertices, liver_vertices, centerline_pos.reshape(-1, 3)]
        raw_valid = [g for g in raw_groups if g.size and np.isfinite(g).all(axis=1).any()]
        if raw_valid:
            all_pts = np.concatenate([g[np.isfinite(g).all(axis=1)] for g in raw_valid], axis=0)
            lo = np.min(all_pts, axis=0)
            hi = np.max(all_pts, axis=0)
            center = (lo + hi) * 0.5
            scale = float(np.max(hi - lo))
        else:
            center = np.zeros(3, dtype=np.float32)
            scale = 1.0

        stl_global = np.concatenate([vessel_feats, spleen_feats, liver_feats]).astype(np.float32)
        stl_mask = np.ones_like(stl_global, dtype=np.float32)
        for offset, valid in ((0, vessel_valid), (8, spleen_valid), (16, liver_valid)):
            stl_mask[offset:offset + 8] = float(valid)
            stl_mask[offset] = 1.0

        return {
            "name": name,
            "subject_id": subject_id_from_name(name),
            "profiles": np.asarray(item["profiles"], dtype=np.float32),
            "point_valid": np.asarray(item["point_valid"], dtype=np.float32),
            "arc_lengths": np.asarray(item["arc_lengths"], dtype=np.float32),
            "segment_mask": np.asarray(item["segment_mask"], dtype=np.float32),
            "aux_scalars": np.asarray(item["aux_scalars"], dtype=np.float32),
            "aux_mask": np.asarray(item["aux_mask"], dtype=np.float32),
            "vessel_points": _normalize_points(vessel_cloud, center, scale),
            "vessel_valid": np.float32(vessel_valid),
            "spleen_points": _normalize_points(spleen_cloud, center, scale),
            "spleen_valid": np.float32(spleen_valid),
            "liver_points": _normalize_points(liver_cloud, center, scale),
            "liver_valid": np.float32(liver_valid),
            "centerline_pos": _normalize_points(centerline_pos.reshape(-1, 3), center, scale).reshape(centerline_pos.shape),
            "centerline_valid": centerline_valid.astype(np.float32),
            "stl_global": stl_global,
            "stl_global_mask": stl_mask,
            "patient_scale_mm": np.float32(scale),
            "label": np.float32(item["label"]),
            "is_post_tips": np.float32(item["is_post_tips"]),
        }

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[int(idx)]
        out = {}
        for key, value in rec.items():
            if isinstance(value, np.ndarray):
                out[key] = torch.from_numpy(value).float()
            elif isinstance(value, (np.floating, float, int)):
                out[key] = torch.tensor(float(value)).float()
            else:
                out[key] = value
        return out


def collate_architecture(items: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    tensor_keys = [k for k, v in items[0].items() if torch.is_tensor(v)]
    for key in tensor_keys:
        out[key] = torch.stack([it[key] for it in items], dim=0)
    out["name"] = [str(it["name"]) for it in items]
    out["subject_id"] = [str(it["subject_id"]) for it in items]
    return out


def apply_normalization(batch: Dict[str, object], norm: FoldNormalizer, device) -> Dict[str, object]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    profile_mean = torch.as_tensor(norm.profile_mean, device=device).view(1, 1, 1, -1)
    profile_std = torch.as_tensor(norm.profile_std, device=device).view(1, 1, 1, -1)
    aux_mean = torch.as_tensor(norm.aux_mean, device=device).view(1, -1)
    aux_std = torch.as_tensor(norm.aux_std, device=device).view(1, -1)
    stl_mean = torch.as_tensor(norm.stl_global_mean, device=device).view(1, -1)
    stl_std = torch.as_tensor(norm.stl_global_std, device=device).view(1, -1)

    out["profiles_norm"] = ((out["profiles"] - profile_mean) / profile_std) * out["point_valid"].unsqueeze(-1)
    out["aux_norm"] = ((out["aux_scalars"] - aux_mean) / aux_std) * out["aux_mask"]
    out["stl_global_norm"] = ((out["stl_global"] - stl_mean) / stl_std) * out["stl_global_mask"]
    out["label_norm"] = (out["label"] - float(norm.label_mean)) / float(norm.label_std)
    return out

