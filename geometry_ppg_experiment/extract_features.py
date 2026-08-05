"""Extract six global portal-vein geometry features for PVP analysis.

This experiment reads existing ``unified_features.json`` files. It does not
rebuild centerlines or cross-sections from STL/medical images.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATA_ROOT = Path(r"F:\PCG data\dataset\test4all_sample")
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)
FEATURE_COLUMNS = [
    "R_total",
    "D_Murray",
    "R_collateral",
    "Ratio_SMV_SV",
    "theta_SMV_SV",
    "Ratio_LPV_RPV",
]
COLLATERAL_SEGMENTS = ("lgv", "pgv")
R_TOTAL_SEGMENTS = ("smv", "sv", "lgv", "mpv", "lpv", "rpv", "tips")
INTRAHEPATIC_SEGMENTS = ("lpv", "rpv")
MURRAY_PARENT_SEGMENTS = ("mpv", "sv", "smv", "lpv", "rpv")
LOCAL_LOSS_LAMBDA = 1.0
COLLATERAL_LOSS_LAMBDA = 1.0
ANGLE_LOSS_K = 1.0


def subject_id_from_name(name: str) -> str:
    core = re.sub(r"^\d+", "", str(name))
    return core.split("#", 1)[0]


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def finite_array(values: Any) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return np.asarray([], dtype=np.float64)
    if arr.ndim != 1:
        return np.asarray([], dtype=np.float64)
    return arr


def _same_length(values: Any, length: int) -> np.ndarray:
    arr = finite_array(values)
    if arr.size != length:
        return np.asarray([], dtype=np.float64)
    return arr


def _profile_length(seg_data: dict[str, Any]) -> int:
    lengths: list[int] = []
    for key in (
        "area",
        "raw_area",
        "eq_diameter",
        "hydraulic_diameter",
        "raw_eq_diameter",
        "owned_radius",
        "inscribed_radius",
        "anchor_radius",
        "arc_length_mm",
    ):
        arr = finite_array(seg_data.get(key))
        if arr.size:
            lengths.append(arr.size)
    position = points_array(seg_data.get("position"))
    if position.shape[0]:
        lengths.append(int(position.shape[0]))
    return max(lengths) if lengths else 0


def _constant_profile(value: Any, length: int) -> list[float]:
    scalar = safe_float(value)
    if not np.isfinite(scalar) or scalar <= 0 or length <= 0:
        return []
    return [float(scalar)] * int(length)


def area_profile(seg_data: dict[str, Any]) -> np.ndarray:
    """Return cross-section area, filling missing values from diameter/radius fields."""
    area = finite_array(seg_data.get("area"))
    candidate_lengths = [area.size] if area.size else []
    for key in (
        "raw_area",
        "_summary_area",
        "eq_diameter",
        "hydraulic_diameter",
        "raw_eq_diameter",
        "_summary_diameter",
        "owned_radius",
        "inscribed_radius",
        "anchor_radius",
        "_summary_radius",
    ):
        arr = finite_array(seg_data.get(key))
        if arr.size:
            candidate_lengths.append(arr.size)
    if not candidate_lengths:
        return area

    n = max(candidate_lengths)
    if area.size != n:
        area = np.full(n, np.nan, dtype=np.float64)
    else:
        area = area.astype(np.float64, copy=True)
    valid = np.isfinite(area) & (area > 0)

    for key in ("raw_area", "_summary_area"):
        raw_area = _same_length(seg_data.get(key), n)
        if not raw_area.size:
            continue
        fill = (~valid) & np.isfinite(raw_area) & (raw_area > 0)
        area[fill] = raw_area[fill]
        valid = np.isfinite(area) & (area > 0)

    for key in ("eq_diameter", "hydraulic_diameter", "raw_eq_diameter", "_summary_diameter"):
        diameter = _same_length(seg_data.get(key), n)
        if not diameter.size:
            continue
        fill = (~valid) & np.isfinite(diameter) & (diameter > 0)
        area[fill] = math.pi * (diameter[fill] / 2.0) ** 2
        valid = np.isfinite(area) & (area > 0)

    for key in ("owned_radius", "inscribed_radius", "anchor_radius", "_summary_radius"):
        radius = _same_length(seg_data.get(key), n)
        if not radius.size:
            continue
        fill = (~valid) & np.isfinite(radius) & (radius > 0)
        area[fill] = math.pi * radius[fill] ** 2
        valid = np.isfinite(area) & (area > 0)

    return area


def points_array(values: Any) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return np.asarray([], dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.asarray([], dtype=np.float64)
    arr = arr[:, :3]
    return arr[np.isfinite(arr).all(axis=1)]


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def first_existing_path(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def load_centerline_nodes(patient_dir: Path) -> dict[int, np.ndarray]:
    for path in (
        patient_dir / "features" / "newcenterline.txt",
        patient_dir / "features" / "centerline.txt",
        patient_dir / "newCenterlist.txt",
        patient_dir / "CenterlinePoints.txt",
    ):
        if not path.exists():
            continue
        nodes: dict[int, np.ndarray] = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                node_id = int(float(parts[0]))
                coord = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
            except ValueError:
                continue
            if np.isfinite(coord).all():
                nodes[node_id] = coord
        if nodes:
            return nodes
    return {}


def segment_record(sources: dict[str, Any], segment: str) -> dict[str, Any]:
    segments = (sources.get("centerline_profiles") or {}).get("segments") or {}
    record = segments.get(segment)
    if isinstance(record, dict):
        return record
    return {}


def segment_path(sources: dict[str, Any], segment: str) -> list[int]:
    path = segment_record(sources, segment).get("path")
    if not isinstance(path, list):
        return []
    out: list[int] = []
    for value in path:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
    return out


def segment_present(sources: dict[str, Any], segment: str) -> bool:
    if len(segment_path(sources, segment)) >= 2:
        return True
    if pointwise_segment(sources, segment):
        return True
    unified = sources.get("unified") or {}
    info = (unified.get("vessel_presence") or {}).get(segment)
    if isinstance(info, dict):
        return bool(info.get("present"))
    return bool(info)


def _pointwise_from_mapping(mapping: dict[str, Any], segment: str) -> dict[str, Any]:
    direct = mapping.get(segment)
    if isinstance(direct, dict):
        return direct
    lower_map = {str(k).lower(): k for k in mapping if isinstance(k, str)}
    key = lower_map.get(segment.lower())
    if key is not None and isinstance(mapping.get(key), dict):
        return mapping[key]
    return {}


def _with_summary_profile_fallback(
    sources: dict[str, Any],
    segment: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    if not record:
        return record
    n = _profile_length(record)
    if n <= 0:
        return record
    portal_features = sources.get("portal_vein_features") or {}
    if not isinstance(portal_features, dict):
        return record

    out = dict(record)
    out["_summary_area"] = _constant_profile(portal_features.get(f"{segment}_mean_area"), n)
    out["_summary_diameter"] = _constant_profile(portal_features.get(f"{segment}_mean_diameter"), n)
    out["_summary_radius"] = _constant_profile(portal_features.get(f"{segment}_effective_radius"), n)
    return out


def pointwise_segment(sources: dict[str, Any], segment: str) -> dict[str, Any]:
    profiles = sources.get("pointwise_profiles") or {}
    out = _pointwise_from_mapping(profiles, segment)
    if out:
        return _with_summary_profile_fallback(sources, segment, out)
    unified = sources.get("unified") or {}
    pointwise = unified.get("pointwise") or {}
    direct = pointwise.get(segment)
    if isinstance(direct, dict):
        return _with_summary_profile_fallback(sources, segment, direct)
    out = _pointwise_from_mapping(pointwise, segment)
    return _with_summary_profile_fallback(sources, segment, out)


def section_mask_by_arc(arc: np.ndarray, length: int, lo: float, hi: float) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    if length == 0:
        return mask
    if arc.size == length and np.isfinite(arc).sum() >= 2:
        amin = float(np.nanmin(arc))
        amax = float(np.nanmax(arc))
        if amax > amin:
            pos = (arc - amin) / (amax - amin)
            return np.isfinite(pos) & (pos >= lo) & (pos <= hi)
    idx = np.linspace(0.0, 1.0, length)
    return (idx >= lo) & (idx <= hi)


def area_values(seg_data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    area = area_profile(seg_data)
    arc = finite_array(seg_data.get("arc_length_mm"))
    n = area.size
    if arc.size != n:
        arc = np.asarray([], dtype=np.float64)
    return area, arc


def profile_arrays(seg_data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    area = area_profile(seg_data)
    n = area.size
    if n == 0:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty

    arc = finite_array(seg_data.get("arc_length_mm"))
    if arc.size != n or not np.isfinite(arc).all():
        total = safe_float(seg_data.get("total_length_mm"))
        end = total if np.isfinite(total) and total > 0 else max(n - 1, 1)
        arc = np.linspace(0.0, float(end), n)

    curvature = finite_array(seg_data.get("curvature"))
    if curvature.size != n:
        curvature = np.full(n, np.nan, dtype=np.float64)

    return area, arc, curvature


def radii_from_area(area: np.ndarray) -> np.ndarray:
    radii = np.full(area.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(area) & (area > 0)
    radii[mask] = np.sqrt(area[mask] / math.pi)
    return radii


def segment_resistance_visc(seg_data: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    area, arc, _curvature = profile_arrays(seg_data)
    if area.size < 2:
        return np.nan, {"status": "missing_area_profile"}

    radius = radii_from_area(area)
    d_arc = np.abs(np.diff(arc))
    valid = np.isfinite(d_arc) & (d_arc > 0) & np.isfinite(radius[:-1]) & (radius[:-1] > 0)
    if not np.any(valid):
        return np.nan, {"status": "no_valid_microsegments"}

    contributions = d_arc[valid] / np.clip(radius[:-1][valid], 1e-9, None) ** 4
    value = float(np.sum(contributions))
    return value, {
        "status": "ok",
        "n_points": int(area.size),
        "n_microsegments": int(np.sum(valid)),
        "formula": "sum(Delta_L_i / r_i^4), r_i=sqrt(A_i/pi)",
    }


def local_loss_factor(seg_data: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    area, _arc, curvature = profile_arrays(seg_data)
    if area.size == 0:
        return np.nan, {"status": "missing_area_profile"}

    radius = radii_from_area(area)
    total = 0.0
    counts = {
        "expansion": 0,
        "contraction": 0,
        "bend": 0,
        "combined_expansion_bend": 0,
    }

    for i in range(area.size):
        exp_zeta = np.nan
        con_zeta = np.nan
        if i + 1 < area.size and np.isfinite(area[i]) and np.isfinite(area[i + 1]) and area[i] > 0:
            ar = area[i + 1] / area[i]
            if np.isfinite(ar) and ar > 1.3:
                exp_zeta = (1.0 - area[i] / area[i + 1]) ** 2
            elif np.isfinite(ar) and ar < 0.7:
                con_zeta = 0.5 * (1.0 - area[i + 1] / area[i])

        bend_zeta = np.nan
        if i < curvature.size and np.isfinite(curvature[i]) and np.isfinite(radius[i]) and radius[i] > 0:
            candidate = curvature[i] * radius[i]
            if candidate > 0.1:
                bend_zeta = candidate

        if np.isfinite(exp_zeta) and np.isfinite(bend_zeta):
            total += float(exp_zeta + bend_zeta + exp_zeta * bend_zeta)
            counts["combined_expansion_bend"] += 1
            continue
        if np.isfinite(exp_zeta):
            total += float(exp_zeta)
            counts["expansion"] += 1
        if np.isfinite(con_zeta):
            total += float(con_zeta)
            counts["contraction"] += 1
        if np.isfinite(bend_zeta):
            total += float(bend_zeta)
            counts["bend"] += 1

    return float(total), {
        "status": "ok",
        "counts": counts,
        "formula": "sum zeta_exp/zeta_con/zeta_bend, with expansion+bend combined as exp+bend+exp*bend",
    }


def compute_effective_resistance(seg_data: dict[str, Any], lambda_loss: float = LOCAL_LOSS_LAMBDA) -> tuple[float, dict[str, Any]]:
    r_visc, visc_report = segment_resistance_visc(seg_data)
    phi_local, loss_report = local_loss_factor(seg_data)
    if not np.isfinite(r_visc):
        return np.nan, {"status": "missing_viscous_resistance", "R_visc": visc_report, "Phi_local": loss_report}
    if not np.isfinite(phi_local):
        phi_local = 0.0
    value = float(r_visc * (1.0 + lambda_loss * phi_local))
    return value, {
        "status": "ok",
        "R_visc": r_visc,
        "Phi_local": phi_local,
        "lambda": lambda_loss,
        "R_visc_report": visc_report,
        "Phi_local_report": loss_report,
        "formula": "R_effective = R_visc * (1 + lambda * Phi_local)",
    }


def precomputed_resistance_integral(sources: dict[str, Any], segment: str) -> float:
    portal_features = sources.get("portal_vein_features") or {}
    return safe_float(portal_features.get(f"{segment}_resistance_integral"))


def parallel_resistance(values: list[float]) -> float:
    usable = [float(value) for value in values if np.isfinite(value) and value > 0]
    denom = sum(1.0 / value for value in usable)
    return float(1.0 / denom) if denom > 0 else np.nan


def required_parallel_resistance(values: list[float]) -> float:
    if not values or any(not np.isfinite(value) or value <= 0 for value in values):
        return np.nan
    return parallel_resistance(values)


def median_area(seg_data: dict[str, Any], lo: float = 0.0, hi: float = 1.0) -> float:
    area, arc = area_values(seg_data)
    if area.size == 0:
        return np.nan
    mask = section_mask_by_arc(arc, area.size, lo, hi)
    mask &= np.isfinite(area) & (area > 0.0)
    if not np.any(mask):
        mask = np.isfinite(area) & (area > 0.0)
    if not np.any(mask):
        return np.nan
    return float(np.nanmedian(area[mask]))


def median_radius(seg_data: dict[str, Any], side: str) -> float:
    if side == "start":
        area = median_area(seg_data, 0.0, 0.2)
    elif side == "end":
        area = median_area(seg_data, 0.8, 1.0)
    else:
        area = median_area(seg_data, 0.0, 1.0)
    if not np.isfinite(area) or area <= 0:
        return np.nan
    return float(math.sqrt(area / math.pi))


def resistance_proxy(seg_data: dict[str, Any]) -> float:
    area, arc = area_values(seg_data)
    if area.size < 2:
        return np.nan
    valid = np.isfinite(area) & (area > 0)
    if arc.size == area.size:
        valid &= np.isfinite(arc)
    if valid.sum() < 2:
        return np.nan
    area = area[valid]
    if arc.size:
        arc = arc[valid]
        ds = np.abs(np.diff(arc, prepend=arc[0]))
        if ds.size > 1:
            ds[0] = ds[1]
        ds = np.clip(ds, 1e-6, None)
    else:
        ds = np.ones_like(area, dtype=np.float64)
    radius = np.sqrt(area / math.pi)
    return float(np.sum(ds / np.clip(radius, 1e-6, None) ** 4))


def path_points(path: list[int], nodes: dict[int, np.ndarray]) -> np.ndarray:
    pts = [nodes[node_id] for node_id in path if node_id in nodes]
    if len(pts) < 2:
        return np.asarray([], dtype=np.float64)
    return np.stack(pts, axis=0)


def segment_points(sources: dict[str, Any], segment: str) -> np.ndarray:
    pts = path_points(segment_path(sources, segment), sources.get("nodes") or {})
    if pts.shape[0] >= 2:
        return pts
    return points_array(pointwise_segment(sources, segment).get("position"))


def attachment_detail(
    sources: dict[str, Any],
    child: str,
    parent: str,
) -> dict[str, Any]:
    parent_pts = segment_points(sources, parent)
    child_pts = segment_points(sources, child)
    if parent_pts.shape[0] < 2 or child_pts.shape[0] < 2:
        return {"status": "missing_points", "parent": parent, "child": child}

    candidates = [("start", child_pts[0]), ("end", child_pts[-1])]
    best: dict[str, Any] | None = None
    for child_side, coord in candidates:
        distances = np.linalg.norm(parent_pts - coord, axis=1)
        idx = int(np.argmin(distances))
        dist = float(distances[idx])
        candidate = {
            "status": "ok",
            "parent": parent,
            "child": child,
            "child_side": child_side,
            "parent_index": idx,
            "parent_fraction": idx / max(parent_pts.shape[0] - 1, 1),
            "distance_mm": dist,
        }
        if best is None or dist < float(best["distance_mm"]):
            best = candidate
    return best or {"status": "missing_points", "parent": parent, "child": child}


def best_parent_attachment(
    sources: dict[str, Any],
    child: str,
    parents: tuple[str, ...] = MURRAY_PARENT_SEGMENTS,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for parent in parents:
        if parent == child or not segment_present(sources, parent):
            continue
        detail = attachment_detail(sources, child, parent)
        if detail.get("status") != "ok":
            continue
        if best is None or float(detail["distance_mm"]) < float(best["distance_mm"]):
            best = detail
    return best or {"status": "no_parent_found", "child": child}


def unit_angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an <= 1e-9 or bn <= 1e-9:
        return np.nan
    cos_value = float(np.dot(a, b) / (an * bn))
    cos_value = max(-1.0, min(1.0, cos_value))
    return float(math.degrees(math.acos(cos_value)))


def undirected_angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an <= 1e-9 or bn <= 1e-9:
        return np.nan
    cos_value = abs(float(np.dot(a, b) / (an * bn)))
    cos_value = max(-1.0, min(1.0, cos_value))
    return float(math.degrees(math.acos(cos_value)))


def vector_pointing_to_attachment(points: np.ndarray, side: str) -> np.ndarray:
    if points.shape[0] < 2:
        return np.asarray([], dtype=np.float64)
    if side == "end":
        return points[-1] - points[-2]
    return points[0] - points[1]


def vector_pointing_to_attachment_fit(points: np.ndarray, side: str, n_fit: int = 10) -> np.ndarray:
    if points.shape[0] < 2:
        return np.asarray([], dtype=np.float64)
    n = min(max(n_fit, 2), points.shape[0])
    if side == "end":
        return points[-1] - np.mean(points[-n:-1], axis=0)
    return points[0] - np.mean(points[1:n], axis=0)


def vector_pointing_from_attachment(points: np.ndarray, side: str) -> np.ndarray:
    if points.shape[0] < 2:
        return np.asarray([], dtype=np.float64)
    if side == "end":
        return points[-2] - points[-1]
    return points[1] - points[0]


def vector_pointing_from_attachment_fit(points: np.ndarray, side: str, n_fit: int = 10) -> np.ndarray:
    return -vector_pointing_to_attachment_fit(points, side, n_fit=n_fit)


def parent_tangent_at(points: np.ndarray, index: int) -> np.ndarray:
    if points.shape[0] < 2:
        return np.asarray([], dtype=np.float64)
    index = int(np.clip(index, 0, points.shape[0] - 1))
    if index == 0:
        return points[1] - points[0]
    if index == points.shape[0] - 1:
        return points[-1] - points[-2]
    return points[index + 1] - points[index - 1]


def area_by_distance_from_attachment(
    seg_data: dict[str, Any],
    side: str,
    lo_mm: float,
    hi_mm: float,
) -> float:
    area, arc = area_values(seg_data)
    if area.size == 0:
        return np.nan
    if arc.size != area.size:
        total = safe_float(seg_data.get("total_length_mm"))
        end = total if np.isfinite(total) and total > 0 else max(area.size - 1, 1)
        arc = np.linspace(0.0, float(end), area.size)

    distance = arc[-1] - arc if side == "end" else arc - arc[0]
    valid = np.isfinite(area) & (area > 0) & np.isfinite(distance)
    mask = valid & (distance >= lo_mm) & (distance <= hi_mm)
    if np.any(mask):
        return float(np.nanmedian(area[mask]))

    if not np.any(valid):
        return np.nan
    target = 0.5 * (lo_mm + hi_mm)
    idx = int(np.nanargmin(np.where(valid, np.abs(distance - target), np.nan)))
    value = area[idx]
    return float(value) if np.isfinite(value) and value > 0 else np.nan


def diameter_from_area_value(area: float) -> float:
    if not np.isfinite(area) or area <= 0:
        return np.nan
    return float(2.0 * math.sqrt(area / math.pi))


def endpoint_radius(seg_data: dict[str, Any], side: str) -> float:
    area, _arc = area_values(seg_data)
    if area.size == 0:
        return np.nan
    indices = range(area.size - 1, -1, -1) if side == "end" else range(area.size)
    for idx in indices:
        value = area[idx]
        if np.isfinite(value) and value > 0:
            return float(math.sqrt(value / math.pi))
    return np.nan


def endpoint_curvature(seg_data: dict[str, Any], side: str) -> float:
    curvature = finite_array(seg_data.get("curvature"))
    if curvature.size == 0:
        return np.nan
    indices = range(curvature.size - 1, -1, -1) if side == "end" else range(curvature.size)
    for idx in indices:
        value = curvature[idx]
        if np.isfinite(value):
            return float(value)
    return np.nan


def path_length(path: list[int], nodes: dict[int, np.ndarray]) -> float:
    pts = path_points(path, nodes)
    if pts.shape[0] < 2:
        return np.nan
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def radius_at_fraction(seg_data: dict[str, Any], fraction: float, width: float = 0.08) -> float:
    area, arc = area_values(seg_data)
    if area.size == 0:
        return np.nan
    fraction = float(np.clip(fraction, 0.0, 1.0))
    if arc.size == area.size and np.isfinite(arc).sum() >= 2 and float(np.nanmax(arc)) > float(np.nanmin(arc)):
        pos = (arc - float(np.nanmin(arc))) / (float(np.nanmax(arc)) - float(np.nanmin(arc)))
    else:
        pos = np.linspace(0.0, 1.0, area.size)
    mask = np.isfinite(area) & (area > 0) & (np.abs(pos - fraction) <= width)
    if not np.any(mask):
        nearest = int(np.nanargmin(np.abs(pos - fraction)))
        value = area[nearest]
        if not np.isfinite(value) or value <= 0:
            return np.nan
        return float(math.sqrt(value / math.pi))
    return float(math.sqrt(float(np.nanmedian(area[mask])) / math.pi))


def radius_at_fraction_point(seg_data: dict[str, Any], fraction: float) -> float:
    area, arc = area_values(seg_data)
    if area.size == 0:
        return np.nan
    fraction = float(np.clip(fraction, 0.0, 1.0))
    if arc.size == area.size and np.isfinite(arc).sum() >= 2 and float(np.nanmax(arc)) > float(np.nanmin(arc)):
        pos = (arc - float(np.nanmin(arc))) / (float(np.nanmax(arc)) - float(np.nanmin(arc)))
    else:
        pos = np.linspace(0.0, 1.0, area.size)
    valid = np.isfinite(area) & (area > 0) & np.isfinite(pos)
    if not np.any(valid):
        return np.nan
    idx = int(np.argmin(np.abs(pos[valid] - fraction)))
    value = area[valid][idx]
    return float(math.sqrt(value / math.pi)) if np.isfinite(value) and value > 0 else np.nan


def attachment_to_parent(
    sources: dict[str, Any],
    child: str,
    parent: str,
) -> tuple[float, str, int | None]:
    parent_path = segment_path(sources, parent)
    child_path = segment_path(sources, child)
    if len(parent_path) < 2 or len(child_path) < 2:
        return np.nan, "mid", None

    parent_index = {node_id: idx for idx, node_id in enumerate(parent_path)}
    child_endpoints = [child_path[0], child_path[-1]]
    for side, node_id in zip(("start", "end"), child_endpoints):
        if node_id in parent_index:
            denom = max(len(parent_path) - 1, 1)
            return parent_index[node_id] / denom, side, node_id

    nodes = sources.get("nodes") or {}
    if not nodes:
        return np.nan, "mid", None
    parent_pts = path_points(parent_path, nodes)
    endpoint_coords = [(side, node_id, nodes[node_id]) for side, node_id in zip(("start", "end"), child_endpoints) if node_id in nodes]
    if parent_pts.shape[0] < 2 or not endpoint_coords:
        return np.nan, "mid", None
    best: tuple[float, str, int | None, int] | None = None
    for side, node_id, coord in endpoint_coords:
        distances = np.linalg.norm(parent_pts - coord, axis=1)
        idx = int(np.argmin(distances))
        dist = float(distances[idx])
        if best is None or dist < best[0]:
            best = (dist, side, node_id, idx)
    if best is None:
        return np.nan, "mid", None
    _, side, node_id, idx = best
    return idx / max(parent_pts.shape[0] - 1, 1), side, node_id


def compute_r_total(sources: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    segment_values: dict[str, float] = {}
    segment_reports: dict[str, Any] = {}
    for segment in R_TOTAL_SEGMENTS:
        if not segment_present(sources, segment):
            segment_reports[segment] = {"status": "segment_absent"}
            segment_values[segment] = np.nan
            continue
        value, report = compute_effective_resistance(pointwise_segment(sources, segment), LOCAL_LOSS_LAMBDA)
        if not np.isfinite(value) or value <= 0:
            fallback = precomputed_resistance_integral(sources, segment)
            if np.isfinite(fallback) and fallback > 0:
                value = fallback
                report = {
                    "status": "fallback_precomputed_resistance_integral",
                    "R_effective": fallback,
                    "source": f"portal_vein_features.{segment}_resistance_integral",
                    "note": "Used because pointwise area profile could not support strict R_visc/Phi_local calculation; local loss multiplier is not available in this fallback.",
                    "failed_strict_report": report,
                }
        segment_values[segment] = value
        segment_reports[segment] = report

    lower_parallel_segments = ["smv", "sv"]
    if np.isfinite(segment_values.get("lgv", np.nan)) and segment_values.get("lgv", np.nan) > 0:
        lower_parallel_segments.append("lgv")
    lower_values = [segment_values.get(segment, np.nan) for segment in lower_parallel_segments]
    r_inflow = required_parallel_resistance(lower_values)

    r_prehepatic = (
        r_inflow + segment_values["mpv"]
        if np.isfinite(r_inflow) and np.isfinite(segment_values.get("mpv", np.nan))
        else np.nan
    )

    upper_parallel_segments = [
        segment
        for segment in ("lpv", "rpv", "tips")
        if np.isfinite(segment_values.get(segment, np.nan)) and segment_values.get(segment, np.nan) > 0
    ]
    r_hepatic = parallel_resistance([segment_values[segment] for segment in upper_parallel_segments])
    r_total = r_prehepatic + r_hepatic if np.isfinite(r_prehepatic) and np.isfinite(r_hepatic) else np.nan
    return r_total, {
        "status": "ok" if np.isfinite(r_total) else "missing_required_tree_resistance",
        "segment_R_effective": segment_values,
        "segment_reports": segment_reports,
        "lower_parallel_segments": lower_parallel_segments,
        "R_inflow_parallel_SMV_SV_optional_LGV": r_inflow,
        "R_prehepatic": r_prehepatic,
        "upper_parallel_segments": upper_parallel_segments,
        "R_upper_parallel_LPV_RPV_TIPS": r_hepatic,
        "formula": "R_total=(SMV||SV||optional_LGV)+Main+(available_LPV||available_RPV||available_TIPS)",
    }


def murray_branch_deviation(
    name: str,
    parent_radius: float,
    child_radii: list[float],
) -> tuple[float, dict[str, Any]]:
    deviation = murray_deviation(parent_radius, child_radii)
    return deviation, {
        "name": name,
        "parent_radius": parent_radius,
        "child_radii": child_radii,
        "deviation": deviation,
        "formula": "abs(1 - r0^3 / (r1^3 + r2^3))",
    }


def continuation_fraction(parent_fraction: float) -> float:
    if not np.isfinite(parent_fraction):
        return np.nan
    step = 0.08
    if parent_fraction <= 1.0 - step:
        return parent_fraction + step
    return max(0.0, parent_fraction - step)


def compute_d_murray(sources: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    deviations: dict[str, float] = {}
    junction_reports: dict[str, Any] = {}

    smv_attach = attachment_detail(sources, "smv", "mpv")
    sv_attach = attachment_detail(sources, "sv", "mpv")
    if smv_attach.get("status") == "ok" and sv_attach.get("status") == "ok":
        parent_fraction = float(np.mean([smv_attach["parent_fraction"], sv_attach["parent_fraction"]]))
        deviation, detail = murray_branch_deviation(
            "SMV_SV_confluence",
            radius_at_fraction_point(pointwise_segment(sources, "mpv"), parent_fraction),
            [
                endpoint_radius(pointwise_segment(sources, "smv"), str(smv_attach["child_side"])),
                endpoint_radius(pointwise_segment(sources, "sv"), str(sv_attach["child_side"])),
            ],
        )
        junction_reports["SMV_SV_confluence"] = detail
        if np.isfinite(deviation):
            deviations["SMV_SV_confluence"] = deviation

    lpv_attach = attachment_detail(sources, "lpv", "mpv")
    rpv_attach = attachment_detail(sources, "rpv", "mpv")
    if lpv_attach.get("status") == "ok" and rpv_attach.get("status") == "ok":
        parent_fraction = float(np.mean([lpv_attach["parent_fraction"], rpv_attach["parent_fraction"]]))
        deviation, detail = murray_branch_deviation(
            "MPV_bifurcation",
            radius_at_fraction_point(pointwise_segment(sources, "mpv"), parent_fraction),
            [
                endpoint_radius(pointwise_segment(sources, "lpv"), str(lpv_attach["child_side"])),
                endpoint_radius(pointwise_segment(sources, "rpv"), str(rpv_attach["child_side"])),
            ],
        )
        junction_reports["MPV_bifurcation"] = detail
        if np.isfinite(deviation):
            deviations["MPV_bifurcation"] = deviation

    for collateral in COLLATERAL_SEGMENTS:
        if not segment_present(sources, collateral):
            continue
        attach = best_parent_attachment(sources, collateral)
        if attach.get("status") != "ok":
            junction_reports[f"{collateral}_origin"] = attach
            continue
        parent = str(attach["parent"])
        parent_fraction = float(attach["parent_fraction"])
        cont_fraction = continuation_fraction(parent_fraction)
        deviation, detail = murray_branch_deviation(
            f"{collateral}_origin_from_{parent}",
            radius_at_fraction_point(pointwise_segment(sources, parent), parent_fraction),
            [
                endpoint_radius(pointwise_segment(sources, collateral), str(attach["child_side"])),
                radius_at_fraction_point(pointwise_segment(sources, parent), cont_fraction),
            ],
        )
        detail.update({"attachment": attach, "continuation_fraction": cont_fraction})
        junction_reports[f"{collateral}_origin"] = detail
        if np.isfinite(deviation):
            deviations[f"{collateral}_origin"] = deviation

    value = float(np.mean(list(deviations.values()))) if deviations else np.nan
    return value, {
        "status": "ok" if deviations else "no_valid_murray_junction",
        "junction_deviations": deviations,
        "junction_reports": junction_reports,
        "formula": "D_Murray = mean_j abs(1 - r0^3/(r1^3+r2^3))",
    }


def compute_r_collateral(sources: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    branch_values: dict[str, float] = {}
    branch_reports: dict[str, Any] = {}

    for collateral in COLLATERAL_SEGMENTS:
        if not segment_present(sources, collateral):
            branch_reports[collateral] = {"status": "segment_absent"}
            continue

        seg_data = pointwise_segment(sources, collateral)
        r_visc, visc_report = segment_resistance_visc(seg_data)
        attach = best_parent_attachment(sources, collateral)
        theta = np.nan
        if attach.get("status") == "ok":
            parent_pts = segment_points(sources, str(attach["parent"]))
            child_pts = segment_points(sources, collateral)
            child_vec = vector_pointing_from_attachment_fit(child_pts, str(attach["child_side"]))
            parent_vec = parent_tangent_at(parent_pts, int(attach["parent_index"]))
            theta = undirected_angle_degrees(child_vec, parent_vec)

        zeta_angle = ANGLE_LOSS_K * (1.0 - math.cos(math.radians(theta))) if np.isfinite(theta) else np.nan
        side = str(attach.get("child_side", "start"))
        k_start = endpoint_curvature(seg_data, side)
        r_start = endpoint_radius(seg_data, side)
        zeta_curvature = k_start * r_start if np.isfinite(k_start) and np.isfinite(r_start) else np.nan
        zeta_entrance = sum(value for value in (zeta_angle, zeta_curvature) if np.isfinite(value))

        if np.isfinite(r_visc) and r_visc > 0:
            r_eff = float(r_visc * (1.0 + COLLATERAL_LOSS_LAMBDA * zeta_entrance))
            branch_values[collateral] = r_eff
        else:
            r_eff = np.nan

        branch_reports[collateral] = {
            "status": "ok" if np.isfinite(r_eff) else "missing_collateral_resistance",
            "R_visc_coll": r_visc,
            "R_visc_report": visc_report,
            "attachment": attach,
            "theta_degrees": theta,
            "zeta_angle": zeta_angle,
            "K_start": k_start,
            "r_coll_start": r_start,
            "zeta_curvature": zeta_curvature,
            "zeta_entrance": zeta_entrance,
            "lambda_coll": COLLATERAL_LOSS_LAMBDA,
            "R_eff_coll": r_eff,
            "formula": "R_eff_coll=R_visc_coll*(1+lambda_coll*(k_angle*(1-cos(theta))+K_start*r_start))",
        }

    value = parallel_resistance(list(branch_values.values()))
    return value, {
        "status": "ok" if np.isfinite(value) else "no_valid_collateral",
        "branch_R_eff": branch_values,
        "branch_reports": branch_reports,
        "formula": "1/R_collateral = sum_c 1/R_eff_coll^(c)",
    }


def compute_ratio_smv_sv(sources: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    smv_attach = attachment_detail(sources, "smv", "mpv")
    sv_attach = attachment_detail(sources, "sv", "mpv")
    smv_side = str(smv_attach.get("child_side", "start"))
    sv_side = str(sv_attach.get("child_side", "start"))
    smv_area = area_by_distance_from_attachment(pointwise_segment(sources, "smv"), smv_side, 20.0, 30.0)
    sv_area = area_by_distance_from_attachment(pointwise_segment(sources, "sv"), sv_side, 20.0, 30.0)
    d_smv = diameter_from_area_value(smv_area)
    d_sv = diameter_from_area_value(sv_area)
    value = d_smv / d_sv if np.isfinite(d_smv) and np.isfinite(d_sv) and d_sv > 0 else np.nan
    return value, {
        "status": "ok" if np.isfinite(value) else "missing_smv_sv_diameter",
        "SMV_attachment": smv_attach,
        "SV_attachment": sv_attach,
        "A_SMV_20_30mm_from_confluence": smv_area,
        "A_SV_20_30mm_from_confluence": sv_area,
        "D_SMV": d_smv,
        "D_SV": d_sv,
        "formula": "Ratio_SMV_SV = (2*sqrt(A_SMV/pi))/(2*sqrt(A_SV/pi))",
    }


def compute_theta_smv_sv(sources: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    smv_attach = attachment_detail(sources, "smv", "mpv")
    sv_attach = attachment_detail(sources, "sv", "mpv")
    theta = np.nan
    source = "computed_from_centerline_vectors"
    unified = sources.get("unified") or {}
    unified_angle = unified.get("sv_smv_angle")
    if isinstance(unified_angle, dict):
        theta = safe_float(unified_angle.get("angle_degrees"))
        source = "unified_features.sv_smv_angle.angle_degrees"
    else:
        theta = safe_float(unified_angle)
        if np.isfinite(theta):
            source = "unified_features.sv_smv_angle"

    angle_path = (sources.get("patient_dir") or Path()) / "sv_smv_angle.json"
    if not np.isfinite(theta) and angle_path.exists():
        theta = safe_float(read_json(angle_path).get("angle_degrees"))
        source = "sv_smv_angle.json"

    if smv_attach.get("status") == "ok" and sv_attach.get("status") == "ok":
        smv_vec = vector_pointing_to_attachment_fit(segment_points(sources, "smv"), str(smv_attach["child_side"]))
        sv_vec = vector_pointing_to_attachment_fit(segment_points(sources, "sv"), str(sv_attach["child_side"]))
        computed_theta = unit_angle_degrees(smv_vec, sv_vec)
        if not np.isfinite(theta):
            theta = computed_theta
        elif np.isfinite(computed_theta):
            source = "sv_smv_angle.json;computed_check_available"

    return theta, {
        "status": "ok" if np.isfinite(theta) else "missing_smv_sv_angle",
        "SMV_attachment": smv_attach,
        "SV_attachment": sv_attach,
        "source": source,
        "formula": "theta=acos((v_SMV dot v_SV)/(|v_SMV|*|v_SV|)) in degrees, vectors point to confluence",
    }


def compute_ratio_lpv_rpv(sources: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    lpv_attach = attachment_detail(sources, "lpv", "mpv")
    rpv_attach = attachment_detail(sources, "rpv", "mpv")
    lpv_side = str(lpv_attach.get("child_side", "start"))
    rpv_side = str(rpv_attach.get("child_side", "start"))
    lpv_area = median_area(pointwise_segment(sources, "lpv"), 0.0, 0.2) if lpv_side == "start" else median_area(pointwise_segment(sources, "lpv"), 0.8, 1.0)
    rpv_area = median_area(pointwise_segment(sources, "rpv"), 0.0, 0.2) if rpv_side == "start" else median_area(pointwise_segment(sources, "rpv"), 0.8, 1.0)
    d_lpv = diameter_from_area_value(lpv_area)
    d_rpv = diameter_from_area_value(rpv_area)
    value = d_lpv / d_rpv if np.isfinite(d_lpv) and np.isfinite(d_rpv) and d_rpv > 0 else np.nan
    return value, {
        "status": "ok" if np.isfinite(value) else "missing_lpv_rpv_diameter",
        "LPV_attachment": lpv_attach,
        "RPV_attachment": rpv_attach,
        "A_LPV_initial_normal_section": lpv_area,
        "A_RPV_initial_normal_section": rpv_area,
        "D_LPV": d_lpv,
        "D_RPV": d_rpv,
        "formula": "Ratio_LPV_RPV = (2*sqrt(A_LPV/pi))/(2*sqrt(A_RPV/pi))",
    }


def compute_r_liver(sources: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    branch_resistances: dict[str, float] = {}
    for segment in INTRAHEPATIC_SEGMENTS:
        if not segment_present(sources, segment):
            continue
        value = resistance_proxy(pointwise_segment(sources, segment))
        if np.isfinite(value) and value > 0:
            branch_resistances[segment] = value
    denom = sum(1.0 / value for value in branch_resistances.values() if value > 0)
    value = float(1.0 / denom) if denom > 0 else np.nan
    return value, {
        "used_segments": sorted(branch_resistances),
        "branch_resistance": branch_resistances,
        "status": "ok" if np.isfinite(value) else "missing_lpv_rpv_resistance",
        "segment_source": "centerline_profiles.path",
    }


def compute_r_area(sources: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    mpv_data = pointwise_segment(sources, "mpv")
    main_area = median_area(mpv_data, 0.1, 0.3)
    main_source = "mpv_10_30_percent"
    if not np.isfinite(main_area) or main_area <= 0:
        main_area = median_area(mpv_data, 0.0, 1.0)
        main_source = "mpv_all"

    included: dict[str, float] = {}
    excluded: dict[str, str] = {}
    for segment in COLLATERAL_SEGMENTS:
        if not segment_present(sources, segment):
            excluded[segment] = "segment_absent"
            continue
        area = median_area(pointwise_segment(sources, segment), 0.2, 0.8)
        if not np.isfinite(area) or area <= 0:
            excluded[segment] = "area_missing"
            continue
        eq_diameter = 2.0 * math.sqrt(area / math.pi)
        if eq_diameter <= 3.0:
            excluded[segment] = f"diameter_le_3mm:{eq_diameter:.3f}"
            continue
        included[segment] = float(area)

    total_collateral_area = float(sum(included.values()))
    value = (
        total_collateral_area / float(main_area)
        if np.isfinite(main_area) and main_area > 0
        else np.nan
    )
    return value, {
        "included_segments": included,
        "excluded_segments": excluded,
        "main_area": float(main_area) if np.isfinite(main_area) else None,
        "main_source": main_source,
        "status": "ok" if np.isfinite(value) else "missing_mpv_reference_area",
        "segment_source": "centerline_profiles.path",
    }


def murray_deviation(parent_radius: float, child_radii: list[float]) -> float:
    if not np.isfinite(parent_radius) or parent_radius <= 0:
        return np.nan
    child_power = sum(r ** 3 for r in child_radii if np.isfinite(r) and r > 0)
    if child_power <= 0:
        return np.nan
    return float(abs(1.0 - (parent_radius ** 3) / child_power))


def compute_d_murray_legacy(sources: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    deviations: dict[str, float] = {}

    confluence = murray_deviation(
        median_radius(pointwise_segment(sources, "mpv"), "start"),
        [
            median_radius(pointwise_segment(sources, "sv"), "start"),
            median_radius(pointwise_segment(sources, "smv"), "start"),
        ],
    )
    if np.isfinite(confluence):
        deviations["sv_smv_to_mpv"] = confluence

    attachments: dict[str, list[tuple[str, float, str]]] = {}
    for child in ("lpv", "rpv", "tips"):
        if not segment_present(sources, child):
            continue
        parent_fraction, child_side, node_id = attachment_to_parent(sources, child, "mpv")
        if not np.isfinite(parent_fraction):
            parent_fraction = 1.0
        key = str(node_id) if node_id is not None else f"{parent_fraction:.2f}"
        attachments.setdefault(key, []).append((child, parent_fraction, child_side))

    for key, group in attachments.items():
        parent_fraction = float(np.mean([item[1] for item in group]))
        parent_radius = radius_at_fraction(pointwise_segment(sources, "mpv"), parent_fraction)
        child_radii = [
            median_radius(pointwise_segment(sources, child), child_side)
            for child, _frac, child_side in group
        ]
        if parent_fraction < 0.92:
            child_radii.append(radius_at_fraction(pointwise_segment(sources, "mpv"), min(parent_fraction + 0.08, 1.0)))
        deviation = murray_deviation(parent_radius, child_radii)
        if np.isfinite(deviation):
            child_names = "_".join(item[0] for item in group)
            deviations[f"mpv_attach_{child_names}_{key}"] = deviation

    value = float(np.mean(list(deviations.values()))) if deviations else np.nan
    return value, {
        "junction_deviations": deviations,
        "status": "ok" if deviations else "no_valid_murray_junction",
        "segment_source": "centerline_profiles.path",
    }


def compute_tortuosity(sources: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    mpv_path = segment_path(sources, "mpv")
    nodes = sources.get("nodes") or {}
    pts = path_points(mpv_path, nodes)
    if pts.shape[0] >= 2:
        length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        chord = float(np.linalg.norm(pts[-1] - pts[0]))
        if chord > 1e-6:
            return length / chord, {"source": "centerline_profiles.mpv.path+newCenterlist", "status": "ok"}

    mpv_data = pointwise_segment(sources, "mpv")
    points = points_array(mpv_data.get("position"))
    if points.shape[0] >= 2:
        length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        chord = float(np.linalg.norm(points[-1] - points[0]))
        if chord > 1e-6:
            return length / chord, {"source": "pointwise.mpv.position_xyz", "status": "ok"}

    unified = sources.get("unified") or {}
    meta = ((unified.get("segments_meta") or {}).get("mpv") or {})
    length = safe_float(meta.get("length_mm"))
    endpoints = points_array(meta.get("endpoints_coord"))
    if np.isfinite(length) and endpoints.shape[0] >= 2:
        chord = float(np.linalg.norm(endpoints[-1] - endpoints[0]))
        if chord > 1e-6:
            return float(length / chord), {"source": "segments_meta.mpv.length_endpoints", "status": "ok"}

    stat = ((unified.get("statistical") or {}).get("mpv") or {})
    value = safe_float(stat.get("tortuosity"))
    return value, {
        "source": "statistical.mpv.tortuosity",
        "status": "ok" if np.isfinite(value) else "missing_mpv_tortuosity",
    }


def load_label(patient_dir: Path) -> float:
    return safe_float((patient_dir / "label" / "PVP.txt").read_text(encoding="utf-8").strip())


def extract_patient(patient_dir: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    report: dict[str, Any] = {"sample": patient_dir.name, "status": "pending", "features": {}}
    if "@" in patient_dir.name or "!" in patient_dir.name:
        report["status"] = "skipped_bad_name"
        return None, report
    label_path = patient_dir / "label" / "PVP.txt"
    features_dir = patient_dir / "features"
    unified_path = first_existing_path(
        features_dir / "unified_features.json",
        patient_dir / "unified_features.json",
    )
    if not label_path.exists():
        report["status"] = "skipped_missing_label"
        return None, report
    if unified_path is None:
        report["status"] = "skipped_missing_unified_features"
        return None, report

    label = load_label(patient_dir)
    if not np.isfinite(label):
        report["status"] = "skipped_invalid_label"
        return None, report

    unified = read_json(unified_path)
    centerline_profiles = read_json(
        first_existing_path(
            features_dir / "segment_assignments.json",
            patient_dir / "centerline_profiles.json",
        )
        or Path()
    )
    pointwise_profiles = read_json(
        first_existing_path(
            features_dir / "pointwise_profiles.json",
            patient_dir / "centerline_pointwise_profiles.json",
        )
        or Path()
    )
    portal_vein_features = read_json(
        first_existing_path(
            patient_dir / "portal_vein_features.json",
            features_dir / "unified_features0.json",
        )
        or Path()
    )
    sources = {
        "unified": unified,
        "centerline_profiles": centerline_profiles,
        "pointwise_profiles": pointwise_profiles,
        "portal_vein_features": portal_vein_features,
        "nodes": load_centerline_nodes(patient_dir),
        "patient_dir": patient_dir,
    }
    report["unified_features_path"] = str(unified_path)
    has_name_tips_marker = "#" in patient_dir.name
    has_tips_tube = segment_present(sources, "tips")
    report["has_name_tips_marker"] = has_name_tips_marker
    report["has_tips_tube"] = has_tips_tube
    if has_name_tips_marker and not has_tips_tube:
        report["status"] = "skipped_hash_without_tips_tube"
        return None, report
    if not has_name_tips_marker and has_tips_tube:
        report["status"] = "skipped_tips_tube_without_hash"
        return None, report

    report["segment_vessels_version"] = centerline_profiles.get("segment_vessels_version")
    report["n_branch_points"] = centerline_profiles.get("n_branch_points")
    report["n_endpoints"] = centerline_profiles.get("n_endpoints")
    report["segment_path_lengths"] = {
        segment: len(segment_path(sources, segment))
        for segment in ("mpv", "sv", "smv", "lpv", "rpv", "lgv", "pgv", "tips")
    }

    r_total, r_total_report = compute_r_total(sources)
    d_murray, d_murray_report = compute_d_murray(sources)
    r_collateral, r_collateral_report = compute_r_collateral(sources)
    ratio_smv_sv, ratio_smv_sv_report = compute_ratio_smv_sv(sources)
    theta_smv_sv, theta_smv_sv_report = compute_theta_smv_sv(sources)
    ratio_lpv_rpv, ratio_lpv_rpv_report = compute_ratio_lpv_rpv(sources)

    values = {
        "R_total": r_total,
        "D_Murray": d_murray,
        "R_collateral": r_collateral,
        "Ratio_SMV_SV": ratio_smv_sv,
        "theta_SMV_SV": theta_smv_sv,
        "Ratio_LPV_RPV": ratio_lpv_rpv,
    }
    report["features"] = {
        "R_total": r_total_report,
        "D_Murray": d_murray_report,
        "R_collateral": r_collateral_report,
        "Ratio_SMV_SV": ratio_smv_sv_report,
        "theta_SMV_SV": theta_smv_sv_report,
        "Ratio_LPV_RPV": ratio_lpv_rpv_report,
    }
    report["finite_feature_count"] = int(sum(np.isfinite(v) for v in values.values()))
    report["status"] = "ok" if report["finite_feature_count"] > 0 else "no_finite_features"

    row = {
        "sample": patient_dir.name,
        "subject_id": subject_id_from_name(patient_dir.name),
        "y_true_mmHg": label,
        "is_post_tips": int(has_name_tips_marker),
        **values,
    }
    return row, report


def write_features_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["sample", "subject_id", "y_true_mmHg", "is_post_tips", *FEATURE_COLUMNS]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in FEATURE_COLUMNS:
                value = out.get(key)
                out[key] = "" if not np.isfinite(safe_float(value)) else f"{float(value):.12g}"
            writer.writerow(out)


def build_summary(rows: list[dict[str, Any]], reports: list[dict[str, Any]], data_root: Path) -> dict[str, Any]:
    finite_counts = {
        key: int(sum(np.isfinite(safe_float(row.get(key))) for row in rows))
        for key in FEATURE_COLUMNS
    }
    statuses: dict[str, int] = {}
    for report in reports:
        status = str(report.get("status"))
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "data_root": str(data_root),
        "n_rows_written": len(rows),
        "feature_finite_counts": finite_counts,
        "status_counts": statuses,
        "patients": reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--features-name", default="features.csv")
    parser.add_argument("--report-name", default="feature_extraction_report.json")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of patient worker processes (default: {DEFAULT_WORKERS}; use 1 for serial)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    patient_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    if args.workers == 1:
        results = map(extract_patient, patient_dirs)
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        results = executor.map(extract_patient, patient_dirs)

    try:
        for row, report in results:
            reports.append(report)
            if row is not None:
                rows.append(row)
    finally:
        if args.workers > 1:
            executor.shutdown()

    out_dir = args.out_dir
    write_features_csv(out_dir / args.features_name, rows)
    summary = build_summary(rows, reports, data_root)
    (out_dir / args.report_name).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[extract] wrote {len(rows)} rows to {out_dir / args.features_name}")
    print(f"[extract] finite feature counts: {summary['feature_finite_counts']}")
    print(f"[extract] report: {out_dir / args.report_name}")


if __name__ == "__main__":
    main()
