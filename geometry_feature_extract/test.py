"""Compare high-curvature cross-section strategies on one vessel branch.

The default case is the SV branch of 0013996314YangTingFu.  The experiment
compares the saved profile with one-plane tangent sections and with tangent
sections restricted to the 3-D Voronoi cell of the current centerline sample.
The restriction removes the part of a planar section owned by another part of
the same curved centerline, so non-local turns cannot cross each other.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import trimesh
from scipy import ndimage
from shapely.geometry import LineString, Point, Polygon

from extract_profiles import (
    _make_orthonormal_basis,
    _pick_polygon_from_geometry,
    _section_one,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_PATIENT = Path(
    r"F:\PCG data\dataset\test4all_sample\0013996314YangTingFu"
)


def _arc_length(coords: np.ndarray) -> np.ndarray:
    return np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))
    )


def _smooth_tangents(coords: np.ndarray, smoothing_mm: float) -> np.ndarray:
    arc = _arc_length(coords)
    step = float(np.median(np.diff(arc)))
    sigma = max(0.0, float(smoothing_mm) / max(step, 1e-9))
    guide = (
        ndimage.gaussian_filter1d(coords, sigma=sigma, axis=0, mode="nearest")
        if sigma > 0.25
        else coords.copy()
    )
    tangent = np.gradient(guide, arc, axis=0, edge_order=2)
    norms = np.linalg.norm(tangent, axis=1)
    tangent /= np.maximum(norms[:, None], 1e-12)
    for idx in range(1, len(tangent)):
        if np.dot(tangent[idx], tangent[idx - 1]) < 0:
            tangent[idx] *= -1.0
    return tangent


def _clip_convex_polygon(
    vertices: np.ndarray, normal: np.ndarray, limit: float, tol: float = 1e-9
) -> np.ndarray:
    """Clip a convex 2-D polygon by normal dot x <= limit."""
    if len(vertices) == 0:
        return vertices
    output = []
    previous = vertices[-1]
    previous_value = float(np.dot(normal, previous) - limit)
    previous_inside = previous_value <= tol
    for current in vertices:
        current_value = float(np.dot(normal, current) - limit)
        current_inside = current_value <= tol
        if current_inside != previous_inside:
            denom = previous_value - current_value
            if abs(denom) > 1e-15:
                fraction = previous_value / denom
                output.append(previous + fraction * (current - previous))
        if current_inside:
            output.append(current)
        previous = current
        previous_value = current_value
        previous_inside = current_inside
    return np.asarray(output, dtype=float)


def _centerline_voronoi_cell(
    coords: np.ndarray,
    index: int,
    normal: np.ndarray,
    extent: float,
    min_arc_separation_mm: float = 0.0,
) -> tuple[Polygon | None, np.ndarray, np.ndarray]:
    """Intersect a section plane with the centerline site's 3-D Voronoi cell."""
    point = coords[index]
    u, v = _make_orthonormal_basis(normal)
    square = np.array(
        [[-extent, -extent], [extent, -extent], [extent, extent], [-extent, extent]],
        dtype=float,
    )
    arc = _arc_length(coords)
    for other_index, other in enumerate(coords):
        if other_index == index:
            continue
        if abs(float(arc[other_index] - arc[index])) <= min_arc_separation_mm:
            continue
        delta = other - point
        halfplane_normal = np.array([np.dot(delta, u), np.dot(delta, v)])
        if np.linalg.norm(halfplane_normal) <= 1e-10:
            continue
        square = _clip_convex_polygon(
            square, halfplane_normal, 0.5 * float(np.dot(delta, delta))
        )
        if len(square) < 3:
            return None, u, v
    cell = Polygon(square)
    if not cell.is_valid:
        cell = cell.buffer(0)
    return (cell if not cell.is_empty and cell.area > 0 else None), u, v


def _owned_section(
    mesh: trimesh.Trimesh,
    coords: np.ndarray,
    index: int,
    normal: np.ndarray,
    min_arc_separation_mm: float | None,
) -> dict:
    result = _section_one(
        mesh,
        coords[index],
        normal,
        ownership_factor=None,
        return_ring=True,
        return_metrics=True,
    )
    _, _, aspect, circularity, ring = result
    if ring is None or len(ring) < 4:
        return {"area": 0.0, "perimeter": 0.0, "ring": None}

    raw = Polygon(ring)
    if not raw.is_valid:
        raw = raw.buffer(0)
    raw = _pick_polygon_from_geometry(raw, Point(0.0, 0.0))
    if raw is None:
        return {"area": 0.0, "perimeter": 0.0, "ring": None}

    u, v = _make_orthonormal_basis(normal)
    selected = raw
    if min_arc_separation_mm is not None:
        extent = max(30.0, 1.25 * float(np.sqrt(raw.area / np.pi)) * 4.0)
        cell, u, v = _centerline_voronoi_cell(
            coords,
            index,
            normal,
            extent=extent,
            min_arc_separation_mm=min_arc_separation_mm,
        )
        if cell is None:
            return {"area": 0.0, "perimeter": 0.0, "ring": None}
        selected = _pick_polygon_from_geometry(raw.intersection(cell), Point(0.0, 0.0))
        if selected is None:
            return {"area": 0.0, "perimeter": 0.0, "ring": None}

    ring_2d = np.asarray(selected.exterior.coords, dtype=float)
    ring_3d = (
        coords[index]
        + ring_2d[:, 0, None] * u[None, :]
        + ring_2d[:, 1, None] * v[None, :]
    )
    return {
        "area": float(selected.area),
        "perimeter": float(selected.exterior.length),
        "ring": ring_3d,
        "aspect_ratio": float(aspect),
        "circularity": float(circularity),
    }


def _line_intervals(geometry, origin_2d: np.ndarray, direction_2d: np.ndarray):
    direction_2d = direction_2d / max(np.linalg.norm(direction_2d), 1e-12)
    span = 1000.0
    line = LineString(
        [origin_2d - span * direction_2d, origin_2d + span * direction_2d]
    )
    intersection = geometry.intersection(line)
    pieces = list(intersection.geoms) if hasattr(intersection, "geoms") else [intersection]
    intervals = []
    for piece in pieces:
        if piece.is_empty or piece.geom_type != "LineString":
            continue
        values = [
            float(np.dot(np.asarray(coord) - origin_2d, direction_2d))
            for coord in piece.coords
        ]
        if values:
            intervals.append((min(values), max(values)))
    return intervals


def _rings_cross(
    point_a: np.ndarray,
    normal_a: np.ndarray,
    ring_a: np.ndarray,
    point_b: np.ndarray,
    normal_b: np.ndarray,
    ring_b: np.ndarray,
    tolerance: float = 0.05,
) -> bool:
    direction = np.cross(normal_a, normal_b)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm < 1e-5:
        return False
    direction /= direction_norm
    matrix = np.vstack([normal_a, normal_b, direction])
    rhs = np.array(
        [np.dot(normal_a, point_a), np.dot(normal_b, point_b), 0.0], dtype=float
    )
    try:
        line_point = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return False

    intervals = []
    for point, normal, ring in (
        (point_a, normal_a, ring_a),
        (point_b, normal_b, ring_b),
    ):
        u, v = _make_orthonormal_basis(normal)
        polygon_2d = Polygon(
            np.column_stack(((ring - point) @ u, (ring - point) @ v))
        )
        origin_2d = np.array([np.dot(line_point - point, u), np.dot(line_point - point, v)])
        direction_2d = np.array([np.dot(direction, u), np.dot(direction, v)])
        intervals.append(_line_intervals(polygon_2d, origin_2d, direction_2d))
    return any(
        min(first[1], second[1]) - max(first[0], second[0]) > tolerance
        for first in intervals[0]
        for second in intervals[1]
    )


def _crossing_count(
    coords: np.ndarray,
    arc: np.ndarray,
    normals: np.ndarray,
    rings: list[np.ndarray | None],
    min_arc_separation_mm: float = 3.0,
) -> int:
    count = 0
    for first in range(len(coords)):
        if rings[first] is None:
            continue
        for second in range(first + 1, len(coords)):
            if arc[second] - arc[first] < min_arc_separation_mm:
                continue
            if rings[second] is None:
                continue
            radius_a = np.max(np.linalg.norm(rings[first] - coords[first], axis=1))
            radius_b = np.max(np.linalg.norm(rings[second] - coords[second], axis=1))
            if np.linalg.norm(coords[second] - coords[first]) > radius_a + radius_b:
                continue
            if _rings_cross(
                coords[first], normals[first], rings[first],
                coords[second], normals[second], rings[second],
            ):
                count += 1
    return count


def _area_metrics(area: np.ndarray, arc: np.ndarray) -> dict:
    valid = np.isfinite(area) & (area > 0)
    clean = area[valid]
    ratios = []
    for first, second in zip(area[:-1], area[1:]):
        if np.isfinite(first) and np.isfinite(second) and first > 0 and second > 0:
            ratios.append(max(first / second, second / first))
    ratios = np.asarray(ratios, dtype=float)
    local_median = ndimage.median_filter(
        np.where(valid, area, np.nanmedian(clean)), size=15, mode="nearest"
    )
    inflation = np.divide(
        area,
        local_median,
        out=np.full_like(area, np.nan, dtype=float),
        where=local_median > 0,
    )
    return {
        "valid_sections": int(np.sum(valid)),
        "median_area_mm2": float(np.median(clean)) if len(clean) else None,
        "max_area_mm2": float(np.max(clean)) if len(clean) else None,
        "max_adjacent_ratio": float(np.max(ratios)) if len(ratios) else None,
        "p95_adjacent_ratio": float(np.percentile(ratios, 95)) if len(ratios) else None,
        "max_local_inflation": float(np.nanmax(inflation)) if len(clean) else None,
        "total_variation_per_mm": (
            float(np.sum(np.abs(np.diff(area[valid]))) / max(arc[valid][-1] - arc[valid][0], 1e-9))
            if np.sum(valid) >= 2
            else None
        ),
    }


def _load_saved_profile(patient: Path, segment: str):
    unified = json.loads(
        (patient / "features" / "unified_features.json").read_text(encoding="utf-8")
    )
    return unified["pointwise"][segment]


def _load_legacy_profile(patient: Path, segment: str):
    path = patient / "features" / "unified_features0.json"
    if not path.exists():
        return None
    unified = json.loads(path.read_text(encoding="utf-8"))
    profile = unified.get("pointwise", {}).get(segment)
    return profile if isinstance(profile, dict) else None


def run(patient: Path, segment: str, output_dir: Path) -> dict:
    saved = _load_saved_profile(patient, segment)
    coords = np.column_stack(
        [saved["centerline_x"], saved["centerline_y"], saved["centerline_z"]]
    ).astype(float)
    arc = np.asarray(saved["arc_length_mm"], dtype=float)
    mesh = trimesh.load(patient / "vessel.stl")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = next(iter(mesh.geometry.values()))

    configurations = {
        "tangent_4mm": (4.0, None),
        "voronoi_4mm_strict": (4.0, 0.0),
        "voronoi_4mm_skip_1mm": (4.0, 1.0),
        "voronoi_4mm_skip_2mm": (4.0, 2.0),
        "voronoi_4mm_skip_3mm": (4.0, 3.0),
        "voronoi_4mm_skip_4mm": (4.0, 4.0),
        "voronoi_4mm_skip_5mm": (4.0, 5.0),
    }
    methods = {
        "saved_baseline": {
            "area": np.asarray(saved["area"], dtype=float),
            "normal": np.column_stack(
                [
                    saved["section_normal_x"],
                    saved["section_normal_y"],
                    saved["section_normal_z"],
                ]
            ),
            "rings": None,
            "elapsed_seconds": 0.0,
        }
    }
    legacy = _load_legacy_profile(patient, segment)
    if legacy is not None and len(legacy.get("area", [])) == len(coords):
        methods["legacy_backup"] = {
            "area": np.asarray(legacy["area"], dtype=float),
            "normal": np.column_stack([
                legacy.get("section_normal_x", saved["section_normal_x"]),
                legacy.get("section_normal_y", saved["section_normal_y"]),
                legacy.get("section_normal_z", saved["section_normal_z"]),
            ]),
            "rings": None,
            "elapsed_seconds": 0.0,
        }

    for name, (smoothing_mm, separation_mm) in configurations.items():
        start = time.perf_counter()
        normals = _smooth_tangents(coords, smoothing_mm)
        sections = [
            _owned_section(mesh, coords, idx, normals[idx], separation_mm)
            for idx in range(len(coords))
        ]
        methods[name] = {
            "area": np.asarray([item["area"] for item in sections], dtype=float),
            "normal": normals,
            "rings": [item["ring"] for item in sections],
            "elapsed_seconds": float(time.perf_counter() - start),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for name, values in methods.items():
        metrics[name] = _area_metrics(values["area"], arc)
        metrics[name]["elapsed_seconds"] = values["elapsed_seconds"]
        if values["rings"] is not None:
            metrics[name]["nonlocal_crossing_pairs"] = _crossing_count(
                coords, arc, values["normal"], values["rings"]
            )

    with (output_dir / "area_profiles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "arc_length_mm", "curvature", *methods.keys()])
        for idx in range(len(coords)):
            writer.writerow(
                [idx, arc[idx], saved["curvature"][idx]]
                + [methods[name]["area"][idx] for name in methods]
            )

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for name, values in methods.items():
        axes[0].plot(arc, values["area"], label=name, linewidth=1.6)
    axes[0].set_ylabel("Area (mm2)")
    axes[0].legend(ncol=2, fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].plot(arc, saved["curvature"], color="black", linewidth=1.2)
    axes[1].set_xlabel("SV arc length (mm)")
    axes[1].set_ylabel("Curvature (1/mm)")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "area_comparison.png", dpi=180)
    plt.close(fig)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", type=Path, default=DEFAULT_PATIENT)
    parser.add_argument("--segment", default="sv")
    parser.add_argument("--output", type=Path, default=Path("outputs") / "sv_section_test")
    args = parser.parse_args()
    metrics = run(args.patient, args.segment.lower(), args.output)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
