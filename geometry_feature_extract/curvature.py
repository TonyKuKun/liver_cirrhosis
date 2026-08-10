"""Physical-scale centerline curvature estimation."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


DEFAULT_CURVATURE_SMOOTHING_SIGMA_MM = 3.0
DEFAULT_CURVATURE_FIT_WINDOW_MM = 8.0
DEFAULT_CURVATURE_MIN_FIT_POINTS = 7
DEFAULT_CURVATURE_POLY_ORDER = 3


def _arc_length(coords: np.ndarray) -> np.ndarray:
    steps = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(steps)))


def _linear_endpoint_padding(
    coords: np.ndarray,
    step_mm: float,
    sigma_samples: float,
    smoothing_sigma_mm: float,
) -> tuple[np.ndarray, int]:
    """Pad with fitted endpoint tangents before Gaussian smoothing."""
    pad = max(1, int(np.ceil(4.0 * sigma_samples)))
    fit_count = min(
        len(coords),
        max(3, int(np.ceil(smoothing_sigma_mm / step_mm)) + 1),
    )
    sample_s = np.arange(len(coords), dtype=float) * step_mm

    left_s = -np.arange(pad, 0, -1, dtype=float) * step_mm
    right_s = sample_s[-1] + np.arange(1, pad + 1, dtype=float) * step_mm
    left = np.empty((pad, 3), dtype=float)
    right = np.empty((pad, 3), dtype=float)
    for axis in range(3):
        left_coef = np.polyfit(
            sample_s[:fit_count], coords[:fit_count, axis], deg=1)
        right_coef = np.polyfit(
            sample_s[-fit_count:], coords[-fit_count:, axis], deg=1)
        left[:, axis] = np.polyval(left_coef, left_s)
        right[:, axis] = np.polyval(right_coef, right_s)
    return np.vstack((left, coords, right)), pad


def _smooth_uniform_coords(
    coords: np.ndarray,
    step_mm: float,
    smoothing_sigma_mm: float,
) -> np.ndarray:
    if smoothing_sigma_mm <= 0 or len(coords) < 3:
        return coords.copy()
    sigma_samples = float(smoothing_sigma_mm) / float(step_mm)
    if sigma_samples <= 0.25:
        return coords.copy()
    padded, pad = _linear_endpoint_padding(
        coords, step_mm, sigma_samples, smoothing_sigma_mm)
    smoothed = ndimage.gaussian_filter1d(
        padded, sigma=sigma_samples, axis=0, mode="nearest", truncate=4.0)
    return smoothed[pad:pad + len(coords)]


def _fit_indices(
    arc: np.ndarray,
    center: float,
    window_mm: float,
    min_fit_points: int,
) -> np.ndarray:
    """Select a fixed physical window, shifting it inward at endpoints."""
    total = float(arc[-1])
    width = min(float(window_mm), total)
    half = 0.5 * width
    start = center - half
    end = center + half
    if start < 0.0:
        end = min(total, end - start)
        start = 0.0
    if end > total:
        start = max(0.0, start - (end - total))
        end = total

    indices = np.flatnonzero((arc >= start - 1e-9) & (arc <= end + 1e-9))
    required = min(len(arc), max(3, int(min_fit_points)))
    if len(indices) < required:
        nearest = np.argsort(np.abs(arc - center), kind="stable")[:required]
        indices = np.sort(nearest)
    return indices


def _local_polynomial_curvature(
    arc: np.ndarray,
    coords: np.ndarray,
    center_index: int,
    fit_window_mm: float,
    min_fit_points: int,
    poly_order: int,
) -> float:
    center = float(arc[center_index])
    indices = _fit_indices(arc, center, fit_window_mm, min_fit_points)
    degree = min(max(2, int(poly_order)), len(indices) - 1)
    if degree < 2:
        return 0.0

    local_s = arc[indices] - center
    scale = max(0.5 * float(fit_window_mm), float(np.median(np.diff(arc))))
    weights = np.exp(-0.5 * (local_s / scale) ** 2)
    coefficients = np.column_stack([
        np.polynomial.polynomial.polyfit(
            local_s, coords[indices, axis], degree, w=weights)
        for axis in range(3)
    ])
    velocity = coefficients[1]
    acceleration = 2.0 * coefficients[2]
    speed = float(np.linalg.norm(velocity))
    if speed <= 1e-10:
        return 0.0
    curvature = float(np.linalg.norm(np.cross(velocity, acceleration)) / speed ** 3)
    return curvature if np.isfinite(curvature) and curvature > 1e-12 else 0.0


def estimate_centerline_curvature(
    coords,
    smoothing_sigma_mm=DEFAULT_CURVATURE_SMOOTHING_SIGMA_MM,
    fit_window_mm=DEFAULT_CURVATURE_FIT_WINDOW_MM,
    min_fit_points=DEFAULT_CURVATURE_MIN_FIT_POINTS,
    poly_order=DEFAULT_CURVATURE_POLY_ORDER,
):
    """Estimate pointwise 3-D curvature using physical-scale local fits.

    Coordinates are reparameterized uniformly by arc length, Gaussian-smoothed
    in millimetres, then fitted with a local polynomial over a physical window.
    At either endpoint the same window is shifted inside the observed curve, so
    the first and last curvature values use one-sided fits instead of zeros.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")
    n_points = len(coords)
    if n_points < 3:
        return np.zeros(n_points, dtype=float)
    if not np.all(np.isfinite(coords)):
        raise ValueError("coords must contain only finite values")
    if smoothing_sigma_mm < 0:
        raise ValueError("smoothing_sigma_mm must be non-negative")
    if fit_window_mm <= 0:
        raise ValueError("fit_window_mm must be positive")

    original_arc = _arc_length(coords)
    keep = np.concatenate(([True], np.diff(original_arc) > 1e-8))
    unique_arc = original_arc[keep]
    unique_coords = coords[keep]
    if len(unique_coords) < 3 or unique_arc[-1] <= 1e-8:
        return np.zeros(n_points, dtype=float)

    uniform_arc = np.linspace(0.0, float(unique_arc[-1]), len(unique_coords))
    uniform_coords = np.column_stack([
        np.interp(uniform_arc, unique_arc, unique_coords[:, axis])
        for axis in range(3)
    ])
    step_mm = float(uniform_arc[1] - uniform_arc[0])
    smoothed_coords = _smooth_uniform_coords(
        uniform_coords,
        step_mm,
        float(smoothing_sigma_mm))

    smoothed_curvature = np.asarray([
        _local_polynomial_curvature(
            uniform_arc,
            smoothed_coords,
            index,
            float(fit_window_mm),
            int(min_fit_points),
            int(poly_order),
        )
        for index in range(len(uniform_arc))
    ])
    if smoothing_sigma_mm > 0:
        # Gaussian padding is deliberately linear and can bias curvature near
        # a boundary. Use the requested one-sided physical fit there, then
        # blend into the Gaussian-smoothed interior estimate.
        one_sided_curvature = np.asarray([
            _local_polynomial_curvature(
                uniform_arc,
                uniform_coords,
                index,
                float(fit_window_mm),
                int(min_fit_points),
                int(poly_order),
            )
            for index in range(len(uniform_arc))
        ])
        boundary_width = min(
            max(0.5 * float(fit_window_mm),
                3.0 * float(smoothing_sigma_mm)),
            0.5 * float(uniform_arc[-1]))
        if boundary_width > 1e-8:
            distance_to_end = np.minimum(
                uniform_arc, float(uniform_arc[-1]) - uniform_arc)
            blend = np.clip(distance_to_end / boundary_width, 0.0, 1.0)
            blend = blend * blend * (3.0 - 2.0 * blend)
            uniform_curvature = (
                (1.0 - blend) * one_sided_curvature
                + blend * smoothed_curvature)
        else:
            uniform_curvature = one_sided_curvature
    else:
        uniform_curvature = smoothed_curvature
    return np.interp(original_arc, uniform_arc, uniform_curvature)
