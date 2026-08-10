import numpy as np

from curvature import estimate_centerline_curvature


def _circle_arc(radius=20.0, n_points=161):
    theta = np.linspace(-0.8, 0.8, n_points)
    return np.column_stack((
        radius * np.cos(theta),
        radius * np.sin(theta),
        np.zeros_like(theta),
    ))


def _legacy_three_point_curvature(coords, half_window=3):
    values = np.zeros(len(coords), dtype=float)
    for index in range(len(coords)):
        low = max(0, index - half_window)
        high = min(len(coords) - 1, index + half_window)
        left = coords[index] - coords[low]
        right = coords[high] - coords[index]
        lengths = (
            np.linalg.norm(left),
            np.linalg.norm(right),
            np.linalg.norm(coords[high] - coords[low]),
        )
        if min(lengths) <= 1e-10:
            continue
        values[index] = (
            2.0 * np.linalg.norm(np.cross(left, right))
            / np.prod(lengths)
        )
    return values


def test_straight_centerline_is_zero_including_endpoints():
    arc = np.linspace(0.0, 40.0, 161)
    coords = np.column_stack((arc, np.zeros_like(arc), np.zeros_like(arc)))

    curvature = estimate_centerline_curvature(coords)

    np.testing.assert_allclose(curvature, 0.0, atol=1e-10)


def test_circle_curvature_is_preserved_at_both_endpoints():
    radius = 20.0
    expected = 1.0 / radius

    curvature = estimate_centerline_curvature(_circle_arc(radius=radius))

    assert np.all(np.isfinite(curvature))
    np.testing.assert_allclose(curvature, expected, rtol=0.11, atol=0.001)
    assert np.isclose(curvature[0], expected, rtol=0.03)
    assert np.isclose(curvature[-1], expected, rtol=0.03)


def test_physical_scale_fit_suppresses_three_point_noise():
    rng = np.random.default_rng(123)
    noisy = _circle_arc() + rng.normal(0.0, 0.05, size=(161, 3))

    legacy = _legacy_three_point_curvature(noisy)
    curvature = estimate_centerline_curvature(noisy)
    legacy_roughness = np.mean(np.abs(np.diff(legacy[10:-10])))
    new_roughness = np.mean(np.abs(np.diff(curvature[10:-10])))

    assert new_roughness < 0.1 * legacy_roughness
    assert np.all(curvature >= 0.0)
    assert curvature[0] > 0.0
    assert curvature[-1] > 0.0
