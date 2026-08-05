import unittest
from unittest.mock import patch

import numpy as np

from extract_profiles import (
    _extract_branch_raw_profile,
    _normal_perturbation_shells,
)


def _section_result(area):
    if area <= 0:
        return (0.0, 0.0, 999.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return (area, 20.0, 1.0, 0.9, area, 20.0, 2.0, 0.0, 1.0)


class SectionNormalFallbackTests(unittest.TestCase):
    def setUp(self):
        self.coords = np.column_stack((
            np.arange(5, dtype=float),
            np.zeros(5),
            np.zeros(5),
        ))

    def _extract(self, section_one, policy=None):
        with (
            patch('extract_profiles._section_one', side_effect=section_one),
            patch(
                'extract_profiles._compute_inscribed_radius_per_point',
                return_value=np.ones(len(self.coords)),
            ),
        ):
            return _extract_branch_raw_profile(
                list(range(len(self.coords))),
                {},
                object(),
                branch_coords=self.coords,
                section_step=1,
                normal_search_policy=policy,
            )

    def test_perturbation_shells_are_bounded_and_angularly_exact(self):
        shells = list(_normal_perturbation_shells(
            np.array([1.0, 0.0, 0.0]),
            step_deg=2.0,
            max_deg=6.0,
            directions=8,
        ))

        self.assertEqual([angle for angle, _ in shells], [2.0, 4.0, 6.0])
        self.assertTrue(all(len(normals) == 8 for _, normals in shells))
        for angle, normals in shells:
            for normal in normals:
                measured = np.rad2deg(np.arccos(np.clip(normal[0], -1.0, 1.0)))
                self.assertAlmostEqual(measured, angle, places=6)

    def test_successful_base_section_does_not_trigger_fallback(self):
        calls = []

        def section_one(_mesh, _point, normal, **_kwargs):
            calls.append(np.asarray(normal))
            return _section_result(100.0)

        result = self._extract(section_one)

        self.assertEqual(len(calls), len(self.coords))
        self.assertEqual(result['_normal_search_counts']['fallback_triggered'], 0)
        self.assertEqual(result['_normal_search_candidate_count'], len(self.coords))
        np.testing.assert_allclose(result['section_normal_offset_deg'], 0.0)

    def test_failed_base_section_is_recovered_at_smallest_angle(self):
        def section_one(_mesh, _point, normal, **_kwargs):
            normal = np.asarray(normal, dtype=float)
            angle = np.rad2deg(np.arccos(np.clip(normal[0], -1.0, 1.0)))
            return _section_result(100.0 + normal[1]) if angle >= 1.9 else _section_result(0.0)

        result = self._extract(section_one)

        self.assertEqual(result['_n_section_failures'], 0)
        self.assertEqual(result['_normal_search_counts']['fallback_triggered'], 5)
        self.assertEqual(result['_normal_search_counts']['fallback_recovered'], 5)
        self.assertEqual(result['_normal_search_counts']['fallback_failed'], 0)
        np.testing.assert_allclose(result['section_normal_offset_deg'], 2.0)
        self.assertTrue(np.all(np.asarray(result['area']) > 0))

    def test_failed_fallback_stops_at_configured_maximum(self):
        calls = []

        def section_one(_mesh, _point, normal, **_kwargs):
            calls.append(np.asarray(normal))
            return _section_result(0.0)

        result = self._extract(section_one, policy={
            'failure_fallback_step_deg': 2.0,
            'failure_fallback_max_deg': 4.0,
            'failure_fallback_directions': 4,
        })

        expected_per_point = 1 + 2 * 4
        self.assertEqual(len(calls), len(self.coords) * expected_per_point)
        self.assertEqual(result['_n_section_failures'], len(self.coords))
        self.assertEqual(result['_normal_search_counts']['fallback_failed'], 5)
        self.assertEqual(
            result['_normal_search_candidate_count'],
            len(self.coords) * expected_per_point,
        )


if __name__ == '__main__':
    unittest.main()
