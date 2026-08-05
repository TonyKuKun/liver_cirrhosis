import inspect
import unittest

import numpy as np
import trimesh

from extract_features import (
    _clean_pointwise_profile_for_unified,
    _global_features,
)
from extract_profiles import (
    _resample_profile,
    _section_one,
    batch_extract_profiles,
    extract_profiles,
)
from system_features import (
    _angle_features,
    _clinical_summary_features,
    _junction_section_values,
    _segment_resistance_integral,
    _tips_takeoff_angle,
    _topology_features,
)


def _nodes(points):
    return {
        index: {'x': float(point[0]), 'y': float(point[1]), 'z': float(point[2])}
        for index, point in enumerate(points)
    }


class ProfileSamplingDefaultsTests(unittest.TestCase):
    def test_all_python_profile_entry_points_default_to_200(self):
        for function in (
                _resample_profile, extract_profiles, batch_extract_profiles):
            default = inspect.signature(function).parameters['n_points'].default
            self.assertEqual(default, 200)


class ClinicalMaskTests(unittest.TestCase):
    def test_endpoint_zeros_do_not_create_severe_pvt(self):
        profile = {
            'area': [0.0, 80.0, 100.0],
            'solidity': [0.0, 0.92, 0.95],
        }
        result = _clinical_summary_features(
            {'mpv': {'path': [0, 1]}}, {}, {'mpv': profile}, {}, {})

        self.assertEqual(result['pvt_severity_grade'], 0)
        self.assertAlmostEqual(
            result['min_lumen_area_to_max_ratio_mpv'], 0.8)


class HydraulicFeatureTests(unittest.TestCase):
    def test_resistance_uses_final_hydraulic_diameter(self):
        profile = {
            'arc_length_mm': [0.0, 1.0, 2.0],
            'area': [np.pi, np.pi, np.pi],
            'hydraulic_diameter': [2.0, 2.0, 2.0],
            'eq_diameter': [20.0, 20.0, 20.0],
            'inscribed_radius': [10.0, 10.0, 10.0],
        }

        resistance, n_used, covered = _segment_resistance_integral(
            profile, 2.0)

        self.assertAlmostEqual(resistance, 2.0)
        self.assertEqual(n_used, 3)
        self.assertAlmostEqual(covered, 2.0)

    def test_resistance_reports_only_observed_coverage(self):
        profile = {
            'arc_length_mm': [0.0, 1.0, 2.0, 3.0],
            'area': [0.0, np.pi, np.pi, np.pi],
            'hydraulic_diameter': [0.0, 2.0, 2.0, 2.0],
        }

        resistance, _, covered = _segment_resistance_integral(profile, 3.0)

        self.assertAlmostEqual(resistance, 2.0)
        self.assertAlmostEqual(covered, 2.0)


class AngleFeatureTests(unittest.TestCase):
    def test_bifurcation_total_is_daughter_opening(self):
        nodes = _nodes([
            (0, 0, 0), (-10, 0, 0), (10, 0, 0), (0, 10, 0),
        ])
        segments = {
            'mpv': {'path': [0, 1]},
            'lpv': {'path': [0, 2]},
            'rpv': {'path': [0, 3]},
        }

        result = _angle_features(segments, nodes)

        self.assertAlmostEqual(result['angle_lpv_rpv'], 90.0)
        self.assertAlmostEqual(result['angle_mpv_bifurc_total'], 90.0)

    def test_tips_takeoff_supports_internal_parent_node(self):
        nodes = _nodes([
            (-10, 0, 0), (0, 0, 0), (10, 0, 0), (0, 10, 0),
        ])
        segments = {
            'lpv': {'path': [0, 1, 2]},
            'tips': {'path': [1, 3]},
        }

        self.assertAlmostEqual(_tips_takeoff_angle(segments, nodes), 90.0)


class PointwiseConsistencyTests(unittest.TestCase):
    def test_unified_resamples_retained_sections_inside_valid_arc(self):
        profile = {
            'position': [0.0, 0.25, 0.5, 0.75, 1.0],
            'arc_length_mm': [0.0, 10.0, 20.0, 30.0, 40.0],
            'total_length_mm': 40.0,
            'profile_sample_index': [0, 1, 2, 3, 4],
            'area': [0.0, 10.0, 20.0, 30.0, 40.0],
            'eq_diameter': [0.0, 2.0, 4.0, 6.0, 8.0],
            'perimeter': [0.0, 4.0, 6.0, 8.0, 10.0],
            'n_components': [0.0, 1.0, 1.0, 1.0, 1.0],
            'centerline_x': [0.0, 10.0, 20.0, 30.0, 40.0],
            'centerline_y': [0.0] * 5,
            'centerline_z': [0.0] * 5,
            'section_normal_x': [0.0] * 5,
            'section_normal_y': [0.0] * 5,
            'section_normal_z': [1.0] * 5,
            'endpoint_junction_mask': [1.0, 0.0, 0.0, 0.0, 0.0],
        }

        unified = _clean_pointwise_profile_for_unified(profile)

        np.testing.assert_allclose(
            unified['area'], [10.0, 17.5, 25.0, 32.5, 40.0])
        np.testing.assert_allclose(
            unified['arc_length_mm'], [0.0, 7.5, 15.0, 22.5, 30.0])
        np.testing.assert_allclose(
            unified['centerline_x'], [10.0, 17.5, 25.0, 32.5, 40.0])
        self.assertEqual(unified['section_valid'], [1.0] * 5)
        self.assertEqual(unified['endpoint_junction_mask'], [0.0] * 5)
        self.assertNotIn('n_components', unified)
        self.assertEqual(unified['total_length_mm'], 30.0)
        self.assertEqual(unified['_point_filter']['source_valid_n_points'], 4)
        self.assertEqual(unified['_point_filter']['interpolated_n_points'], 1)

    def test_unified_interpolates_internal_failed_section(self):
        profile = {
            'arc_length_mm': [0.0, 10.0, 20.0],
            'area': [10.0, 0.0, 30.0],
            'eq_diameter': [2.0, 0.0, 6.0],
            'perimeter': [4.0, 0.0, 8.0],
        }

        unified = _clean_pointwise_profile_for_unified(profile)

        np.testing.assert_allclose(unified['area'], [10.0, 20.0, 30.0])
        self.assertEqual(unified['section_valid'], [1.0, 1.0, 1.0])
        self.assertEqual(
            unified['_point_filter']['dropped_internal_invalid_n_points'], 1)

    def test_unified_drops_explicitly_invalid_positive_section_before_sampling(self):
        profile = {
            'arc_length_mm': [0.0, 10.0, 20.0, 30.0, 40.0],
            'area': [10.0, 20.0, 999.0, 40.0, 50.0],
            'eq_diameter': [2.0, 3.0, 99.0, 5.0, 6.0],
            'perimeter': [4.0, 5.0, 99.0, 7.0, 8.0],
            'section_valid': [1.0, 1.0, 0.0, 1.0, 1.0],
        }

        unified = _clean_pointwise_profile_for_unified(profile)

        np.testing.assert_allclose(
            unified['area'], [10.0, 20.0, 30.0, 40.0, 50.0])
        self.assertEqual(unified['section_valid'], [1.0] * 5)
        self.assertNotIn(999.0, unified['area'])
        self.assertEqual(unified['_point_filter']['source_valid_n_points'], 4)
        self.assertEqual(
            unified['_point_filter']['dropped_internal_invalid_n_points'], 1)

    def test_unified_omits_profile_with_fewer_than_two_valid_sections(self):
        profile = {
            'arc_length_mm': [0.0, 10.0, 20.0],
            'area': [0.0, 10.0, 0.0],
            'eq_diameter': [0.0, 2.0, 0.0],
            'perimeter': [0.0, 4.0, 0.0],
        }

        self.assertIsNone(_clean_pointwise_profile_for_unified(profile))

    def test_unified_restores_old_compacted_profile_to_original_count(self):
        profile = {
            'arc_length_mm': [10.0, 20.0, 30.0, 40.0],
            'area': [10.0, 20.0, 30.0, 40.0],
            'eq_diameter': [2.0, 4.0, 6.0, 8.0],
            'perimeter': [4.0, 6.0, 8.0, 10.0],
            '_point_filter': {'original_n_points': 4},
        }

        unified = _clean_pointwise_profile_for_unified(
            profile, target_n_points=5)

        self.assertEqual(len(unified['area']), 5)
        np.testing.assert_allclose(
            unified['area'], [10.0, 17.5, 25.0, 32.5, 40.0])
        self.assertEqual(unified['_point_filter']['source_serialized_n_points'], 4)
        self.assertEqual(unified['_point_filter']['interpolated_n_points'], 1)

    def test_other_vessels_on_plane_do_not_change_owned_section(self):
        first = trimesh.creation.cylinder(radius=1.0, height=4.0, sections=32)
        second = first.copy()
        second.apply_translation([4.0, 0.0, 0.0])
        mesh = trimesh.util.concatenate([first, second])

        area, _, solidity = _section_one(
            mesh, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]),
            return_extras=True)

        self.assertGreater(area, 0.0)
        self.assertGreater(solidity, 0.0)

    def test_failed_section_is_not_interpolated(self):
        raw = {
            'arc_length': np.array([0.0, 1.0, 2.0]),
            'area': np.array([10.0, 0.0, 20.0]),
            'perimeter': np.array([4.0, 0.0, 6.0]),
            'eq_diameter': np.array([3.0, 0.0, 5.0]),
            'hydraulic_diameter': np.array([10.0, 0.0, 13.0]),
            'circularity': np.array([0.9, 0.0, 0.8]),
            'solidity': np.array([1.0, 0.0, 0.95]),
            'raw_area': np.array([10.0, 0.0, 20.0]),
            'raw_perimeter': np.array([4.0, 0.0, 6.0]),
            'raw_eq_diameter': np.array([3.0, 0.0, 5.0]),
            'anchor_radius': np.array([1.0, 0.0, 2.0]),
            'owned_radius': np.array([1.0, 0.0, 2.0]),
            'r_insc_to_r_eq_ratio': np.array([0.8, 0.0, 0.7]),
            'curvature': np.zeros(3),
            'inscribed_radius': np.ones(3),
            'torsion': np.zeros(3),
        }

        profile = _resample_profile(raw, n_points=3)

        for channel in (
                'area', 'perimeter', 'eq_diameter', 'hydraulic_diameter',
                'solidity', 'r_insc_to_r_eq_ratio',
                'dA_ds_norm'):
            self.assertEqual(profile[channel][1], 0.0, channel)


class JunctionWindowTests(unittest.TestCase):
    @staticmethod
    def _segments():
        return {
            'mpv': {'path': [0, 1]},
            'sv': {'path': [0, 2]},
            'smv': {'path': [0, 3]},
        }

    def test_each_vessel_uses_its_own_first_valid_section(self):
        delayed = {
            'arc_length_mm': [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            'eq_diameter': [0.0, 0.0, 7.0, 8.0, 9.0, 10.0],
            'area': [0.0, 0.0, 17.0, 18.0, 19.0, 20.0],
        }
        regular = {
            'arc_length_mm': [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            'eq_diameter': [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            'area': [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
        }
        profiles = {'mpv': delayed, 'sv': regular, 'smv': regular}

        result = _junction_section_values(
            self._segments(), profiles, ('mpv', 'sv', 'smv'))

        self.assertEqual(result['mpv'], {'diameter': 7.0, 'area': 17.0})
        self.assertEqual(result['sv'], {'diameter': 3.0, 'area': 13.0})
        self.assertEqual(result['smv'], {'diameter': 3.0, 'area': 13.0})

    def test_shared_endpoint_at_path_end_uses_last_valid_section(self):
        segments = self._segments()
        segments['mpv'] = {'path': [1, 0]}
        profile = {
            'arc_length_mm': [0.0, 10.0, 20.0],
            'eq_diameter': [3.0, 4.0, 5.0],
            'area': [13.0, 14.0, 15.0],
        }
        profiles = {name: dict(profile) for name in ('mpv', 'sv', 'smv')}

        result = _junction_section_values(
            segments, profiles, ('mpv', 'sv', 'smv'))

        self.assertEqual(result['mpv'], {'diameter': 5.0, 'area': 15.0})
        self.assertEqual(result['sv'], {'diameter': 3.0, 'area': 13.0})


class RobustTopologyTests(unittest.TestCase):
    def test_taper_uses_terminal_five_mm_medians(self):
        profile = {
            'arc_length_mm': [0.0, 2.5, 5.0, 5.1, 7.5, 10.0],
            'area': [1.0] * 6,
            'eq_diameter': [100.0, 10.0, 10.0, 20.0, 20.0, 200.0],
        }
        result = _topology_features(
            {}, {'mpv_length': 10.0, 'total_centerline_length': 10.0},
            {'mpv': profile}, [], {})

        self.assertAlmostEqual(result['mpv_proximal_diameter'], 10.0)
        self.assertAlmostEqual(result['mpv_distal_diameter'], 20.0)
        self.assertAlmostEqual(result['mpv_taper_coefficient'], -1.0)

    def test_taper_is_oriented_from_confluence_to_bifurcation(self):
        profile = {
            'arc_length_mm': [0.0, 2.5, 5.0, 7.5, 10.0],
            'area': [1.0] * 5,
            'eq_diameter': [10.0, 10.0, 15.0, 20.0, 20.0],
        }
        segments = {
            'mpv': {'path': [10, 20]},
            'sv': {'path': [20, 30]},
            'smv': {'path': [20, 40]},
        }

        result = _topology_features(
            segments,
            {'mpv_length': 10.0, 'total_centerline_length': 10.0},
            {'mpv': profile}, [], {})

        self.assertAlmostEqual(result['mpv_proximal_diameter'], 20.0)
        self.assertAlmostEqual(result['mpv_distal_diameter'], 10.0)
        self.assertAlmostEqual(result['mpv_taper_coefficient'], 1.0)

    def test_global_length_uses_assigned_anatomical_segments(self):
        nodes = _nodes([(0, 0, 0), (100, 0, 0)])
        adjacency = {0: [1], 1: [0]}
        features = {'mpv_length': 10.0, 'sv_length': 20.0}

        result = _global_features(nodes, adjacency, features, {})

        self.assertAlmostEqual(result['total_centerline_length'], 30.0)


if __name__ == '__main__':
    unittest.main()
