import unittest

import numpy as np

from smooth_centerline import (
    _anatomical_atomic_join_candidates,
    _atomic_centerline_segments,
    _junction_kink_metrics,
    _junction_turn_angle_degrees,
    _smooth_path_at_junction,
)


class CenterlineJunctionSmoothingTests(unittest.TestCase):
    def test_atomic_join_candidates_only_join_same_anatomical_vessel(self):
        adjacency = {
            0: {1},
            1: {0, 2},
            2: {1, 3, 4},
            3: {2},
            4: {2},
        }
        atoms = _atomic_centerline_segments(adjacency)
        self.assertEqual(
            {atom['id'] for atom in atoms}, {'0:2', '2:3', '2:4'})
        assignments = {
            'segments': {
                'mpv': {'path': [0, 1, 2, 3]},
                'lgv': {'path': [2, 4]},
            }
        }
        candidates = _anatomical_atomic_join_candidates(
            assignments, adjacency)
        self.assertEqual(candidates['mpv'], [{
            'junction_node_id': 2,
            'atomic_segment_ids': ['0:2', '2:3'],
        }])
        self.assertNotIn('lgv', candidates)

    def test_manual_atom_assignment_is_used_before_path_subset_matching(self):
        adjacency = {
            0: {1}, 1: {0, 2}, 2: {1, 3, 4}, 3: {2}, 4: {2},
        }
        assignments = {
            'segments': {
                'mpv': {'path': [0, 1, 2, 3]},
                'lgv': {'path': [2, 4]},
            },
            'assignments': {
                '0:2': {'vessel': 'mpv'},
                '2:3': {'vessel': 'mpv'},
                '2:4': {'vessel': 'lgv'},
            },
        }
        candidates = _anatomical_atomic_join_candidates(
            assignments, adjacency)
        self.assertEqual(candidates['mpv'][0]['junction_node_id'], 2)
        self.assertEqual(candidates['mpv'][0]['atomic_segment_ids'],
                         ['0:2', '2:3'])

    def test_local_hermite_smoothing_keeps_shared_node_and_removes_kink(self):
        spacing = 0.2
        left = np.column_stack((
            np.arange(-6.0, spacing, spacing),
            np.zeros(31),
            np.zeros(31),
        ))
        right_distance = np.arange(spacing, 6.0 + spacing, spacing)
        right = np.column_stack((
            right_distance / np.sqrt(2.0),
            right_distance / np.sqrt(2.0),
            np.zeros(30),
        ))
        coords = np.vstack((left, right))
        junction_index = 30

        before = _junction_turn_angle_degrees(
            coords, junction_index, tangent_span_mm=2.0)
        smoothed, detail = _smooth_path_at_junction(
            coords, junction_index,
            half_window_mm=6.0,
            tangent_span_mm=2.0,
        )
        after = _junction_turn_angle_degrees(
            smoothed, junction_index, tangent_span_mm=0.25)

        self.assertGreater(before, 40.0)
        self.assertLess(after, 5.0)
        self.assertLess(detail['angle_after_deg'], before)
        np.testing.assert_allclose(
            smoothed[junction_index], coords[junction_index])
        np.testing.assert_allclose(smoothed[0], coords[0])
        np.testing.assert_allclose(smoothed[-1], coords[-1])
        self.assertGreater(detail['max_displacement_mm'], 0.0)

    def test_localized_24_degree_join_is_detected_below_old_threshold(self):
        angle = np.radians(24.0)
        left = np.column_stack((
            np.arange(-6.0, 0.2, 0.2),
            np.zeros(31),
            np.zeros(31),
        ))
        distance = np.arange(0.2, 6.0 + 0.2, 0.2)
        right = np.column_stack((
            distance * np.cos(angle),
            distance * np.sin(angle),
            np.zeros(30),
        ))
        coords = np.vstack((left, right))

        kink = _junction_kink_metrics(coords, 30)

        self.assertGreater(kink['vertex_angle_deg'], 15.0)
        self.assertGreater(kink['angle_excess_deg'], 10.0)


if __name__ == '__main__':
    unittest.main()
