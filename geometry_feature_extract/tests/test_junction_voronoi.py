import unittest

import numpy as np
from shapely.geometry import Point, Polygon

from extract_profiles import (
    _adjacent_valid_median,
    _build_clinical_junction_plan,
    _build_network_voronoi_centerlines,
    _clip_section_to_centerline_voronoi,
    _mask_endpoint_junction_sections,
)


class JunctionPlanTests(unittest.TestCase):
    def test_shared_endpoint_marks_every_vessel_endpoint(self):
        segments = {
            "mpv": {"path": [1, 2]},
            "sv": {"path": [1, 3]},
            "smv": {"path": [1, 4]},
        }
        coords = {
            "mpv": np.array([[0, 0, 0], [1, 0, 0]], dtype=float),
            "sv": np.array([[0, 0, 0], [0, 1, 0]], dtype=float),
            "smv": np.array([[0, 0, 0], [0, -1, 0]], dtype=float),
        }
        plan = _build_clinical_junction_plan(
            {"segments": segments}, {}, coords,
            {"mpv": 5.0, "sv": 4.0, "smv": 4.0})

        for vessel in segments:
            junctions = plan[vessel]["endpoint_junctions"]
            self.assertEqual(len(junctions), 1)
            self.assertEqual(junctions[0]["junction_type"], "shared_endpoint")
            self.assertEqual(plan[vessel]["side_branch_anchors"], [])

    def test_portal_bifurcation_marks_mpv_lpv_and_rpv(self):
        segments = {
            "mpv": {"path": [10, 11]},
            "lpv": {"path": [10, 12]},
            "rpv": {"path": [10, 13]},
        }
        coords = {
            "mpv": np.array([[0, 0, 0], [0, -1, 0]], dtype=float),
            "lpv": np.array([[0, 0, 0], [-1, 1, 0]], dtype=float),
            "rpv": np.array([[0, 0, 0], [1, 1, 0]], dtype=float),
        }
        plan = _build_clinical_junction_plan(
            {"segments": segments}, {}, coords,
            {"mpv": 8.0, "lpv": 5.0, "rpv": 5.0})

        for vessel in segments:
            junctions = plan[vessel]["endpoint_junctions"]
            self.assertEqual(len(junctions), 1)
            self.assertEqual(junctions[0]["junction_node_id"], 10)
            self.assertEqual(junctions[0]["junction_type"], "shared_endpoint")
            self.assertEqual(
                set(junctions[0]["connected_vessels"]),
                set(segments) - {vessel},
            )

    def test_internal_side_branch_splits_parent_and_branch_policy(self):
        segments = {
            "mpv": {"path": [10, 11, 12]},
            "lgv": {"path": [11, 13]},
        }
        nodes = {
            10: {"x": -1, "y": 0, "z": 0},
            11: {"x": 0, "y": 0, "z": 0},
            12: {"x": 1, "y": 0, "z": 0},
            13: {"x": 0, "y": 1, "z": 0},
        }
        coords = {
            "mpv": np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=float),
            "lgv": np.array([[0, 0, 0], [0, 1, 0]], dtype=float),
        }
        plan = _build_clinical_junction_plan(
            {"segments": segments}, nodes, coords,
            {"mpv": 5.0, "lgv": 2.0})

        self.assertEqual(plan["mpv"]["endpoint_junctions"], [])
        self.assertEqual(
            [item["side_branch"] for item in plan["mpv"]["side_branch_anchors"]],
            ["lgv"],
        )
        lgv_start = plan["lgv"]["endpoint_junctions"][0]
        self.assertEqual(lgv_start["junction_type"], "side_branch_endpoint")
        self.assertEqual(lgv_start["receiving_vessel"], "mpv")


class VoronoiSectionTests(unittest.TestCase):
    def test_side_branch_uses_same_local_exclusion_as_parent_curve(self):
        branch = np.column_stack((
            np.zeros(11), np.arange(11, dtype=float), np.zeros(11)))
        competitors = _build_network_voronoi_centerlines(
            [{
                "side_branch": "lgv",
                "side_branch_junction_side": "start",
                "junction_node_id": 11,
            }],
            {"lgv": branch},
            {"lgv": 1.5},
            n_points=11,
            junction_exclusion_mm=5.0,
        )

        self.assertEqual(len(competitors), 1)
        self.assertGreater(competitors[0]["centerline_coords"][0, 1], 5.0)

    def test_network_competitor_clips_only_its_side_of_parent_section(self):
        parent = np.array([[-10, 0, 0], [0, 0, 0], [10, 0, 0]], dtype=float)
        side_branch = np.array([[0, 0, 0], [0, 2, 0], [0, 4, 0]], dtype=float)
        raw = Polygon([(-2, -2), (2, -2), (2, 2), (-2, 2)])

        clipped = _clip_section_to_centerline_voronoi(
            raw,
            Point(0, 0),
            parent[1],
            np.array([1, 0, 0], dtype=float),
            centerline_coords=parent,
            centerline_index=1,
            local_exclusion_mm=5.0,
            competing_centerlines=[side_branch],
        )

        self.assertIsNotNone(clipped)
        self.assertAlmostEqual(clipped.area, 12.0, places=6)
        self.assertTrue(clipped.covers(Point(0, 0)))

    def test_network_power_weights_preserve_the_larger_parent_lumen(self):
        parent = np.array([[-10, 0, 0], [0, 0, 0], [10, 0, 0]], dtype=float)
        side_branch = np.array([[0, 0, 0], [0, 2, 0], [0, 4, 0]], dtype=float)
        raw = Polygon([(-2, -2), (2, -2), (2, 2), (-2, 2)])

        weighted = _clip_section_to_centerline_voronoi(
            raw,
            Point(0, 0),
            parent[1],
            np.array([1, 0, 0], dtype=float),
            centerline_coords=parent,
            centerline_index=1,
            local_exclusion_mm=5.0,
            competing_centerlines=[{
                "centerline_coords": side_branch,
                "radius_mm": 1.0,
            }],
            site_radius_mm=5.0,
        )

        self.assertIsNotNone(weighted)
        self.assertAlmostEqual(weighted.area, raw.area, places=6)

    def test_endpoint_area_ratio_keeps_padding_rule(self):
        n = 200
        area = np.r_[np.full(21, 200.0), np.full(n - 21, 100.0)]
        profile = {
            "position": np.linspace(0, 1, n).tolist(),
            "arc_length_mm": np.arange(n, dtype=float).tolist(),
            "area": area.tolist(),
            "raw_area": area.tolist(),
            "perimeter": np.full(n, 10.0).tolist(),
            "eq_diameter": np.full(n, 10.0).tolist(),
        }
        result = _mask_endpoint_junction_sections(
            profile,
            ratio_threshold=1.6,
            allow_terminal_start=True,
            allow_terminal_end=False,
            terminal_padding_sections=6,
        )

        event = result["area_jump_events"][0]
        self.assertEqual(
            event["masked_end_index"],
            event["critical_index"] + 6,
        )
        masked = np.where(np.asarray(result["endpoint_junction_mask"]) > 0)[0]
        np.testing.assert_array_equal(
            masked, np.arange(event["masked_end_index"] + 1))
        self.assertTrue(np.all(np.asarray(result["area"])[masked] == 0))
        self.assertGreater(result["area"][event["masked_end_index"] + 1], 0)

    def test_endpoint_area_ratio_uses_strongest_multistep_transition(self):
        n = 200
        area = np.r_[
            np.full(24, 700.0),
            np.full(12, 520.0),
            np.full(6, 320.0),
            np.full(n - 42, 80.0),
        ]
        profile = {
            "position": np.linspace(0, 1, n).tolist(),
            "arc_length_mm": np.arange(n, dtype=float).tolist(),
            "area": area.tolist(),
            "raw_area": area.tolist(),
            "perimeter": np.full(n, 10.0).tolist(),
            "eq_diameter": np.full(n, 10.0).tolist(),
        }
        result = _mask_endpoint_junction_sections(
            profile,
            ratio_threshold=1.6,
            allow_terminal_start=True,
            allow_terminal_end=False,
            terminal_padding_sections=6,
        )

        event = result["area_jump_events"][0]
        self.assertGreaterEqual(event["critical_index"], 41)
        self.assertEqual(event["masked_end_index"], event["critical_index"] + 6)
        self.assertEqual(result["area"][event["masked_end_index"] + 1], 80.0)

    def test_endpoint_area_ratio_reaches_short_rpv_transition(self):
        n = 200
        area = np.r_[
            np.linspace(528.0, 254.0, 76),
            np.linspace(138.0, 123.0, n - 76),
        ]
        profile = {
            "position": np.linspace(0, 1, n).tolist(),
            "arc_length_mm": np.linspace(0, 24.823, n).tolist(),
            "area": area.tolist(),
            "raw_area": area.tolist(),
            "perimeter": np.full(n, 10.0).tolist(),
            "eq_diameter": np.full(n, 10.0).tolist(),
        }
        result = _mask_endpoint_junction_sections(
            profile,
            ratio_threshold=1.6,
            allow_terminal_start=True,
            allow_terminal_end=False,
            terminal_padding_sections=6,
        )

        event = result["area_jump_events"][0]
        self.assertGreater(
            event["detected_arc_end_mm"],
            0.35 * profile["arc_length_mm"][-1],
        )
        self.assertEqual(event["masked_end_index"], event["critical_index"] + 6)
        self.assertTrue(np.all(np.asarray(result["area"])[:event["masked_end_index"] + 1] == 0))
        self.assertGreater(result["area"][event["masked_end_index"] + 1], 0)

    def test_endpoint_area_ratio_accepts_short_terminal_contamination(self):
        n = 200
        area = np.r_[
            np.linspace(1672.0, 1508.0, 7),
            np.linspace(545.0, 300.0, n - 7),
        ]
        profile = {
            "position": np.linspace(0, 1, n).tolist(),
            "arc_length_mm": np.linspace(0, 69.287, n).tolist(),
            "area": area.tolist(),
            "raw_area": area.tolist(),
            "perimeter": np.full(n, 10.0).tolist(),
            "eq_diameter": np.full(n, 10.0).tolist(),
        }

        result = _mask_endpoint_junction_sections(
            profile,
            ratio_threshold=1.6,
            allow_terminal_start=True,
            allow_terminal_end=False,
            terminal_padding_sections=6,
        )

        event = result["area_jump_events"][0]
        self.assertLess(event["detected_arc_end_mm"], 4.0)
        self.assertEqual(event["masked_end_index"], event["critical_index"] + 6)
        self.assertGreater(result["n_endpoint_junction_zeroed"], 0)

    def test_endpoint_area_ratio_handles_junctions_at_both_ends(self):
        n = 200
        area = np.r_[
            np.linspace(400.0, 330.0, 10),
            np.full(n - 20, 180.0),
            np.linspace(300.0, 360.0, 10),
        ]
        profile = {
            "position": np.linspace(0, 1, n).tolist(),
            "arc_length_mm": np.linspace(0, 70.0, n).tolist(),
            "area": area.tolist(),
            "raw_area": area.tolist(),
            "perimeter": np.full(n, 10.0).tolist(),
            "eq_diameter": np.full(n, 10.0).tolist(),
        }

        result = _mask_endpoint_junction_sections(
            profile,
            ratio_threshold=1.6,
            allow_terminal_start=True,
            allow_terminal_end=True,
            terminal_padding_sections=6,
        )

        event_types = {event["type"] for event in result["area_jump_events"]}
        self.assertEqual(
            event_types,
            {"endpoint_start_interval_zeroed", "endpoint_end_interval_zeroed"},
        )

    def test_endpoint_mask_does_not_cross_internal_side_branch(self):
        n = 200
        area = np.r_[np.full(120, 100.0), np.full(80, 300.0)]
        profile = {
            "position": np.linspace(0, 1, n).tolist(),
            "arc_length_mm": np.arange(n, dtype=float).tolist(),
            "area": area.tolist(),
            "raw_area": area.tolist(),
            "perimeter": np.full(n, 10.0).tolist(),
            "eq_diameter": np.full(n, 10.0).tolist(),
        }

        result = _mask_endpoint_junction_sections(
            profile,
            ratio_threshold=1.6,
            allow_terminal_start=False,
            allow_terminal_end=True,
            terminal_padding_sections=6,
            protected_side_branch_arcs=[150.0],
            side_branch_protection_mm=5.0,
        )

        event = result["area_jump_events"][0]
        self.assertTrue(event["side_branch_topology_clamped"])
        self.assertEqual(event["masked_start_index"], 155)
        self.assertGreater(result["area"][150], 0.0)
        self.assertTrue(np.all(np.asarray(result["area"])[155:] == 0.0))

    def test_endpoint_area_ratio_accepts_one_terminal_section(self):
        n = 200
        area = np.r_[300.0, np.full(n - 1, 100.0)]
        profile = {
            "position": np.linspace(0, 1, n).tolist(),
            "arc_length_mm": np.arange(n, dtype=float).tolist(),
            "area": area.tolist(),
            "raw_area": area.tolist(),
            "perimeter": np.full(n, 10.0).tolist(),
            "eq_diameter": np.full(n, 10.0).tolist(),
        }

        result = _mask_endpoint_junction_sections(
            profile,
            ratio_threshold=1.6,
            allow_terminal_start=True,
            allow_terminal_end=False,
            terminal_padding_sections=6,
        )

        event = result["area_jump_events"][0]
        self.assertEqual(event["critical_index"], 0)
        self.assertEqual(event["masked_end_index"], 6)

    def test_endpoint_area_ratio_uses_raw_not_voronoi_area(self):
        n = 200
        owned_area = np.r_[np.full(10, 300.0), np.full(n - 10, 100.0)]
        raw_area = np.full(n, 100.0)
        profile = {
            "position": np.linspace(0, 1, n).tolist(),
            "arc_length_mm": np.arange(n, dtype=float).tolist(),
            "area": owned_area.tolist(),
            "raw_area": raw_area.tolist(),
            "perimeter": np.full(n, 10.0).tolist(),
            "eq_diameter": np.full(n, 10.0).tolist(),
        }

        result = _mask_endpoint_junction_sections(
            profile,
            ratio_threshold=1.6,
            allow_terminal_start=True,
            allow_terminal_end=False,
            terminal_padding_sections=6,
        )

        self.assertEqual(result["area_jump_events"], [])
        self.assertEqual(result["area_jump_parameters"]["area_channel"], "raw_area")

    def test_gradual_terminal_taper_does_not_mask_from_remote_drop(self):
        """A smooth caliber change is not an endpoint junction."""
        n = 200
        area = np.r_[
            np.linspace(250.0, 180.0, 120),
            np.linspace(175.0, 120.0, n - 120),
        ]
        profile = {
            "position": np.linspace(0, 1, n).tolist(),
            "arc_length_mm": np.arange(n, dtype=float).tolist(),
            "area": area.tolist(),
            "raw_area": area.tolist(),
            "perimeter": np.full(n, 10.0).tolist(),
            "eq_diameter": np.full(n, 10.0).tolist(),
        }

        result = _mask_endpoint_junction_sections(
            profile,
            ratio_threshold=1.6,
            allow_terminal_start=True,
            allow_terminal_end=False,
            terminal_padding_sections=6,
        )

        self.assertEqual(result["area_jump_events"], [])
        self.assertEqual(result["n_endpoint_junction_zeroed"], 0)

    def test_local_step_does_not_require_enlarged_endpoint_sample(self):
        area = np.r_[
            np.linspace(350.0, 620.0, 60),
            np.full(140, 300.0),
        ]
        profile = {
            "position": np.linspace(0, 1, len(area)).tolist(),
            "arc_length_mm": np.arange(len(area), dtype=float).tolist(),
            "area": area.tolist(),
            "raw_area": area.tolist(),
            "perimeter": np.full(len(area), 10.0).tolist(),
            "eq_diameter": np.full(len(area), 10.0).tolist(),
        }

        result = _mask_endpoint_junction_sections(
            profile,
            ratio_threshold=1.6,
            allow_terminal_start=True,
            allow_terminal_end=False,
            terminal_padding_sections=6,
        )

        self.assertEqual(len(result["area_jump_events"]), 1)
        self.assertGreaterEqual(
            result["area_jump_events"][0]["critical_index"], 58)

    def test_wide_terminal_transition_is_detected_at_second_scale(self):
        area = np.r_[
            np.linspace(385.0, 242.0, 21),
            np.full(179, 185.0),
        ]
        profile = {
            "position": np.linspace(0, 1, len(area)).tolist(),
            "arc_length_mm": np.arange(len(area), dtype=float).tolist(),
            "area": area.tolist(),
            "raw_area": area.tolist(),
            "perimeter": np.full(len(area), 10.0).tolist(),
            "eq_diameter": np.full(len(area), 10.0).tolist(),
        }

        result = _mask_endpoint_junction_sections(
            profile,
            ratio_threshold=1.6,
            allow_terminal_start=True,
            allow_terminal_end=False,
            terminal_padding_sections=6,
            terminal_reference_points=5,
        )

        self.assertEqual(len(result["area_jump_events"]), 1)
        self.assertEqual(
            result["area_jump_parameters"]["transition_window_scales_points"],
            [5, 10],
        )

    def test_internal_side_branch_bump_does_not_mask_endpoint(self):
        area = np.r_[
            np.full(50, 100.0),
            np.full(20, 300.0),
            np.full(130, 100.0),
        ]
        profile = {
            "position": np.linspace(0, 1, len(area)).tolist(),
            "arc_length_mm": np.arange(len(area), dtype=float).tolist(),
            "area": area.tolist(),
            "raw_area": area.tolist(),
            "perimeter": np.full(len(area), 10.0).tolist(),
            "eq_diameter": np.full(len(area), 10.0).tolist(),
        }

        result = _mask_endpoint_junction_sections(
            profile,
            ratio_threshold=1.6,
            allow_terminal_start=True,
            allow_terminal_end=False,
        )

        self.assertEqual(result["area_jump_events"], [])

    def test_internal_thrombus_valley_does_not_mask_endpoint(self):
        area = np.r_[
            np.full(50, 200.0),
            np.full(20, 50.0),
            np.full(130, 200.0),
        ]
        profile = {
            "position": np.linspace(0, 1, len(area)).tolist(),
            "arc_length_mm": np.arange(len(area), dtype=float).tolist(),
            "area": area.tolist(),
            "raw_area": area.tolist(),
            "perimeter": np.full(len(area), 10.0).tolist(),
            "eq_diameter": np.full(len(area), 10.0).tolist(),
        }

        result = _mask_endpoint_junction_sections(
            profile,
            ratio_threshold=1.6,
            allow_terminal_start=True,
            allow_terminal_end=False,
        )

        self.assertEqual(result["area_jump_events"], [])

    def test_terminal_expansion_survives_internal_side_branch_pairing(self):
        area = np.r_[
            np.full(20, 300.0),
            np.full(40, 100.0),
            np.full(20, 300.0),
            np.full(120, 100.0),
        ]
        profile = {
            "position": np.linspace(0, 1, len(area)).tolist(),
            "arc_length_mm": np.arange(len(area), dtype=float).tolist(),
            "area": area.tolist(),
            "raw_area": area.tolist(),
            "perimeter": np.full(len(area), 10.0).tolist(),
            "eq_diameter": np.full(len(area), 10.0).tolist(),
        }

        result = _mask_endpoint_junction_sections(
            profile,
            ratio_threshold=1.6,
            allow_terminal_start=True,
            allow_terminal_end=False,
            terminal_padding_sections=6,
        )

        self.assertEqual(len(result["area_jump_events"]), 1)
        self.assertLess(result["area_jump_events"][0]["critical_index"], 30)

    def test_weaker_return_step_closes_internal_area_bump(self):
        area = np.r_[
            np.full(140, 170.0),
            np.full(20, 250.0),
            np.full(40, 145.0),
        ]
        profile = {
            "position": np.linspace(0, 1, len(area)).tolist(),
            "arc_length_mm": np.arange(len(area), dtype=float).tolist(),
            "area": area.tolist(),
            "raw_area": area.tolist(),
            "perimeter": np.full(len(area), 10.0).tolist(),
            "eq_diameter": np.full(len(area), 10.0).tolist(),
        }

        result = _mask_endpoint_junction_sections(
            profile,
            ratio_threshold=1.6,
            allow_terminal_start=True,
            allow_terminal_end=False,
        )

        self.assertEqual(result["area_jump_events"], [])
        self.assertEqual(
            result["area_jump_parameters"]["n_strong_local_transitions"], 1)
        self.assertEqual(
            result["area_jump_parameters"]["n_paired_local_transitions"], 2)

    def test_broad_terminal_expansion_is_not_paired_as_local_bump(self):
        area = np.r_[
            np.linspace(350.0, 625.0, 60),
            np.full(140, 310.0),
        ]
        profile = {
            "position": np.linspace(0, 1, len(area)).tolist(),
            "arc_length_mm": np.arange(len(area), dtype=float).tolist(),
            "area": area.tolist(),
            "raw_area": area.tolist(),
            "perimeter": np.full(len(area), 10.0).tolist(),
            "eq_diameter": np.full(len(area), 10.0).tolist(),
        }

        result = _mask_endpoint_junction_sections(
            profile,
            ratio_threshold=1.6,
            allow_terminal_start=True,
            allow_terminal_end=False,
            terminal_reference_points=5,
        )

        self.assertEqual(len(result["area_jump_events"]), 1)
        self.assertGreater(
            result["area_jump_events"][0]["critical_index"], 50)
        self.assertEqual(
            result["area_jump_parameters"][
                "transition_pairing_max_span_points"],
            40,
        )

    def test_broad_low_trunk_does_not_hide_start_junction(self):
        area = np.r_[
            np.linspace(420.0, 180.0, 21),
            np.full(80, 190.0),
            np.linspace(225.0, 300.0, 20),
            np.full(79, 300.0),
        ]
        profile = {
            "position": np.linspace(0, 1, len(area)).tolist(),
            "arc_length_mm": np.arange(len(area), dtype=float).tolist(),
            "area": area.tolist(),
            "raw_area": area.tolist(),
            "perimeter": np.full(len(area), 10.0).tolist(),
            "eq_diameter": np.full(len(area), 10.0).tolist(),
        }

        result = _mask_endpoint_junction_sections(
            profile,
            ratio_threshold=1.6,
            allow_terminal_start=True,
            allow_terminal_end=True,
            terminal_reference_points=5,
        )

        event_types = {event["type"] for event in result["area_jump_events"]}
        self.assertIn("endpoint_start_interval_zeroed", event_types)

    def test_remote_stronger_drop_does_not_move_monotonic_boundary(self):
        area = np.r_[
            np.linspace(580.0, 140.0, 61),
            np.linspace(140.0, 100.0, 10),
            np.linspace(100.0, 23.0, 98),
            np.linspace(23.0, 6.0, 31),
        ]
        profile = {
            "position": np.linspace(0, 1, len(area)).tolist(),
            "arc_length_mm": np.arange(len(area), dtype=float).tolist(),
            "area": area.tolist(),
            "raw_area": area.tolist(),
            "perimeter": np.full(len(area), 10.0).tolist(),
            "eq_diameter": np.full(len(area), 10.0).tolist(),
        }

        result = _mask_endpoint_junction_sections(
            profile,
            ratio_threshold=1.6,
            allow_terminal_start=True,
            allow_terminal_end=False,
            terminal_reference_points=5,
        )

        event = result["area_jump_events"][0]
        self.assertLess(event["critical_index"], 80)
        self.assertGreaterEqual(event["area_ratio"], 1.6)

    def test_endpoint_area_ratio_accepts_two_interior_reference_sections(self):
        values = np.asarray([300.0, 300.0, 100.0, 100.0])
        valid = np.ones(len(values), dtype=bool)

        reference = _adjacent_valid_median(
            values, valid, boundary=1, side="start", n_points=5)

        self.assertEqual(reference, 100.0)


if __name__ == "__main__":
    unittest.main()
