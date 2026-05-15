from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pretrain.preprocess import (
    PRETRAIN_ALGORITHM_VERSION,
    ask_for_cleanup_plan,
    ask_for_coarse_plan,
    _apply_llm_removal,
    _cleanup_mask_by_region_growth,
    _crop_slices,
    _default_plan,
    _filter_components_by_reference_bbox,
    _portal_seed_from_cleanup_plan,
    _portal_seed_from_reference,
    _portal_seed_from_plan,
    _portal_seed_from_volume,
    _reference_envelope_mask,
    _reference_crop_from_stl,
    _should_run_region_growth,
    _segment_once,
    _sanitize_plan,
    _should_rebuild_pretrain,
    _should_skip_existing_pretrain,
)
from pretrain.threshold_scorer import _score_coronal_tree, _select_best_candidate, search_best_threshold
from pretrain.z_standardizer import _validate_z_range, standardize_z_range
from utils.common import DicomVolume, PatientCase, discover_patients, mask_to_stl, write_binary_stl


class _CaptureClient:
    enabled = True

    def __init__(self, response: dict | None = None) -> None:
        self.response = response or {}
        self.calls: list[tuple[str, str, list[Path]]] = []

    def chat_json(self, system: str, prompt: str, image_paths: list[Path] | None = None) -> dict:
        self.calls.append((system, prompt, image_paths or []))
        return dict(self.response)


def _binary_stl_points(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    triangles = struct.unpack("<I", raw[80:84])[0]
    arr = np.frombuffer(
        raw,
        dtype=np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")]),
        offset=84,
        count=triangles,
    )
    return arr["v"].reshape(-1, 3)


class PretrainRoiTests(unittest.TestCase):
    def test_single_patient_root_is_discovered_from_dcm_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            patient = Path(tmp) / "20201224WangMingLian"
            (patient / "dcm").mkdir(parents=True)

            cases = discover_patients(patient)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].name, "20201224WangMingLian")
        self.assertEqual(cases[0].dcm_dir.name, "dcm")

    def test_patient_names_with_invalid_markers_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("skip@case", "skip!case", "skip&case", "keep_case"):
                (root / name / "dcm").mkdir(parents=True)

            cases = discover_patients(root)

        self.assertEqual([case.name for case in cases], ["keep_case"])

    def test_fallback_crop_is_tighter_than_old_abdominal_box(self) -> None:
        crop = _default_plan(is_post_tips=False)["crop"]

        self.assertLessEqual(crop["z"][1] - crop["z"][0], 0.46)
        self.assertLessEqual(crop["y"][1] - crop["y"][0], 0.40)
        self.assertLessEqual(crop["x"][1] - crop["x"][0], 0.66)

    def test_sanitize_clamps_overwide_model_crop(self) -> None:
        plan = _sanitize_plan(
            {
                "hu_low": 20,
                "hu_high": 900,
                "crop": {"z": [0.0, 1.0], "y": [0.0, 1.0], "x": [0.0, 1.0]},
                "notes": "too broad",
            },
            is_post_tips=False,
        )

        self.assertGreaterEqual(plan["hu_low"], 80)
        self.assertLessEqual(plan["hu_high"], 420)
        self.assertLessEqual(plan["crop"]["z"][1] - plan["crop"]["z"][0], 0.46)
        self.assertLessEqual(plan["crop"]["y"][1] - plan["crop"]["y"][0], 0.40)

    def test_tips_plan_keeps_high_hu_but_limits_space(self) -> None:
        plan = _sanitize_plan(
            {
                "hu_low": 20,
                "hu_high": 1500,
                "crop": {"z": [0.0, 1.0], "y": [0.0, 1.0], "x": [0.0, 1.0]},
            },
            is_post_tips=True,
        )

        self.assertLessEqual(plan["hu_high"], 720)
        self.assertLessEqual(plan["crop"]["z"][1] - plan["crop"]["z"][0], 0.58)
        self.assertLessEqual(plan["crop"]["y"][1] - plan["crop"]["y"][0], 0.48)
        self.assertLessEqual(plan["crop"]["x"][1] - plan["crop"]["x"][0], 0.70)

    def test_tips_plan_clamps_model_low_portal_threshold_to_search_floor(self) -> None:
        plan = _sanitize_plan(
            {
                "hu_low": 75,
                "hu_high": 260,
                "crop": {"z": [0.4, 0.8], "y": [0.35, 0.65], "x": [0.25, 0.7]},
                "portal_seed": {"z": 0.55, "y": 0.5, "x": 0.45},
            },
            is_post_tips=True,
        )

        self.assertEqual(plan["hu_low"], 80.0)
        self.assertEqual(plan["hu_high"], 260.0)
        self.assertEqual(plan["portal_seed"], {"z": 0.55, "y": 0.5, "x": 0.45})

    def test_tips_segmentation_uses_adaptive_low_threshold_below_130(self) -> None:
        vol = np.zeros((12, 12, 12), dtype=np.float32)
        vol[4:8, 4:8, 4:8] = 95.0
        plan = {
            "hu_low": 75.0,
            "hu_high": 120.0,
            "crop": {"z": [0.2, 0.9], "y": [0.2, 0.9], "x": [0.2, 0.9]},
            "notes": "adaptive low HU portal",
        }

        mask = _segment_once(vol, plan, is_post_tips=True)

        self.assertGreater(int(mask.sum()), 0)
        self.assertTrue(mask[5, 5, 5])

    def test_post_tips_segmentation_honors_model_crop_instead_of_fixed_intersection(self) -> None:
        vol = np.zeros((40, 40, 40), dtype=np.float32)
        vol[18:24, 3:9, 10:16] = 105.0
        plan = {
            "hu_low": 80.0,
            "hu_high": 140.0,
            "crop": {"z": [0.40, 0.70], "y": [0.05, 0.30], "x": [0.20, 0.50]},
            "include_tips": False,
        }

        mask = _segment_once(vol, plan, is_post_tips=True)

        self.assertGreater(int(mask.sum()), 0)
        self.assertTrue(mask[20, 5, 12])

    def test_coarse_prompt_is_tips_specific_and_asks_for_sv_extent(self) -> None:
        client = _CaptureClient()

        ask_for_coarse_plan(client, "case#", True, {"p50": 40.0}, [])
        tips_system, tips_prompt, _ = client.calls[-1]
        ask_for_coarse_plan(client, "case", False, {"p50": 40.0}, [])
        non_tips_system, non_tips_prompt, _ = client.calls[-1]

        self.assertIn("cirrhosis", tips_system.lower())
        self.assertIn("portal hypertension", tips_system.lower())
        self.assertIn("TIPS stent", tips_prompt)
        self.assertIn("bright gastric", tips_prompt)
        self.assertIn("splenic vein", tips_prompt)
        self.assertIn("long enough", tips_prompt)
        self.assertIn("non-TIPS", non_tips_prompt)
        self.assertIn("avoid bone", non_tips_prompt)

    def test_crop_slices_use_normalized_bounds(self) -> None:
        slices = _crop_slices((100, 200, 300), {"z": [0.25, 0.75], "y": [0.1, 0.6], "x": [0.2, 0.8]})
        mask = np.zeros((100, 200, 300), dtype=bool)
        mask[slices] = True

        self.assertEqual(mask.sum(), 50 * 100 * 180)
        self.assertTrue(mask[25, 20, 60])
        self.assertFalse(mask[24, 20, 60])

    def test_binary_stl_writer_uses_compact_standard_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "one_triangle.stl"
            vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
            faces = np.array([[0, 1, 2]], dtype=np.int64)

            write_binary_stl(out, vertices, faces)
            raw = out.read_bytes()

        self.assertEqual(len(raw), 84 + 50)
        self.assertEqual(struct.unpack("<I", raw[80:84])[0], 1)

    def test_mask_to_stl_applies_dicom_origin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_out = Path(tmp) / "base.stl"
            shifted_out = Path(tmp) / "shifted.stl"
            mask = np.zeros((3, 3, 3), dtype=np.uint8)
            mask[1, 1, 1] = 1

            mask_to_stl(mask, (2.0, 3.0, 4.0), base_out)
            mask_to_stl(mask, (2.0, 3.0, 4.0), shifted_out, origin_xyz=(-10.0, -20.0, -30.0))
            base_points = _binary_stl_points(base_out)
            shifted_points = _binary_stl_points(shifted_out)

        self.assertGreater(len(shifted_points), 0)
        np.testing.assert_allclose(shifted_points.min(axis=0) - base_points.min(axis=0), [-10.0, -20.0, -30.0], atol=1e-5)
        np.testing.assert_allclose(shifted_points.max(axis=0) - base_points.max(axis=0), [-10.0, -20.0, -30.0], atol=1e-5)

    def test_reference_pre_stl_debug_crop_builds_patient_space_crop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            patient = Path(tmp) / "case#"
            (patient / "dcm").mkdir(parents=True)
            case = PatientCase("case#", patient, patient / "dcm", patient / "vessel.stl", patient / "pretrain.stl", patient / "predict.stl", True)
            vertices = np.array(
                [
                    [-8.0, -18.0, -28.0],
                    [8.0, -18.0, -28.0],
                    [8.0, -6.0, -28.0],
                    [-8.0, -6.0, -28.0],
                    [-8.0, -18.0, -8.0],
                    [8.0, -18.0, -8.0],
                    [8.0, -6.0, -8.0],
                    [-8.0, -6.0, -8.0],
                ],
                dtype=np.float32,
            )
            faces = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]], dtype=np.int64)
            write_binary_stl(patient / "pre.stl", vertices, faces)
            volume = DicomVolume(np.zeros((40, 50, 60), dtype=np.float32), (1.0, 2.0, 4.0), (-20.0, -30.0, -40.0))

            crop = _reference_crop_from_stl(case, volume, padding_zyx=(0.0, 0.0, 0.0))

        self.assertIsNotNone(crop)
        assert crop is not None
        np.testing.assert_allclose(crop["z"], [0.3, 0.8], atol=1e-6)
        np.testing.assert_allclose(crop["y"], [0.12, 0.24], atol=1e-6)
        np.testing.assert_allclose(crop["x"], [0.05, 0.1166666667], atol=1e-6)

    def test_region_growth_keeps_only_component_connected_to_portal_seed(self) -> None:
        mask = np.zeros((8, 8, 8), dtype=bool)
        mask[1:3, 1:3, 1:3] = True
        mask[5:7, 5:7, 5:7] = True

        cleaned, info = _cleanup_mask_by_region_growth(mask, seed_zyx=(1.5, 1.5, 1.5))

        self.assertEqual(int(cleaned.sum()), 8)
        self.assertTrue(cleaned[1, 1, 1])
        self.assertFalse(cleaned[5, 5, 5])
        self.assertEqual(info["removed_components"], 1)
        self.assertEqual(info["seed_source"], "explicit")

    def test_reference_bbox_filter_removes_remote_bright_tips_component(self) -> None:
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[2:4, 2:4, 2:4] = True
        mask[7:9, 7:9, 7:9] = True
        volume = DicomVolume(np.zeros((10, 10, 10), dtype=np.float32), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
        reference_bounds = (np.array([1.5, 1.5, 1.5], dtype=np.float32), np.array([4.5, 4.5, 4.5], dtype=np.float32))

        filtered, info = _filter_components_by_reference_bbox(mask, volume, reference_bounds, max_distance_mm=1.0, min_voxels=1)

        self.assertEqual(int(filtered.sum()), 8)
        self.assertTrue(filtered[2, 2, 2])
        self.assertFalse(filtered[7, 7, 7])
        self.assertEqual(info["kept_components"], 1)
        self.assertEqual(info["removed_components"], 1)

    def test_reference_stls_are_not_used_as_portal_seed_priors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            patient = Path(tmp) / "case#"
            (patient / "dcm").mkdir(parents=True)
            case = PatientCase("case#", patient, patient / "dcm", patient / "vessel.stl", patient / "pretrain.stl", patient / "predict.stl", True)
            volume = DicomVolume(np.zeros((20, 20, 20), dtype=np.float32), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
            vertices = np.array([[1, 1, 1], [3, 1, 1], [1, 3, 1], [1, 1, 3]], dtype=np.float32)
            faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
            write_binary_stl(patient / "pre.stl", vertices + 10, faces)
            write_binary_stl(patient / "vessel.stl", vertices, faces)

            seed, source = _portal_seed_from_reference(case, volume)

        self.assertIsNone(source)
        self.assertIsNone(seed)

    def test_dicom_portal_seed_uses_volume_not_reference_stl(self) -> None:
        volume = DicomVolume(np.zeros((20, 30, 40), dtype=np.float32), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
        volume.volume_hu[10:13, 8:11, 18:21] = 160.0
        plan = {"hu_low": 80.0, "hu_high": 350.0, "crop": {"z": [0.2, 0.8], "y": [0.1, 0.7], "x": [0.2, 0.8]}}

        seed, source, info = _portal_seed_from_volume(volume, plan, spine_center_yx=(0.8, 0.5))

        self.assertEqual(source, "dicom_portal_hu_density")
        self.assertIsNotNone(seed)
        assert seed is not None
        self.assertGreater(seed[0], 9)
        self.assertEqual(info["source"], "dicom_portal_hu_density")

    def test_model_portal_seed_can_drive_region_growth_without_reference(self) -> None:
        volume = DicomVolume(np.zeros((20, 30, 40), dtype=np.float32), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
        plan = {"portal_seed": {"z": 0.78, "y": 0.72, "x": 0.72}}
        mask = np.zeros((20, 30, 40), dtype=bool)
        mask[2:5, 2:5, 2:5] = True
        mask[14:17, 20:23, 28:31] = True

        seed, source = _portal_seed_from_plan(plan, volume)
        cleaned, info = _cleanup_mask_by_region_growth(mask, seed, seed_source=source)

        self.assertEqual(source, "model_portal_seed")
        self.assertTrue(cleaned[15, 21, 29])
        self.assertFalse(cleaned[3, 3, 3])
        self.assertEqual(info["seed_source"], "model_portal_seed")

    def test_cleanup_prompt_and_seed_support_final_region_growth(self) -> None:
        client = _CaptureClient(response={"cleanup_seed": {"z": 0.22, "y": 0.33, "x": 0.44}, "notes": "keep PV"})
        volume = DicomVolume(np.zeros((20, 30, 40), dtype=np.float32), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))

        cleanup_plan = ask_for_cleanup_plan(client, "case#", True, {"mask_voxels": 10}, {"hu_low": 80}, [])
        seed, source = _portal_seed_from_cleanup_plan(cleanup_plan, volume)
        system, prompt, _ = client.calls[-1]

        self.assertIn("final cleanup", system.lower())
        self.assertIn("portal vein", prompt.lower())
        self.assertIn("TIPS", prompt)
        self.assertEqual(source, "model_cleanup_seed")
        np.testing.assert_allclose(seed, (0.22 * 19, 0.33 * 29, 0.44 * 39), atol=1e-6)

    def test_llm_removal_skips_overwide_region_boxes(self) -> None:
        mask = np.ones((10, 10, 10), dtype=bool)
        vol = np.zeros(mask.shape, dtype=np.float32)
        refine = {
            "assessment": "needs_cleanup",
            "remove_regions": [{"z_frac": [0, 1], "y_frac": [0, 1], "x_frac": [0, 1]}],
        }

        cleaned, info = _apply_llm_removal(mask, vol, refine, None, (1.0, 1.0, 1.0))

        self.assertEqual(int(cleaned.sum()), int(mask.sum()))
        self.assertEqual(info["regions_removed"], 0)
        self.assertEqual(info["skipped_regions"][0]["reason"], "region_too_large")

    def test_llm_disconnected_cleanup_rolls_back_when_too_destructive(self) -> None:
        mask = np.zeros((12, 12, 12), dtype=bool)
        mask[1:3, 1:3, 1:3] = True
        mask[6:11, 6:11, 6:11] = True
        vol = np.zeros(mask.shape, dtype=np.float32)
        refine = {"assessment": "needs_cleanup", "remove_disconnected_blobs": True}

        cleaned, info = _apply_llm_removal(mask, vol, refine, (1.5, 1.5, 1.5), (1.0, 1.0, 1.0))

        self.assertEqual(int(cleaned.sum()), int(mask.sum()))
        self.assertTrue(cleaned[8, 8, 8])
        self.assertEqual(info["region_grow_after"]["skipped_reason"], "would_remove_too_much")

    def test_llm_removal_preserves_high_hu_tips_voxels(self) -> None:
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[2:7, 2:7, 2:7] = True
        vol = np.zeros(mask.shape, dtype=np.float32)
        vol[3, 3, 3] = 650.0
        refine = {
            "assessment": "needs_cleanup",
            "remove_regions": [{"z_frac": [0.2, 0.4], "y_frac": [0.2, 0.4], "x_frac": [0.2, 0.4]}],
        }

        cleaned, info = _apply_llm_removal(mask, vol, refine, None, (1.0, 1.0, 1.0))

        self.assertTrue(cleaned[3, 3, 3])
        self.assertLess(int(cleaned.sum()), int(mask.sum()))
        self.assertEqual(info["regions_removed"], 1)

    def test_reference_envelope_mask_is_disabled_for_pretrain_priors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            patient = Path(tmp) / "case#"
            (patient / "dcm").mkdir(parents=True)
            case = PatientCase("case#", patient, patient / "dcm", patient / "vessel.stl", patient / "pretrain.stl", patient / "predict.stl", True)
            volume = DicomVolume(np.zeros((20, 20, 20), dtype=np.float32), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
            vertices = np.array([[4, 4, 4], [5, 4, 4], [4, 5, 4], [4, 4, 5]], dtype=np.float32)
            faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
            write_binary_stl(patient / "pre.stl", vertices, faces)

            envelope, info = _reference_envelope_mask(case, volume, radius_mm=2.0)

        self.assertIsNone(envelope)
        self.assertFalse(info["enabled"])
        self.assertTrue(info["debug_only"])
        self.assertEqual(info["skipped_reason"], "pre.stl_not_allowed_as_pretrain_prior")

    def test_region_growth_is_skipped_when_pre_stl_envelope_applies(self) -> None:
        self.assertTrue(_should_run_region_growth({"applied": True, "source": "pre.stl"}))
        self.assertTrue(_should_run_region_growth({"applied": False}))

    def test_short_z_standardization_falls_back_to_wide_portal_range(self) -> None:
        ok, reason = _validate_z_range(
            0, 38, 1.25, diaphragm_z=16, iliac_z=22,
            iliac_info={"status": "detected", "width_at_iliac_mm": 55.4},
        )

        self.assertFalse(ok)
        self.assertIn(reason, {"z_range_too_short", "diaphragm_iliac_too_close", "iliac_edge_width_unstable"})

    def test_threshold_search_returns_same_mask_used_for_best_candidate(self) -> None:
        vol = np.zeros((16, 16, 16), dtype=np.float32)
        vol[4:8, 4:8, 4:8] = 100.0
        vol[10:12, 10:12, 10:12] = 180.0
        plan = {"hu_low": 100.0, "hu_high": 220.0, "crop": {"z": [0, 1], "y": [0, 1], "x": [0, 1]}}

        best_plan, info, mask = search_best_threshold(
            vol, plan, None, (5.0, 5.0, 5.0), (1.0, 1.0, 1.0), False,
            _segment_once, region_grow_fn=lambda m, s, sp: (m, {"output_voxels": int(m.sum())}),
            hu_low_candidates=[80, 100, 180], return_best_mask=True,
        )

        self.assertEqual(best_plan["hu_low"], info["best_hu_low"])
        self.assertEqual(int(mask.sum()), info["best_voxels"])

    def test_coronal_tree_score_prefers_tree_over_blob_projection(self) -> None:
        tree = np.zeros((40, 12, 40), dtype=bool)
        tree[8:32, 5:7, 19:21] = True
        tree[18:22, 5:7, 8:32] = True
        blob = np.zeros_like(tree)
        blob[8:32, 4:9, 8:32] = True

        tree_score, tree_info = _score_coronal_tree(tree, (1.0, 1.0, 1.0))
        blob_score, blob_info = _score_coronal_tree(blob, (1.0, 1.0, 1.0))

        self.assertGreater(tree_score, blob_score)
        self.assertLess(tree_info["fill_ratio"], blob_info["fill_ratio"])
        self.assertGreaterEqual(tree_info["z_extent_mm"], 20)
        self.assertGreaterEqual(tree_info["x_extent_mm"], 20)

    def test_coronal_selection_prefers_viable_lower_threshold_over_tiny_high_threshold(self) -> None:
        candidates = [
            {
                "hu_low": 150.0,
                "selection_score": 77.0,
                "voxels": 100_000,
                "median_area_mm2": 320.0,
                "coronal_tree": 15.0,
                "coronal_fill_ratio": 0.56,
                "coronal_z_extent_mm": 160.0,
                "coronal_x_extent_mm": 63.0,
                "main_fraction": 0.70,
                "coronal_low_threshold_candidate": True,
            },
            {
                "hu_low": 170.0,
                "selection_score": 88.0,
                "voxels": 24_000,
                "median_area_mm2": 125.0,
                "coronal_tree": 15.0,
                "coronal_fill_ratio": 0.55,
                "coronal_z_extent_mm": 160.0,
                "coronal_x_extent_mm": 33.0,
                "main_fraction": 0.65,
                "coronal_low_threshold_candidate": False,
            },
        ]

        best = _select_best_candidate(candidates)

        self.assertEqual(best["hu_low"], 150.0)

    def test_pretrain_rebuilds_when_stl_missing_meta_missing_or_version_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "case"
            (root / "dcm").mkdir(parents=True)
            case = discover_patients(root)[0]
            meta = root / "vkan_work" / "pretrain_meta.json"
            meta.parent.mkdir()

            self.assertTrue(_should_rebuild_pretrain(case, meta, input_mtime=1.0)[0])
            write_binary_stl(case.pretrain_stl, np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64))
            meta.write_text('{"algorithm_version":"old"}', encoding="utf-8")
            self.assertTrue(_should_rebuild_pretrain(case, meta, input_mtime=1.0)[0])
            meta.write_text(
                f'{{"algorithm_version":"{PRETRAIN_ALGORITHM_VERSION}","input_mtime":1.0}}',
                encoding="utf-8",
            )
            self.assertFalse(_should_rebuild_pretrain(case, meta, input_mtime=1.0)[0])

    def test_skip_existing_pretrain_option_defaults_to_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "case"
            (root / "dcm").mkdir(parents=True)
            case = discover_patients(root)[0]
            write_binary_stl(case.pretrain_stl, np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64))

            self.assertFalse(_should_skip_existing_pretrain(case, skip_existing_pretrain=False))
            self.assertTrue(_should_skip_existing_pretrain(case, skip_existing_pretrain=True))


if __name__ == "__main__":
    unittest.main()
