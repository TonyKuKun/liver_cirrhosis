from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import struct

import numpy as np

from pretrain.preprocess import (
    DicomVolume,
    PRETRAIN_ALGORITHM_VERSION,
    _crop_slices,
    _default_plan,
    _sanitize_plan,
    _should_rebuild_pretrain,
    _write_status,
    load_nifti_volume,
    save_nifti_volume,
)
from utils.common import discover_patients, write_binary_stl


class PretrainRoiTests(unittest.TestCase):
    def test_single_patient_root_is_discovered(self) -> None:
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

        self.assertGreaterEqual(plan["hu_low"], 90)
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

    def test_nifti_cache_round_trips_hu_spacing_and_origin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "patient.nii.gz"
            vol = DicomVolume(
                volume_hu=np.array([[[-1024, 100], [220, 3071]], [[0, 420], [680, 900]]], dtype=np.float32),
                spacing_zyx=(1.5, 0.7, 0.8),
                origin_xyz=(-10.0, 20.0, 30.0),
            )

            save_nifti_volume(vol, path)
            loaded = load_nifti_volume(path)

        self.assertEqual(loaded.volume_hu.shape, vol.volume_hu.shape)
        np.testing.assert_array_equal(loaded.volume_hu, vol.volume_hu.astype(np.int16))
        self.assertEqual(loaded.spacing_zyx, vol.spacing_zyx)
        self.assertEqual(loaded.origin_xyz, vol.origin_xyz)

    def test_pretrain_rebuilds_when_meta_missing_or_version_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stl = root / "pretrain.stl"
            meta = root / "vkan_work" / "pretrain_meta.json"
            nii = root / "patient.nii.gz"
            meta.parent.mkdir()
            stl.write_bytes(b"old")
            nii.write_bytes(b"nii")

            self.assertTrue(_should_rebuild_pretrain(stl, meta, nii, input_mtime=1.0)[0])
            meta.write_text('{"algorithm_version":"old"}', encoding="utf-8")
            self.assertTrue(_should_rebuild_pretrain(stl, meta, nii, input_mtime=1.0)[0])
            meta.write_text(
                '{"algorithm_version":"%s","input_mtime":1.0}' % PRETRAIN_ALGORITHM_VERSION,
                encoding="utf-8",
            )
            self.assertFalse(_should_rebuild_pretrain(stl, meta, nii, input_mtime=1.0)[0])

    def test_write_status_distinguishes_new_and_existing_outputs(self) -> None:
        self.assertEqual(_write_status(old_stl_exists=False), "wrote")
        self.assertEqual(_write_status(old_stl_exists=True), "regenerated")


if __name__ == "__main__":
    unittest.main()
