from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pretrain.preprocess import _pretrain_quality
from utils.common import PatientCase, write_binary_stl


def _write_cube_stl(path: Path, offset: float = 0.0) -> Path:
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.float32,
    )
    vertices += offset
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.int64,
    )
    return write_binary_stl(path, vertices, faces)


class STLPipelineTests(unittest.TestCase):
    def test_stl_dataset_reads_pretrain_and_vessel_stl(self) -> None:
        try:
            import torch  # noqa: F401
            import trimesh  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"optional dependency missing: {exc}")

        from refinement.dataset import VesselSTLDataset

        with tempfile.TemporaryDirectory() as tmp:
            patient = Path(tmp) / "case"
            (patient / "dcm").mkdir(parents=True)
            _write_cube_stl(patient / "pretrain.stl")
            _write_cube_stl(patient / "vessel.stl", offset=0.1)

            ds = VesselSTLDataset(tmp, grid_size=8)
            item = ds[0]

        self.assertEqual(item["input"].shape, (1, 8, 8, 8))
        self.assertEqual(item["label"].shape, (1, 8, 8, 8))
        self.assertGreater(float(item["input"].sum()), 0.0)
        self.assertGreater(float(item["label"].sum()), 0.0)

    def test_nii_dataset_crops_and_resizes_masks(self) -> None:
        try:
            import nibabel as nib
            import torch  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"optional dependency missing: {exc}")

        from refinement.dataset import VesselNiiDataset

        with tempfile.TemporaryDirectory() as tmp:
            patient = Path(tmp) / "case"
            patient.mkdir(parents=True)
            affine = np.eye(4, dtype=np.float32)
            pre = np.zeros((24, 24, 16), dtype=np.uint8)
            label = np.zeros_like(pre)
            pre[8:12, 9:13, 4:8] = 1
            label[9:13, 10:14, 5:9] = 1
            nib.save(nib.Nifti1Image(pre, affine), patient / "pretrain.nii.gz")
            nib.save(nib.Nifti1Image(label, affine), patient / "mask.nii.gz")

            ds = VesselNiiDataset(tmp, grid_size=12, roi_margin=2)
            item = ds[0]

        self.assertEqual(item["input"].shape, (1, 12, 12, 12))
        self.assertEqual(item["label"].shape, (1, 12, 12, 12))
        self.assertGreater(float(item["input"].sum()), 0.0)
        self.assertGreater(float(item["label"].sum()), 0.0)
        self.assertLess(int((item["crop_slices"][:, 1] - item["crop_slices"][:, 0]).prod()), 24 * 24 * 16)

    def test_nii_dataset_always_skips_at_marked_cases(self) -> None:
        try:
            import nibabel as nib
            import torch  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"optional dependency missing: {exc}")

        from refinement.dataset import VesselNiiDataset

        with tempfile.TemporaryDirectory() as tmp:
            affine = np.eye(4, dtype=np.float32)
            mask = np.zeros((8, 8, 4), dtype=np.uint8)
            mask[2:5, 2:5, 1:3] = 1
            for name in ("keep_case", "skip@case"):
                patient = Path(tmp) / name
                patient.mkdir(parents=True)
                nib.save(nib.Nifti1Image(mask, affine), patient / "pretrain.nii.gz")
                nib.save(nib.Nifti1Image(mask, affine), patient / "mask.nii.gz")

            ds = VesselNiiDataset(tmp, grid_size=8, include_invalid=True)

        self.assertEqual([case.name for case in ds.cases], ["keep_case"])

    def test_overlay_mask_is_renamed_and_derived(self) -> None:
        try:
            import nibabel as nib
        except ImportError as exc:
            self.skipTest(f"optional dependency missing: {exc}")

        from pretrain.derive_mask_from_overlay import convert_patient

        with tempfile.TemporaryDirectory() as tmp:
            patient = Path(tmp) / "case"
            patient.mkdir(parents=True)
            affine = np.eye(4, dtype=np.float32)
            orig = np.zeros((8, 8, 4), dtype=np.float32)
            overlay = orig.copy()
            overlay[2:5, 3:6, 1:3] = 10.0
            nib.save(nib.Nifti1Image(orig, affine), patient / "orig.nii.gz")
            nib.save(nib.Nifti1Image(overlay, affine), patient / "mask.nii.gz")

            result = convert_patient(patient, threshold=0.5)
            mask = np.asarray(nib.load(str(patient / "mask.nii.gz")).dataobj)
            origm_exists = (patient / "origm.nii.gz").exists()

            self.assertEqual(result.status, "wrote")
            self.assertTrue(origm_exists)
            self.assertEqual(int(mask.sum()), 18)
            self.assertEqual(set(np.unique(mask).tolist()), {0, 1})

    def test_prediction_outputs_only_predict_stl(self) -> None:
        from refinement.predict import _save_prediction_outputs

        with tempfile.TemporaryDirectory() as tmp:
            patient = Path(tmp) / "case"
            (patient / "dcm").mkdir(parents=True)
            case = PatientCase(
                name="case",
                path=patient,
                dcm_dir=patient / "dcm",
                label_stl=patient / "vessel.stl",
                pretrain_stl=patient / "pretrain.stl",
                predict_stl=patient / "predict.stl",
                is_post_tips=False,
            )
            prob = np.zeros((4, 4, 4), dtype=np.float32)
            prob[1:3, 1:3, 1:3] = 0.9

            out = _save_prediction_outputs(case, prob, np.array([[0, 0, 0], [4, 4, 4]], dtype=np.float32), threshold=0.5)
            exists = out.exists()
            wrote_mask = (patient / "predict_mask.nii.gz").exists()

        self.assertEqual(out, patient / "predict.stl")
        self.assertTrue(exists)
        self.assertFalse(wrote_mask)

    def test_large_pretrain_stl_marks_review_quality(self) -> None:
        mask = np.zeros((8, 16, 16), dtype=np.uint8)
        mask[:, 2:14, 2:14] = 1

        quality, issues, _stats = _pretrain_quality(mask, stl_bytes=21_000 * 1024, max_voxels=10_000_000)

        self.assertEqual(quality, "review")
        self.assertIn("stl_over_20mb", issues)
