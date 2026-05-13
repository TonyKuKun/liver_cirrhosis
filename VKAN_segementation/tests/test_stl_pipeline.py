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

        quality, issues = _pretrain_quality(mask, stl_bytes=21_000 * 1024)

        self.assertEqual(quality, "review")
        self.assertIn("stl_over_20000kb", issues)
