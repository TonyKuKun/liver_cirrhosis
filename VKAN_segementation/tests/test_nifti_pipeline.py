from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pretrain.preprocess import DicomVolume, mask_label_nifti_path, pretrain_nifti_path
from utils.common import PatientCase, load_nifti_volume, save_nifti_volume


def _write_tiny_dicom(path: Path, pixels: np.ndarray, z: float = 0.0) -> None:
    pixels = np.asarray(pixels, dtype=np.uint16)
    rows, cols = pixels.shape

    def elem(group: int, tag: int, vr: str, value: bytes) -> bytes:
        if len(value) % 2:
            value += b" "
        if vr in {"OB", "OW", "OF", "SQ", "UN", "UT"}:
            return struct.pack("<HH2s2sI", group, tag, vr.encode("ascii"), b"\0\0", len(value)) + value
        return struct.pack("<HH2sH", group, tag, vr.encode("ascii"), len(value)) + value

    body = b"".join(
        [
            elem(0x0028, 0x0010, "US", struct.pack("<H", rows)),
            elem(0x0028, 0x0011, "US", struct.pack("<H", cols)),
            elem(0x0028, 0x0103, "US", struct.pack("<H", 0)),
            elem(0x0028, 0x0030, "DS", b"1.0\\1.0"),
            elem(0x0028, 0x1052, "DS", b"0"),
            elem(0x0028, 0x1053, "DS", b"1"),
            elem(0x0018, 0x0050, "DS", b"1.0"),
            elem(0x0020, 0x0032, "DS", f"0\\0\\{z}".encode("ascii")),
            elem(0x0020, 0x0013, "IS", str(int(z) + 1).encode("ascii")),
            elem(0x7FE0, 0x0010, "OW", pixels.astype("<u2").tobytes()),
        ]
    )
    path.write_bytes(b"\0" * 128 + b"DICM" + body)


class NiftiPipelineTests(unittest.TestCase):
    def test_mask_folder_raw_pixel_diff_converts_to_binary_nifti(self) -> None:
        from pretrain.preprocess import convert_mask_folder_to_nifti

        with tempfile.TemporaryDirectory() as tmp:
            patient = Path(tmp) / "case"
            dcm_dir = patient / "dcm"
            mask_dir = patient / "mask"
            dcm_dir.mkdir(parents=True)
            mask_dir.mkdir()
            base = np.array([[10, 10, 10], [10, 10, 10]], dtype=np.uint16)
            marked = base.copy()
            marked[0, 1] = 2000
            _write_tiny_dicom(dcm_dir / "slice000.dcm", base, z=0)
            _write_tiny_dicom(mask_dir / "slice000.dcm", marked, z=0)

            out = convert_mask_folder_to_nifti(dcm_dir, mask_dir, patient / "mask.nii.gz", min_voxels=1)
            loaded = load_nifti_volume(out)

        self.assertEqual(loaded.volume_hu.shape, (1, 2, 3))
        self.assertEqual(int(loaded.volume_hu.sum()), 1)
        self.assertEqual(float(loaded.volume_hu[0, 0, 1]), 1.0)

    def test_pretrain_nifti_round_trips_binary_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pretrain.nii.gz"
            mask = np.zeros((3, 4, 5), dtype=np.uint8)
            mask[1, 2, 3] = 1

            save_nifti_volume(DicomVolume(mask, (1.0, 0.5, 0.5), (0.0, 0.0, 0.0)), path)
            loaded = load_nifti_volume(path)

        self.assertEqual(int(loaded.volume_hu.sum()), 1)
        self.assertEqual(float(loaded.volume_hu[1, 2, 3]), 1.0)

    def test_nifti_dataset_reads_masks_without_stl_voxelization(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed in this environment")
        from refinement.dataset import VesselNiftiDataset

        with tempfile.TemporaryDirectory() as tmp:
            patient = Path(tmp) / "case"
            (patient / "dcm").mkdir(parents=True)
            pretrain = np.zeros((4, 4, 4), dtype=np.uint8)
            label = np.zeros((4, 4, 4), dtype=np.uint8)
            pretrain[1:3, 1:3, 1:3] = 1
            label[2, 2, 2] = 1
            save_nifti_volume(DicomVolume(pretrain, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)), patient / "pretrain.nii.gz")
            save_nifti_volume(DicomVolume(label, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)), patient / "mask.nii.gz")

            ds = VesselNiftiDataset(tmp, grid_size=4)
            item = ds[0]

        self.assertEqual(item["input"].shape, (1, 4, 4, 4))
        self.assertEqual(item["label"].shape, (1, 4, 4, 4))
        self.assertEqual(float(item["input"].sum()), 8.0)
        self.assertEqual(float(item["label"].sum()), 1.0)

    def test_predict_case_writes_predict_mask_nifti(self) -> None:
        from refinement.predict import _save_prediction_outputs

        with tempfile.TemporaryDirectory() as tmp:
            patient = Path(tmp) / "case"
            patient.mkdir()
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
            loaded = load_nifti_volume(patient / "predict_mask.nii.gz")

        self.assertEqual(out, patient / "predict.stl")
        self.assertEqual(int(loaded.volume_hu.sum()), 8)

    def test_case_nifti_paths_are_fixed_names(self) -> None:
        case = PatientCase("abc", Path("p"), Path("p/dcm"), Path("p/vessel.stl"), Path("p/pretrain.stl"), Path("p/predict.stl"), False)

        self.assertEqual(pretrain_nifti_path(case), Path("p/pretrain.nii.gz"))
        self.assertEqual(mask_label_nifti_path(case), Path("p/mask.nii.gz"))

    def test_large_pretrain_stl_marks_review_quality(self) -> None:
        from pretrain.preprocess import _pretrain_quality

        mask = np.zeros((8, 16, 16), dtype=np.uint8)
        mask[:, 2:14, 2:14] = 1

        quality, issues = _pretrain_quality(mask, stl_bytes=21_000 * 1024)

        self.assertEqual(quality, "review")
        self.assertIn("stl_over_20000kb", issues)


if __name__ == "__main__":
    unittest.main()
