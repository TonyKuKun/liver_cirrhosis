import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import nibabel as nib

from pretrain.preprocess import (
    _get_exclusion_mask_fast,
    _get_tips_exclusion_mask_fast,
    _limit_to_portal_reference_neighborhood,
    _save_pretrain_nifti,
    _standardize_z_from_bone,
)
from utils.common import PatientCase


def _case(patient: Path) -> PatientCase:
    return PatientCase(
        patient.name,
        patient,
        patient / "dcm",
        patient / "vessel.stl",
        patient / "pretrain.stl",
        patient / "predict.stl",
        False,
    )


def _touch_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


class PreprocessVertebraZTests(unittest.TestCase):
    def test_pretrain_nifti_preserves_orig_orientation_forms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            orig_path = root / "orig.nii.gz"
            out_path = root / "pretrain.nii.gz"
            affine = np.array(
                [
                    [-0.7, 0.0, 0.0, 156.0],
                    [0.0, 0.8, 0.0, -120.0],
                    [0.0, 0.0, 1.2, -280.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            ref_img = nib.Nifti1Image(np.zeros((5, 6, 7), dtype=np.float32), affine)
            ref_img.set_qform(affine, 1)
            ref_img.set_sform(affine, 1)
            nib.save(ref_img, str(orig_path))

            mask_zyx = np.zeros((7, 6, 5), dtype=bool)
            mask_zyx[2, 3, 4] = True
            _save_pretrain_nifti(mask_zyx, orig_path, out_path)

            out_img = nib.load(str(out_path))
            self.assertEqual(out_img.shape, ref_img.shape)
            self.assertTrue(np.allclose(out_img.affine, affine))
            self.assertTrue(np.allclose(out_img.get_qform(), affine))
            self.assertTrue(np.allclose(out_img.get_sform(), affine))
            self.assertEqual(int(out_img.header["qform_code"]), 1)
            self.assertEqual(int(out_img.header["sform_code"]), 1)
            self.assertEqual(int(np.asarray(out_img.dataobj)[4, 3, 2]), 1)

    def test_z_range_uses_totalseg_t8_to_l3_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            patient = Path(tmp) / "case"
            (patient / "dcm").mkdir(parents=True)
            ts_output = patient / "segmentation" / "totalseg_output"
            t8 = np.zeros((80, 8, 8), dtype=bool)
            l3 = np.zeros_like(t8)
            t8[10:16, 2:6, 2:6] = True
            l3[50:56, 2:6, 2:6] = True
            _touch_mask(ts_output / "vertebrae_T8.nii.gz")
            _touch_mask(ts_output / "vertebrae_L3.nii.gz")
            masks = {"vertebrae_T8.nii.gz": t8, "vertebrae_L3.nii.gz": l3}

            with patch("pretrain.preprocess._load_mask_nii", side_effect=lambda path, *_args: masks[path.name]):
                z_start, z_end, info = _standardize_z_from_bone(
                    _case(patient), t8.shape, (1.0, 1.0, 1.0), margin_mm=20.0, cache={}
                )

        self.assertEqual((z_start, z_end), (10, 55))
        self.assertEqual(info["source"], "totalseg_vertebrae_nii")
        self.assertEqual(info["upper_source"], "vertebrae_T8")
        self.assertEqual(info["lower_source"], "vertebrae_L3_inferior_edge")

    def test_z_range_without_t8_t9_uses_highest_loaded_vertebra_point(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            patient = Path(tmp) / "case"
            (patient / "dcm").mkdir(parents=True)
            ts_output = patient / "segmentation" / "totalseg_output"
            t12 = np.zeros((80, 8, 8), dtype=bool)
            l2 = np.zeros_like(t12)
            l3 = np.zeros_like(t12)
            t12[20:26, 2:6, 2:6] = True
            l2[40:46, 2:6, 2:6] = True
            l3[50:56, 2:6, 2:6] = True
            _touch_mask(ts_output / "vertebrae_T12.nii.gz")
            _touch_mask(ts_output / "vertebrae_L2.nii.gz")
            _touch_mask(ts_output / "vertebrae_L3.nii.gz")
            masks = {"vertebrae_T12.nii.gz": t12, "vertebrae_L2.nii.gz": l2, "vertebrae_L3.nii.gz": l3}

            with patch("pretrain.preprocess._load_mask_nii", side_effect=lambda path, *_args: masks[path.name]):
                z_start, z_end, info = _standardize_z_from_bone(
                    _case(patient), t12.shape, (1.0, 1.0, 1.0), cache={}
                )

        self.assertEqual((z_start, z_end), (20, 55))
        self.assertEqual(info["upper_source"], "highest_loaded_vertebra_point")

    def test_liver_exclusion_preserves_portal_vein_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            patient = Path(tmp) / "case"
            (patient / "dcm").mkdir(parents=True)
            ts_output = patient / "segmentation" / "totalseg_output"
            liver = np.zeros((20, 10, 10), dtype=bool)
            portal = np.zeros_like(liver)
            liver[5:15, 2:8, 2:8] = True
            portal[8:10, 4:6, 4:6] = True
            _touch_mask(ts_output / "liver.nii.gz")
            _touch_mask(ts_output / "portal_vein.nii.gz")
            masks = {"liver.nii.gz": liver, "portal_vein.nii.gz": portal}

            with patch("pretrain.preprocess._load_mask_nii", side_effect=lambda path, *_args: masks[path.name]):
                exclusion, info = _get_exclusion_mask_fast(
                    _case(patient), liver.shape, dilate_bone=0, dilate_organ=0, cache={}
                )

        self.assertTrue(exclusion[6, 3, 3])
        self.assertFalse(exclusion[8, 4, 4])
        self.assertEqual(info["portal_protection"]["status"], "ok")

    def test_exclusion_loads_requested_segmentation_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            patient = Path(tmp) / "case"
            (patient / "dcm").mkdir(parents=True)
            seg_dir = patient / "segmentation"
            names = (
                "bone_all",
                "spleen",
                "liver",
                "kidney_left",
                "kidney_right",
                "inferior_vena_cava",
                "aorta",
                "portal_vein",
            )
            masks = {}
            for idx, name in enumerate(names):
                mask = np.zeros((20, 10, 10), dtype=bool)
                mask[idx : idx + 1, 1:3, 1:3] = True
                masks[f"{name}.nii.gz"] = mask
                _touch_mask(seg_dir / f"{name}.nii.gz")

            individual_bone = np.zeros((20, 10, 10), dtype=bool)
            individual_bone[15:16, 1:3, 1:3] = True
            masks["vertebrae_L3.nii.gz"] = individual_bone
            _touch_mask(seg_dir / "vertebrae_L3.nii.gz")
            loaded_names = []

            def fake_load(path: Path, *_args):
                loaded_names.append(path.name)
                return masks[path.name]

            with patch("pretrain.preprocess._load_mask_nii", side_effect=fake_load):
                _exclusion, info = _get_exclusion_mask_fast(
                    _case(patient), (20, 10, 10), dilate_bone=0, dilate_organ=0, cache={}
                )

        loaded_structures = {item["name"] for item in info["loaded"]}
        self.assertTrue({"bone_all", "spleen", "liver", "kidney_left", "kidney_right", "inferior_vena_cava", "aorta"} <= loaded_structures)
        self.assertIn("bone_all.nii.gz", loaded_names)
        self.assertNotIn("vertebrae_L3.nii.gz", loaded_names)

    def test_portal_cleanup_clips_large_component_far_from_reference(self) -> None:
        mask = np.zeros((30, 30, 30), dtype=bool)
        portal = np.zeros_like(mask)
        portal[14:16, 14:16, 14:16] = True
        mask[13:17, 13:17, 13:17] = True
        mask[13:17, 17:28, 14:16] = True
        mask[13:17, 26:30, 13:17] = True

        cleaned, info = _limit_to_portal_reference_neighborhood(
            mask, portal, (1.0, 1.0, 1.0), radius_mm=4.0, seed_dilate=1
        )

        self.assertLess(int(cleaned.sum()), int(mask.sum()))
        self.assertTrue(cleaned[14, 14, 14])
        self.assertFalse(cleaned[14, 28, 14])
        self.assertEqual(info["status"], "ok")

    def test_tips_exclusion_does_not_remove_liver_or_ivc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            patient = Path(tmp) / "case#"
            (patient / "dcm").mkdir(parents=True)
            seg_dir = patient / "segmentation"
            liver = np.zeros((20, 10, 10), dtype=bool)
            ivc = np.zeros_like(liver)
            bone = np.zeros_like(liver)
            liver[5:15, 2:8, 2:8] = True
            ivc[7:13, 4:6, 4:6] = True
            bone[1:4, 1:4, 1:4] = True
            for name in ("liver", "inferior_vena_cava", "bone_all"):
                _touch_mask(seg_dir / f"{name}.nii.gz")
            masks = {
                "liver.nii.gz": liver,
                "inferior_vena_cava.nii.gz": ivc,
                "bone_all.nii.gz": bone,
            }

            with patch("pretrain.preprocess._load_mask_nii", side_effect=lambda path, *_args: masks[path.name]):
                exclusion, info = _get_tips_exclusion_mask_fast(_case(patient), liver.shape, cache={})

        self.assertTrue(exclusion[2, 2, 2])
        self.assertFalse(exclusion[8, 4, 4])
        loaded = {item["name"] for item in info["loaded"]}
        self.assertIn("bone_all", loaded)
        self.assertNotIn("liver", loaded)
        self.assertNotIn("inferior_vena_cava", loaded)


if __name__ == "__main__":
    unittest.main()
