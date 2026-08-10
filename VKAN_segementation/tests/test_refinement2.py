from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from refinement2.dataset import CTHUVesselDataset, discover_cases
from refinement2.inference import _remove_small_components
from refinement2.model import CTPretrainNNVNet, create_loss, dice_per_case
from refinement2.train import grouped_split, patient_group_key


def _write_case(root: Path, name: str, review: bool = False) -> None:
    case = root / name
    case.mkdir(parents=True)
    affine = np.eye(4, dtype=np.float32)
    orig = np.linspace(-200, 700, num=12 * 12 * 12, dtype=np.float32).reshape(12, 12, 12)
    pretrain = np.zeros_like(orig, dtype=np.uint8)
    label = np.zeros_like(orig, dtype=np.uint8)
    pretrain[3:8, 3:8, 3:8] = 1
    label[4:9, 4:9, 4:9] = 1
    nib.save(nib.Nifti1Image(orig, affine), case / "orig.nii.gz")
    nib.save(nib.Nifti1Image(pretrain, affine), case / "pretrain.nii.gz")
    nib.save(nib.Nifti1Image(label, affine), case / "mask.nii.gz")
    if review:
        work = case / "vkan_work"
        work.mkdir()
        (work / "pretrain_meta.json").write_text(json.dumps({"pretrain_quality": "review"}), encoding="utf-8")


class Refinement2DatasetTests(unittest.TestCase):
    def test_dataset_filters_only_dollar_marked_cases_and_builds_two_channels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_case(root, "20240101Good")
            _write_case(root, "20240102Review", review=True)
            _write_case(root, "20240103PvpMetadata@")
            _write_case(root, "20240104SegmentationExcluded$")

            cases = discover_cases(root)
            dataset = CTHUVesselDataset(
                root, grid_size=16, roi_margin=1, hu_min=-100, hu_max=600, cache_dir=root / "cache"
            )
            sample = dataset[0]
            cached_sample = dataset[0]

        self.assertEqual(
            [case.name for case in cases],
            ["20240101Good", "20240102Review", "20240103PvpMetadata@"],
        )
        self.assertEqual(len(dataset), 3)
        self.assertEqual(tuple(sample["input"].shape), (2, 16, 16, 16))
        self.assertEqual(tuple(sample["label"].shape), (1, 16, 16, 16))
        self.assertGreaterEqual(float(sample["input"][0].min()), 0.0)
        self.assertLessEqual(float(sample["input"][0].max()), 1.0)
        self.assertEqual(set(torch.unique(sample["input"][1]).tolist()), {0.0, 1.0})
        self.assertTrue(torch.equal(sample["input"], cached_sample["input"]))

    def test_grouped_split_keeps_patient_variants_together(self) -> None:
        self.assertEqual(patient_group_key("20240101Wang#"), patient_group_key("20240201Wang"))
        cases = [
            type("Case", (), {"name": "20240101Wang#"})(),
            type("Case", (), {"name": "20240201Wang"})(),
            type("Case", (), {"name": "20240101Li"})(),
            type("Case", (), {"name": "20240101Zhang"})(),
        ]
        train_indices, val_indices = grouped_split(cases, val_ratio=0.25, seed=7)
        self.assertTrue(train_indices)
        self.assertTrue(val_indices)
        self.assertFalse(({0, 1} & set(train_indices)) and ({0, 1} & set(val_indices)))


class Refinement2ModelTests(unittest.TestCase):
    def test_model_losses_and_component_filter(self) -> None:
        model = CTPretrainNNVNet(base_channels=4)
        inputs = torch.rand(1, 2, 16, 16, 16)
        target = torch.zeros(1, 1, 16, 16, 16)
        target[:, :, 4:8, 4:8, 4:8] = 1.0
        logits = model(inputs)

        self.assertEqual(tuple(logits.shape), (1, 1, 16, 16, 16))
        for name in ("dice_bce", "tversky", "focal_tversky", "dice_focal_tversky"):
            self.assertTrue(torch.isfinite(create_loss(name)(logits, target)))
        self.assertEqual(tuple(dice_per_case(logits, target).shape), (1,))

        mask = np.zeros((8, 8, 8), dtype=bool)
        mask[1:4, 1:4, 1:4] = True
        mask[7, 7, 7] = True
        filtered = _remove_small_components(mask, min_voxels=2)
        self.assertEqual(int(filtered.sum()), 27)


if __name__ == "__main__":
    unittest.main()
