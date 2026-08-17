from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import torch

from refinement.augmentation import (
    FILL_VARIANTS_PER_CASE,
    FILL_LEARNING_MODES,
    FixedFillAugmentedDataset,
    apply_fill_learning_mode,
    load_smv_centerline_geometry,
)
from refinement.train import (
    _add_input_augmentation_arguments,
    _cldice_weight_for_epoch,
    _epochs_without_improvement,
    _plot_training_history,
)


class TrainUtilityTests(unittest.TestCase):
    def test_input_error_augmentation_is_disabled_by_default(self) -> None:
        parser = argparse.ArgumentParser()
        _add_input_augmentation_arguments(parser)

        self.assertFalse(parser.parse_args([]).input_error_augmentation)
        self.assertTrue(parser.parse_args(["--input_error_augmentation"]).input_error_augmentation)
        self.assertFalse(parser.parse_args(["--no_input_error_augmentation"]).input_error_augmentation)

    @staticmethod
    def _make_smv_case(data_root: Path) -> dict:
        coordinates = torch.arange(24)
        xx, yy, zz = torch.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
        label_mask = ((xx - 12) ** 2 + (yy - 12) ** 2 <= 9) & (zz >= 3) & (zz <= 20)
        input_mask = ((xx - 12) ** 2 + (yy - 12) ** 2 <= 16) & (zz >= 3) & (zz <= 20)
        item = {
            "name": "case",
            "input": input_mask[None].float(),
            "label": label_mask[None].float(),
            "affine": torch.eye(4),
            "crop_slices": torch.tensor([[0, 24], [0, 24], [0, 24]]),
        }

        z_values = torch.linspace(3.0, 20.0, 200).tolist()
        feature_path = data_root / "case" / "features" / "unified_features.json"
        feature_path.parent.mkdir(parents=True)
        feature_path.write_text(
            json.dumps(
                {
                    "sv_smv_angle": {"confluence_point_physical": [-12.0, -12.0, 3.0]},
                    "pointwise": {
                        "smv": {
                            "centerline_x": [-12.0] * 200,
                            "centerline_y": [-12.0] * 200,
                            "centerline_z": z_values,
                            "arc_length_mm": [value - 3.0 for value in z_values],
                            "inscribed_radius": [3.0] * 200,
                            "hydraulic_diameter": [6.0] * 200,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return item

    def test_pretrain_error_augmentation_has_five_fill_modes(self) -> None:
        self.assertEqual(len(FILL_LEARNING_MODES), 5)
        self.assertEqual(FILL_VARIANTS_PER_CASE, 6)

    def test_pretrain_error_modes_only_remove_input_voxels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir)
            item = self._make_smv_case(data_root)
            geometry = load_smv_centerline_geometry(
                item,
                data_root / "case" / "features" / "unified_features.json",
            )
            self.assertIsNotNone(geometry)
            for mode in FILL_LEARNING_MODES:
                candidate = apply_fill_learning_mode(
                    item["input"],
                    item["label"],
                    mode=mode,
                    geometry=geometry,
                )
                removed = item["input"] > candidate
                self.assertTrue(removed.any(), mode)
                self.assertTrue((candidate <= item["input"]).all(), mode)
                self.assertTrue((item["label"][removed] > 0.5).all(), mode)

    def test_fixed_fill_dataset_expands_each_case_to_six_deterministic_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir)
            item = self._make_smv_case(data_root)
            dataset = FixedFillAugmentedDataset([item], data_root=data_root)

            self.assertEqual(len(dataset), 6)
            self.assertTrue(torch.equal(dataset[0]["input"], item["input"]))
            variants = []
            for index in range(1, 6):
                first = dataset[index]
                repeated = dataset[index]
                variants.append(first["input"])
                self.assertTrue(torch.equal(first["input"], repeated["input"]))
                self.assertFalse(torch.equal(first["input"], item["input"]))
                self.assertTrue((first["input"] <= item["input"]).all())
                self.assertTrue(torch.equal(first["label"], item["label"]))
            for left in range(len(variants)):
                for right in range(left + 1, len(variants)):
                    self.assertFalse(torch.equal(variants[left], variants[right]))

    def test_cldice_weight_uses_warmup_and_linear_ramp(self) -> None:
        self.assertEqual(_cldice_weight_for_epoch(10, 0.2, 10, 30), 0.0)
        self.assertAlmostEqual(_cldice_weight_for_epoch(25, 0.2, 10, 30), 0.1)
        self.assertAlmostEqual(_cldice_weight_for_epoch(40, 0.2, 10, 30), 0.2)
        self.assertAlmostEqual(_cldice_weight_for_epoch(80, 0.2, 10, 30), 0.2)

    def test_cldice_weight_can_skip_the_ramp(self) -> None:
        self.assertEqual(_cldice_weight_for_epoch(1, 0.2, 0, 0), 0.2)

    def test_epochs_without_improvement_uses_strict_dice_improvement(self) -> None:
        history = [
            {"epoch": 1, "val": {"dice": 0.60}},
            {"epoch": 2, "val": {"dice": 0.62}},
            {"epoch": 3, "val": {"dice": 0.62}},
            {"epoch": 4, "val": {"dice": 0.61}},
        ]

        self.assertEqual(_epochs_without_improvement(history), 2)

    def test_plot_training_history_writes_png(self) -> None:
        history = [
            {"epoch": 1, "train": {"loss": 0.8, "dice": 0.4}, "val": {"loss": 0.9, "dice": 0.3}},
            {"epoch": 2, "train": {"loss": 0.6, "dice": 0.6}, "val": {"loss": 0.7, "dice": 0.5}},
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            plot_path = Path(tmp_dir) / "curve.png"

            result = _plot_training_history(history, plot_path)

            self.assertEqual(result, plot_path)
            self.assertTrue(plot_path.exists())
            self.assertGreater(plot_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
