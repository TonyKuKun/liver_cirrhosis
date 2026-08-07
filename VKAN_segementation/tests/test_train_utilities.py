from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from refinement.train import _epochs_without_improvement, _plot_training_history


class TrainUtilityTests(unittest.TestCase):
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
