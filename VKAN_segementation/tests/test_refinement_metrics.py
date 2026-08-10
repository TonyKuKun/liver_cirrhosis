from __future__ import annotations

import unittest

import torch

from refinement.model import dice_score


class RefinementMetricTests(unittest.TestCase):
    def test_dice_score_removes_disconnected_false_positive_component(self) -> None:
        logits = torch.full((1, 1, 8, 8, 8), -10.0)
        logits[:, :, 1:3, 1:3, 1:3] = 10.0
        logits[:, :, 6, 6, 6] = 10.0
        target = torch.zeros_like(logits)
        target[:, :, 1:3, 1:3, 1:3] = 1.0

        self.assertAlmostEqual(dice_score(logits, target), 17.0 / 18.0)
        self.assertEqual(dice_score(logits, target, largest_component=True), 1.0)

    def test_dice_score_filters_each_case_independently(self) -> None:
        logits = torch.full((2, 1, 8, 8, 8), -10.0)
        logits[0, :, 1:3, 1:3, 1:3] = 10.0
        logits[1, :, 5:7, 5:7, 5:7] = 10.0
        target = torch.zeros_like(logits)
        target[0, :, 1:3, 1:3, 1:3] = 1.0
        target[1, :, 5:7, 5:7, 5:7] = 1.0

        self.assertEqual(dice_score(logits, target, largest_component=True), 1.0)


if __name__ == "__main__":
    unittest.main()
