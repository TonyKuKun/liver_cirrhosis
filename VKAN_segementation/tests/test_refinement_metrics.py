from __future__ import annotations

import unittest

import torch

from refinement.model import DiceBCECLDiceLoss, DiceBCELoss, SoftCLDiceLoss, dice_score


class RefinementMetricTests(unittest.TestCase):
    def test_soft_cldice_penalizes_a_broken_centerline(self) -> None:
        target = torch.zeros((1, 1, 9, 9, 9))
        target[:, :, 2:7, 4, 4] = 1.0
        perfect = torch.where(target > 0, torch.tensor(10.0), torch.tensor(-10.0))
        broken = perfect.clone()
        broken[:, :, 4, 4, 4] = -10.0
        criterion = SoftCLDiceLoss(iterations=3)

        self.assertLess(criterion(perfect, target).item(), criterion(broken, target).item())

    def test_combined_loss_matches_base_loss_when_cldice_is_disabled(self) -> None:
        logits = torch.randn((1, 1, 8, 8, 8), requires_grad=True)
        target = (torch.rand_like(logits) > 0.8).float()
        base = DiceBCELoss()(logits, target)
        combined = DiceBCECLDiceLoss(cldice_weight=0.0)(logits, target)

        self.assertTrue(torch.allclose(base, combined))

    def test_combined_loss_backpropagates_through_soft_skeleton(self) -> None:
        logits = torch.randn((1, 1, 8, 8, 8), requires_grad=True)
        target = torch.zeros_like(logits)
        target[:, :, 2:6, 4, 4] = 1.0

        loss = DiceBCECLDiceLoss(cldice_weight=0.2, skeleton_iterations=2)(logits, target)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

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
