import unittest

import torch

from dataset import N_AUX, N_SEGMENTS, SEG_INDEX
from model import PortalCircuitLayer


class PortalCircuitLayerTest(unittest.TestCase):
    def test_more_tips_conductance_lowers_pressure(self):
        q_in = torch.tensor([4.0])
        g_hepatic = torch.tensor([1.0])
        g_low_tips = torch.tensor([0.2])
        g_high_tips = torch.tensor([2.0])
        g_collateral = torch.tensor([0.1])
        base = torch.tensor([0.0])
        disease = torch.tensor([0.0])
        severity = torch.tensor([0.0])

        p_low_tips = PortalCircuitLayer.compute_pressure(
            q_in, g_hepatic, g_low_tips, g_collateral, base, disease, severity
        )
        p_high_tips = PortalCircuitLayer.compute_pressure(
            q_in, g_hepatic, g_high_tips, g_collateral, base, disease, severity
        )

        self.assertLess(float(p_high_tips.item()), float(p_low_tips.item()))

    def test_more_hepatic_conductance_lowers_pressure(self):
        q_in = torch.tensor([4.0])
        g_low_hepatic = torch.tensor([0.4])
        g_high_hepatic = torch.tensor([2.0])
        g_tips = torch.tensor([0.0])
        g_collateral = torch.tensor([0.5])
        base = torch.tensor([0.0])
        disease = torch.tensor([0.0])
        severity = torch.tensor([0.0])

        p_low_hepatic = PortalCircuitLayer.compute_pressure(
            q_in, g_low_hepatic, g_tips, g_collateral, base, disease, severity
        )
        p_high_hepatic = PortalCircuitLayer.compute_pressure(
            q_in, g_high_hepatic, g_tips, g_collateral, base, disease, severity
        )

        self.assertLess(float(p_high_hepatic.item()), float(p_low_hepatic.item()))

    def test_collateral_severity_and_conductance_are_independent(self):
        q_in = torch.tensor([4.0])
        g_hepatic = torch.tensor([1.0])
        g_tips = torch.tensor([0.0])
        g_collateral_low = torch.tensor([0.1])
        g_collateral_high = torch.tensor([1.5])
        base = torch.tensor([0.0])
        disease = torch.tensor([0.0])
        severity_low = torch.tensor([0.0])
        severity_high = torch.tensor([1.0])

        p_low_bypass = PortalCircuitLayer.compute_pressure(
            q_in, g_hepatic, g_tips, g_collateral_low, base, disease, severity_low
        )
        p_high_bypass = PortalCircuitLayer.compute_pressure(
            q_in, g_hepatic, g_tips, g_collateral_high, base, disease, severity_low
        )
        p_high_severity = PortalCircuitLayer.compute_pressure(
            q_in, g_hepatic, g_tips, g_collateral_low, base, disease, severity_high
        )

        self.assertLess(float(p_high_bypass.item()), float(p_low_bypass.item()))
        self.assertGreater(float(p_high_severity.item()), float(p_low_bypass.item()))

    def test_tips_absent_forces_zero_tips_conductance(self):
        layer = PortalCircuitLayer(d_branch=4, d_aux=N_AUX, d_hidden=8)
        branch_embeds = torch.zeros(2, N_SEGMENTS, 4)
        aux_norm = torch.zeros(2, N_AUX)
        segment_mask = torch.ones(2, N_SEGMENTS)
        segment_mask[0, SEG_INDEX["tips"]] = 0.0
        junction_diameters = torch.ones(2, N_SEGMENTS) * 8.0
        branch_resistance = torch.ones(2, N_SEGMENTS)
        branch_lengths = torch.ones(2, N_SEGMENTS) * 50.0
        is_post_tips = torch.tensor([1.0, 1.0])

        out = layer(
            branch_embeds,
            aux_norm,
            segment_mask,
            junction_diameters,
            branch_resistance,
            branch_lengths,
            is_post_tips,
        )

        self.assertAlmostEqual(float(out["g_tips"][0].item()), 0.0, places=6)
        self.assertGreater(float(out["g_tips"][1].item()), 0.0)


if __name__ == "__main__":
    unittest.main()
