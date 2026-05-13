import unittest

import torch

from dataset import N_AUX, N_SEGMENTS, SEG_INDEX
from model import FlowRateEstimator


def _base_inputs():
    batch = 2
    hidden = 4
    branch_embeds = torch.zeros(batch, N_SEGMENTS, hidden)
    aux_norm = torch.zeros(batch, N_AUX)
    segment_mask = torch.ones(batch, N_SEGMENTS)
    junction_diameters = torch.full((batch, N_SEGMENTS), 10.0)
    branch_resistance = torch.ones(batch, N_SEGMENTS)
    return branch_embeds, aux_norm, segment_mask, junction_diameters, branch_resistance


class FlowPriorTest(unittest.TestCase):
    def test_low_resistance_tips_receives_more_bifurcation_flow(self):
        flow_est = FlowRateEstimator(d_branch=4, d_aux=N_AUX, d_hidden=8)
        inputs = list(_base_inputs())
        branch_resistance = inputs[-1]
        branch_resistance[0, SEG_INDEX["tips"]] = 0.1
        branch_resistance[1, SEG_INDEX["tips"]] = 10.0

        out = flow_est(*inputs)

        self.assertGreater(
            out["tips_fraction"][0].item(),
            out["tips_fraction"][1].item(),
        )

    def test_low_resistance_collaterals_increase_collateral_fraction(self):
        flow_est = FlowRateEstimator(d_branch=4, d_aux=N_AUX, d_hidden=8)
        inputs = list(_base_inputs())
        branch_resistance = inputs[-1]
        branch_resistance[0, SEG_INDEX["lgv"]] = 0.1
        branch_resistance[0, SEG_INDEX["pgv"]] = 0.1
        branch_resistance[1, SEG_INDEX["lgv"]] = 10.0
        branch_resistance[1, SEG_INDEX["pgv"]] = 10.0

        out = flow_est(*inputs)

        self.assertGreater(
            out["collateral_fraction"][0].item(),
            out["collateral_fraction"][1].item(),
        )

    def test_absent_tips_has_zero_flow_and_preserves_bifurcation_mass(self):
        flow_est = FlowRateEstimator(d_branch=4, d_aux=N_AUX, d_hidden=8)
        inputs = list(_base_inputs())
        segment_mask = inputs[2]
        segment_mask[:, SEG_INDEX["tips"]] = 0.0

        out = flow_est(*inputs)
        q = out["Q"]

        self.assertTrue(torch.allclose(q[:, SEG_INDEX["tips"]], torch.zeros(2)))
        self.assertTrue(
            torch.allclose(
                q[:, SEG_INDEX["lpv"]] + q[:, SEG_INDEX["rpv"]],
                q[:, SEG_INDEX["mpv"]],
                atol=1e-6,
            )
        )


if __name__ == "__main__":
    unittest.main()
