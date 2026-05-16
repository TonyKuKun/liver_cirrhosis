import unittest

import torch

from dataset import (
    N_HIGH_PRESSURE_AUX,
    N_PEAK_PROFILE_FEATURES,
    N_SEGMENTS,
    SEG_INDEX,
)
from model import PortalPressureNet
from train import make_cv_splits


class TailFeatureTest(unittest.TestCase):
    def test_subject_split_stratifies_by_pressure_bin_when_labels_exist(self):
        data = []
        labels = [18.0, 27.0, 32.0, 38.0] * 3
        for i, label in enumerate(labels):
            data.append({
                "name": f"202001{i:02d}Subj{i}",
                "is_post_tips": bool(i % 2),
                "label": label,
            })

        _, info = make_cv_splits(data, n_folds=3, seed=7, split_mode="subject")

        self.assertIn("pressure_bins", info)
        self.assertEqual(info["stratification"], "post_tips+pvp_bin")

    def test_tail_head_returns_positive_delta_and_risk_scores(self):
        model = PortalPressureNet(d_hidden=8, dropout=0.0, gnn_layers=1)
        batch = {
            "profiles": torch.ones(2, N_SEGMENTS, 12, 11),
            "profiles_norm": torch.zeros(2, N_SEGMENTS, 12, 11),
            "arc_lengths": torch.arange(12, dtype=torch.float32).view(1, 1, 12).repeat(2, N_SEGMENTS, 1),
            "point_valid": torch.ones(2, N_SEGMENTS, 12),
            "segment_mask": torch.ones(2, N_SEGMENTS),
            "aux_norm": torch.zeros(2, 26),
            "high_pressure_aux_norm": torch.zeros(2, N_HIGH_PRESSURE_AUX),
            "high_pressure_aux_mask": torch.ones(2, N_HIGH_PRESSURE_AUX),
            "peak_profile_norm": torch.zeros(2, N_PEAK_PROFILE_FEATURES),
            "peak_profile_mask": torch.ones(2, N_PEAK_PROFILE_FEATURES),
            "is_post_tips": torch.tensor([0.0, 1.0]),
        }
        batch["segment_mask"][0, SEG_INDEX["tips"]] = 0.0

        out = model(batch)

        self.assertIn("pvp_base", out)
        self.assertIn("tail_delta", out)
        self.assertIn("risk_30", out)
        self.assertIn("risk_35", out)
        self.assertTrue(torch.all(out["tail_delta"] >= 0))


if __name__ == "__main__":
    unittest.main()
