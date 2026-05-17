import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from diagnostics import (
    PREDICTION_COLUMNS,
    load_model_state_compat,
    subject_id_from_name,
    summarize_prediction_rows,
    write_prediction_csv,
)
from train import make_cv_splits


class DiagnosticsTest(unittest.TestCase):
    def test_subject_id_groups_pre_and_post_tips_names(self):
        self.assertEqual(
            subject_id_from_name("20210909WuJinHeng"),
            subject_id_from_name("20210921WuJinHeng#"),
        )
        self.assertEqual(
            subject_id_from_name("20210510HanShengLi#H"),
            subject_id_from_name("20210428HanShengLi"),
        )

    def test_subject_split_keeps_subjects_out_of_both_train_and_val(self):
        data = [
            {"name": "20200101Alpha", "is_post_tips": False},
            {"name": "20200201Alpha#", "is_post_tips": True},
            {"name": "20200101Beta", "is_post_tips": False},
            {"name": "20200201Beta#", "is_post_tips": True},
            {"name": "20200101Gamma", "is_post_tips": False},
            {"name": "20200201Gamma#", "is_post_tips": True},
            {"name": "20200101Delta", "is_post_tips": False},
            {"name": "20200201Delta#", "is_post_tips": True},
        ]

        splits, split_info = make_cv_splits(data, n_folds=2, seed=42, split_mode="subject")

        self.assertEqual(split_info["split_mode"], "subject")
        for train_idx, val_idx in splits:
            train_subjects = {subject_id_from_name(data[i]["name"]) for i in train_idx}
            val_subjects = {subject_id_from_name(data[i]["name"]) for i in val_idx}
            self.assertTrue(train_subjects.isdisjoint(val_subjects))
            self.assertGreaterEqual(np.array([data[i]["is_post_tips"] for i in val_idx]).sum(), 1)

    def test_prediction_csv_schema_and_summary(self):
        row = {
            "fold": 0,
            "name": "20210921WuJinHeng#",
            "subject_id": "WuJinHeng",
            "label": 20.0,
            "pred": 24.0,
            "err": 4.0,
            "abs_err": 4.0,
            "post_tips": 1,
            "has_collateral": 0,
            "has_lgv": 0,
            "has_pgv": 0,
            "has_rpv": 1,
            "q_mpv": 1.0,
            "q_lpv": 0.4,
            "q_rpv": 0.2,
            "q_tips": 0.4,
            "tips_fraction": 0.4,
            "collateral_fraction": 0.0,
            "liver_fraction": 0.6,
            "q_in": 1.2,
            "g_hepatic": 0.8,
            "g_tips": 0.4,
            "g_collateral": 0.1,
            "p_circuit": 23.0,
            "disease_offset": 1.5,
            "collateral_severity_offset": 0.0,
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "oof_predictions.csv"
            write_prediction_csv([row], path)
            with path.open("r", newline="", encoding="utf-8") as f:
                loaded = list(csv.DictReader(f))

        self.assertEqual(loaded[0].keys(), set(PREDICTION_COLUMNS))
        summary = summarize_prediction_rows([row])
        self.assertEqual(summary["overall"]["n"], 1)
        self.assertAlmostEqual(summary["overall"]["mae"], 4.0)
        self.assertIn("post_tips=1", summary["groups"])
        self.assertAlmostEqual(
            summary["groups"]["post_tips=1"]["circuit_means"]["g_tips"],
            0.4,
        )

    def test_load_model_state_compat_strips_module_prefix(self):
        model = torch.nn.Linear(2, 1)
        state = {f"module.{k}": v.clone() for k, v in model.state_dict().items()}

        loaded = torch.nn.Linear(2, 1)
        result = load_model_state_compat(loaded, state)

        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(loaded.state_dict()[key], value))


if __name__ == "__main__":
    unittest.main()
