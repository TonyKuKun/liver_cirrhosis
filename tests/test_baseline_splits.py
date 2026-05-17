import json
import tempfile
import unittest
from pathlib import Path

from baseline.features import subject_id_from_name
from baseline.run_baselines import load_cv_splits


class BaselineSplitTest(unittest.TestCase):
    def test_loaded_split_keeps_subjects_disjoint(self):
        data = [
            {"name": "20200101Alpha", "label": 22.0, "is_post_tips": False},
            {"name": "20200201Alpha#", "label": 18.0, "is_post_tips": True},
            {"name": "20200101Beta", "label": 30.0, "is_post_tips": False},
            {"name": "20200101Gamma", "label": 26.0, "is_post_tips": False},
        ]
        payload = {
            "split_info": {"method": "fixture"},
            "folds": [
                {
                    "fold": 0,
                    "train_names": ["20200101Beta", "20200101Gamma"],
                    "val_names": ["20200101Alpha", "20200201Alpha#"],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "splits.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            splits, info = load_cv_splits(str(path), data, n_folds=1, seed=42)

        self.assertEqual(info["method"], "fixture")
        train_idx, val_idx = splits[0]
        train_subjects = {subject_id_from_name(data[i]["name"]) for i in train_idx}
        val_subjects = {subject_id_from_name(data[i]["name"]) for i in val_idx}
        self.assertTrue(train_subjects.isdisjoint(val_subjects))


if __name__ == "__main__":
    unittest.main()

