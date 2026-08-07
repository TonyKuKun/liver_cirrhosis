from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from refinement.predict import _select_checkpoint_test_cases, _select_test_cases


class PredictSplitTests(unittest.TestCase):
    def test_seeded_split_requires_training_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "best.pt"

            with self.assertRaisesRegex(FileNotFoundError, "cases.json"):
                _select_checkpoint_test_cases(
                    [Path("patients") / "case_a", Path("patients") / "case_b"],
                    checkpoint_path,
                    seed=30,
                    val_ratio=0.5,
                )

    def test_checkpoint_split_skips_dollar_cases_before_random_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            checkpoint_path = run_dir / "best.pt"
            manifest_names = ["case_a", "skip$case", "case_b@review", "case_c", "case_d", "case_e"]
            (run_dir / "cases.json").write_text(
                json.dumps({"cases": manifest_names}),
                encoding="utf-8",
            )
            cases = [Path("patients") / name for name in manifest_names]

            selected, split_source_count = _select_checkpoint_test_cases(
                cases,
                checkpoint_path,
                seed=30,
                val_ratio=0.4,
            )

            eligible_names = [name for name in manifest_names if "$" not in name]
            expected_names = _select_test_cases(eligible_names, seed=30, val_ratio=0.4)
            self.assertEqual([case.name for case in selected], expected_names)
            self.assertEqual(split_source_count, len(eligible_names))
            self.assertNotIn("skip$case", expected_names)

    def test_missing_selected_case_does_not_change_the_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            checkpoint_path = run_dir / "best.pt"
            manifest_names = ["case_a", "case_b", "case_c", "case_d", "case_e"]
            (run_dir / "cases.json").write_text(
                json.dumps({"cases": manifest_names}),
                encoding="utf-8",
            )
            selected_names = _select_test_cases(manifest_names, seed=30, val_ratio=0.4)
            missing_name = selected_names[0]
            available = [Path("patients") / name for name in manifest_names if name != missing_name]

            with self.assertRaisesRegex(RuntimeError, missing_name):
                _select_checkpoint_test_cases(
                    available,
                    checkpoint_path,
                    seed=30,
                    val_ratio=0.4,
                )


if __name__ == "__main__":
    unittest.main()
