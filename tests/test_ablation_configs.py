import tempfile
import unittest
from pathlib import Path

from ablation.experiments import DEFAULT_VARIANTS, select_variants
from ablation.run_ablations import build_command, parse_args


class AblationConfigTest(unittest.TestCase):
    def test_variant_names_are_unique_and_reference_exists(self):
        names = [v.name for v in DEFAULT_VARIANTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("full_model", names)
        self.assertIn("module_no_aux", names)
        self.assertIn("loss_main_only", names)

    def test_select_variants_rejects_unknown_name(self):
        with self.assertRaises(ValueError):
            select_variants(["not_a_real_ablation"])

    def test_build_command_contains_train_script_and_variant_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = parse_args([
                "--dry_run",
                "--python", "python",
                "--data_root", "DATA",
                "--out_root", tmp,
                "--variants", "module_no_aux",
                "--epochs", "1",
            ])
            variant = select_variants(["module_no_aux"])[0]
            cmd = build_command(args, variant, Path(tmp) / variant.name)
        self.assertEqual(cmd[0], "python")
        self.assertIn("train.py", cmd[1])
        self.assertIn("--no_aux", cmd)
        self.assertIn("--epochs", cmd)
        self.assertIn("1", cmd)


if __name__ == "__main__":
    unittest.main()

