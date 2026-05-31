from __future__ import annotations

import inspect
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from pretrain import preprocess


class PreprocessInterfaceTests(unittest.TestCase):
    def test_pretrain_patient_no_longer_accepts_llm_client(self) -> None:
        params = inspect.signature(preprocess.pretrain_patient).parameters

        self.assertNotIn("client", params)

    def test_preprocess_help_has_no_model_api_options(self) -> None:
        script = Path(preprocess.__file__).resolve()
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertNotIn("--model", result.stdout)
        self.assertNotIn("--api" + "_key", result.stdout)
        self.assertNotIn("--api" + "_base_url", result.stdout)

    def test_only_dollar_patients_is_opt_in_filter(self) -> None:
        cases = [
            SimpleNamespace(name="caseA"),
            SimpleNamespace(name="case$B"),
            SimpleNamespace(name="caseC"),
        ]

        self.assertEqual([case.name for case in preprocess._only_dollar_patients(cases)], ["case$B"])


if __name__ == "__main__":
    unittest.main()
