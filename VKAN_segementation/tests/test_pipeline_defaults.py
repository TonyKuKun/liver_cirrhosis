from __future__ import annotations

import unittest

import pipeline


class PipelineDefaultsTests(unittest.TestCase):
    def test_deepseek_defaults_are_used_for_planning_model(self) -> None:
        self.assertEqual(pipeline.DEFAULT_API_BASE_URL, "https://api.deepseek.com")
        self.assertEqual(pipeline.DEFAULT_MODEL, "deepseek-v4-pro")
        self.assertEqual(pipeline.DEFAULT_API_KEY_ENV, "DEEPSEEK_API_KEY")


if __name__ == "__main__":
    unittest.main()
