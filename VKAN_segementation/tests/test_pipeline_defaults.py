from __future__ import annotations

import unittest

import pipeline


class PipelineDefaultsTests(unittest.TestCase):
    def test_pipeline_defaults_match_nnvnet_workflow(self) -> None:
        self.assertEqual(pipeline.DEFAULT_OUT_DIR, "VKAN_segementation/runs/nnVnet3")
        self.assertEqual(pipeline.DEFAULT_REFINEMENT_MODEL, "nnVnet")


if __name__ == "__main__":
    unittest.main()
