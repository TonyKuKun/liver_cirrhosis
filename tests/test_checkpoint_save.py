import tempfile
import unittest
from pathlib import Path

import torch

from train import safe_torch_save


class CheckpointSaveTest(unittest.TestCase):
    def test_safe_torch_save_can_replace_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"

            safe_torch_save({"epoch": 1}, path)
            safe_torch_save({"epoch": 2}, path)

            saved = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(saved["epoch"], 2)


if __name__ == "__main__":
    unittest.main()
