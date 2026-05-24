from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.common import zyx_mask_to_stl


def _binary_stl_points(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    triangles = struct.unpack("<I", raw[80:84])[0]
    arr = np.frombuffer(
        raw,
        dtype=np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")]),
        offset=84,
        count=triangles,
    )
    return arr["v"].reshape(-1, 3)


class NiftiOrientationTests(unittest.TestCase):
    def test_zyx_mask_to_stl_uses_full_nifti_affine(self) -> None:
        try:
            import nibabel  # noqa: F401
            import skimage  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"optional dependency missing: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "affine.stl"
            mask = np.zeros((4, 4, 4), dtype=np.uint8)
            mask[1:3, 1:3, 1:3] = 1
            affine = np.array(
                [
                    [-2.0, 0.0, 0.0, 10.0],
                    [0.0, 3.0, 0.0, -20.0],
                    [0.0, 0.0, 4.0, 30.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )

            zyx_mask_to_stl(mask, affine, out)
            points = _binary_stl_points(out)

        self.assertGreater(len(points), 0)
        self.assertLess(points[:, 0].min(), points[:, 0].max())
        self.assertLess(points[:, 0].max(), 10.0)
        self.assertGreater(points[:, 2].min(), 30.0)


if __name__ == "__main__":
    unittest.main()
