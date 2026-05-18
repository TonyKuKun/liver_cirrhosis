import tempfile
import unittest
from pathlib import Path

import torch

from architecture_benchmark.configs import select_experiments
from architecture_benchmark.datasets import find_stl_path, read_stl_mesh, sample_mesh_points
from architecture_benchmark.models import build_model
from architecture_benchmark.train_benchmark import safe_torch_save
from dataset import N_AUX, N_PROFILE_FEAT, N_SEGMENTS


ASCII_STL = """solid tetra
facet normal 0 0 1
 outer loop
  vertex 0 0 0
  vertex 1 0 0
  vertex 0 1 0
 endloop
endfacet
endsolid tetra
"""


def synthetic_batch(batch=2, n=12, centerline=8, vessel_points=32, organ_points=16):
    return {
        "profiles": torch.ones(batch, N_SEGMENTS, n, N_PROFILE_FEAT),
        "profiles_norm": torch.zeros(batch, N_SEGMENTS, n, N_PROFILE_FEAT),
        "point_valid": torch.ones(batch, N_SEGMENTS, n),
        "segment_mask": torch.ones(batch, N_SEGMENTS),
        "aux_norm": torch.zeros(batch, N_AUX),
        "vessel_points": torch.randn(batch, vessel_points, 3),
        "vessel_valid": torch.ones(batch),
        "spleen_points": torch.randn(batch, organ_points, 3),
        "spleen_valid": torch.ones(batch),
        "liver_points": torch.randn(batch, organ_points, 3),
        "liver_valid": torch.ones(batch),
        "centerline_pos": torch.randn(batch, N_SEGMENTS, centerline, 3),
        "centerline_valid": torch.ones(batch, N_SEGMENTS, centerline),
        "stl_global_norm": torch.zeros(batch, 24),
    }


class ArchitectureBenchmarkTest(unittest.TestCase):
    def test_ascii_stl_reader_and_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mesh.stl"
            path.write_text(ASCII_STL, encoding="utf-8")
            vertices, faces = read_stl_mesh(path)
            points, valid = sample_mesh_points(vertices, faces, n_points=10, seed=1)
        self.assertEqual(vertices.shape[1], 3)
        self.assertEqual(faces.shape[1], 3)
        self.assertEqual(points.shape, (10, 3))
        self.assertEqual(valid, 1.0)

    def test_portal_vein_fallback_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            seg = Path(tmp) / "segmentation"
            seg.mkdir()
            (seg / "portal_vein.stl").write_text(ASCII_STL, encoding="utf-8")
            self.assertTrue(find_stl_path(tmp, ("vessel.stl", "portal_vein.stl")).endswith("portal_vein.stl"))

    def test_core_model_forwards(self):
        batch = synthetic_batch()
        for name in ["numeric_cnn_gnn", "stl_centerline_gnn", "fusion_numeric_stl"]:
            with self.subTest(name=name):
                model = build_model(name, d_hidden=16, dropout=0.0)
                out = model(batch)
                self.assertEqual(tuple(out.shape), (2, 1))
                self.assertTrue(torch.isfinite(out).all())

    def test_default_experiment_selection(self):
        selected = select_experiments(["numeric_cnn_gnn", "fusion_numeric_stl"])
        self.assertEqual([e.name for e in selected], ["numeric_cnn_gnn", "fusion_numeric_stl"])

    def test_safe_torch_save_replaces_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            safe_torch_save({"epoch": 1}, path)
            safe_torch_save({"epoch": 2}, path)
            saved = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(saved["epoch"], 2)


if __name__ == "__main__":
    unittest.main()
