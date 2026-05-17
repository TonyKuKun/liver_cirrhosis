import unittest

import numpy as np

from baseline.features import build_feature_table_from_records, indices_for_feature_set
from dataset import N_SEGMENTS, N_PROFILE_FEAT, SEG_INDEX, AUX_KEYS


def _record(name, label, scale=1.0):
    n = 12
    profiles = np.ones((N_SEGMENTS, n, N_PROFILE_FEAT), dtype=np.float32) * scale
    profiles[:, :, 0] = 60.0 * scale
    profiles[:, :, 1] = 8.0 * scale
    profiles[:, :, 2] = 25.0 * scale
    profiles[:, :, 5] = 3.5 * scale
    profiles[:, :, 6] = 0.9
    profiles[:, :, 7] = 0.8
    profiles[:, :, 9] = 0.85
    profiles[:, :, 10] = 1.0
    point_valid = np.ones((N_SEGMENTS, n), dtype=np.float32)
    segment_mask = np.ones(N_SEGMENTS, dtype=np.float32)
    arc_lengths = np.tile(np.linspace(0, 100, n, dtype=np.float32), (N_SEGMENTS, 1))
    aux = np.zeros(len(AUX_KEYS), dtype=np.float32)
    aux_mask = np.ones(len(AUX_KEYS), dtype=np.float32)
    aux[AUX_KEYS.index("has_lgv")] = 1.0
    aux[AUX_KEYS.index("has_pgv")] = 0.0
    aux[AUX_KEYS.index("has_tips")] = 1.0
    aux[AUX_KEYS.index("pvt_severity_grade")] = 1.0
    return {
        "name": name,
        "profiles": profiles,
        "point_valid": point_valid,
        "arc_lengths": arc_lengths,
        "segment_mask": segment_mask,
        "aux_scalars": aux,
        "aux_mask": aux_mask,
        "endpoints_3d": np.ones((N_SEGMENTS, 2, 3), dtype=np.float32),
        "label": float(label),
        "is_post_tips": "#" in name,
        "extras_for_eval": {"mpv_resistance_integral": 1.5 * scale},
    }


class BaselineFeatureTest(unittest.TestCase):
    def test_feature_schema_is_stable_and_finite_columns_remain(self):
        table = build_feature_table_from_records([
            _record("20200101Alpha", 20.0, 1.0),
            _record("20200201Beta#", 24.0, 1.1),
            _record("20200301Gamma", 28.0, 1.2),
        ])

        self.assertEqual(table.X.shape[0], 3)
        self.assertEqual(len(table.feature_names), table.X.shape[1])
        self.assertEqual(len(set(table.feature_names)), len(table.feature_names))
        self.assertFalse(np.any(np.all(~np.isfinite(table.X), axis=0)))
        self.assertGreater(len(indices_for_feature_set(table, "geometry")), 0)
        self.assertGreater(len(indices_for_feature_set(table, "physics")), 0)
        self.assertGreater(len(indices_for_feature_set(table, "aux")), 0)
        self.assertGreater(len(indices_for_feature_set(table, "combined")), len(indices_for_feature_set(table, "geometry")))


if __name__ == "__main__":
    unittest.main()

