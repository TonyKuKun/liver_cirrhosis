import unittest

import numpy as np

from baseline.models import build_model_registry, fit_baseline_model


class BaselineSmokeTest(unittest.TestCase):
    def test_core_models_fit_and_predict(self):
        rng = np.random.RandomState(7)
        X = rng.normal(size=(18, 6))
        X[0, 1] = np.nan
        y = 2.0 * X[:, 0] - 0.5 * np.nan_to_num(X[:, 1]) + rng.normal(scale=0.1, size=18)

        registry = build_model_registry(seed=7, n_inner_folds=2)
        for name in ["mean", "ridge_cv", "random_forest"]:
            with self.subTest(name=name):
                fitted = fit_baseline_model(registry[name], X[:12], y[:12], seed=7, n_inner_folds=2)
                pred = fitted.predict(X[12:])
                self.assertEqual(pred.shape, (6,))
                self.assertTrue(np.all(np.isfinite(pred)))


if __name__ == "__main__":
    unittest.main()

