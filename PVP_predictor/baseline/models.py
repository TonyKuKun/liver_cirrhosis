"""Model registry and fitting helpers for traditional baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    param_grid: Optional[Mapping[str, Iterable[object]]] = None
    scale: bool = True


def _inner_cv(n_samples: int, n_inner_folds: int, seed: int) -> KFold:
    n_splits = max(2, min(int(n_inner_folds), int(n_samples)))
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def _pipeline(spec: ModelSpec) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if spec.scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", spec.estimator))
    return Pipeline(steps)


def build_model_registry(seed: int = 42, n_inner_folds: int = 3) -> Dict[str, ModelSpec]:
    cv = max(2, int(n_inner_folds))
    return {
        "mean": ModelSpec("mean", DummyRegressor(strategy="mean"), scale=False),
        "median": ModelSpec("median", DummyRegressor(strategy="median"), scale=False),
        "linear": ModelSpec("linear", LinearRegression(), scale=True),
        "ridge_cv": ModelSpec(
            "ridge_cv",
            RidgeCV(alphas=np.logspace(-3, 3, 13), cv=cv),
            scale=True,
        ),
        "lasso_cv": ModelSpec(
            "lasso_cv",
            LassoCV(alphas=np.logspace(-3, 1, 20), cv=cv, max_iter=20000, random_state=seed),
            scale=True,
        ),
        "elasticnet_cv": ModelSpec(
            "elasticnet_cv",
            ElasticNetCV(
                alphas=np.logspace(-3, 1, 16),
                l1_ratio=[0.15, 0.5, 0.85],
                cv=cv,
                max_iter=20000,
                random_state=seed,
            ),
            scale=True,
        ),
        "svr_rbf": ModelSpec(
            "svr_rbf",
            SVR(kernel="rbf"),
            param_grid={
                "model__C": [0.3, 1.0, 3.0, 10.0],
                "model__epsilon": [0.5, 1.0, 2.0],
                "model__gamma": ["scale", "auto"],
            },
            scale=True,
        ),
        "knn": ModelSpec(
            "knn",
            KNeighborsRegressor(weights="distance"),
            param_grid={"model__n_neighbors": [1, 3, 5, 7]},
            scale=True,
        ),
        "random_forest": ModelSpec(
            "random_forest",
            RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
            param_grid={"model__max_features": ["sqrt", 0.5, 1.0], "model__min_samples_leaf": [1, 3]},
            scale=False,
        ),
        "extra_trees": ModelSpec(
            "extra_trees",
            ExtraTreesRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
            param_grid={"model__max_features": ["sqrt", 0.5, 1.0], "model__min_samples_leaf": [1, 3]},
            scale=False,
        ),
        "gradient_boosting": ModelSpec(
            "gradient_boosting",
            GradientBoostingRegressor(random_state=seed),
            param_grid={"model__n_estimators": [100, 250], "model__learning_rate": [0.03, 0.08], "model__max_depth": [2, 3]},
            scale=False,
        ),
        "hist_gradient_boosting": ModelSpec(
            "hist_gradient_boosting",
            HistGradientBoostingRegressor(random_state=seed, max_iter=250),
            param_grid={"model__learning_rate": [0.03, 0.08], "model__l2_regularization": [0.0, 0.1, 1.0]},
            scale=False,
        ),
        "adaboost": ModelSpec(
            "adaboost",
            AdaBoostRegressor(
                estimator=DecisionTreeRegressor(max_depth=3, random_state=seed),
                n_estimators=250,
                random_state=seed,
            ),
            param_grid={"model__learning_rate": [0.03, 0.1, 0.3]},
            scale=False,
        ),
    }


def _filtered_grid(spec: ModelSpec, n_train: int, n_inner_folds: int) -> Optional[Mapping[str, Iterable[object]]]:
    if not spec.param_grid:
        return None
    grid = {k: list(v) for k, v in spec.param_grid.items()}
    if "model__n_neighbors" in grid:
        cv_train_size = max(1, int(n_train * (max(2, n_inner_folds) - 1) / max(2, n_inner_folds)))
        grid["model__n_neighbors"] = [k for k in grid["model__n_neighbors"] if int(k) <= cv_train_size]
        if not grid["model__n_neighbors"]:
            grid["model__n_neighbors"] = [1]
    return grid


def fit_baseline_model(
    spec: ModelSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    n_inner_folds: int = 3,
):
    estimator = _pipeline(spec)
    param_grid = _filtered_grid(spec, len(y_train), n_inner_folds)
    if not param_grid:
        return estimator.fit(X_train, y_train)

    scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    search = GridSearchCV(
        estimator,
        param_grid=param_grid,
        scoring=scorer,
        cv=_inner_cv(len(y_train), n_inner_folds, seed),
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    return search.fit(X_train, y_train)


def best_estimator(fitted):
    return getattr(fitted, "best_estimator_", fitted)


def extract_feature_importance(fitted, feature_names):
    estimator = best_estimator(fitted)
    if hasattr(estimator, "named_steps"):
        model = estimator.named_steps.get("model")
    else:
        model = estimator
    if model is None:
        return {}

    values = None
    kind = None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        kind = "feature_importance"
    elif hasattr(model, "coef_"):
        values = np.ravel(np.asarray(model.coef_, dtype=float))
        kind = "coefficient"

    if values is None or values.size != len(feature_names):
        return {}
    return {
        name: {"value": float(value), "abs_value": float(abs(value)), "kind": kind}
        for name, value in zip(feature_names, values)
        if np.isfinite(value)
    }

