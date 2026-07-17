"""Run LOOCV linear and nested-CV Lasso regression for geometry features."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "R_total",
    "D_Murray",
    "R_collateral",
    "Ratio_SMV_SV",
    "theta_SMV_SV",
    "Ratio_LPV_RPV",
]
DEFAULT_FEATURES = Path(__file__).resolve().parent / "features.csv"


def make_imputer() -> SimpleImputer:
    try:
        return SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:
        return SimpleImputer(strategy="median")


def make_linear_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", make_imputer()),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def make_lasso_pipeline(alpha: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            ("imputer", make_imputer()),
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=alpha, max_iter=50000, random_state=0)),
        ]
    )


def load_feature_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"sample", "subject_id", "y_true_mmHg", *FEATURE_COLUMNS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    for key in ["y_true_mmHg", *FEATURE_COLUMNS]:
        df[key] = pd.to_numeric(df[key], errors="coerce")
    usable = df["y_true_mmHg"].notna() & df[FEATURE_COLUMNS].notna().any(axis=1)
    return df.loc[usable].reset_index(drop=True)


def prediction_rows(
    df: pd.DataFrame,
    preds: np.ndarray,
    model_name: str,
    extra: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    extra = extra or [{} for _ in range(len(df))]
    for i, (_, src) in enumerate(df.iterrows()):
        label = float(src["y_true_mmHg"])
        pred = float(preds[i])
        err = pred - label
        row = {
            "model": model_name,
            "fold": i,
            "sample": src["sample"],
            "subject_id": src["subject_id"],
            "y_true_mmHg": label,
            "pred_mmHg": pred,
            "err_mmHg": err,
            "abs_err_mmHg": abs(err),
        }
        row.update(extra[i])
        rows.append(row)
    return rows


def run_linear_loocv(df: pd.DataFrame, x: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
    preds = np.zeros(len(y), dtype=np.float64)
    loo = LeaveOneOut()
    for fold, (train_idx, test_idx) in enumerate(loo.split(x)):
        model = make_linear_pipeline()
        model.fit(x[train_idx], y[train_idx])
        preds[test_idx[0]] = float(model.predict(x[test_idx])[0])
    return prediction_rows(df, preds, "linear")


def run_lasso_nested_loocv(
    df: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> list[dict[str, Any]]:
    preds = np.zeros(len(y), dtype=np.float64)
    extras: list[dict[str, Any]] = [{} for _ in range(len(y))]
    loo = LeaveOneOut()
    alpha_grid = np.logspace(-4, 2, 60)

    for fold, (train_idx, test_idx) in enumerate(loo.split(x)):
        n_train = len(train_idx)
        n_splits = min(5, n_train)
        if n_splits < 2:
            raise RuntimeError("Need at least 3 total samples for nested LOOCV Lasso")
        inner_cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed + fold)
        search = GridSearchCV(
            make_lasso_pipeline(),
            param_grid={"model__alpha": alpha_grid},
            scoring="neg_mean_absolute_error",
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        search.fit(x[train_idx], y[train_idx])
        preds[test_idx[0]] = float(search.predict(x[test_idx])[0])
        extras[test_idx[0]] = {
            "best_alpha": float(search.best_params_["model__alpha"]),
            "inner_cv_mae": float(-search.best_score_),
        }

    return prediction_rows(df, preds, "lasso", extras)


def metrics_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y = np.asarray([float(row["y_true_mmHg"]) for row in rows], dtype=np.float64)
    pred = np.asarray([float(row["pred_mmHg"]) for row in rows], dtype=np.float64)
    err = pred - y
    if len(y) >= 2 and np.std(y) > 0 and np.std(pred) > 0:
        pearson_r, pearson_p = stats.pearsonr(y, pred)
    else:
        pearson_r, pearson_p = np.nan, np.nan
    diff_std = float(np.std(err, ddof=1)) if len(err) > 1 else 0.0
    bias = float(np.mean(err))
    return {
        "n": int(len(y)),
        "pearson_r": float(pearson_r) if np.isfinite(pearson_r) else None,
        "pearson_p": float(pearson_p) if np.isfinite(pearson_p) else None,
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "bias": bias,
        "bland_altman_mean_diff": bias,
        "bland_altman_loa_lower": bias - 1.96 * diff_std,
        "bland_altman_loa_upper": bias + 1.96 * diff_std,
    }


def write_prediction_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "fold",
        "sample",
        "subject_id",
        "y_true_mmHg",
        "pred_mmHg",
        "err_mmHg",
        "abs_err_mmHg",
        "best_alpha",
        "inner_cv_mae",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def final_linear_coefficients(df: pd.DataFrame, x: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
    pipeline = make_linear_pipeline()
    pipeline.fit(x, y)
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]

    coef_std = np.ravel(model.coef_).astype(np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    coef_raw = coef_std / scale
    intercept_raw = float(model.intercept_ - np.sum(coef_std * mean / scale))

    rows: list[dict[str, Any]] = [
        {
            "feature": "intercept",
            "coefficient_nonstandard": intercept_raw,
            "coefficient_standardized": float(model.intercept_),
            "impute_median": "",
            "scaler_mean_after_impute": "",
            "scaler_std_after_impute": "",
        }
    ]
    for i, name in enumerate(FEATURE_COLUMNS):
        rows.append(
            {
                "feature": name,
                "coefficient_nonstandard": float(coef_raw[i]),
                "coefficient_standardized": float(coef_std[i]),
                "impute_median": float(imputer.statistics_[i]),
                "scaler_mean_after_impute": float(mean[i]),
                "scaler_std_after_impute": float(scale[i]),
            }
        )
    return rows


def write_coefficients(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "feature",
        "coefficient_nonstandard",
        "coefficient_standardized",
        "impute_median",
        "scaler_mean_after_impute",
        "scaler_std_after_impute",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def grouped_predictions(rows_by_model: dict[str, list[dict[str, Any]]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model_name, rows in rows_by_model.items():
        y = np.asarray([float(row["y_true_mmHg"]) for row in rows], dtype=np.float64)
        pred = np.asarray([float(row["pred_mmHg"]) for row in rows], dtype=np.float64)
        out[model_name] = (y, pred)
    return out


def save_plots(out_dir: Path, rows_by_model: dict[str, list[dict[str, Any]]]) -> None:
    preds = grouped_predictions(rows_by_model)
    colors = {"linear": "#2563eb", "lasso": "#dc2626"}

    plt.figure(figsize=(6.2, 5.2))
    all_y = []
    all_pred = []
    for model_name, (y, pred) in preds.items():
        all_y.append(y)
        all_pred.append(pred)
        plt.scatter(y, pred, s=28, alpha=0.8, label=model_name, color=colors.get(model_name))
    ycat = np.concatenate(all_y)
    pcat = np.concatenate(all_pred)
    lo = float(min(np.min(ycat), np.min(pcat)))
    hi = float(max(np.max(ycat), np.max(pcat)))
    plt.plot([lo, hi], [lo, hi], color="#111827", linewidth=1.0, linestyle="--")
    plt.xlabel("Measured PPG/PVP (mmHg)")
    plt.ylabel("LOOCV predicted PPG/PVP (mmHg)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_actual_vs_pred.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6.2, 5.2))
    for model_name, (y, pred) in preds.items():
        mean = (y + pred) / 2.0
        diff = pred - y
        bias = float(np.mean(diff))
        sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
        plt.scatter(mean, diff, s=28, alpha=0.8, label=model_name, color=colors.get(model_name))
        plt.axhline(bias, color=colors.get(model_name), linewidth=1.0)
        plt.axhline(bias + 1.96 * sd, color=colors.get(model_name), linewidth=0.8, linestyle="--")
        plt.axhline(bias - 1.96 * sd, color=colors.get(model_name), linewidth=0.8, linestyle="--")
    plt.xlabel("Mean of measured and predicted (mmHg)")
    plt.ylabel("Prediction - measured (mmHg)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "bland_altman.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6.2, 5.2))
    for model_name, (y, pred) in preds.items():
        residual = pred - y
        plt.scatter(pred, residual, s=28, alpha=0.8, label=model_name, color=colors.get(model_name))
    plt.axhline(0.0, color="#111827", linewidth=1.0, linestyle="--")
    plt.xlabel("Predicted PPG/PVP (mmHg)")
    plt.ylabel("Residual (mmHg)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "residuals.png", dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="+", choices=["linear", "lasso"], default=["linear", "lasso"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features_path = args.features.resolve()
    out_dir = (args.out_dir or features_path.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_feature_table(features_path)
    if len(df) < 3:
        raise RuntimeError(f"Need at least 3 usable rows, found {len(df)}")
    x = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = df["y_true_mmHg"].to_numpy(dtype=np.float64)

    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    if "linear" in args.models:
        rows = run_linear_loocv(df, x, y)
        rows_by_model["linear"] = rows
        write_prediction_csv(out_dir / "predictions_linear.csv", rows)
    if "lasso" in args.models:
        rows = run_lasso_nested_loocv(df, x, y, seed=args.seed)
        rows_by_model["lasso"] = rows
        write_prediction_csv(out_dir / "predictions_lasso.csv", rows)

    metrics = {
        "features": str(features_path),
        "n_samples": int(len(df)),
        "feature_columns": FEATURE_COLUMNS,
        "models": {model_name: metrics_for_rows(rows) for model_name, rows in rows_by_model.items()},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    write_coefficients(out_dir / "final_coefficients.csv", final_linear_coefficients(df, x, y))
    save_plots(out_dir, rows_by_model)

    print(f"[regression] usable rows: {len(df)}")
    for model_name, values in metrics["models"].items():
        print(
            f"[regression] {model_name}: "
            f"r={values['pearson_r']}, MAE={values['mae']:.4f}, RMSE={values['rmse']:.4f}"
        )
    print(f"[regression] outputs: {out_dir}")


if __name__ == "__main__":
    main()
