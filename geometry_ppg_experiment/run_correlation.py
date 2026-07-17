"""Analyze pairwise correlations between six global geometry features and PVP."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


FEATURE_COLUMNS = [
    "R_total",
    "D_Murray",
    "R_collateral",
    "Ratio_SMV_SV",
    "theta_SMV_SV",
    "Ratio_LPV_RPV",
]
TARGET_COLUMN = "y_true_mmHg"
DEFAULT_FEATURES = Path(__file__).resolve().parent / "features.csv"


def load_feature_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"sample", "subject_id", TARGET_COLUMN, *FEATURE_COLUMNS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    for key in [TARGET_COLUMN, *FEATURE_COLUMNS]:
        df[key] = pd.to_numeric(df[key], errors="coerce")
    return df[df[TARGET_COLUMN].notna()].reset_index(drop=True)


def finite_pair(df: pd.DataFrame, feature: str) -> tuple[np.ndarray, np.ndarray]:
    pair = df[[feature, TARGET_COLUMN]].replace([np.inf, -np.inf], np.nan).dropna()
    return pair[feature].to_numpy(dtype=np.float64), pair[TARGET_COLUMN].to_numpy(dtype=np.float64)


def corr_or_nan(x: np.ndarray, y: np.ndarray, method: str) -> tuple[float, float]:
    if len(x) < 3 or np.nanstd(x) <= 0 or np.nanstd(y) <= 0:
        return np.nan, np.nan
    if method == "pearson":
        return stats.pearsonr(x, y)
    if method == "spearman":
        return stats.spearmanr(x, y)
    raise ValueError(method)


def correlation_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in FEATURE_COLUMNS:
        x, y = finite_pair(df, feature)
        pearson_r, pearson_p = corr_or_nan(x, y, "pearson")
        spearman_r, spearman_p = corr_or_nan(x, y, "spearman")
        rows.append(
            {
                "feature": feature,
                "n": int(len(x)),
                "missing": int(len(df) - len(x)),
                "feature_mean": float(np.mean(x)) if len(x) else np.nan,
                "feature_std": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
                "pvp_mean": float(np.mean(y)) if len(y) else np.nan,
                "pearson_r": float(pearson_r) if np.isfinite(pearson_r) else np.nan,
                "pearson_p": float(pearson_p) if np.isfinite(pearson_p) else np.nan,
                "spearman_r": float(spearman_r) if np.isfinite(spearman_r) else np.nan,
                "spearman_p": float(spearman_p) if np.isfinite(spearman_p) else np.nan,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "feature",
        "n",
        "missing",
        "feature_mean",
        "feature_std",
        "pvp_mean",
        "pearson_r",
        "pearson_p",
        "spearman_r",
        "spearman_p",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_zscore_features(path: Path, df: pd.DataFrame) -> None:
    out = df.copy()
    for feature in FEATURE_COLUMNS:
        values = out[feature]
        mean = values.mean(skipna=True)
        std = values.std(skipna=True, ddof=0)
        out[f"{feature}_z"] = (values - mean) / std if np.isfinite(std) and std > 0 else np.nan
    out.to_csv(path, index=False, encoding="utf-8")


def finite_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [finite_json(val) for val in value]
    return value


def save_scatter_plot(out_dir: Path, df: pd.DataFrame, rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2))
    axes = axes.ravel()
    row_by_feature = {row["feature"]: row for row in rows}
    for ax, feature in zip(axes, FEATURE_COLUMNS):
        pair = df[[feature, TARGET_COLUMN]].replace([np.inf, -np.inf], np.nan).dropna()
        x = pair[feature].to_numpy(dtype=np.float64)
        y = pair[TARGET_COLUMN].to_numpy(dtype=np.float64)
        ax.scatter(x, y, s=24, color="#2563eb", alpha=0.78, edgecolors="none")
        if len(x) >= 2 and np.nanstd(x) > 0:
            slope, intercept = np.polyfit(x, y, deg=1)
            xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            ax.plot(xs, slope * xs + intercept, color="#dc2626", linewidth=1.2)
        stats_row = row_by_feature[feature]
        r = stats_row["pearson_r"]
        p = stats_row["pearson_p"]
        title = f"{feature}\nr={r:.3f}, p={p:.3g}, n={int(stats_row['n'])}" if np.isfinite(r) else f"{feature}\nr=NA"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(feature)
        ax.set_ylabel("PVP (mmHg)")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_pvp_scatter.png", dpi=180)
    plt.close(fig)


def save_heatmap(out_dir: Path, df: pd.DataFrame) -> None:
    corr_df = df[[TARGET_COLUMN, *FEATURE_COLUMNS]].corr(method="pearson", min_periods=3)
    labels = ["PVP", *FEATURE_COLUMNS]
    matrix = corr_df.loc[[TARGET_COLUMN, *FEATURE_COLUMNS], [TARGET_COLUMN, *FEATURE_COLUMNS]].to_numpy(dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = "" if not np.isfinite(value) else f"{value:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color="#111827")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_correlation_heatmap.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features_path = args.features.resolve()
    out_dir = (args.out_dir or features_path.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_feature_table(features_path)
    rows = correlation_rows(df)
    write_csv(out_dir / "feature_pvp_correlations.csv", rows)
    write_zscore_features(out_dir / "features_zscore.csv", df)
    save_scatter_plot(out_dir, df, rows)
    save_heatmap(out_dir, df)

    metrics = {
        "features": str(features_path),
        "n_samples_with_pvp": int(len(df)),
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "correlations": rows,
    }
    (out_dir / "correlation_metrics.json").write_text(
        json.dumps(finite_json(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[correlation] rows with PVP: {len(df)}")
    for row in rows:
        print(
            f"[correlation] {row['feature']}: "
            f"Pearson r={row['pearson_r']:.4f}, p={row['pearson_p']:.4g}, n={row['n']}"
        )
    print(f"[correlation] outputs: {out_dir}")


if __name__ == "__main__":
    main()
