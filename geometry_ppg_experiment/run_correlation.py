"""Build the three requested feature-PVP correlation tables.

Outputs:
- correlation_tables.csv: long-form table with combined, TIPS, and non-TIPS rows.
- correlation_tables.md: Markdown report containing the three tables.
- correlation_metrics.json: machine-readable copy of the same results.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

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
GROUP_COLUMN = "is_post_tips"
DEFAULT_FEATURES = Path(__file__).resolve().parent / "features.csv"


def load_feature_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"sample", "subject_id", TARGET_COLUMN, GROUP_COLUMN, *FEATURE_COLUMNS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    for key in [TARGET_COLUMN, GROUP_COLUMN, *FEATURE_COLUMNS]:
        df[key] = pd.to_numeric(df[key], errors="coerce")
    df = df[df[TARGET_COLUMN].notna()].copy()
    df[GROUP_COLUMN] = df[GROUP_COLUMN].fillna(0).astype(int)
    return df.reset_index(drop=True)


def finite_pair(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    return df[[feature, TARGET_COLUMN]].replace([np.inf, -np.inf], np.nan).dropna()


def correlation_values(pair: pd.DataFrame, feature: str) -> tuple[float, float, float, float]:
    if len(pair) < 3:
        return np.nan, np.nan, np.nan, np.nan
    x = pair[feature].to_numpy(dtype=np.float64)
    y = pair[TARGET_COLUMN].to_numpy(dtype=np.float64)
    if np.std(x) <= 0 or np.std(y) <= 0:
        return np.nan, np.nan, np.nan, np.nan
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    return float(pearson_r), float(pearson_p), float(spearman_r), float(spearman_p)


def group_rows(df: pd.DataFrame, group_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in FEATURE_COLUMNS:
        pair = finite_pair(df, feature)
        pearson_r, pearson_p, spearman_r, spearman_p = correlation_values(pair, feature)
        rows.append(
            {
                "group": group_name,
                "feature": feature,
                "n_used": int(len(pair)),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
            }
        )
    return rows


def all_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        *group_rows(df, "combined"),
        *group_rows(df[df[GROUP_COLUMN] == 1], "tips"),
        *group_rows(df[df[GROUP_COLUMN] == 0], "non_tips"),
    ]


def finite_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [finite_json(val) for val in value]
    return value


def format_number(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(numeric):
        return ""
    return f"{numeric:.4f}"


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = ["feature", "n_used", "pearson_r", "pearson_p", "spearman_r", "spearman_p"]
    lines = [
        "| 指标 | n_used | Pearson r | Pearson p | Spearman r | Spearman p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = [
            str(row["feature"]),
            str(row["n_used"]),
            format_number(row["pearson_r"]),
            format_number(row["pearson_p"]),
            format_number(row["spearman_r"]),
            format_number(row["spearman_p"]),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown(path: Path, df: pd.DataFrame, rows: list[dict[str, Any]]) -> None:
    by_group = {
        "combined": [row for row in rows if row["group"] == "combined"],
        "tips": [row for row in rows if row["group"] == "tips"],
        "non_tips": [row for row in rows if row["group"] == "non_tips"],
    }
    n_combined = len(df)
    n_tips = int((df[GROUP_COLUMN] == 1).sum())
    n_non_tips = int((df[GROUP_COLUMN] == 0).sum())
    text = "\n\n".join(
        [
            "# Feature-PVP Correlation Tables",
            f"Total samples with PVP: {n_combined}",
            f"TIPS samples: {n_tips}",
            f"Non-TIPS samples: {n_non_tips}",
            "## Combined",
            markdown_table(by_group["combined"]),
            "## TIPS",
            markdown_table(by_group["tips"]),
            "## Non-TIPS",
            markdown_table(by_group["non_tips"]),
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


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
    rows = all_rows(df)

    pd.DataFrame(rows).to_csv(out_dir / "correlation_tables.csv", index=False, encoding="utf-8")
    write_markdown(out_dir / "correlation_tables.md", df, rows)

    metrics = {
        "features": str(features_path),
        "n_samples_with_pvp": int(len(df)),
        "n_tips": int((df[GROUP_COLUMN] == 1).sum()),
        "n_non_tips": int((df[GROUP_COLUMN] == 0).sum()),
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "rows": rows,
    }
    (out_dir / "correlation_metrics.json").write_text(
        json.dumps(finite_json(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[correlation] wrote {out_dir / 'correlation_tables.csv'}")
    print(f"[correlation] wrote {out_dir / 'correlation_tables.md'}")


if __name__ == "__main__":
    main()
