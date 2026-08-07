"""Build the three requested feature-PVP correlation tables.

Outputs:
- correlation_tables.csv: long-form table with combined, TIPS, and non-TIPS rows.
- correlation_metrics.json: machine-readable copy of the same results.
- resistance_peak_tuning.csv: alpha sweep for the peak resistance contribution.
- local_loss_sum_max_tuning.csv: lambda1/lambda2 sweep for cumulative and peak local loss.
- collateral_resistance_peak_tuning.csv: the same alpha sweep for R_collateral.
- collateral_local_loss_sum_max_tuning.csv: the same lambda1/lambda2 sweep for R_collateral.
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
DEFAULT_RESULT_DIR = Path(__file__).resolve().parent / "result"
DEFAULT_FEATURES = DEFAULT_RESULT_DIR / "features.csv"
DEFAULT_REPORT = DEFAULT_RESULT_DIR / "feature_extraction_report.json"
RESISTANCE_PEAK_ALPHAS = [
    0.0,
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    50.0,
    100.0,
]
LOCAL_LOSS_SUM_LAMBDAS = [0.0, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0]
LOCAL_LOSS_MAX_LAMBDAS = [
    0.0,
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    50.0,
    100.0,
]
CANDIDATE_COLUMN = "R_total_candidate"
COLLATERAL_CANDIDATE_COLUMN = "R_collateral_candidate"


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


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def parallel_resistance(values: list[float]) -> float:
    usable = [float(value) for value in values if math.isfinite(value) and value > 0]
    denominator = sum(1.0 / value for value in usable)
    return float(1.0 / denominator) if denominator > 0 else np.nan


def required_parallel_resistance(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        return np.nan
    return parallel_resistance(values)


def load_patient_reports(path: Path) -> dict[str, dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    patients = report.get("patients") or []
    return {
        str(patient["sample"]): patient
        for patient in patients
        if patient.get("status") == "ok" and isinstance(patient.get("features"), dict)
    }


def segment_candidate_resistance(
    detail: dict[str, Any],
    resistance_peak_alpha: float,
    local_loss_sum_lambda: float,
    local_loss_max_lambda: float,
) -> float:
    if detail.get("status") == "fallback_precomputed_resistance_integral":
        return safe_float(detail.get("R_effective"))
    if detail.get("status") != "ok":
        return np.nan

    r_visc = safe_float(detail.get("R_visc"))
    if not math.isfinite(r_visc) or r_visc <= 0:
        return np.nan
    peak_term = safe_float(detail.get("R_visc_peak_length_term"), 0.0)
    phi_sum = safe_float(detail.get("Phi_local_sum", detail.get("Phi_local")), 0.0)
    phi_max = safe_float(detail.get("Phi_local_max"), 0.0)
    resistance_base = r_visc + resistance_peak_alpha * peak_term
    local_multiplier = 1.0 + local_loss_sum_lambda * phi_sum + local_loss_max_lambda * phi_max
    value = resistance_base * local_multiplier
    return float(value) if math.isfinite(value) and value > 0 else np.nan


def r_total_candidate(
    patient_report: dict[str, Any],
    resistance_peak_alpha: float,
    local_loss_sum_lambda: float,
    local_loss_max_lambda: float,
) -> float:
    total_report = ((patient_report.get("features") or {}).get("R_total") or {})
    segment_reports = total_report.get("segment_reports") or {}
    segment_values = {
        segment: segment_candidate_resistance(
            segment_reports.get(segment) or {},
            resistance_peak_alpha,
            local_loss_sum_lambda,
            local_loss_max_lambda,
        )
        for segment in ("smv", "sv", "lgv", "mpv", "lpv", "rpv", "tips")
    }

    lower_segments = total_report.get("lower_parallel_segments") or ["smv", "sv"]
    inflow = required_parallel_resistance([segment_values.get(segment, np.nan) for segment in lower_segments])
    mpv = segment_values["mpv"]
    prehepatic = inflow + mpv if math.isfinite(inflow) and math.isfinite(mpv) else np.nan

    upper_segments = total_report.get("upper_parallel_segments") or []
    upper = parallel_resistance([segment_values.get(segment, np.nan) for segment in upper_segments])
    return float(prehepatic + upper) if math.isfinite(prehepatic) and math.isfinite(upper) else np.nan


def candidate_frame(
    df: pd.DataFrame,
    reports: dict[str, dict[str, Any]],
    resistance_peak_alpha: float,
    local_loss_sum_lambda: float,
    local_loss_max_lambda: float,
) -> pd.DataFrame:
    candidate = df[["sample", TARGET_COLUMN, GROUP_COLUMN]].copy()
    candidate[CANDIDATE_COLUMN] = [
        r_total_candidate(
            reports.get(str(sample), {}),
            resistance_peak_alpha,
            local_loss_sum_lambda,
            local_loss_max_lambda,
        )
        for sample in candidate["sample"]
    ]
    return candidate


def collateral_branch_candidate(
    detail: dict[str, Any],
    resistance_peak_alpha: float,
    local_loss_sum_lambda: float,
    local_loss_max_lambda: float,
) -> float:
    if detail.get("status") != "ok":
        return np.nan

    r_visc = safe_float(detail.get("R_visc_coll"))
    peak_term = safe_float(detail.get("R_visc_peak_length_term"))
    phi_sum = safe_float(detail.get("Phi_local_sum"))
    phi_max = safe_float(detail.get("Phi_local_max"))
    zeta_entrance = safe_float(detail.get("zeta_entrance"))
    lambda_collateral = safe_float(detail.get("lambda_coll"))
    required = (r_visc, peak_term, phi_sum, phi_max, zeta_entrance, lambda_collateral)
    if any(not math.isfinite(value) for value in required) or r_visc <= 0:
        return np.nan

    resistance_base = r_visc + resistance_peak_alpha * peak_term
    loss_multiplier = (
        1.0
        + lambda_collateral * zeta_entrance
        + local_loss_sum_lambda * phi_sum
        + local_loss_max_lambda * phi_max
    )
    value = resistance_base * loss_multiplier
    return float(value) if math.isfinite(value) and value > 0 else np.nan


def r_collateral_candidate(
    patient_report: dict[str, Any],
    resistance_peak_alpha: float,
    local_loss_sum_lambda: float,
    local_loss_max_lambda: float,
) -> float:
    collateral_report = ((patient_report.get("features") or {}).get("R_collateral") or {})
    branch_reports = collateral_report.get("branch_reports") or {}
    branch_values = [
        collateral_branch_candidate(
            branch_reports.get(segment) or {},
            resistance_peak_alpha,
            local_loss_sum_lambda,
            local_loss_max_lambda,
        )
        for segment in ("lgv", "pgv")
    ]
    return parallel_resistance(branch_values)


def collateral_candidate_frame(
    df: pd.DataFrame,
    reports: dict[str, dict[str, Any]],
    resistance_peak_alpha: float,
    local_loss_sum_lambda: float,
    local_loss_max_lambda: float,
) -> pd.DataFrame:
    candidate = df[["sample", TARGET_COLUMN, GROUP_COLUMN]].copy()
    candidate[COLLATERAL_CANDIDATE_COLUMN] = [
        r_collateral_candidate(
            reports.get(str(sample), {}),
            resistance_peak_alpha,
            local_loss_sum_lambda,
            local_loss_max_lambda,
        )
        for sample in candidate["sample"]
    ]
    return candidate


def validate_zero_coefficient_baselines(
    df: pd.DataFrame,
    reports: dict[str, dict[str, Any]],
) -> None:
    candidates = {
        "R_total": candidate_frame(df, reports, 0.0, 0.0, 0.0)[CANDIDATE_COLUMN],
        "R_collateral": collateral_candidate_frame(df, reports, 0.0, 0.0, 0.0)[
            COLLATERAL_CANDIDATE_COLUMN
        ],
    }
    for feature, candidate in candidates.items():
        expected = df[feature].to_numpy(dtype=np.float64)
        actual = candidate.to_numpy(dtype=np.float64)
        if not np.array_equal(np.isfinite(expected), np.isfinite(actual)):
            raise ValueError(f"Zero-coefficient {feature} candidate has a different missing-value pattern")
        finite = np.isfinite(expected)
        if not np.allclose(expected[finite], actual[finite], rtol=1e-9, atol=1e-12):
            max_error = float(np.max(np.abs(expected[finite] - actual[finite])))
            raise ValueError(f"Zero-coefficient {feature} candidate differs from features.csv: {max_error}")


def candidate_correlation_rows(
    df: pd.DataFrame,
    candidate_column: str = CANDIDATE_COLUMN,
) -> list[dict[str, Any]]:
    groups = {
        "combined": df,
        "tips": df[df[GROUP_COLUMN] == 1],
        "non_tips": df[df[GROUP_COLUMN] == 0],
    }
    rows: list[dict[str, Any]] = []
    for group, current in groups.items():
        pair = current[[candidate_column, TARGET_COLUMN]].replace([np.inf, -np.inf], np.nan).dropna()
        pearson_r, pearson_p, spearman_r, spearman_p = correlation_values(pair, candidate_column)
        rows.append(
            {
                "group": group,
                "n_used": int(len(pair)),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
            }
        )
    return rows


def build_resistance_peak_tuning(
    df: pd.DataFrame,
    reports: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    thresholds = {
        safe_float(((patient.get("features") or {}).get("R_total") or {}).get("stenosis_relative_threshold"))
        for patient in reports.values()
    }
    thresholds = {value for value in thresholds if math.isfinite(value)}
    if len(thresholds) != 1:
        raise ValueError(f"Expected one stenosis threshold across reports, found: {sorted(thresholds)}")
    stenosis_threshold = thresholds.pop()

    rows: list[dict[str, Any]] = []
    for alpha in RESISTANCE_PEAK_ALPHAS:
        candidate = candidate_frame(df, reports, alpha, 0.0, 0.0)
        for result in candidate_correlation_rows(candidate):
            rows.append(
                {
                    "stenosis_relative_peak_threshold": stenosis_threshold,
                    "resistance_peak_alpha": alpha,
                    **result,
                }
            )
    return rows


def build_local_loss_sum_max_tuning(
    df: pd.DataFrame,
    reports: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lambda_sum in LOCAL_LOSS_SUM_LAMBDAS:
        for lambda_max in LOCAL_LOSS_MAX_LAMBDAS:
            if lambda_max < lambda_sum:
                continue
            candidate = candidate_frame(df, reports, 0.0, lambda_sum, lambda_max)
            for result in candidate_correlation_rows(candidate):
                rows.append(
                    {
                        "local_loss_sum_lambda": lambda_sum,
                        "local_loss_max_lambda": lambda_max,
                        **result,
                    }
                )
    return rows


def one_collateral_report_parameter(
    reports: dict[str, dict[str, Any]],
    key: str,
) -> float:
    values = {
        safe_float(((patient.get("features") or {}).get("R_collateral") or {}).get(key))
        for patient in reports.values()
    }
    values = {value for value in values if math.isfinite(value)}
    if len(values) != 1:
        raise ValueError(f"Expected one collateral {key} across reports, found: {sorted(values)}")
    return values.pop()


def build_collateral_resistance_peak_tuning(
    df: pd.DataFrame,
    reports: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    stenosis_threshold = one_collateral_report_parameter(reports, "stenosis_relative_threshold")
    collateral_loss_lambda = one_collateral_report_parameter(reports, "collateral_loss_lambda")
    rows: list[dict[str, Any]] = []
    for alpha in RESISTANCE_PEAK_ALPHAS:
        candidate = collateral_candidate_frame(df, reports, alpha, 0.0, 0.0)
        for result in candidate_correlation_rows(candidate, COLLATERAL_CANDIDATE_COLUMN):
            rows.append(
                {
                    "stenosis_relative_peak_threshold": stenosis_threshold,
                    "collateral_loss_lambda": collateral_loss_lambda,
                    "resistance_peak_alpha": alpha,
                    **result,
                }
            )
    return rows


def build_collateral_local_loss_sum_max_tuning(
    df: pd.DataFrame,
    reports: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    collateral_loss_lambda = one_collateral_report_parameter(reports, "collateral_loss_lambda")
    rows: list[dict[str, Any]] = []
    for lambda_sum in LOCAL_LOSS_SUM_LAMBDAS:
        for lambda_max in LOCAL_LOSS_MAX_LAMBDAS:
            if lambda_max < lambda_sum:
                continue
            candidate = collateral_candidate_frame(df, reports, 0.0, lambda_sum, lambda_max)
            for result in candidate_correlation_rows(candidate, COLLATERAL_CANDIDATE_COLUMN):
                rows.append(
                    {
                        "collateral_loss_lambda": collateral_loss_lambda,
                        "local_loss_sum_lambda": lambda_sum,
                        "local_loss_max_lambda": lambda_max,
                        **result,
                    }
                )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features_path = args.features.resolve()
    report_path = args.report.resolve()
    out_dir = (args.out_dir or features_path.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not report_path.is_file():
        raise FileNotFoundError(f"Feature extraction report not found: {report_path}")

    df = load_feature_table(features_path)
    rows = all_rows(df)
    reports = load_patient_reports(report_path)
    missing_reports = sorted(set(df["sample"].astype(str)) - set(reports))
    if missing_reports:
        raise ValueError(f"Missing usable extraction reports for {len(missing_reports)} feature rows")
    validate_zero_coefficient_baselines(df, reports)

    pd.DataFrame(rows).to_csv(out_dir / "correlation_tables.csv", index=False, encoding="utf-8")
    peak_rows = build_resistance_peak_tuning(df, reports)
    pd.DataFrame(peak_rows).to_csv(out_dir / "resistance_peak_tuning.csv", index=False, encoding="utf-8")
    loss_rows = build_local_loss_sum_max_tuning(df, reports)
    pd.DataFrame(loss_rows).to_csv(out_dir / "local_loss_sum_max_tuning.csv", index=False, encoding="utf-8")
    collateral_peak_rows = build_collateral_resistance_peak_tuning(df, reports)
    pd.DataFrame(collateral_peak_rows).to_csv(
        out_dir / "collateral_resistance_peak_tuning.csv",
        index=False,
        encoding="utf-8",
    )
    collateral_loss_rows = build_collateral_local_loss_sum_max_tuning(df, reports)
    pd.DataFrame(collateral_loss_rows).to_csv(
        out_dir / "collateral_local_loss_sum_max_tuning.csv",
        index=False,
        encoding="utf-8",
    )

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
    print(f"[correlation] wrote {out_dir / 'correlation_metrics.json'}")
    print(f"[correlation] wrote {out_dir / 'resistance_peak_tuning.csv'}")
    print(f"[correlation] wrote {out_dir / 'local_loss_sum_max_tuning.csv'}")
    print(f"[correlation] wrote {out_dir / 'collateral_resistance_peak_tuning.csv'}")
    print(f"[correlation] wrote {out_dir / 'collateral_local_loss_sum_max_tuning.csv'}")


if __name__ == "__main__":
    main()
