"""Fold-safe calibration analysis for OOF PVP predictions.

This script does not retrain the neural model. It tests whether a simple
post-hoc affine calibration can reduce systematic bias without leaking the
held-out fold into the calibration fit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    return {
        "n": int(len(y_true)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "bias": float(np.mean(err)),
    }


def _fit_ridge_affine(x: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xtx = x.T @ x
    reg = ridge * np.eye(xtx.shape[0], dtype=np.float64)
    reg[0, 0] = 0.0
    return np.linalg.solve(xtx + reg, x.T @ y)


def _design_global(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        np.ones(len(df), dtype=np.float64),
        df["pred"].to_numpy(dtype=np.float64),
    ])


def _design_tips_interaction(df: pd.DataFrame) -> np.ndarray:
    post = df["post_tips"].to_numpy(dtype=np.float64)
    pred = df["pred"].to_numpy(dtype=np.float64)
    return np.column_stack([
        np.ones(len(df), dtype=np.float64),
        pred,
        post,
        pred * post,
    ])


def calibrate_oof(df: pd.DataFrame, mode: str, ridge: float) -> pd.DataFrame:
    out_parts = []
    for fold in sorted(df["fold"].unique()):
        train = df[df["fold"] != fold].copy()
        val = df[df["fold"] == fold].copy()
        y_train = train["label"].to_numpy(dtype=np.float64)

        if mode == "global":
            beta = _fit_ridge_affine(_design_global(train), y_train, ridge=ridge)
            val["pred_calibrated"] = _design_global(val) @ beta
        elif mode == "tips_interaction":
            beta = _fit_ridge_affine(_design_tips_interaction(train), y_train, ridge=ridge)
            val["pred_calibrated"] = _design_tips_interaction(val) @ beta
        else:
            raise ValueError(f"Unknown calibration mode: {mode}")

        out_parts.append(val)
    out = pd.concat(out_parts, ignore_index=True)
    out["err_calibrated"] = out["pred_calibrated"] - out["label"]
    out["abs_err_calibrated"] = out["err_calibrated"].abs()
    return out.sort_values(["fold", "name"]).reset_index(drop=True)


def summarize(df: pd.DataFrame, pred_col: str) -> dict:
    payload = {"overall": _metrics(df["label"].to_numpy(), df[pred_col].to_numpy())}
    groups = {}
    for key in ["post_tips", "has_lgv", "has_pgv", "has_rpv", "pvt_severity"]:
        if key not in df.columns:
            continue
        for value, sub in df.groupby(key):
            if len(sub) == 0:
                continue
            groups[f"{key}={value}"] = _metrics(
                sub["label"].to_numpy(),
                sub[pred_col].to_numpy(),
            )
    payload["groups"] = groups
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--ridge", type=float, default=1e-3)
    args = ap.parse_args()

    pred_csv = Path(args.pred_csv)
    out_dir = Path(args.out_dir) if args.out_dir else pred_csv.parent / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_csv)
    required = {"fold", "name", "label", "pred"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    summaries = {"raw": summarize(df, "pred")}
    for mode in ["global", "tips_interaction"]:
        calibrated = calibrate_oof(df, mode=mode, ridge=args.ridge)
        calibrated.to_csv(out_dir / f"{mode}_calibrated_predictions.csv", index=False)
        summaries[mode] = summarize(calibrated, "pred_calibrated")

    with (out_dir / "calibration_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    rows = []
    raw_mae = summaries["raw"]["overall"]["mae"]
    for mode, payload in summaries.items():
        row = {"mode": mode, **payload["overall"]}
        row["delta_mae_vs_raw"] = row["mae"] - raw_mae
        rows.append(row)
    pd.DataFrame(rows).sort_values("mae").to_csv(out_dir / "calibration_comparison.csv", index=False)
    print(pd.DataFrame(rows).sort_values("mae").to_string(index=False))


if __name__ == "__main__":
    main()
