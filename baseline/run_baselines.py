"""Run traditional PVP baselines with the same folds as the deep model."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.features import (  # noqa: E402
    FeatureTable,
    build_feature_table,
    feature_schema,
    indices_for_feature_set,
)
from baseline.models import (  # noqa: E402
    build_model_registry,
    extract_feature_importance,
    fit_baseline_model,
)
from dataset import PortalVeinDataset  # noqa: E402
from train import compute_metrics, make_cv_splits  # noqa: E402


PREDICTION_COLUMNS = [
    "feature_set",
    "model",
    "fold",
    "name",
    "subject_id",
    "label",
    "pred",
    "err",
    "abs_err",
    "post_tips",
    "has_lgv",
    "has_pgv",
    "has_rpv",
    "pvt_severity",
]


def load_cv_splits(
    split_json: str | None,
    data: Sequence[dict],
    n_folds: int,
    seed: int,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict[str, object]]:
    if split_json and os.path.exists(split_json):
        with open(split_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        name_to_idx = {str(d["name"]): i for i, d in enumerate(data)}
        splits = []
        for fold in payload["folds"]:
            train_idx = np.asarray([name_to_idx[n] for n in fold["train_names"] if n in name_to_idx], dtype=int)
            val_idx = np.asarray([name_to_idx[n] for n in fold["val_names"] if n in name_to_idx], dtype=int)
            splits.append((train_idx, val_idx))
        info = dict(payload.get("split_info", {}))
        info["source"] = split_json
        return splits, info
    splits, info = make_cv_splits(data, n_folds=n_folds, seed=seed, split_mode="subject")
    info = dict(info)
    info["source"] = "generated"
    return splits, info


def _write_csv(path: str | os.PathLike, rows: Sequence[dict], columns: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(os.fspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def _prediction_rows(
    table: FeatureTable,
    feature_set: str,
    model_name: str,
    fold: int,
    val_idx: np.ndarray,
    preds: np.ndarray,
) -> List[dict]:
    rows = []
    for idx, pred in zip(val_idx, preds):
        label = float(table.y[idx])
        err = float(pred - label)
        meta = table.metadata[int(idx)]
        rows.append({
            "feature_set": feature_set,
            "model": model_name,
            "fold": int(fold),
            "name": meta["name"],
            "subject_id": meta["subject_id"],
            "label": label,
            "pred": float(pred),
            "err": err,
            "abs_err": abs(err),
            "post_tips": int(meta["post_tips"]),
            "has_lgv": int(meta["has_lgv"]),
            "has_pgv": int(meta["has_pgv"]),
            "has_rpv": int(meta["has_rpv"]),
            "pvt_severity": int(meta["pvt_severity"]),
        })
    return rows


def _summarize_baseline(rows: Sequence[dict]) -> Dict[str, object]:
    labels = np.asarray([float(r["label"]) for r in rows], dtype=float)
    preds = np.asarray([float(r["pred"]) for r in rows], dtype=float)
    mae, rmse, r2 = compute_metrics(preds, labels)
    err = preds - labels
    return {
        "n": int(len(rows)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "bias": float(np.mean(err)),
        "label_mean": float(np.mean(labels)),
        "pred_mean": float(np.mean(preds)),
    }


def _summarize_group(rows: Sequence[dict]) -> Dict[str, object]:
    if not rows:
        return {"n": 0}
    return _summarize_baseline(rows)


def summarize_prediction_rows(rows: Sequence[dict]) -> Dict[str, object]:
    groups = {
        "overall": list(rows),
        "pre_tips": [r for r in rows if int(r.get("post_tips", 0)) == 0],
        "post_tips": [r for r in rows if int(r.get("post_tips", 0)) == 1],
        "has_lgv": [r for r in rows if int(r.get("has_lgv", 0)) == 1],
        "has_pgv": [r for r in rows if int(r.get("has_pgv", 0)) == 1],
        "has_rpv": [r for r in rows if int(r.get("has_rpv", 0)) == 1],
        "no_rpv": [r for r in rows if int(r.get("has_rpv", 0)) == 0],
    }
    return {name: _summarize_group(group_rows) for name, group_rows in groups.items()}


def _aggregate_importance(importance_accum: Dict[tuple, List[dict]]) -> List[dict]:
    rows = []
    for (feature_set, model_name), entries in sorted(importance_accum.items()):
        by_feature = defaultdict(list)
        kinds = {}
        for entry in entries:
            for name, values in entry.items():
                by_feature[name].append(float(values["abs_value"]))
                kinds[name] = values["kind"]
        for name, values in by_feature.items():
            rows.append({
                "feature_set": feature_set,
                "model": model_name,
                "feature": name,
                "importance_kind": kinds.get(name, ""),
                "mean_abs_importance": float(np.mean(values)),
                "std_abs_importance": float(np.std(values)),
                "n_folds": len(values),
            })
    rows.sort(key=lambda r: (r["feature_set"], r["model"], -r["mean_abs_importance"], r["feature"]))
    return rows


def run_baselines(args) -> Dict[str, object]:
    os.makedirs(args.out_dir, exist_ok=True)
    dataset = PortalVeinDataset(
        args.data_root,
        n_points=args.n_points,
        verbose=True,
        include_00_prefix_samples=args.include_00_prefix_samples,
    )
    table = build_feature_table(dataset)
    splits, split_info = load_cv_splits(args.split_json, dataset.data, args.n_folds, args.seed)

    with open(os.path.join(args.out_dir, "feature_schema.json"), "w", encoding="utf-8") as f:
        json.dump(feature_schema(table), f, indent=2, ensure_ascii=False)

    registry = build_model_registry(seed=args.seed, n_inner_folds=args.n_inner_folds)
    all_prediction_rows: List[dict] = []
    summary_rows: List[dict] = []
    summary_json: Dict[str, object] = {
        "data_root": args.data_root,
        "n_samples": int(len(table.y)),
        "split_info": split_info,
        "baselines": {},
    }
    group_summary: Dict[str, object] = {}
    importance_accum: Dict[tuple, List[dict]] = defaultdict(list)

    for feature_set in args.feature_sets:
        indices = indices_for_feature_set(table, feature_set)
        if not indices:
            raise RuntimeError(f"Feature set '{feature_set}' has no usable columns")
        X = table.X[:, indices]
        feature_names = [table.feature_names[i] for i in indices]

        for model_name, spec in registry.items():
            baseline_id = f"{feature_set}/{model_name}"
            print(f"[Baseline] {baseline_id}: {X.shape[1]} features")
            baseline_rows: List[dict] = []
            fold_metrics: List[dict] = []

            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                fitted = fit_baseline_model(
                    spec,
                    X[train_idx],
                    table.y[train_idx],
                    seed=args.seed + fold_idx,
                    n_inner_folds=args.n_inner_folds,
                )
                preds = fitted.predict(X[val_idx])
                rows = _prediction_rows(table, feature_set, model_name, fold_idx, val_idx, preds)
                baseline_rows.extend(rows)
                all_prediction_rows.extend(rows)

                labels = table.y[val_idx]
                mae, rmse, r2 = compute_metrics(np.asarray(preds, dtype=float), labels)
                fold_metrics.append({
                    "fold": int(fold_idx),
                    "n_val": int(len(val_idx)),
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                    "bias": float(np.mean(np.asarray(preds, dtype=float) - labels)),
                })

                importance = extract_feature_importance(fitted, feature_names)
                if importance:
                    importance_accum[(feature_set, model_name)].append(importance)

            overall = _summarize_baseline(baseline_rows)
            summary_json["baselines"][baseline_id] = {
                "feature_set": feature_set,
                "model": model_name,
                "n_features": int(X.shape[1]),
                "overall": overall,
                "folds": fold_metrics,
            }
            group_summary[baseline_id] = summarize_prediction_rows(baseline_rows)

            fold_mae = np.asarray([m["mae"] for m in fold_metrics], dtype=float)
            fold_rmse = np.asarray([m["rmse"] for m in fold_metrics], dtype=float)
            fold_r2 = np.asarray([m["r2"] for m in fold_metrics], dtype=float)
            summary_rows.append({
                "feature_set": feature_set,
                "model": model_name,
                "n_features": int(X.shape[1]),
                "n": overall["n"],
                "mae": overall["mae"],
                "rmse": overall["rmse"],
                "r2": overall["r2"],
                "bias": overall["bias"],
                "fold_mae_mean": float(np.mean(fold_mae)),
                "fold_mae_std": float(np.std(fold_mae)),
                "fold_rmse_mean": float(np.mean(fold_rmse)),
                "fold_rmse_std": float(np.std(fold_rmse)),
                "fold_r2_mean": float(np.mean(fold_r2)),
                "fold_r2_std": float(np.std(fold_r2)),
            })

    _write_csv(os.path.join(args.out_dir, "oof_predictions.csv"), all_prediction_rows, PREDICTION_COLUMNS)
    _write_csv(
        os.path.join(args.out_dir, "summary.csv"),
        summary_rows,
        [
            "feature_set",
            "model",
            "n_features",
            "n",
            "mae",
            "rmse",
            "r2",
            "bias",
            "fold_mae_mean",
            "fold_mae_std",
            "fold_rmse_mean",
            "fold_rmse_std",
            "fold_r2_mean",
            "fold_r2_std",
        ],
    )
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "per_group_summary.json"), "w", encoding="utf-8") as f:
        json.dump(group_summary, f, indent=2, ensure_ascii=False)

    importance_rows = _aggregate_importance(importance_accum)
    _write_csv(
        os.path.join(args.out_dir, "feature_importance.csv"),
        importance_rows,
        ["feature_set", "model", "feature", "importance_kind", "mean_abs_importance", "std_abs_importance", "n_folds"],
    )
    return summary_json


def parse_args(argv: Sequence[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", type=str, default=r"F:\PCG data\dataset\test4all_sample")
    ap.add_argument("--split_json", type=str, default=os.path.join("runs", "v5.1", "splits.json"))
    ap.add_argument("--out_dir", type=str, default=os.path.join("runs", "baseline_v1"))
    ap.add_argument("--n_points", type=int, default=200)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_inner_folds", type=int, default=3)
    ap.add_argument("--include_00_prefix_samples", action="store_true", default=False)
    ap.add_argument(
        "--feature_sets",
        nargs="+",
        default=["geometry", "physics", "aux", "combined"],
        choices=["geometry", "physics", "aux", "combined"],
    )
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_baselines(args)


if __name__ == "__main__":
    main()
