"""Generate CSPH comparison report for the eight-vessel model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def auc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    cmp = pos[:, None] - neg[None, :]
    return float((cmp > 0).mean() + 0.5 * (cmp == 0).mean())


def metrics(scores, labels, threshold):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    n = max(len(labels), 1)
    return {
        "auc": auc(scores, labels),
        "accuracy": (tp + tn) / n,
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "ppv": tp / max(tp + fp, 1),
        "npv": tn / max(tn + fn, 1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "threshold": float(threshold),
    }


def best_youden(scores, labels):
    values = sorted(set(float(x) for x in scores))
    candidates = [min(values) - 1e-9, *values, max(values) + 1e-9]
    best = None
    for threshold in candidates:
        row = metrics(scores, labels, threshold)
        youden = row["sensitivity"] + row["specificity"] - 1.0
        if best is None or youden > best[0] or (
            abs(youden - best[0]) < 1e-12 and row["accuracy"] > best[1]["accuracy"]
        ):
            best = (youden, row)
    return best[1]


def load_csph_predictions(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    labels = np.array([int(r["csph_label"]) for r in rows], dtype=int)
    scores = np.array([float(r["csph_prob"]) for r in rows], dtype=float)
    return labels, scores


def load_regression_predictions(path, pvp_threshold):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    labels = np.array([1 if float(r["label"]) >= pvp_threshold else 0 for r in rows], dtype=int)
    scores = np.array([float(r["pred"]) for r in rows], dtype=float)
    return labels, scores


def load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(x):
    return "" if x is None else f"{float(x):.4f}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--pvp_threshold", type=float, default=20.0)
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args(argv)

    root = args.root
    runs_root = root / "ablation" / "runs"
    out_dir = args.out_dir or runs_root / "csph_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        ("8v_csph_physics_loss_off", runs_root / "csph_final_best"),
    ]
    rows = []
    for name, run_dir in runs:
        summary = load_summary(run_dir / "summary.json")
        labels, scores = load_csph_predictions(run_dir / "oof_predictions.csv")
        rows.append({
            "name": name,
            "n": int(len(labels)),
            "positive": int(labels.sum()),
            "fold_auc_mean": summary.get("val_auc_mean"),
            "fold_auc_std": summary.get("val_auc_std"),
            "fold_accuracy_mean": summary.get("val_accuracy_mean"),
            "fold_sensitivity_mean": summary.get("val_sensitivity_mean"),
            "fold_specificity_mean": summary.get("val_specificity_mean"),
            "oof_threshold_0_5": metrics(scores, labels, 0.5),
            "oof_best_youden": best_youden(scores, labels),
        })

    labels, scores = load_regression_predictions(
        runs_root / "pvp_final_best" / "oof_predictions.csv",
        args.pvp_threshold,
    )
    rows.append({
        "name": "8v_pvp_regression_threshold",
        "n": int(len(labels)),
        "positive": int(labels.sum()),
        "fold_auc_mean": None,
        "fold_auc_std": None,
        "fold_accuracy_mean": None,
        "fold_sensitivity_mean": None,
        "fold_specificity_mean": None,
        "oof_threshold_0_5": metrics(scores, labels, args.pvp_threshold),
        "oof_best_youden": best_youden(scores, labels),
    })

    with open(out_dir / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    lines = [
        "# CSPH comparison",
        "",
        f"CSPH label: PPG >= 10 mmHg, using PPG = PVP - 10 mmHg, so PVP >= {args.pvp_threshold:.1f} mmHg.",
        "",
        "## Cross-validated folds",
        "",
        "| Model | Fold AUC | Fold accuracy | Fold sensitivity | Fold specificity |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {fmt(row['fold_auc_mean'])} | "
            f"{fmt(row['fold_accuracy_mean'])} | {fmt(row['fold_sensitivity_mean'])} | "
            f"{fmt(row['fold_specificity_mean'])} |"
        )

    lines.extend([
        "",
        "## OOF threshold metrics",
        "",
        "| Model | Threshold | AUC | Accuracy | Sensitivity | Specificity | PPV | NPV | Confusion |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in rows:
        for label, key in (("default", "oof_threshold_0_5"), ("best_youden", "oof_best_youden")):
            m = row[key]
            lines.append(
                f"| {row['name']} ({label}) | {m['threshold']:.4f} | {m['auc']:.4f} | "
                f"{m['accuracy']:.4f} | {m['sensitivity']:.4f} | {m['specificity']:.4f} | "
                f"{m['ppv']:.4f} | {m['npv']:.4f} | TP {m['tp']} / TN {m['tn']} / FP {m['fp']} / FN {m['fn']} |"
            )

    lines.extend([
        "",
        "## Literature reference",
        "",
        "| Study/model | Dataset | AUC | Accuracy | Sensitivity | Specificity |",
        "|---|---|---:|---:|---:|---:|",
        "| CT/MRI vascular model, Radiology 2023 | Internal test | 0.90 | 0.84 | 0.87 | 0.83 |",
        "| CT/MRI vascular model, Radiology 2023 | External test 1 | 0.84 | 0.88 | 0.94 | 0.69 |",
        "| CT/MRI vascular model, Radiology 2023 | External test 2 | 0.87 | 0.91 | 0.92 | 0.90 |",
    ])
    with open(out_dir / "comparison.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(out_dir / "comparison.md")


if __name__ == "__main__":
    main()
