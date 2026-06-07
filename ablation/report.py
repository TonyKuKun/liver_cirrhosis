"""Summarize final PVP predictor ablation runs."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


def load_variant_summary(out_root: str | os.PathLike, variant_name: str):
    path = Path(out_root) / variant_name / "summary.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "name": variant_name,
        "n_folds": data.get("n_folds"),
        "n_samples": data.get("n_samples"),
        "val_mae_mean": data.get("val_mae_mean"),
        "val_mae_std": data.get("val_mae_std"),
        "val_rmse_mean": data.get("val_rmse_mean"),
        "val_r2_mean": data.get("val_r2_mean"),
    }


def summarize(out_root: str | os.PathLike, manifest: List[Mapping[str, object]]):
    out_root = Path(out_root)
    rows = []
    for variant in manifest:
        row = load_variant_summary(out_root, str(variant["name"]))
        if row is None:
            row = {"name": variant["name"], "status": "missing"}
        else:
            row["status"] = "done"
        row["category"] = variant.get("category", "")
        row["changed_component"] = variant.get("changed_component", "")
        row["hypothesis"] = variant.get("hypothesis", "")
        rows.append(row)

    full = next(
        (
            r for r in rows
            if r.get("status") == "done"
            and (r.get("category") == "reference" or r.get("name") == "full_model")
        ),
        None,
    )
    ref_name = full.get("name", "reference") if full else "reference"
    for row in rows:
        if full and row.get("status") == "done" and row.get("name") != ref_name:
            row["delta_mae_vs_full"] = float(row["val_mae_mean"] - full["val_mae_mean"])
            row["delta_rmse_vs_full"] = float(row["val_rmse_mean"] - full["val_rmse_mean"])
            row["delta_r2_vs_full"] = float(row["val_r2_mean"] - full["val_r2_mean"])
            if row.get("category") == "loss":
                evidence = int(row["delta_mae_vs_full"] < 0)
                evidence += int(row["delta_rmse_vs_full"] < 0)
                evidence += int(row["delta_r2_vs_full"] > 0)
            else:
                evidence = int(row["delta_mae_vs_full"] > 0)
                evidence += int(row["delta_rmse_vs_full"] > 0)
                evidence += int(row["delta_r2_vs_full"] < 0)
            row["module_effective_direction"] = "helpful" if evidence >= 2 else "not_helpful_or_noisy"
        else:
            row["delta_mae_vs_full"] = None
            row["delta_rmse_vs_full"] = None
            row["delta_r2_vs_full"] = None
            row["module_effective_direction"] = "reference" if row.get("name") == ref_name else "unknown"

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    fieldnames = [
        "name", "status", "category", "changed_component", "n_folds", "n_samples",
        "val_mae_mean", "val_mae_std", "val_rmse_mean", "val_r2_mean",
        "delta_mae_vs_full", "delta_rmse_vs_full", "delta_r2_vs_full",
        "module_effective_direction", "hypothesis",
    ]
    with open(out_root / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    lines = ["# PVP predictor ablation analysis", ""]
    if full:
        lines.append(
            f"- {ref_name}: MAE {full['val_mae_mean']:.4f}, "
            f"RMSE {full['val_rmse_mean']:.4f}, R2 {full['val_r2_mean']:.4f}"
        )
    else:
        lines.append("- reference variant is missing; run it before interpreting deltas.")
    done = [r for r in rows if r.get("status") == "done" and r.get("name") != ref_name]
    helpful = [r for r in done if r.get("module_effective_direction") == "helpful"]
    noisy = [r for r in done if r.get("module_effective_direction") == "not_helpful_or_noisy"]
    if helpful:
        lines.append(f"- Variants with supportive one-factor evidence versus {ref_name}:")
        for row in sorted(helpful, key=lambda r: -r["delta_mae_vs_full"]):
            lines.append(
                f"  - {row['name']}: dMAE {row['delta_mae_vs_full']:.4f}, "
                f"dRMSE {row['delta_rmse_vs_full']:.4f}, dR2 {row['delta_r2_vs_full']:.4f}"
            )
    if noisy:
        lines.append(f"- Variants without supportive one-factor evidence versus {ref_name}:")
        for row in sorted(noisy, key=lambda r: r["delta_mae_vs_full"]):
            lines.append(
                f"  - {row['name']}: dMAE {row['delta_mae_vs_full']:.4f}, "
                f"dRMSE {row['delta_rmse_vs_full']:.4f}, dR2 {row['delta_r2_vs_full']:.4f}"
            )
    missing = [r for r in rows if r.get("status") != "done"]
    if missing:
        lines.append("- Missing variants:")
        for row in missing:
            lines.append(f"  - {row['name']}")
    lines.append("")
    lines.append("Interpretation note: decisions use MAE, RMSE, and R2 together. Smoke runs are for wiring and stability; full 5-fold runs are used for module/loss decisions.")
    with open(out_root / "analysis.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return rows


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args(argv)
    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    summarize(args.out_root, manifest)


if __name__ == "__main__":
    main()
