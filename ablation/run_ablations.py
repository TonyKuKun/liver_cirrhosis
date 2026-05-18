"""Run module/loss ablations and summarize their effect.

Example:

    conda run -n pytorch python ablation/run_ablations.py \
      --data_root "F:\PCG data\dataset\test4all_sample" \
      --out_root runs/ablations_v1

Use ``--variants`` to run a subset and ``--dry_run`` to inspect commands.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation.experiments import AblationVariant, select_variants  # noqa: E402


SUMMARY_COLUMNS = [
    "name",
    "category",
    "changed_component",
    "n_folds",
    "mae",
    "rmse",
    "r2",
    "mae_std",
    "rmse_std",
    "r2_std",
    "delta_mae_vs_full",
    "delta_rmse_vs_full",
    "delta_r2_vs_full",
    "hypothesis",
    "out_dir",
]


def _common_train_args(args, out_dir: Path) -> List[str]:
    return [
        "--data_root", args.data_root,
        "--out_dir", str(out_dir),
        "--n_points", str(args.n_points),
        "--n_folds", str(args.n_folds),
        "--seed", str(args.seed),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--patience", str(args.patience),
        "--print_every", str(args.print_every),
        "--d_hidden", str(args.d_hidden),
        "--dropout", str(args.dropout),
        "--gnn_layers", str(args.gnn_layers),
        "--huber_delta", str(args.huber_delta),
        "--lambda_murray", str(args.lambda_murray),
        "--lambda_press", str(args.lambda_press),
        "--lambda_smooth", str(args.lambda_smooth),
        "--lambda_physio", str(args.lambda_physio),
        "--lambda_mono", str(args.lambda_mono),
        "--lambda_spread", str(args.lambda_spread),
        "--extremity_alpha", str(args.extremity_alpha),
        "--post_tips_high_alpha", str(args.post_tips_high_alpha),
        "--post_tips_high_threshold", str(args.post_tips_high_threshold),
        "--lambda_residual", str(args.lambda_residual),
        "--sample_power", str(args.sample_power),
    ]


def build_command(args, variant: AblationVariant, out_dir: Path) -> List[str]:
    train_script = str((ROOT / args.train_script).resolve())
    return [args.python, train_script] + _common_train_args(args, out_dir) + list(variant.args)


def _load_summary(out_dir: Path) -> Dict[str, object] | None:
    path = out_dir / "summary.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_csv(path: Path, rows: Sequence[dict], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def _summary_row(variant: AblationVariant, out_dir: Path, summary: Dict[str, object]) -> dict:
    return {
        "name": variant.name,
        "category": variant.category,
        "changed_component": variant.changed_component,
        "n_folds": summary.get("n_folds", ""),
        "mae": summary.get("val_mae_mean", ""),
        "rmse": summary.get("val_rmse_mean", ""),
        "r2": summary.get("val_r2_mean", ""),
        "mae_std": summary.get("val_mae_std", ""),
        "rmse_std": summary.get("val_rmse_std", ""),
        "r2_std": summary.get("val_r2_std", ""),
        "hypothesis": variant.hypothesis,
        "out_dir": str(out_dir),
    }


def _with_deltas(rows: List[dict]) -> List[dict]:
    full = next((r for r in rows if r["name"] == "full_model"), None)
    if full is None or full.get("mae") == "":
        return rows
    full_mae = float(full["mae"])
    full_rmse = float(full["rmse"])
    full_r2 = float(full["r2"])
    for row in rows:
        if row.get("mae") == "":
            continue
        row["delta_mae_vs_full"] = float(row["mae"]) - full_mae
        row["delta_rmse_vs_full"] = float(row["rmse"]) - full_rmse
        row["delta_r2_vs_full"] = float(row["r2"]) - full_r2
    return rows


def write_analysis_report(path: Path, rows: Sequence[dict]) -> None:
    complete = [r for r in rows if r.get("mae") != ""]
    full = next((r for r in complete if r["name"] == "full_model"), None)
    sorted_rows = sorted(complete, key=lambda r: float(r.get("delta_mae_vs_full", 0.0)), reverse=True)
    improved = [r for r in sorted_rows if float(r.get("delta_mae_vs_full", 0.0)) < -1e-6]
    degraded = [r for r in sorted_rows if float(r.get("delta_mae_vs_full", 0.0)) > 1e-6]

    lines = ["# Ablation Analysis", ""]
    if full:
        lines.append(
            f"Full model: MAE {float(full['mae']):.3f}, RMSE {float(full['rmse']):.3f}, R2 {float(full['r2']):.3f}."
        )
        lines.append("")

    lines.append("## Largest Accuracy Drops")
    if degraded:
        for row in degraded[:8]:
            lines.append(
                f"- {row['name']}: delta MAE {float(row['delta_mae_vs_full']):+.3f}, "
                f"delta RMSE {float(row['delta_rmse_vs_full']):+.3f}, "
                f"delta R2 {float(row['delta_r2_vs_full']):+.3f}. "
                f"Component: {row['changed_component']}."
            )
    else:
        lines.append("- No completed ablation was worse than the full model.")

    lines.append("")
    lines.append("## Ablations That Improve MAE")
    if improved:
        for row in sorted(improved, key=lambda r: float(r["delta_mae_vs_full"]))[:8]:
            lines.append(
                f"- {row['name']}: delta MAE {float(row['delta_mae_vs_full']):+.3f}. "
                "This suggests the removed component may be noisy, over-regularizing, or acting as a shortcut."
            )
    else:
        lines.append("- No completed ablation improved MAE over the full model.")

    lines.append("")
    lines.append("## Interpretation Rules")
    lines.append("- If removing a module increases MAE, that module is carrying useful predictive signal.")
    lines.append("- If removing a module improves MAE, inspect whether it is overfitting the 62-sample dataset.")
    lines.append("- If `module_no_aux` barely changes performance, the model is learning mainly geometry/physics; if it collapses, it depends heavily on system/status shortcuts.")
    lines.append("- If `module_no_branch_embed` barely changes performance, pointwise learned geometry is not adding much beyond hand-coded physics signals.")
    lines.append("- If `loss_main_only` improves performance, the physics losses may be too strong for the current data scale.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_manifest(out_root: Path, manifest: Sequence[dict]) -> List[dict]:
    lookup = {v.name: v for v in select_variants([item["name"] for item in manifest])}
    rows = []
    for item in manifest:
        variant = lookup[item["name"]]
        out_dir = Path(item["out_dir"])
        summary = _load_summary(out_dir)
        if summary is not None:
            rows.append(_summary_row(variant, out_dir, summary))
    rows = _with_deltas(rows)
    _write_csv(out_root / "comparison.csv", rows, SUMMARY_COLUMNS)
    with (out_root / "comparison.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    write_analysis_report(out_root / "analysis.md", rows)
    return rows


def run(args) -> List[dict]:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    variants = select_variants(args.variants)
    manifest = []

    for variant in variants:
        out_dir = out_root / variant.name
        cmd = build_command(args, variant, out_dir)
        manifest.append({
            "name": variant.name,
            "category": variant.category,
            "changed_component": variant.changed_component,
            "hypothesis": variant.hypothesis,
            "out_dir": str(out_dir),
            "command": cmd,
        })
    with (out_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    if args.summarize_only:
        return summarize_manifest(out_root, manifest)

    for item in manifest:
        variant = next(v for v in variants if v.name == item["name"])
        out_dir = Path(item["out_dir"])
        cmd = item["command"]
        print(f"[Ablation] {variant.name}")
        print("  " + " ".join(cmd))
        if args.dry_run:
            continue
        if (out_dir / "summary.json").exists() and not args.force:
            print("  summary.json exists; skipping. Use --force to rerun.")
            summarize_manifest(out_root, manifest)
            continue
        subprocess.run(cmd, cwd=str(ROOT), check=True)
        summarize_manifest(out_root, manifest)

    return summarize_manifest(out_root, manifest)


def parse_args(argv: Sequence[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", type=str, default=r"F:\PCG data\dataset\test4all_sample")
    ap.add_argument("--out_root", type=str, default=os.path.join("runs", "ablations_v2"))
    ap.add_argument("--train_script", type=str, default="train.py")
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--summarize_only", action="store_true")

    ap.add_argument("--n_points", type=int, default=200)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--print_every", type=int, default=10)

    ap.add_argument("--d_hidden", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--gnn_layers", type=int, default=2)

    ap.add_argument("--huber_delta", type=float, default=1.0)
    ap.add_argument("--lambda_murray", type=float, default=0.0)
    ap.add_argument("--lambda_press", type=float, default=0.0)
    ap.add_argument("--lambda_smooth", type=float, default=0.0)
    ap.add_argument("--lambda_physio", type=float, default=0.0)
    ap.add_argument("--lambda_mono", type=float, default=0.0)
    ap.add_argument("--lambda_spread", type=float, default=0.0)
    ap.add_argument("--extremity_alpha", type=float, default=1.0)
    ap.add_argument("--post_tips_high_alpha", type=float, default=0.0)
    ap.add_argument("--post_tips_high_threshold", type=float, default=0.5)
    ap.add_argument("--lambda_residual", type=float, default=0.0)
    ap.add_argument("--sample_power", type=float, default=1.5)
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
