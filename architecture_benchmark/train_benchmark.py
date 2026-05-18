"""Train and compare numeric/STL architecture benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture_benchmark.configs import BenchmarkExperiment, select_experiments  # noqa: E402
from architecture_benchmark.datasets import (  # noqa: E402
    ArchitectureDataset,
    apply_normalization,
    collate_architecture,
    compute_normalizer,
)
from architecture_benchmark.models import build_model  # noqa: E402
from diagnostics import subject_id_from_name  # noqa: E402
from train import compute_metrics, make_cv_splits  # noqa: E402


CURRENT_MODEL_MAE = 3.2488491535
TRADITIONAL_BASELINE_MAE = 3.4815416619
DEFAULT_OUT_ROOT = ROOT / "runs" / "architecture_benchmark_v1"
DEFAULT_SPLIT_JSON = ROOT / "runs" / "v5.1" / "splits.json"

PRED_COLUMNS = [
    "dataset_mode", "model_name", "experiment", "fold", "name", "subject_id",
    "label", "pred", "err", "abs_err", "post_tips", "high_pvp", "low_pvp",
    "liver_valid", "spleen_valid",
]

COMPARISON_COLUMNS = [
    "dataset_mode", "model_name", "experiment", "n", "mae", "rmse", "r2", "bias",
    "delta_mae_vs_current", "delta_mae_vs_baseline", "fold_mae_mean", "fold_mae_std",
    "fold_rmse_mean", "fold_rmse_std", "fold_r2_mean", "fold_r2_std", "question", "out_dir",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_torch_save(obj, path):
    path = os.fspath(path)
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}-{time.time_ns()}"
    try:
        torch.save(obj, tmp)
        last_error = None
        for _ in range(50):
            try:
                os.replace(tmp, path)
                last_error = None
                break
            except PermissionError as e:
                last_error = e
                try:
                    if os.path.exists(path):
                        os.chmod(path, 0o666)
                        os.remove(path)
                    os.replace(tmp, path)
                    last_error = None
                    break
                except PermissionError as inner:
                    last_error = inner
                    time.sleep(0.1)
        if last_error is not None:
            raise last_error
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_splits(split_json: str | None, ds: ArchitectureDataset, n_folds: int, seed: int):
    if split_json and os.path.exists(split_json):
        with open(split_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        name_to_idx = {str(r["name"]): i for i, r in enumerate(ds.records)}
        splits = []
        missing_names = []
        for fold in payload["folds"]:
            missing_names.extend([n for n in fold["train_names"] if n not in name_to_idx])
            missing_names.extend([n for n in fold["val_names"] if n not in name_to_idx])
            train_idx = np.asarray([name_to_idx[n] for n in fold["train_names"] if n in name_to_idx], dtype=int)
            val_idx = np.asarray([name_to_idx[n] for n in fold["val_names"] if n in name_to_idx], dtype=int)
            splits.append((train_idx, val_idx))
        info = dict(payload.get("split_info", {}))
        info["source"] = split_json
        info["n_loaded_samples"] = int(len(ds.records))
        info["n_missing_split_names"] = int(len(set(missing_names)))
        info["missing_split_names"] = sorted(set(missing_names))
        if missing_names:
            print(
                f"[Benchmark] Warning: {len(set(missing_names))} split names are not loaded "
                f"by the current dataset. Results use {len(ds.records)} samples."
            )
        return splits, info
    splits, info = make_cv_splits(ds.base.data, n_folds=n_folds, seed=seed, split_mode="subject")
    info = dict(info)
    info["source"] = "generated"
    info["n_loaded_samples"] = int(len(ds.records))
    info["n_missing_split_names"] = 0
    info["missing_split_names"] = []
    return splits, info


def metrics_from_rows(rows: Sequence[dict]) -> dict:
    labels = np.asarray([float(r["label"]) for r in rows], dtype=float)
    preds = np.asarray([float(r["pred"]) for r in rows], dtype=float)
    mae, rmse, r2 = compute_metrics(preds, labels)
    return {
        "n": int(len(rows)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "bias": float(np.mean(preds - labels)),
        "label_mean": float(np.mean(labels)),
        "pred_mean": float(np.mean(preds)),
    }


def group_summary(rows: Sequence[dict]) -> Dict[str, dict]:
    groups = {"overall": metrics_from_rows(rows), "groups": {}, "folds": {}}
    for key in ("post_tips", "high_pvp", "low_pvp", "liver_valid", "spleen_valid"):
        for value in (0, 1):
            subset = [r for r in rows if int(r[key]) == value]
            if subset:
                groups["groups"][f"{key}={value}"] = metrics_from_rows(subset)
    for fold in sorted({int(r["fold"]) for r in rows}):
        subset = [r for r in rows if int(r["fold"]) == fold]
        groups["folds"][str(fold)] = metrics_from_rows(subset)
    return groups


def run_epoch(model, loader, optimizer, criterion, normalizer, device):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    rows = []
    n_seen = 0
    for raw in loader:
        batch = apply_normalization(raw, normalizer, device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            pred_norm = model(batch).squeeze(-1)
            loss = criterion(pred_norm, batch["label_norm"])
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
        bsz = pred_norm.numel()
        total_loss += float(loss.detach().cpu()) * bsz
        n_seen += bsz
        pred = pred_norm.detach().cpu().numpy() * normalizer.label_std + normalizer.label_mean
        labels = batch["label"].detach().cpu().numpy()
        for i, name in enumerate(raw["name"]):
            err = float(pred[i] - labels[i])
            rows.append({
                "fold": -1,
                "name": name,
                "subject_id": subject_id_from_name(name),
                "label": float(labels[i]),
                "pred": float(pred[i]),
                "err": err,
                "abs_err": abs(err),
                "post_tips": int(float(raw["is_post_tips"][i]) > 0.5),
                "high_pvp": int(float(labels[i]) >= 30.0),
                "low_pvp": int(float(labels[i]) < 20.0),
                "liver_valid": int(float(raw["liver_valid"][i]) > 0.5),
                "spleen_valid": int(float(raw["spleen_valid"][i]) > 0.5),
            })
    m = metrics_from_rows(rows)
    m["loss"] = total_loss / max(n_seen, 1)
    return m, rows


def train_one_fold(exp: BenchmarkExperiment, ds, train_idx, val_idx, fold_idx, args, device):
    out_fold = Path(args.out_root) / exp.name / f"fold_{fold_idx}"
    out_fold.mkdir(parents=True, exist_ok=True)
    normalizer = compute_normalizer(ds.records, train_idx)
    train_loader = DataLoader(
        Subset(ds, train_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_architecture,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        Subset(ds, val_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_architecture,
        num_workers=0,
        drop_last=False,
    )
    set_seed(args.seed + fold_idx)
    model = build_model(exp.model_name, d_hidden=args.d_hidden, dropout=args.dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.01)
    criterion = nn.MSELoss()

    best = {"mae": float("inf"), "epoch": 0, "rows": []}
    history = ["epoch,phase,loss,mae,rmse,r2,bias\n"]
    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics, _ = run_epoch(model, train_loader, optimizer, criterion, normalizer, device)
        scheduler.step()
        with torch.no_grad():
            val_metrics, val_rows = run_epoch(model, val_loader, None, criterion, normalizer, device)
        for phase, metrics in (("train", train_metrics), ("val", val_metrics)):
            history.append(
                f"{epoch},{phase},{metrics['loss']:.6f},{metrics['mae']:.6f},"
                f"{metrics['rmse']:.6f},{metrics['r2']:.6f},{metrics['bias']:.6f}\n"
            )
        if val_metrics["mae"] < best["mae"] - 1e-6:
            best = {"mae": val_metrics["mae"], "epoch": epoch, "rows": val_rows}
            no_improve = 0
            safe_torch_save(
                {
                    "model_state_dict": model.state_dict(),
                    "normalizer": normalizer.__dict__,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "experiment": exp.__dict__,
                    "args": vars(args),
                },
                out_fold / "best.pt",
            )
        else:
            no_improve += 1
        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"[{exp.name} fold {fold_idx} ep {epoch}] "
                f"train_mae={train_metrics['mae']:.3f} val_mae={val_metrics['mae']:.3f} "
                f"best={best['mae']:.3f}@{best['epoch']}"
            )
        if no_improve >= args.patience:
            break
    with (out_fold / "history.csv").open("w", encoding="utf-8") as f:
        f.writelines(history)
    rows = []
    for row in best["rows"]:
        row = dict(row)
        row["fold"] = int(fold_idx)
        row["dataset_mode"] = exp.dataset_mode
        row["model_name"] = exp.model_name
        row["experiment"] = exp.name
        rows.append(row)
    return {
        "fold": int(fold_idx),
        "best_epoch": int(best["epoch"]),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        **metrics_from_rows(rows),
    }, rows


def write_csv(path: Path, rows: Sequence[dict], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def summarize_experiment(exp: BenchmarkExperiment, out_dir: Path) -> dict | None:
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    return summary


def write_root_comparison(args, experiments: Sequence[BenchmarkExperiment]) -> List[dict]:
    out_root = Path(args.out_root)
    rows = []
    group_payload = {}
    for exp in experiments:
        summary = summarize_experiment(exp, out_root / exp.name)
        if summary is None:
            continue
        overall = summary["overall"]
        fold_metrics = summary.get("fold_results", [])
        fold_mae = np.asarray([f["mae"] for f in fold_metrics], dtype=float)
        fold_rmse = np.asarray([f["rmse"] for f in fold_metrics], dtype=float)
        fold_r2 = np.asarray([f["r2"] for f in fold_metrics], dtype=float)
        rows.append({
            "dataset_mode": exp.dataset_mode,
            "model_name": exp.model_name,
            "experiment": exp.name,
            "n": overall["n"],
            "mae": overall["mae"],
            "rmse": overall["rmse"],
            "r2": overall["r2"],
            "bias": overall["bias"],
            "delta_mae_vs_current": overall["mae"] - args.current_mae,
            "delta_mae_vs_baseline": overall["mae"] - args.baseline_mae,
            "fold_mae_mean": float(fold_mae.mean()) if len(fold_mae) else "",
            "fold_mae_std": float(fold_mae.std()) if len(fold_mae) else "",
            "fold_rmse_mean": float(fold_rmse.mean()) if len(fold_rmse) else "",
            "fold_rmse_std": float(fold_rmse.std()) if len(fold_rmse) else "",
            "fold_r2_mean": float(fold_r2.mean()) if len(fold_r2) else "",
            "fold_r2_std": float(fold_r2.std()) if len(fold_r2) else "",
            "question": exp.question,
            "out_dir": str(out_root / exp.name),
        })
        group_payload[exp.name] = summary.get("group_summary", {})
    rows.sort(key=lambda r: float(r["mae"]))
    write_csv(out_root / "comparison.csv", rows, COMPARISON_COLUMNS)
    with (out_root / "comparison.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    with (out_root / "per_group_summary.json").open("w", encoding="utf-8") as f:
        json.dump(group_payload, f, indent=2, ensure_ascii=False)
    write_analysis(out_root / "analysis.md", rows, args)
    return rows


def write_analysis(path: Path, rows: Sequence[dict], args) -> None:
    lines = ["# Architecture Benchmark Analysis", ""]
    if not rows:
        lines.append("No completed experiments found.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    best = rows[0]
    n_values = sorted({int(r["n"]) for r in rows})
    lines.append(
        f"Best by MAE: `{best['experiment']}` ({best['dataset_mode']} / {best['model_name']}) "
        f"MAE {float(best['mae']):.3f}, RMSE {float(best['rmse']):.3f}, R2 {float(best['r2']):.3f}."
    )
    if n_values and n_values[-1] != 62:
        lines.append("")
        lines.append(
            f"Note: completed experiments use n={n_values[-1]} loaded samples. "
            "Compare against older 62-sample runs with caution unless the same dataset filtering is restored."
        )
    lines.append("")
    lines.append("## Top Experiments")
    for row in rows[:8]:
        lines.append(
            f"- {row['experiment']}: MAE {float(row['mae']):.3f}, RMSE {float(row['rmse']):.3f}, "
            f"R2 {float(row['r2']):.3f}, delta vs current {float(row['delta_mae_vs_current']):+.3f}, "
            f"delta vs traditional baseline {float(row['delta_mae_vs_baseline']):+.3f}."
        )
    lines.append("")
    lines.append("## Reading Guide")
    lines.append("- If `numeric_cnn_gnn` is best, centerline numeric profiles plus topology are sufficient for now.")
    lines.append("- If `stl_centerline_gnn` beats numeric models, 3D centerline geometry is carrying signal beyond hand-crafted profile channels.")
    lines.append("- If `stl_pointnet` is weak but centerline GNN is strong, vessel surface STL is probably noisy and centerline structure is the better 3D input.")
    lines.append("- If `fusion_numeric_stl` wins, STL and numeric features are complementary and should be fused in the next main model.")
    lines.append("- Compare liver_valid and spleen_valid groups in `per_group_summary.json` before trusting liver-driven improvements.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(exp: BenchmarkExperiment, ds, splits, split_info, args, device):
    out_dir = Path(args.out_root) / exp.name
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "summary.json").exists() and not args.force:
        print(f"[Benchmark] {exp.name}: summary exists; skipping.")
        return
    all_rows = []
    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        fold_result, rows = train_one_fold(exp, ds, train_idx, val_idx, fold_idx, args, device)
        fold_results.append(fold_result)
        all_rows.extend(rows)
    write_csv(out_dir / "oof_predictions.csv", all_rows, PRED_COLUMNS)
    summary = {
        "experiment": exp.__dict__,
        "n_folds": len(splits),
        "split_info": split_info,
        "overall": metrics_from_rows(all_rows),
        "fold_results": fold_results,
        "group_summary": group_summary(all_rows),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def run(args):
    args.out_root = resolve_project_path(args.out_root)
    args.split_json = resolve_project_path(args.split_json)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    experiments = select_experiments(args.experiments)
    manifest = [e.__dict__ for e in experiments]
    with (out_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    if args.summarize_only:
        return write_root_comparison(args, experiments)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[Benchmark] Device: {device}")
    ds = ArchitectureDataset(
        args.data_root,
        n_profile_points=args.n_points,
        vessel_points=args.vessel_points,
        organ_points=args.organ_points,
        centerline_points=args.centerline_points,
        verbose=True,
    )
    splits, split_info = load_splits(args.split_json, ds, args.n_folds, args.seed)
    for exp in experiments:
        print(f"[Benchmark] Running {exp.name} ({exp.dataset_mode}/{exp.model_name})")
        run_experiment(exp, ds, splits, split_info, args, device)
        write_root_comparison(args, experiments)
    return write_root_comparison(args, experiments)


def parse_args(argv: Sequence[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", type=str, default=r"F:\PCG data\dataset\test4all_sample")
    ap.add_argument("--out_root", type=str, default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--split_json", type=str, default=str(DEFAULT_SPLIT_JSON))
    ap.add_argument("--experiments", nargs="*", default=None)
    ap.add_argument("--summarize_only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cpu", action="store_true")

    ap.add_argument("--n_points", type=int, default=200)
    ap.add_argument("--vessel_points", type=int, default=4096)
    ap.add_argument("--organ_points", type=int, default=2048)
    ap.add_argument("--centerline_points", type=int, default=64)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--print_every", type=int, default=10)
    ap.add_argument("--d_hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--current_mae", type=float, default=CURRENT_MODEL_MAE)
    ap.add_argument("--baseline_mae", type=float, default=TRADITIONAL_BASELINE_MAE)
    return ap.parse_args(argv)


def resolve_project_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str(ROOT / path)


def main(argv: Sequence[str] | None = None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
