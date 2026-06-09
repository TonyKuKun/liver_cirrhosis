"""Train the new physics-constrained PVP model."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from openpyxl import Workbook

from dataset import AUX_KEYS, SEG_INDEX, PortalVeinDataset, collate_fn
from model import NewPhysicsLoss, NewPortalPressureNet, count_params


PREDICTION_COLUMNS = [
    "fold", "name", "subject_id", "label", "pred", "err", "abs_err",
    "post_tips", "has_lgv", "has_pgv", "has_rpv", "pvt_severity",
    "q_mpv", "q_lpv", "q_rpv", "q_tips", "tips_fraction",
    "collateral_fraction", "collateral_type", "liver_fraction", "physics_gate",
    "physics_anchor_norm", "physics_raw_norm", "physics_calibrated_norm",
    "physics_delta_norm",
]

def safe_torch_save(obj, path):
    path = os.fspath(path)
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp_path = f"{path}.tmp-{os.getpid()}"
    try:
        torch.save(obj, tmp_path)
        for _ in range(20):
            try:
                os.replace(tmp_path, path)
                break
            except PermissionError:
                time.sleep(0.1)
        else:
            os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def subject_id_from_name(name):
    core = re.sub(r"^\d+", "", str(name))
    return core.split("#", 1)[0]


def compute_metrics(preds, labels):
    err = preds - labels
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((labels - labels.mean()) ** 2) + 1e-9
    return float(mae), float(rmse), float(1.0 - ss_res / ss_tot)


def make_cv_splits(data, n_folds, seed, split_mode="subject"):
    rng = np.random.RandomState(seed)
    indices = np.arange(len(data))
    if split_mode == "sample":
        rng.shuffle(indices)
        folds = np.array_split(indices, n_folds)
        splits = []
        for fi in range(n_folds):
            val_idx = np.array(sorted(folds[fi]))
            train_idx = np.array(sorted(np.setdiff1d(indices, val_idx)))
            splits.append((train_idx, val_idx))
        method = "sample KFold"
        n_subjects = len({subject_id_from_name(d["name"]) for d in data})
    else:
        subject_samples: Dict[str, List[int]] = {}
        subject_max_pvp: Dict[str, float] = {}
        for i, d in enumerate(data):
            sid = subject_id_from_name(d["name"])
            subject_samples.setdefault(sid, []).append(i)
            subject_max_pvp[sid] = max(subject_max_pvp.get(sid, -np.inf), float(d["label"]))
        subjects = sorted(subject_samples)
        if len(subjects) < n_folds:
            raise RuntimeError(f"Need >= {n_folds} subjects, have {len(subjects)}")
        rng.shuffle(subjects)
        subjects.sort(key=lambda s: -subject_max_pvp[s])
        fold_pvp_sum = np.zeros(n_folds)
        fold_assignment = {}
        for sid in subjects:
            target = int(np.argmin(fold_pvp_sum))
            fold_assignment[sid] = target
            fold_pvp_sum[target] += subject_max_pvp[sid]
        splits = []
        for fi in range(n_folds):
            val_sids = {s for s, f in fold_assignment.items() if f == fi}
            val_idx = np.array([i for i in indices if subject_id_from_name(data[i]["name"]) in val_sids])
            train_idx = np.array([i for i in indices if i not in set(val_idx)])
            splits.append((train_idx, val_idx))
        method = "GreedyBalancedGroupKFold(PVP)"
        n_subjects = len(subjects)

    labels = np.array([float(d["label"]) for d in data])
    fold_stats = []
    for fi, (_, va) in enumerate(splits):
        fold_stats.append({
            "fold": fi,
            "val_mean_pvp": float(labels[va].mean()),
            "val_std_pvp": float(labels[va].std()),
            "val_min": float(labels[va].min()),
            "val_max": float(labels[va].max()),
        })
    return splits, {
        "split_mode": split_mode,
        "method": method,
        "n_subjects": int(n_subjects),
        "n_folds": int(n_folds),
        "post_tips": int(sum(1 for d in data if d["is_post_tips"])),
        "pre_tips": int(sum(1 for d in data if not d["is_post_tips"])),
        "fold_stats": fold_stats,
    }


def _make_sampler(full_ds, train_idx, power=1.5):
    if power <= 0:
        return None
    labels = np.array([full_ds.data[i]["label"] for i in train_idx])
    median = np.median(labels)
    std = max(np.std(labels), 1e-6)
    weights = 1.0 + (np.abs(labels - median) / std) ** power
    weights = weights / weights.mean()
    return WeightedRandomSampler(torch.from_numpy(weights).double(), len(train_idx), replacement=True)


def _aux_flag(aux, key, threshold=0.5):
    try:
        return int(aux[AUX_KEYS.index(key)] > threshold)
    except ValueError:
        return 0


def prediction_rows_from_batch(out, batch, fold, label_mean, label_std):
    pvp_norm = out["pvp_pred"].detach().squeeze(-1).cpu().numpy()
    preds = pvp_norm * label_std + label_mean
    labels = batch["label"].detach().cpu().numpy()
    seg_mask = batch["segment_mask"].detach().cpu().numpy()
    aux = batch["aux_scalars"].detach().cpu().numpy()
    q = out["Q"].detach().cpu().numpy()
    tips_fraction = out["flow_out"]["tips_fraction"].detach().cpu().numpy()
    collateral_fraction = out["flow_out"]["collateral_fraction"].detach().cpu().numpy()
    liver_fraction = out["flow_out"]["liver_fraction"].detach().cpu().numpy()
    collateral_type = out.get("collateral_type", torch.zeros_like(batch["label"], dtype=torch.long))
    collateral_type = collateral_type.detach().cpu().numpy()
    gate = out["pvp_physics_gate"].detach().squeeze(-1).cpu().numpy()
    anchor = out["pvp_baseline_norm"].detach().squeeze(-1).cpu().numpy()
    raw = out["pvp_baseline_raw_norm"].detach().squeeze(-1).cpu().numpy()
    calibrated = out["pvp_physics_calibrated_norm"].detach().squeeze(-1).cpu().numpy()
    delta = out["pvp_physics_delta_norm"].detach().squeeze(-1).cpu().numpy()
    rows = []
    for i, name in enumerate(batch["name"]):
        err = float(preds[i] - labels[i])
        rows.append({
            "fold": int(fold),
            "name": name,
            "subject_id": subject_id_from_name(name),
            "label": float(labels[i]),
            "pred": float(preds[i]),
            "err": err,
            "abs_err": abs(err),
            "post_tips": int(float(batch["is_post_tips"][i].detach().cpu()) > 0.5),
            "has_lgv": _aux_flag(aux[i], "has_lgv"),
            "has_pgv": _aux_flag(aux[i], "has_pgv"),
            "has_rpv": int(seg_mask[i, SEG_INDEX["rpv"]] > 0.5),
            "pvt_severity": _aux_flag(aux[i], "pvt_severity_grade", threshold=-0.5),
            "q_mpv": float(q[i, SEG_INDEX["mpv"]]),
            "q_lpv": float(q[i, SEG_INDEX["lpv"]]),
            "q_rpv": float(q[i, SEG_INDEX["rpv"]]),
            "q_tips": float(q[i, SEG_INDEX["tips"]]),
            "tips_fraction": float(tips_fraction[i]),
            "collateral_fraction": float(collateral_fraction[i]),
            "collateral_type": int(collateral_type[i]),
            "liver_fraction": float(liver_fraction[i]),
            "physics_gate": float(gate[i]),
            "physics_anchor_norm": float(anchor[i]),
            "physics_raw_norm": float(raw[i]),
            "physics_calibrated_norm": float(calibrated[i]),
            "physics_delta_norm": float(delta[i]),
        })
    return rows


def collect_prediction_rows(model, loader, device, fold, label_mean, label_std):
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
            out = model(batch)
            rows.extend(prediction_rows_from_batch(out, batch, fold, label_mean, label_std))
    return rows


def write_prediction_outputs(rows, csv_path):
    csv_path = os.fspath(csv_path)
    xlsx_path = os.path.splitext(csv_path)[0] + ".xlsx"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    table_rows = [{k: row.get(k, "") for k in PREDICTION_COLUMNS} for row in rows]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        writer.writerows(table_rows)

    wb = Workbook()
    ws = wb.active
    ws.title = "predictions"
    ws.append(PREDICTION_COLUMNS)
    for row in table_rows:
        ws.append([row.get(k, "") for k in PREDICTION_COLUMNS])
    wb.save(xlsx_path)


def write_prediction_csv(rows, path):
    write_prediction_outputs(rows, path)


def summarize_rows(rows):
    if not rows:
        return {"overall": {"n": 0, "mae": None, "rmse": None, "bias": None}}
    err = np.array([float(r["err"]) for r in rows])
    labels = np.array([float(r["label"]) for r in rows])
    preds = np.array([float(r["pred"]) for r in rows])
    out = {
        "overall": {
            "n": int(len(rows)),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "bias": float(np.mean(err)),
            "label_mean": float(labels.mean()),
            "pred_mean": float(preds.mean()),
        },
        "folds": {
            str(f): {
                "n": int(sum(int(r["fold"]) == f for r in rows)),
                "mae": float(np.mean([r["abs_err"] for r in rows if int(r["fold"]) == f])),
            }
            for f in sorted(set(int(r["fold"]) for r in rows))
        },
    }
    return out


def write_group_summary(rows, path):
    os.makedirs(os.path.dirname(os.fspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summarize_rows(rows), f, indent=2, ensure_ascii=False)


def build_model(args, full_ds, device):
    return NewPortalPressureNet(
        d_hidden=args.d_hidden,
        dropout=args.dropout,
        flow_gnn_layers=args.flow_gnn_layers,
        use_organ_flow_scale=args.use_organ_flow_scale,
        use_global_flow_corrector=args.use_global_flow_corrector,
        use_flow_graph=args.use_flow_graph,
        fixed_physics_params=args.fixed_physics_params,
        use_all_profile_channels=args.use_all_profile_channels,
        use_unreliable_raw_lengths=args.use_unreliable_raw_lengths,
        use_organ_global_features=args.use_organ_global_features,
        disable_organ_features=args.disable_organ_features,
        use_six_vessel_layout=args.use_six_vessel_layout,
        use_three_vessel_layout=args.use_three_vessel_layout,
        use_organ_branch_scales=args.use_organ_branch_scales,
        label_mean=full_ds.label_mean,
        label_std=full_ds.label_std,
    ).to(device)


def run_epoch(
    model,
    loader,
    criterion,
    device,
    label_mean,
    label_std,
    optimizer=None,
    scheduler=None,
):
    is_train = optimizer is not None
    model.train(is_train)
    loss_log_sum = {}
    preds_real, labels_real = [], []
    n_seen = 0
    for batch in loader:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)
        bsz = int(batch["label_norm"].size(0))
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            out = model(batch)
            loss, log = criterion(out, batch["label_norm"], batch)
        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        for k, v in log.items():
            loss_log_sum[k] = loss_log_sum.get(k, 0.0) + v * bsz
        pred_n = out["pvp_pred"].detach().squeeze(-1).cpu().numpy()
        preds_real.append(pred_n * label_std + label_mean)
        labels_real.append(batch["label"].detach().cpu().numpy())
        n_seen += bsz
    if is_train and scheduler is not None:
        scheduler.step()
    preds = np.concatenate(preds_real)
    labels = np.concatenate(labels_real)
    mae, rmse, r2 = compute_metrics(preds, labels)
    avg = {k: v / max(n_seen, 1) for k, v in loss_log_sum.items()}
    avg.update({"mae": mae, "rmse": rmse, "r2": r2})
    return avg


def train_fold(fold_idx, train_idx, val_idx, full_ds, args, device):
    out_fold = os.path.join(args.out_dir, f"fold_{fold_idx}")
    os.makedirs(out_fold, exist_ok=True)
    train_ld = DataLoader(
        Subset(full_ds, train_idx.tolist()),
        batch_size=args.batch_size,
        sampler=_make_sampler(full_ds, train_idx.tolist(), args.sample_power),
        shuffle=args.sample_power <= 0,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_ld = DataLoader(
        Subset(full_ds, val_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    model = build_model(args, full_ds, device)
    if fold_idx == 0:
        total, trainable = count_params(model)
        print(f"[NewModel] Params: {total:,} trainable={trainable:,}")
        print(f"[NewModel] Selected geometry: {model.selected_profile_names}")
        print(f"[NewModel] Global aux excludes flags: {[k for k in ['has_lgv','has_pgv','has_tips'] if k not in model.global_aux_names]}")
    criterion = NewPhysicsLoss(
        lambda_shunt=args.lambda_shunt,
        split_loss_mode=args.split_loss_mode,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.01)

    best_val_mae = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    history = ["epoch,phase,total,main,shunt,mae,rmse,r2\n"]
    for epoch in range(1, args.epochs + 1):
        train_log = run_epoch(
            model,
            train_ld,
            criterion,
            device,
            full_ds.label_mean,
            full_ds.label_std,
            optimizer,
            scheduler,
        )
        val_log = run_epoch(
            model,
            val_ld,
            criterion,
            device,
            full_ds.label_mean,
            full_ds.label_std,
        )
        for phase, log in (("train", train_log), ("val", val_log)):
            history.append(
                f"{epoch},{phase},{log['total']:.5f},{log['main']:.5f},"
                f"{log['shunt']:.5f},{log['mae']:.4f},{log['rmse']:.4f},{log['r2']:.4f}\n"
            )

        improved = val_log["mae"] < best_val_mae - 1e-4
        if improved:
            best_val_mae = val_log["mae"]
            best_epoch = epoch
            epochs_no_improve = 0
            ckpt_payload = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_mae": best_val_mae,
                "val_rmse": val_log["rmse"],
                "val_r2": val_log["r2"],
                "args": vars(args),
                "selected_profile_names": model.selected_profile_names,
                "global_aux_names": model.global_aux_names,
            }
            safe_torch_save(ckpt_payload, os.path.join(out_fold, "best.pt"))
        else:
            epochs_no_improve += 1
        if epoch % args.print_every == 0 or epoch == 1:
            print(
                f"[Fold {fold_idx} | Ep {epoch:3d}] "
                f"train total={train_log['total']:.4f} mae={train_log['mae']:.2f} | "
                f"val total={val_log['total']:.4f} mae={val_log['mae']:.2f} "
                f"r2={val_log['r2']:.2f} | best={best_val_mae:.2f}@{best_epoch}"
            )
        if epochs_no_improve >= args.patience:
            print(f"[Fold {fold_idx}] Early stop at ep {epoch}; best mae={best_val_mae:.3f}@{best_epoch}")
            break

    with open(os.path.join(out_fold, "history.csv"), "w", encoding="utf-8") as f:
        f.writelines(history)

    ckpt = torch.load(os.path.join(out_fold, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    val_rows = collect_prediction_rows(
        model,
        val_ld,
        device,
        fold_idx,
        full_ds.label_mean,
        full_ds.label_std,
    )
    write_prediction_csv(val_rows, os.path.join(out_fold, "val_predictions.csv"))
    result = {
        "fold": fold_idx,
        "best_epoch": int(ckpt["epoch"]),
        "val_mae": float(ckpt["val_mae"]),
        "val_rmse": float(ckpt["val_rmse"]),
        "val_r2": float(ckpt["val_r2"]),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
    }
    return result, val_rows


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", type=str, default=r"F:\PCG data\dataset\test4all_sample")
    ap.add_argument("--out_dir", type=str, default=str(ROOT / "runs" / "full_model"))
    ap.add_argument("--n_points", type=int, default=200)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=40)
    ap.add_argument("--split_mode", choices=["subject", "sample"], default="subject")
    ap.add_argument("--include_00_prefix_samples", action="store_true", default=False)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--print_every", type=int, default=10)
    ap.add_argument("--d_hidden", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--flow_gnn_layers", type=int, default=2)
    ap.add_argument("--use_organ_flow_scale", action="store_true", default=False)
    ap.add_argument("--no_organ_flow_scale", dest="use_organ_flow_scale", action="store_false")
    ap.add_argument("--use_global_flow_corrector", action="store_true", default=True)
    ap.add_argument("--no_global_flow_corrector", dest="use_global_flow_corrector", action="store_false")
    ap.add_argument("--use_flow_graph", action="store_true", default=True)
    ap.add_argument("--no_flow_graph", dest="use_flow_graph", action="store_false")
    ap.add_argument("--fixed_physics_params", action="store_true", default=False)
    ap.add_argument("--use_all_profile_channels", action="store_true", default=False)
    ap.add_argument("--use_unreliable_raw_lengths", action="store_true", default=False)
    ap.add_argument("--use_organ_global_features", action="store_true", default=True)
    ap.add_argument("--no_organ_global_features", dest="use_organ_global_features", action="store_false")
    ap.add_argument("--disable_organ_features", action="store_true", default=False)
    ap.add_argument("--use_six_vessel_layout", action="store_true", default=False)
    ap.add_argument("--use_three_vessel_layout", action="store_true", default=False)
    ap.add_argument("--use_organ_branch_scales", action="store_true", default=True)
    ap.add_argument("--no_organ_branch_scales", dest="use_organ_branch_scales", action="store_false")
    ap.add_argument("--use_eight_vessel_layout", action="store_true", default=False)
    ap.add_argument("--lambda_shunt", type=float, default=0.03)
    ap.add_argument("--split_loss_mode", choices=["full", "core_confluence"], default="core_confluence")
    ap.add_argument("--sample_power", type=float, default=1.5)
    args = ap.parse_args(argv)
    if args.use_eight_vessel_layout:
        args.use_six_vessel_layout = False
        args.use_three_vessel_layout = False
    if args.use_six_vessel_layout and args.use_three_vessel_layout:
        raise ValueError("Choose at most one compact layout: --use_six_vessel_layout or --use_three_vessel_layout.")
    return args


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[NewModel] Device: {device}")
    full_ds = PortalVeinDataset(
        args.data_root,
        n_points=args.n_points,
        verbose=True,
        include_00_prefix_samples=args.include_00_prefix_samples,
    )
    if len(full_ds) < args.n_folds:
        raise RuntimeError(f"Need >= {args.n_folds} patients, have {len(full_ds)}")
    safe_torch_save({
        "profile_mean": full_ds.profile_mean,
        "profile_std": full_ds.profile_std,
        "aux_mean": full_ds.aux_mean,
        "aux_std": full_ds.aux_std,
        "label_mean": full_ds.label_mean,
        "label_std": full_ds.label_std,
    }, os.path.join(args.out_dir, "normalization.pt"))
    splits, split_info = make_cv_splits(full_ds.data, args.n_folds, args.seed, args.split_mode)
    with open(os.path.join(args.out_dir, "splits.json"), "w", encoding="utf-8") as f:
        json.dump({"split_info": split_info}, f, indent=2, ensure_ascii=False)

    fold_results = []
    all_rows = []
    for fi, (train_idx, val_idx) in enumerate(splits):
        print(f"\n[NewModel] Fold {fi}/{args.n_folds - 1}: train={len(train_idx)} val={len(val_idx)}")
        res, rows = train_fold(fi, train_idx, val_idx, full_ds, args, device)
        fold_results.append(res)
        all_rows.extend(rows)
    write_prediction_csv(all_rows, os.path.join(args.out_dir, "oof_predictions.csv"))
    write_group_summary(all_rows, os.path.join(args.out_dir, "oof_group_summary.json"))

    maes = np.array([r["val_mae"] for r in fold_results])
    rmses = np.array([r["val_rmse"] for r in fold_results])
    r2s = np.array([r["val_r2"] for r in fold_results])
    oof_summary = summarize_rows(all_rows)
    summary = {
        "n_folds": args.n_folds,
        "n_samples": len(full_ds),
        "fold_results": fold_results,
        "val_mae_mean": float(maes.mean()),
        "val_mae_std": float(maes.std()),
        "val_rmse_mean": float(rmses.mean()),
        "val_rmse_std": float(rmses.std()),
        "val_r2_mean": float(r2s.mean()),
        "val_r2_std": float(r2s.std()),
        "oof_summary": oof_summary,
        "split_info": split_info,
        "args": vars(args),
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[NewModel] Done. MAE={summary['val_mae_mean']:.3f} +/- {summary['val_mae_std']:.3f}")


if __name__ == "__main__":
    main()
