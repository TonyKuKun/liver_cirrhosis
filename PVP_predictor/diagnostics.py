import csv
import json
import os
import re
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset import AUX_KEYS, SEG_INDEX, PortalVeinDataset, collate_fn
from model import PortalPressureNet


PREDICTION_COLUMNS = [
    "fold",
    "name",
    "subject_id",
    "label",
    "pred",
    "err",
    "abs_err",
    "post_tips",
    "has_collateral",
    "has_lgv",
    "has_pgv",
    "has_rpv",
    "q_mpv",
    "q_lpv",
    "q_rpv",
    "q_tips",
    "tips_fraction",
    "collateral_fraction",
    "liver_fraction",
]


def subject_id_from_name(name):
    """Remove leading date and TIPS suffix so paired samples share an id."""
    core = re.sub(r"^\d+", "", str(name))
    return core.split("#", 1)[0]


def _as_float(x):
    return float(x.detach().cpu().item() if torch.is_tensor(x) else x)


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
            "post_tips": int(_as_float(batch["is_post_tips"][i]) > 0.5),
            "has_collateral": int(aux[i, AUX_KEYS.index("has_compensation_vessel")] > 0.5),
            "has_lgv": int(aux[i, AUX_KEYS.index("has_lgv")] > 0.5),
            "has_pgv": int(aux[i, AUX_KEYS.index("has_pgv")] > 0.5),
            "has_rpv": int(seg_mask[i, SEG_INDEX["rpv"]] > 0.5),
            "q_mpv": float(q[i, SEG_INDEX["mpv"]]),
            "q_lpv": float(q[i, SEG_INDEX["lpv"]]),
            "q_rpv": float(q[i, SEG_INDEX["rpv"]]),
            "q_tips": float(q[i, SEG_INDEX["tips"]]),
            "tips_fraction": float(tips_fraction[i]),
            "collateral_fraction": float(collateral_fraction[i]),
            "liver_fraction": float(liver_fraction[i]),
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


def write_prediction_csv(rows, path):
    path = os.fspath(path)
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in PREDICTION_COLUMNS})


def _metrics(rows):
    if not rows:
        return {"n": 0, "mae": None, "rmse": None, "bias": None, "label_mean": None, "pred_mean": None}
    err = np.array([float(r["err"]) for r in rows], dtype=float)
    labels = np.array([float(r["label"]) for r in rows], dtype=float)
    preds = np.array([float(r["pred"]) for r in rows], dtype=float)
    return {
        "n": int(len(rows)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
        "label_mean": float(np.mean(labels)),
        "pred_mean": float(np.mean(preds)),
    }


def summarize_prediction_rows(rows):
    subject_counts = Counter(r["subject_id"] for r in rows)
    summary = {
        "overall": _metrics(rows),
        "folds": {},
        "groups": {},
        "subject_stats": {
            "n_subjects": int(len(subject_counts)),
            "n_repeated_subjects": int(sum(1 for c in subject_counts.values() if c > 1)),
            "repeated_subjects": sorted([s for s, c in subject_counts.items() if c > 1]),
        },
    }

    for fold in sorted(set(int(r["fold"]) for r in rows)):
        summary["folds"][str(fold)] = _metrics([r for r in rows if int(r["fold"]) == fold])

    for key in ["post_tips", "has_collateral", "has_lgv", "has_pgv", "has_rpv"]:
        for value in [0, 1]:
            group_rows = [r for r in rows if int(r[key]) == value]
            summary["groups"][f"{key}={value}"] = _metrics(group_rows)

    return summary


def write_group_summary(rows, path):
    path = os.fspath(path)
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summarize_prediction_rows(rows), f, indent=2, ensure_ascii=False)


def load_trained_dataset(checkpoint_dir, data_root, n_points=100, verbose=False):
    ds = PortalVeinDataset(data_root, n_points=n_points, verbose=verbose)
    norm = torch.load(os.path.join(checkpoint_dir, "normalization.pt"),
                      map_location="cpu", weights_only=False)
    ds.profile_mean = norm["profile_mean"]
    ds.profile_std = norm["profile_std"]
    ds.aux_mean = norm["aux_mean"]
    ds.aux_std = norm["aux_std"]
    ds.label_mean = norm["label_mean"]
    ds.label_std = norm["label_std"]
    return ds


def collect_oof_predictions(checkpoint_dir, data_root, n_points=100, batch_size=8, device="cpu"):
    ds = load_trained_dataset(checkpoint_dir, data_root, n_points=n_points, verbose=False)
    with open(os.path.join(checkpoint_dir, "splits.json"), "r", encoding="utf-8") as f:
        splits = json.load(f)["folds"]
    name_to_idx = {d["name"]: i for i, d in enumerate(ds.data)}

    all_rows = []
    for fold in splits:
        fold_idx = int(fold["fold"])
        val_idx = [name_to_idx[n] for n in fold["val_names"] if n in name_to_idx]
        ckpt = torch.load(os.path.join(checkpoint_dir, f"fold_{fold_idx}", "best.pt"),
                          map_location=device, weights_only=False)
        args = ckpt.get("args", {})
        model = PortalPressureNet(
            d_hidden=args.get("d_hidden", 32),
            dropout=args.get("dropout", 0.3),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        loader = DataLoader(Subset(ds, val_idx), batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0, drop_last=False)
        all_rows.extend(collect_prediction_rows(
            model, loader, device, fold_idx, ds.label_mean, ds.label_std
        ))

    return all_rows
