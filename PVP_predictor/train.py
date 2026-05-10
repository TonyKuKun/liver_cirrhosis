"""
K-Fold Cross-Validation Trainer for PVP Prediction
====================================================
- Stratified K-fold by `is_post_tips` (avoids TIPS-imbalanced folds)
- AdamW optimizer + cosine annealing schedule
- Early stopping based on validation MAE
- Saves: best model per fold, normalization stats, per-fold metrics

Usage
─────
    python train.py --data_root /path/to/patients \\
                    --out_dir   ./runs/v3_full \\
                    --n_folds 5 --epochs 300 --batch_size 8

Outputs
─────
    out_dir/
      ├── fold_0/best.pt, history.csv
      ├── fold_1/best.pt, history.csv
      ├── ...
      ├── normalization.pt        (profile_mean/std, aux_mean/std, label_mean/std)
      ├── splits.json             (which patient is in which fold)
      └── summary.json            (mean/std MAE, RMSE, R² across folds)
"""

import os
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold

from dataset import PortalVeinDataset, collate_fn
from diagnostics import (
    collect_prediction_rows,
    subject_id_from_name,
    summarize_prediction_rows,
    write_group_summary,
    write_prediction_csv,
)
from model import PortalPressureNet, PhysicsInformedLoss, count_params


# =====================================================================
# Checkpoint I/O
# =====================================================================
def safe_torch_save(obj, path):
    """Write checkpoints via a temporary file, then atomically replace target."""
    path = os.fspath(path)
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp_path = f"{path}.tmp-{os.getpid()}"
    try:
        torch.save(obj, tmp_path)
        last_error = None
        for _ in range(20):
            try:
                os.replace(tmp_path, path)
                last_error = None
                break
            except PermissionError as e:
                last_error = e
                try:
                    if os.path.exists(path):
                        os.chmod(path, 0o666)
                        os.remove(path)
                    os.replace(tmp_path, path)
                    last_error = None
                    break
                except PermissionError as inner:
                    last_error = inner
                    time.sleep(0.1)
        if last_error is not None:
            raise last_error
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# =====================================================================
# Metrics
# =====================================================================
def compute_metrics(preds, labels):
    """preds, labels: 1-D numpy arrays of REAL-SCALE PVP values (mmHg)."""
    err = preds - labels
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((labels - labels.mean()) ** 2) + 1e-9
    r2 = 1.0 - ss_res / ss_tot
    return float(mae), float(rmse), float(r2)


def make_cv_splits(data, n_folds, seed, split_mode="subject"):
    indices = np.arange(len(data))
    strat_y = np.array([int(d["is_post_tips"]) for d in data])
    min_class = min((strat_y == 0).sum(), (strat_y == 1).sum())

    if split_mode == "subject":
        groups = np.array([subject_id_from_name(d["name"]) for d in data])
        n_groups = len(set(groups))
        if n_groups < n_folds:
            raise RuntimeError(f"Need at least {n_folds} subject groups, have {n_groups}")
        if min_class >= n_folds:
            splitter = StratifiedGroupKFold(
                n_splits=n_folds, shuffle=True, random_state=seed
            )
            split_iter = splitter.split(indices, strat_y, groups)
            method = "StratifiedGroupKFold"
        else:
            splitter = GroupKFold(n_splits=n_folds)
            split_iter = splitter.split(indices, strat_y, groups)
            method = "GroupKFold"
        splits = [(train_idx, val_idx) for train_idx, val_idx in split_iter]
        return splits, {
            "split_mode": split_mode,
            "method": method,
            "n_subjects": int(n_groups),
            "post_tips": int((strat_y == 1).sum()),
            "pre_tips": int((strat_y == 0).sum()),
        }

    if split_mode == "sample":
        if min_class >= n_folds:
            splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            split_iter = splitter.split(indices, strat_y)
            method = "StratifiedKFold"
        else:
            from sklearn.model_selection import KFold
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            split_iter = splitter.split(indices)
            method = "KFold"
        splits = [(train_idx, val_idx) for train_idx, val_idx in split_iter]
        return splits, {
            "split_mode": split_mode,
            "method": method,
            "post_tips": int((strat_y == 1).sum()),
            "pre_tips": int((strat_y == 0).sum()),
        }

    raise ValueError("--split_mode must be either 'subject' or 'sample'")


# =====================================================================
# One epoch
# =====================================================================
def run_epoch(model, loader, criterion, device, mu_y, sigma_y,
              optimizer=None, scheduler=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_main = 0.0, 0.0
    loss_log_sum = {}
    preds_real, labels_real = [], []
    n_seen = 0

    for batch in loader:
        # Move tensors to device
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)
        bsz = batch['label_norm'].size(0)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            out = model(batch)
            L, log = criterion(out, batch['label_norm'], batch)

        if is_train:
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += L.item() * bsz
        total_main += log['main'] * bsz
        for k, v in log.items():
            loss_log_sum[k] = loss_log_sum.get(k, 0.0) + v * bsz

        # Real-scale predictions for metrics
        pvp_n = out['pvp_pred'].detach().squeeze(-1).cpu().numpy()
        preds_real.append(pvp_n * sigma_y + mu_y)
        labels_real.append(batch['label'].detach().cpu().numpy())
        n_seen += bsz

    if is_train and scheduler is not None:
        scheduler.step()

    preds_real = np.concatenate(preds_real)
    labels_real = np.concatenate(labels_real)
    mae, rmse, r2 = compute_metrics(preds_real, labels_real)

    avg_log = {k: v / max(n_seen, 1) for k, v in loss_log_sum.items()}
    avg_log['mae'] = mae
    avg_log['rmse'] = rmse
    avg_log['r2'] = r2
    return avg_log


# =====================================================================
# Train one fold
# =====================================================================
def train_fold(fold_idx, train_idx, val_idx, full_ds, args, device):
    out_fold = os.path.join(args.out_dir, f'fold_{fold_idx}')
    os.makedirs(out_fold, exist_ok=True)

    train_ds = Subset(full_ds, train_idx.tolist())
    val_ds   = Subset(full_ds, val_idx.tolist())
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate_fn, num_workers=0, drop_last=False)
    val_ld   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          collate_fn=collate_fn, num_workers=0, drop_last=False)

    model = PortalPressureNet(d_hidden=args.d_hidden, dropout=args.dropout).to(device)
    if fold_idx == 0:
        total, train = count_params(model)
        print(f"[Train] Model params: {total:,} (trainable {train:,})")

    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs,
                                  eta_min=args.lr * 0.01)
    criterion = PhysicsInformedLoss(
        lambda_murray=args.lambda_murray,
        lambda_press=args.lambda_press,
        lambda_smooth=args.lambda_smooth,
        lambda_physio=args.lambda_physio,
        lambda_mono=args.lambda_mono,
        huber_delta=args.huber_delta,
    ).to(device)

    mu_y = full_ds.label_mean
    sigma_y = full_ds.label_std

    best_val_mae = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    history_lines = ['epoch,phase,total,main,murray,press,smooth,physio,mono,mae,rmse,r2\n']

    for epoch in range(1, args.epochs + 1):
        train_log = run_epoch(model, train_ld, criterion, device, mu_y, sigma_y,
                              optimizer=optimizer, scheduler=scheduler)
        val_log   = run_epoch(model, val_ld,   criterion, device, mu_y, sigma_y,
                              optimizer=None, scheduler=None)

        for phase, log in (('train', train_log), ('val', val_log)):
            history_lines.append(
                f"{epoch},{phase},{log['total']:.5f},{log['main']:.5f},"
                f"{log.get('murray',0):.5f},{log.get('press',0):.5f},"
                f"{log.get('smooth',0):.5f},{log.get('physio',0):.5f},"
                f"{log.get('mono',0):.5f},{log['mae']:.4f},"
                f"{log['rmse']:.4f},{log['r2']:.4f}\n"
            )

        if val_log['mae'] < best_val_mae - 1e-4:
            best_val_mae = val_log['mae']
            best_epoch = epoch
            epochs_no_improve = 0
            safe_torch_save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_mae': best_val_mae,
                'val_rmse': val_log['rmse'],
                'val_r2': val_log['r2'],
                'args': vars(args),
            }, os.path.join(out_fold, 'best.pt'))
        else:
            epochs_no_improve += 1

        if epoch % args.print_every == 0 or epoch == 1:
            print(f"[Fold {fold_idx} | Ep {epoch:3d}] "
                  f"train={train_log['total']:.4f} "
                  f"(main={train_log['main']:.3f} murr={train_log['murray']:.3f} "
                  f"press={train_log['press']:.3f} smth={train_log['smooth']:.3f} "
                  f"phys={train_log['physio']:.3f}) mae={train_log['mae']:.2f} "
                  f"| val={val_log['total']:.4f} "
                  f"(main={val_log['main']:.3f} murr={val_log['murray']:.3f} "
                  f"press={val_log['press']:.3f} phys={val_log['physio']:.3f}) "
                  f"mae={val_log['mae']:.2f} r2={val_log['r2']:.2f} "
                  f"| best={best_val_mae:.2f}@{best_epoch}")

        if epochs_no_improve >= args.patience:
            print(f"[Fold {fold_idx}] Early stopping at epoch {epoch} "
                  f"(best mae={best_val_mae:.3f} @ ep {best_epoch})")
            break

    with open(os.path.join(out_fold, 'history.csv'), 'w') as f:
        f.writelines(history_lines)

    # Reload best for final report and OOF diagnostics
    ckpt = torch.load(os.path.join(out_fold, 'best.pt'), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    val_rows = collect_prediction_rows(
        model, val_ld, device, fold_idx, full_ds.label_mean, full_ds.label_std
    )
    write_prediction_csv(val_rows, os.path.join(out_fold, 'val_predictions.csv'))

    return {
        'fold':     fold_idx,
        'best_epoch': ckpt['epoch'],
        'val_mae':  ckpt['val_mae'],
        'val_rmse': ckpt['val_rmse'],
        'val_r2':   ckpt['val_r2'],
        'n_train':  len(train_idx),
        'n_val':    len(val_idx),
    }, val_rows


# =====================================================================
# Main
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, default=r"F:\PCG data\dataset\test4all_sample")
    ap.add_argument('--out_dir',   type=str, default='./runs/v3')
    ap.add_argument('--n_points',  type=int, default=100)
    ap.add_argument('--n_folds',   type=int, default=5)
    ap.add_argument('--seed',      type=int, default=42)
    ap.add_argument('--split_mode', choices=['subject', 'sample'], default='subject')
    # Optimization
    ap.add_argument('--epochs',       type=int,   default=300)
    ap.add_argument('--batch_size',   type=int,   default=8)
    ap.add_argument('--lr',           type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--patience',     type=int,   default=40)
    ap.add_argument('--print_every',  type=int,   default=10)
    # Model
    ap.add_argument('--d_hidden',     type=int,   default=32)
    ap.add_argument('--dropout',      type=float, default=0.3)
    # Loss weights
    ap.add_argument('--huber_delta',   type=float, default=1.0)
    ap.add_argument('--lambda_murray', type=float, default=0.10)
    ap.add_argument('--lambda_press',  type=float, default=0.05)
    ap.add_argument('--lambda_smooth', type=float, default=0.01)
    ap.add_argument('--lambda_physio', type=float, default=0.01)
    ap.add_argument('--lambda_mono',   type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Device: {device}")

    # Load full dataset (computes normalization once on the whole set)
    full_ds = PortalVeinDataset(args.data_root, n_points=args.n_points, verbose=True)
    if len(full_ds) < args.n_folds:
        raise RuntimeError(f"Need at least {args.n_folds} patients, have {len(full_ds)}")

    # Save normalization stats (loaded for inference)
    torch.save({
        'profile_mean': full_ds.profile_mean,
        'profile_std':  full_ds.profile_std,
        'aux_mean':     full_ds.aux_mean,
        'aux_std':      full_ds.aux_std,
        'label_mean':   full_ds.label_mean,
        'label_std':    full_ds.label_std,
    }, os.path.join(args.out_dir, 'normalization.pt'))

    splits, split_info = make_cv_splits(
        full_ds.data, args.n_folds, args.seed, split_mode=args.split_mode
    )
    strat_y = np.array([int(d['is_post_tips']) for d in full_ds.data])
    print(f"[Train] Using {split_info['method']} "
          f"(mode={split_info['split_mode']}, post-TIPS={split_info['post_tips']}, "
          f"pre-TIPS={split_info['pre_tips']})")

    splits_dict = {'split_info': split_info, 'folds': []}
    fold_results = []
    all_oof_rows = []
    for fi, (train_idx, val_idx) in enumerate(splits):
        train_names = [full_ds.data[i]['name'] for i in train_idx]
        val_names   = [full_ds.data[i]['name'] for i in val_idx]
        splits_dict['folds'].append({
            'fold': fi,
            'train_names': train_names,
            'val_names': val_names,
            'train_subject_ids': sorted({subject_id_from_name(n) for n in train_names}),
            'val_subject_ids': sorted({subject_id_from_name(n) for n in val_names}),
        })
        print(f"\n========== Fold {fi}/{args.n_folds-1} "
              f"(n_train={len(train_idx)}, n_val={len(val_idx)}, "
              f"val_post_tips={int(strat_y[val_idx].sum())}) ==========")
        res, val_rows = train_fold(fi, train_idx, val_idx, full_ds, args, device)
        fold_results.append(res)
        all_oof_rows.extend(val_rows)

    with open(os.path.join(args.out_dir, 'splits.json'), 'w') as f:
        json.dump(splits_dict, f, indent=2)
    write_prediction_csv(all_oof_rows, os.path.join(args.out_dir, 'oof_predictions.csv'))
    write_group_summary(all_oof_rows, os.path.join(args.out_dir, 'oof_group_summary.json'))

    # Aggregate metrics
    maes  = np.array([r['val_mae']  for r in fold_results])
    rmses = np.array([r['val_rmse'] for r in fold_results])
    r2s   = np.array([r['val_r2']   for r in fold_results])
    summary = {
        'n_folds':  args.n_folds,
        'fold_results': fold_results,
        'val_mae_mean':  float(maes.mean()),
        'val_mae_std':   float(maes.std()),
        'val_rmse_mean': float(rmses.mean()),
        'val_rmse_std':  float(rmses.std()),
        'val_r2_mean':   float(r2s.mean()),
        'val_r2_std':    float(r2s.std()),
        'split_info': split_info,
        'oof_group_summary': summarize_prediction_rows(all_oof_rows),
    }
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("\n========== Summary ==========")
    print(f"  MAE  : {summary['val_mae_mean']:.3f} ± {summary['val_mae_std']:.3f}")
    print(f"  RMSE : {summary['val_rmse_mean']:.3f} ± {summary['val_rmse_std']:.3f}")
    print(f"  R2   : {summary['val_r2_mean']:.3f} +/- {summary['val_r2_std']:.3f}")


if __name__ == '__main__':
    main()
