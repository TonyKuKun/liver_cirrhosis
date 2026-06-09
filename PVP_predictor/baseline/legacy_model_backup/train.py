"""
K-Fold Cross-Validation Trainer for PVP Prediction
====================================================
v4.1 — PVP-stratified splits: each fold's train AND val cover
the full PVP range (low/mid/high), preventing the "never seen
a high PVP during training" failure mode.
"""

import os
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import KFold  # only used as fallback

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
    err = preds - labels
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((labels - labels.mean()) ** 2) + 1e-9
    r2 = 1.0 - ss_res / ss_tot
    return float(mae), float(rmse), float(r2)


# =====================================================================
# PVP-BALANCED SPLITS (greedy partition)
# =====================================================================
def make_cv_splits(data, n_folds, seed, split_mode="subject"):
    """
    Greedy balanced group K-fold by PVP value.

    Algorithm (like dealing poker):
      1. Group samples by subject (pre/post TIPS pairs stay together)
      2. Sort subjects by max PVP, descending
      3. Deal each subject to the fold with the currently lowest PVP sum

    This GUARANTEES each fold gets a mix of high/mid/low PVP subjects.
    No bin discretization, no StratifiedGroupKFold approximation — just
    direct numerical balancing.
    """
    rng = np.random.RandomState(seed)
    indices = np.arange(len(data))

    # ── Group samples by subject ──────────────────────────
    subject_samples = {}  # sid → [sample_indices]
    subject_max_pvp = {}  # sid → max PVP
    for i, d in enumerate(data):
        sid = subject_id_from_name(d['name'])
        subject_samples.setdefault(sid, []).append(i)
        subject_max_pvp[sid] = max(subject_max_pvp.get(sid, -np.inf), float(d['label']))

    subjects = sorted(subject_samples.keys())
    n_subjects = len(subjects)
    if n_subjects < n_folds:
        raise RuntimeError(f"Need ≥{n_folds} subjects, have {n_subjects}")

    # ── Sort by max PVP descending, with tie-breaking shuffle ──
    # Shuffle first so equal-PVP subjects get random order
    rng.shuffle(subjects)
    subjects.sort(key=lambda s: -subject_max_pvp[s])

    # ── Greedy deal: assign each subject to the fold with lowest PVP sum ──
    fold_pvp_sum = np.zeros(n_folds)
    fold_assignment = {}  # sid → fold_idx

    for sid in subjects:
        # Pick the fold with the lowest total PVP so far
        target_fold = int(np.argmin(fold_pvp_sum))
        fold_assignment[sid] = target_fold
        fold_pvp_sum[target_fold] += subject_max_pvp[sid]

    # ── Build (train_idx, val_idx) pairs ──────────────────
    splits = []
    for fi in range(n_folds):
        val_sids = {s for s, f in fold_assignment.items() if f == fi}
        val_idx = np.array([i for i in indices
                            if subject_id_from_name(data[i]['name']) in val_sids])
        train_idx = np.array([i for i in indices if i not in set(val_idx)])
        splits.append((train_idx, val_idx))

    # ── Summary info ──────────────────────────────────────
    labels = np.array([float(d['label']) for d in data])
    fold_stats = []
    for fi, (tr, va) in enumerate(splits):
        fold_stats.append({
            'fold': fi,
            'val_mean_pvp': float(labels[va].mean()),
            'val_std_pvp':  float(labels[va].std()),
            'val_min':      float(labels[va].min()),
            'val_max':      float(labels[va].max()),
        })

    return splits, {
        "split_mode": split_mode,
        "method": "GreedyBalancedGroupKFold(PVP)",
        "n_subjects": n_subjects,
        "n_folds": n_folds,
        "post_tips": int(sum(1 for d in data if d['is_post_tips'])),
        "pre_tips":  int(sum(1 for d in data if not d['is_post_tips'])),
        "fold_stats": fold_stats,
    }


def _print_fold_pvp_distribution(data, splits, label='PVP'):
    """Print per-fold PVP distribution so the user can verify balance."""
    print(f"\n[Splits] Per-fold {label} distribution:")
    labels = np.array([float(d['label']) for d in data])
    for fi, (train_idx, val_idx) in enumerate(splits):
        tr_l = labels[train_idx]
        va_l = labels[val_idx]
        tr_lo = (tr_l < 20).sum(); tr_hi = (tr_l >= 30).sum()
        va_lo = (va_l < 20).sum(); va_hi = (va_l >= 30).sum()
        print(f"  Fold {fi}: train n={len(train_idx):2d} "
              f"(<20:{tr_lo} 20-30:{len(train_idx)-tr_lo-tr_hi} >30:{tr_hi}) "
              f"[{tr_l.min():.0f},{tr_l.max():.0f}]  |  "
              f"val n={len(val_idx):2d} "
              f"(<20:{va_lo} 20-30:{len(val_idx)-va_lo-va_hi} >30:{va_hi}) "
              f"[{va_l.min():.0f},{va_l.max():.0f}]")


# =====================================================================
# Extreme-value weighted sampling (optional, mild)
# =====================================================================
def _make_sampler(full_ds, train_idx, power=1.5):
    """
    Optional: oversample extreme PVP during training.
    power=1.5 is mild; set power=0 to disable (plain shuffle).
    """
    if power <= 0:
        return None  # caller should use shuffle=True
    labels = np.array([full_ds.data[i]['label'] for i in train_idx])
    median = np.median(labels)
    std = max(np.std(labels), 1e-6)
    dist = np.abs(labels - median) / std
    weights = 1.0 + dist ** power
    weights = weights / weights.mean()
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=len(train_idx),
        replacement=True,
    )


# =====================================================================
# One epoch
# =====================================================================
def run_epoch(model, loader, criterion, device, mu_y, sigma_y,
              optimizer=None, scheduler=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    loss_log_sum = {}
    preds_real, labels_real = [], []
    n_seen = 0

    for batch in loader:
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
        for k, v in log.items():
            loss_log_sum[k] = loss_log_sum.get(k, 0.0) + v * bsz

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

    # Extreme-value oversampling (mild)
    sampler = _make_sampler(full_ds, train_idx.tolist(), power=args.sample_power)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size,
                          sampler=sampler if sampler else None,
                          shuffle=(sampler is None),
                          collate_fn=collate_fn, num_workers=0, drop_last=False)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0, drop_last=False)

    model = PortalPressureNet(
        d_hidden=args.d_hidden, dropout=args.dropout,
        gnn_layers=args.gnn_layers, use_residual=args.use_residual,
        use_q_scale=args.use_q_scale,
        use_physics_baseline=args.use_physics_baseline,
        use_aux=args.use_aux,
        use_flow_features=args.use_flow_features,
        use_branch_embed=args.use_branch_embed,
        use_profile_transformer=args.use_profile_transformer,
        use_tips_head=args.use_tips_head,
        use_aux_mask=args.use_aux_mask,
        physics_mode=args.physics_mode,
    ).to(device)
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
        lambda_spread=args.lambda_spread,
        extremity_alpha=args.extremity_alpha,
        post_tips_high_alpha=args.post_tips_high_alpha,
        post_tips_high_threshold=args.post_tips_high_threshold,
        lambda_residual=args.lambda_residual,
        huber_delta=args.huber_delta,
    ).to(device)

    mu_y = full_ds.label_mean
    sigma_y = full_ds.label_std

    best_val_mae = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    history_lines = ['epoch,phase,total,main,murray,press,smooth,physio,mono,spread,mae,rmse,r2\n']

    for epoch in range(1, args.epochs + 1):
        train_log = run_epoch(model, train_ld, criterion, device, mu_y, sigma_y,
                              optimizer=optimizer, scheduler=scheduler)
        val_log   = run_epoch(model, val_ld,   criterion, device, mu_y, sigma_y)

        for phase, log in (('train', train_log), ('val', val_log)):
            history_lines.append(
                f"{epoch},{phase},{log['total']:.5f},{log['main']:.5f},"
                f"{log.get('murray',0):.5f},{log.get('press',0):.5f},"
                f"{log.get('smooth',0):.5f},{log.get('physio',0):.5f},"
                f"{log.get('mono',0):.5f},{log.get('spread',0):.5f},"
                f"{log['mae']:.4f},{log['rmse']:.4f},{log['r2']:.4f}\n"
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
                  f"sprd={train_log.get('spread',0):.3f}) "
                  f"mae={train_log['mae']:.2f} "
                  f"| val={val_log['total']:.4f} mae={val_log['mae']:.2f} "
                  f"r2={val_log['r2']:.2f} | best={best_val_mae:.2f}@{best_epoch}")

        if epochs_no_improve >= args.patience:
            print(f"[Fold {fold_idx}] Early stop at ep {epoch} "
                  f"(best mae={best_val_mae:.3f} @ ep {best_epoch})")
            break

    with open(os.path.join(out_fold, 'history.csv'), 'w') as f:
        f.writelines(history_lines)

    # Reload best for OOF diagnostics
    ckpt = torch.load(os.path.join(out_fold, 'best.pt'), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    val_rows = collect_prediction_rows(
        model, val_ld, device, fold_idx, full_ds.label_mean, full_ds.label_std)
    write_prediction_csv(val_rows, os.path.join(out_fold, 'val_predictions.csv'))

    return {
        'fold': fold_idx,
        'best_epoch': ckpt['epoch'],
        'val_mae': ckpt['val_mae'],
        'val_rmse': ckpt['val_rmse'],
        'val_r2': ckpt['val_r2'],
        'n_train': len(train_idx),
        'n_val': len(val_idx),
    }, val_rows


# =====================================================================
# Main
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, default=r"F:\PCG data\dataset\test4all_sample")
    ap.add_argument('--out_dir',   type=str, default='./runs/v5.3')
    ap.add_argument('--n_points',  type=int, default=200)
    ap.add_argument('--n_folds',   type=int, default=5)
    ap.add_argument('--seed',      type=int, default=40)
    ap.add_argument('--split_mode', choices=['subject', 'sample'], default='subject')
    ap.add_argument('--include_00_prefix_samples', action='store_true', default=False,
                    help='Include sample folders whose names start with 00.')
    ap.add_argument('--exclude_00_prefix_samples', dest='include_00_prefix_samples',
                    action='store_false',
                    help='Exclude sample folders whose names start with 00.')
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
    ap.add_argument('--gnn_layers',   type=int,   default=2)
    ap.add_argument('--use_residual', action='store_true', default=True)
    ap.add_argument('--no_residual',  dest='use_residual', action='store_false')
    ap.add_argument('--use_q_scale', action='store_true', default=True)
    ap.add_argument('--no_q_scale',  dest='use_q_scale', action='store_false')
    ap.add_argument('--use_physics_baseline', action='store_true', default=False)
    ap.add_argument('--no_physics_baseline',  dest='use_physics_baseline', action='store_false')
    ap.add_argument('--physics_mode', choices=['none', 'fixed', 'learnable'], default=None,
                    help='Physics anchor: none, fixed Poiseuille baseline, or learnable reduced-order calibration.')
    ap.add_argument('--use_aux', action='store_true', default=True)
    ap.add_argument('--no_aux',  dest='use_aux', action='store_false')
    ap.add_argument('--use_flow_features', action='store_true', default=False)
    ap.add_argument('--no_flow_features',  dest='use_flow_features', action='store_false')
    ap.add_argument('--use_branch_embed', action='store_true', default=True)
    ap.add_argument('--no_branch_embed',  dest='use_branch_embed', action='store_false')
    ap.add_argument('--use_profile_transformer', action='store_true', default=True)
    ap.add_argument('--no_profile_transformer',  dest='use_profile_transformer', action='store_false')
    ap.add_argument('--use_tips_head', action='store_true', default=True)
    ap.add_argument('--no_tips_head',  dest='use_tips_head', action='store_false')
    ap.add_argument('--use_aux_mask', action='store_true', default=True)
    ap.add_argument('--no_aux_mask',  dest='use_aux_mask', action='store_false')
    # Loss weights
    ap.add_argument('--huber_delta',      type=float, default=1.0)
    ap.add_argument('--lambda_murray',    type=float, default=0.0)
    ap.add_argument('--lambda_press',     type=float, default=0.0)
    ap.add_argument('--lambda_smooth',    type=float, default=0.0)
    ap.add_argument('--lambda_physio',    type=float, default=0.0)
    ap.add_argument('--lambda_mono',      type=float, default=0.0)
    ap.add_argument('--lambda_spread',    type=float, default=0.0)
    ap.add_argument('--extremity_alpha',  type=float, default=1.0)
    ap.add_argument('--post_tips_high_alpha', type=float, default=0.0)
    ap.add_argument('--post_tips_high_threshold', type=float, default=0.5)
    ap.add_argument('--lambda_residual',  type=float, default=0.0)
    # Sampling
    ap.add_argument('--sample_power',     type=float, default=1.5,
                    help='Extreme-value oversampling power (0=disabled)')
    args = ap.parse_args()
    if args.physics_mode is None:
        args.physics_mode = 'fixed' if args.use_physics_baseline else 'none'
    args.use_physics_baseline = args.physics_mode != 'none'

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Device: {device}")

    full_ds = PortalVeinDataset(
        args.data_root,
        n_points=args.n_points,
        verbose=True,
        include_00_prefix_samples=args.include_00_prefix_samples,
    )
    if len(full_ds) < args.n_folds:
        raise RuntimeError(f"Need ≥{args.n_folds} patients, have {len(full_ds)}")

    torch.save({
        'profile_mean': full_ds.profile_mean,
        'profile_std':  full_ds.profile_std,
        'aux_mean':     full_ds.aux_mean,
        'aux_std':      full_ds.aux_std,
        'label_mean':   full_ds.label_mean,
        'label_std':    full_ds.label_std,
    }, os.path.join(args.out_dir, 'normalization.pt'))

    splits, split_info = make_cv_splits(
        full_ds.data, args.n_folds, args.seed, split_mode=args.split_mode)

    print(f"[Train] Split method: {split_info['method']}")
    print(f"  Stratified by: {split_info.get('stratify_by', 'N/A')}")
    print(f"  Bin edges: {split_info.get('bin_edges', [])}")
    _print_fold_pvp_distribution(full_ds.data, splits)

    data_filter = {
        'include_00_prefix_samples': bool(args.include_00_prefix_samples),
        'n_00_prefix_excluded': int(len(getattr(full_ds, 'excluded_00_prefix_names', []))),
        'excluded_00_prefix_names': list(getattr(full_ds, 'excluded_00_prefix_names', [])),
    }
    splits_dict = {'split_info': split_info, 'data_filter': data_filter, 'folds': []}
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
        # Per-fold PVP stats
        train_labels = [full_ds.data[i]['label'] for i in train_idx]
        val_labels = [full_ds.data[i]['label'] for i in val_idx]
        print(f"\n{'='*60}")
        print(f"Fold {fi}/{args.n_folds-1}: "
              f"train n={len(train_idx)} [{min(train_labels):.0f}-{max(train_labels):.0f}], "
              f"val n={len(val_idx)} [{min(val_labels):.0f}-{max(val_labels):.0f}]")
        print(f"{'='*60}")
        res, val_rows = train_fold(fi, train_idx, val_idx, full_ds, args, device)
        fold_results.append(res)
        all_oof_rows.extend(val_rows)

    with open(os.path.join(args.out_dir, 'splits.json'), 'w') as f:
        json.dump(splits_dict, f, indent=2)
    write_prediction_csv(all_oof_rows, os.path.join(args.out_dir, 'oof_predictions.csv'))
    write_group_summary(all_oof_rows, os.path.join(args.out_dir, 'oof_group_summary.json'))

    maes  = np.array([r['val_mae']  for r in fold_results])
    rmses = np.array([r['val_rmse'] for r in fold_results])
    r2s   = np.array([r['val_r2']   for r in fold_results])
    summary = {
        'n_folds': args.n_folds,
        'n_samples': len(full_ds),
        'data_filter': data_filter,
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
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  MAE  : {summary['val_mae_mean']:.3f} ± {summary['val_mae_std']:.3f}")
    print(f"  RMSE : {summary['val_rmse_mean']:.3f} ± {summary['val_rmse_std']:.3f}")
    print(f"  R²   : {summary['val_r2_mean']:.3f} ± {summary['val_r2_std']:.3f}")


if __name__ == '__main__':
    main()
