"""
Training Pipeline for Portal Vein Pressure Prediction
======================================================

Features:
    - K-fold cross-validation (small dataset friendly)
    - Early stopping with patience
    - Physics-informed loss with continuity + monotonicity regularization
    - Comprehensive logging and visualization
    - Ablation study support
    - Attention weight visualization

Usage:
    python train.py --data_root /path/to/data --n_folds 5 --epochs 300
"""

import os
import sys
import json
import argparse
import random
import warnings
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

from dataset import PortalVeinDataset, BRANCHES
from model import PortalPressureNet, PhysicsInformedLoss, count_params

warnings.filterwarnings('ignore')


# =====================================================================
# Reproducibility
# =====================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# Custom collate function
# =====================================================================
def collate_fn(batch):
    """Stack batch items into tensors."""
    return {
        'profiles_raw':  torch.stack([b['profiles_raw'] for b in batch]),
        'profiles_norm': torch.stack([b['profiles_norm'] for b in batch]),
        'arc_lengths':   torch.stack([b['arc_lengths'] for b in batch]),
        'branch_mask':   torch.stack([b['branch_mask'] for b in batch]),
        'stat_features': torch.stack([b['stat_features'] for b in batch]),
        'label':         torch.stack([b['label'] for b in batch]),
        'name':          [b['name'] for b in batch],
    }


# =====================================================================
# Trainer
# =====================================================================
class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Trainer] Device: {self.device}")

        # ── Output directory ─────────────────────────────────────────
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
        os.makedirs(self.output_dir, exist_ok=True)

        # Save args
        with open(os.path.join(self.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)

    def build_model(self):
        model = PortalPressureNet(
            d_hidden=self.args.d_hidden,
            dropout=self.args.dropout,
        ).to(self.device)
        return model

    def train_one_epoch(self, model, loader, optimizer, criterion, scheduler=None):
        model.train()
        epoch_losses = defaultdict(float)
        n_batches = 0

        for batch in loader:
            profiles_raw  = batch['profiles_raw'].to(self.device)
            profiles_norm = batch['profiles_norm'].to(self.device)
            arc_lengths   = batch['arc_lengths'].to(self.device)
            branch_mask   = batch['branch_mask'].to(self.device)
            stat_features = batch['stat_features'].to(self.device)
            labels        = batch['label'].to(self.device)

            optimizer.zero_grad()

            pvp_pred, attn_w, corrected = model(
                profiles_raw, profiles_norm, arc_lengths, branch_mask, stat_features
            )

            loss, loss_dict = criterion(
                pvp_pred, labels, corrected, branch_mask, profiles_raw
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k, v in loss_dict.items():
                epoch_losses[k] += v
            n_batches += 1

        if scheduler is not None:
            scheduler.step()

        return {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}

    @torch.no_grad()
    def evaluate(self, model, loader):
        model.eval()
        all_preds = []
        all_labels = []
        all_names = []
        all_attn = []

        for batch in loader:
            profiles_raw  = batch['profiles_raw'].to(self.device)
            profiles_norm = batch['profiles_norm'].to(self.device)
            arc_lengths   = batch['arc_lengths'].to(self.device)
            branch_mask   = batch['branch_mask'].to(self.device)
            stat_features = batch['stat_features'].to(self.device)

            pvp_pred, attn_w, _ = model(
                profiles_raw, profiles_norm, arc_lengths, branch_mask, stat_features
            )

            all_preds.append(pvp_pred.cpu().squeeze(-1))
            all_labels.append(batch['label'])
            all_names.extend(batch['name'])
            all_attn.append(attn_w.cpu())

        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        attn = torch.cat(all_attn).numpy()

        # Guard against NaN in predictions
        nan_mask = ~np.isfinite(preds)
        if nan_mask.any():
            print(f"  [Warning] {nan_mask.sum()} NaN predictions replaced with mean of valid preds")
            valid_mean = np.nanmean(preds) if np.isfinite(np.nanmean(preds)) else np.mean(labels)
            preds[nan_mask] = valid_mean

        metrics = self.compute_metrics(labels, preds)

        return preds, labels, all_names, attn, metrics

    @staticmethod
    def compute_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # Pearson correlation
        if len(y_true) > 2:
            r_val, p_val = stats.pearsonr(y_true, y_pred)
        else:
            r_val, p_val = 0.0, 1.0

        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Pearson_r': r_val,
            'Pearson_p': p_val,
            'n_samples': len(y_true),
        }

    def run_kfold(self, dataset):
        """Run K-fold or Leave-One-Out cross-validation."""
        n_samples = len(dataset)
        args = self.args

        if args.n_folds == -1 or args.n_folds >= n_samples:
            # Leave-One-Out
            splitter = LeaveOneOut()
            n_splits = n_samples
            print(f"\n[CV] Leave-One-Out cross-validation ({n_samples} folds)")
        else:
            splitter = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
            n_splits = args.n_folds
            print(f"\n[CV] {args.n_folds}-fold cross-validation")

        # ── Collect all fold results ─────────────────────────────────
        all_fold_preds = np.zeros(n_samples)
        all_fold_labels = np.zeros(n_samples)
        all_fold_names = [''] * n_samples
        all_fold_attn = {}
        fold_metrics = []

        for fold_i, (train_idx, val_idx) in enumerate(splitter.split(range(n_samples))):
            print(f"\n{'='*60}")
            print(f"  Fold {fold_i+1}/{n_splits}  |  Train: {len(train_idx)}  Val: {len(val_idx)}")
            print(f"{'='*60}")

            # ── Data loaders ─────────────────────────────────────────
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            train_loader = DataLoader(
                train_subset, batch_size=args.batch_size, shuffle=True,
                collate_fn=collate_fn, drop_last=False,
            )
            val_loader = DataLoader(
                val_subset, batch_size=len(val_idx), shuffle=False,
                collate_fn=collate_fn,
            )

            # ── Model, optimizer, scheduler, criterion ───────────────
            model = self.build_model()
            optimizer = optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
            )
            criterion = PhysicsInformedLoss(
                lambda_cont=args.lambda_cont,
                lambda_mono=args.lambda_mono,
                use_huber=True,
            )

            # ── Training loop with early stopping ────────────────────
            best_val_loss = float('inf')
            best_state = None
            patience_counter = 0

            for epoch in range(1, args.epochs + 1):
                train_losses = self.train_one_epoch(
                    model, train_loader, optimizer, criterion, scheduler
                )

                # Evaluate on validation
                val_preds, val_labels, val_names, val_attn, val_metrics = \
                    self.evaluate(model, val_loader)

                val_loss = val_metrics['MAE']

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_metrics = val_metrics
                    best_preds = val_preds
                    best_attn = val_attn
                    patience_counter = 0
                else:
                    patience_counter += 1

                if epoch % args.log_every == 0 or epoch == 1:
                    alphas = {name: model.branch_encoders[name].residual_module.alpha.item()
                              for name in BRANCHES}
                    print(f"  Epoch {epoch:3d} | "
                          f"Train: {train_losses['total']:.4f} "
                          f"(main={train_losses['main']:.4f}, "
                          f"cont={train_losses['continuity']:.4f}) | "
                          f"Val MAE: {val_metrics['MAE']:.2f} "
                          f"R²: {val_metrics['R2']:.3f} | "
                          f"α=[{alphas['mpv']:.3f},{alphas['sv']:.3f},{alphas['smv']:.3f}]")

                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

            # ── Store fold results ───────────────────────────────────
            model.load_state_dict(best_state)

            # Re-evaluate with best model
            val_preds, val_labels, val_names, val_attn, val_metrics = \
                self.evaluate(model, val_loader)

            print(f"\n  Fold {fold_i+1} Best:  MAE={val_metrics['MAE']:.2f}  "
                  f"RMSE={val_metrics['RMSE']:.2f}  R²={val_metrics['R2']:.3f}  "
                  f"r={val_metrics['Pearson_r']:.3f}")

            # Map predictions back to global indices
            for local_i, global_i in enumerate(val_idx):
                all_fold_preds[global_i] = val_preds[local_i]
                all_fold_labels[global_i] = val_labels[local_i]
                all_fold_names[global_i] = val_names[local_i]
                all_fold_attn[val_names[local_i]] = val_attn[local_i]

            fold_metrics.append(val_metrics)

            # Save fold model
            fold_dir = os.path.join(self.output_dir, f'fold_{fold_i+1}')
            os.makedirs(fold_dir, exist_ok=True)
            torch.save(best_state, os.path.join(fold_dir, 'model.pt'))

        # ── Overall cross-validation results ─────────────────────────
        overall_metrics = self.compute_metrics(all_fold_labels, all_fold_preds)
        print(f"\n{'='*60}")
        print(f"  OVERALL CV RESULTS ({n_splits} folds)")
        print(f"{'='*60}")
        print(f"  MAE  = {overall_metrics['MAE']:.2f} mmHg")
        print(f"  RMSE = {overall_metrics['RMSE']:.2f} mmHg")
        print(f"  R²   = {overall_metrics['R2']:.3f}")
        print(f"  r    = {overall_metrics['Pearson_r']:.3f} (p={overall_metrics['Pearson_p']:.4f})")

        # ── Save results ─────────────────────────────────────────────
        results = {
            'overall': overall_metrics,
            'per_fold': fold_metrics,
            'predictions': {
                name: {'pred': float(pred), 'true': float(label)}
                for name, pred, label in zip(all_fold_names, all_fold_preds, all_fold_labels)
            },
        }

        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(os.path.join(self.output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=convert)

        # ── Save predictions table ───────────────────────────────────
        with open(os.path.join(self.output_dir, 'predictions.csv'), 'w') as f:
            f.write('patient,true_pvp,pred_pvp,error\n')
            for name, pred, label in zip(all_fold_names, all_fold_preds, all_fold_labels):
                f.write(f'{name},{label:.2f},{pred:.2f},{abs(pred-label):.2f}\n')

        # ── Save attention weights ───────────────────────────────────
        np.savez(
            os.path.join(self.output_dir, 'attention_weights.npz'),
            **{k: v for k, v in all_fold_attn.items()},
        )

        # ── Visualization ────────────────────────────────────────────
        self.plot_results(all_fold_labels, all_fold_preds, all_fold_names, all_fold_attn)

        return overall_metrics

    def plot_results(self, labels, preds, names, attn_dict):
        """Generate result plots."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("[Trainer] matplotlib not available, skipping plots.")
            return

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # ── 1. Scatter plot: predicted vs true ───────────────────────
        ax = axes[0]
        ax.scatter(labels, preds, c='steelblue', alpha=0.7, edgecolors='k', s=60)
        lims = [min(labels.min(), preds.min()) - 2,
                max(labels.max(), preds.max()) + 2]
        ax.plot(lims, lims, 'r--', lw=1.5, label='Identity')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('True PVP (mmHg)', fontsize=12)
        ax.set_ylabel('Predicted PVP (mmHg)', fontsize=12)
        r_val = stats.pearsonr(labels, preds)[0]
        ax.set_title(f'Prediction vs True (r={r_val:.3f})', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ── 2. Bland-Altman plot ─────────────────────────────────────
        ax = axes[1]
        mean_vals = (labels + preds) / 2
        diff_vals = preds - labels
        mean_diff = diff_vals.mean()
        std_diff = diff_vals.std()

        ax.scatter(mean_vals, diff_vals, c='steelblue', alpha=0.7, edgecolors='k', s=60)
        ax.axhline(mean_diff, color='red', lw=1.5, label=f'Mean: {mean_diff:.2f}')
        ax.axhline(mean_diff + 1.96 * std_diff, color='gray', ls='--', lw=1)
        ax.axhline(mean_diff - 1.96 * std_diff, color='gray', ls='--', lw=1)
        ax.set_xlabel('Mean of True & Predicted (mmHg)', fontsize=12)
        ax.set_ylabel('Predicted - True (mmHg)', fontsize=12)
        ax.set_title('Bland-Altman Plot', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ── 3. Attention heatmap (mean across patients) ──────────────
        ax = axes[2]
        attn_matrix = []
        for name in names:
            if name in attn_dict:
                attn_matrix.append(attn_dict[name])  # (3, N)
        if attn_matrix:
            attn_mean = np.stack(attn_matrix).mean(axis=0)  # (3, N)
            im = ax.imshow(attn_mean, aspect='auto', cmap='hot',
                           interpolation='bilinear')
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(['MPV', 'SV', 'SMV'])
            ax.set_xlabel('Centerline Position (normalized)', fontsize=12)
            ax.set_title('Mean Attention Weights', fontsize=13)
            plt.colorbar(im, ax=ax, shrink=0.8)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'results.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n[Trainer] Results saved to {self.output_dir}/")

    # ── Ablation study ───────────────────────────────────────────────
    def run_ablation(self, dataset):
        """
        Ablation study:
            1. Full model (physics prior + residual + stat features)
            2. No physics residual (physics prior only + stat features)
            3. No physics at all (raw geometry + stat features)
            4. No stat features (physics prior + residual only)
            5. Stat features only (no profile features)
        """
        print("\n" + "=" * 60)
        print("  ABLATION STUDY")
        print("=" * 60)

        ablation_results = {}

        # Run full model
        print("\n[Ablation 1/5] Full model")
        set_seed(42)
        metrics = self.run_kfold(dataset)
        ablation_results['full_model'] = metrics

        # TODO: implement other ablation variants by modifying model forward pass
        # These would require additional flags in the model, e.g.:
        #   - skip_residual=True  → use physics_norm directly, skip Module 2
        #   - skip_physics=True   → use geo_norm only, no physics layer
        #   - skip_stat=True      → zero out stat_features
        #   - skip_profile=True   → zero out branch embeddings

        print("\n[Ablation] Full model results:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")

        return ablation_results


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description='Portal Vein Pressure Prediction')

    # Data
    parser.add_argument('--data_root', type=str, default='F:\PCG data\dataset\zhengzhou_vkan_qian47',
                        help='Root folder with patient subfolders')
    parser.add_argument('--label_key', type=str, default='PVP',
                        choices=['PVP', 'PCG'])
    parser.add_argument('--n_points', type=int, default=100,
                        help='Profile resample length')

    # Model
    parser.add_argument('--d_hidden', type=int, default=32,
                        help='Hidden dimension (keep small for small datasets)')
    parser.add_argument('--dropout', type=float, default=0.3)

    # Training
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')

    # Physics loss
    parser.add_argument('--lambda_cont', type=float, default=0.1,
                        help='Continuity loss weight')
    parser.add_argument('--lambda_mono', type=float, default=0.05,
                        help='Monotonicity loss weight')

    # Cross-validation
    parser.add_argument('--n_folds', type=int, default=5,
                        help='K for K-fold CV; -1 for LOO')

    # Output
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory')
    parser.add_argument('--log_every', type=int, default=20)

    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ablation', action='store_true',
                        help='Run ablation study')

    args = parser.parse_args()
    set_seed(args.seed)

    # ── Load dataset ─────────────────────────────────────────────────
    print(f"\n[Main] Loading dataset from: {args.data_root}")
    dataset = PortalVeinDataset(
        root_dir=args.data_root,
        n_points=args.n_points,
        label_key=args.label_key,
    )

    if len(dataset) < 5:
        print(f"[Main] ERROR: Only {len(dataset)} samples found. Need at least 5.")
        sys.exit(1)

    # Auto-adjust folds for very small datasets
    if args.n_folds > len(dataset):
        print(f"[Main] n_folds ({args.n_folds}) > n_samples ({len(dataset)}), using LOO.")
        args.n_folds = -1

    # ── Model info ───────────────────────────────────────────────────
    model = PortalPressureNet(d_hidden=args.d_hidden, dropout=args.dropout)
    total_params, trainable_params = count_params(model)
    print(f"\n[Main] Model: {trainable_params:,} trainable parameters")
    print(f"[Main] Samples: {len(dataset)}, Ratio: {len(dataset)/trainable_params:.4f}")

    if len(dataset) / trainable_params < 0.01:
        print("[Main] WARNING: very low sample/parameter ratio. Consider reducing d_hidden.")

    del model

    # ── Run ───────────────────────────────────────────────────────────
    trainer = Trainer(args)

    if args.ablation:
        trainer.run_ablation(dataset)
    else:
        trainer.run_kfold(dataset)


if __name__ == '__main__':
    main()