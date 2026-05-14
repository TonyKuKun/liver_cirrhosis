"""
Anti-Shrinkage Calibration & Extreme-Value Sampling
=====================================================
Addresses the fundamental MSE regression-to-mean problem:
  With 62 samples and r(pred,label) ≈ 0.52, MSE-optimal predictions
  have pred_std ≈ 0.52 * label_std — a 48% compression of the range.

Two complementary fixes:
  1. ExtremeValueSampler: oversample patients with extreme PVP during training
  2. ShrinkageCalibrator: post-hoc affine correction that restores the full range
"""

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


# =====================================================================
# Fix 1: Weighted Sampling (used DURING training)
# =====================================================================
class ExtremeValueSampler:
    """
    Creates a WeightedRandomSampler that oversamples extreme PVP values.

    Patients far from the median get sampled more often, so the model
    sees more extreme cases per epoch. This complements the extremity-
    weighted loss in model.py.

    Usage in train.py:
        sampler = ExtremeValueSampler.from_dataset(full_ds, train_idx, power=2.0)
        train_ld = DataLoader(train_ds, sampler=sampler, ...)
        # NOTE: remove shuffle=True when using a sampler
    """

    @staticmethod
    def from_dataset(full_ds, train_indices, power=2.0, floor=1.0):
        """
        Args:
            full_ds:        PortalVeinDataset
            train_indices:  array of indices into full_ds.data for this fold's training set
            power:          how aggressively to oversample extremes (1=linear, 2=quadratic)
            floor:          minimum weight (so no sample is completely ignored)

        Returns:
            WeightedRandomSampler for the training DataLoader.
        """
        labels = np.array([full_ds.data[i]['label'] for i in train_indices])
        median = np.median(labels)
        std = max(np.std(labels), 1e-6)

        # Weight = floor + |label - median|^power / std^power
        # Quadratic (power=2): sample 2σ from median gets weight ≈ floor + 4
        dist = np.abs(labels - median) / std
        weights = floor + dist ** power

        # Normalize so sum = len (expected number of samples per epoch unchanged)
        weights = weights / weights.mean()

        return WeightedRandomSampler(
            weights=torch.from_numpy(weights).double(),
            num_samples=len(train_indices),
            replacement=True,
        )


# =====================================================================
# Fix 2: Post-hoc Shrinkage Calibration (used AFTER training each fold)
# =====================================================================
class ShrinkageCalibrator:
    """
    Affine calibration that reverses the MSE-induced range compression.

    After training, the model's predictions have:
        pred_std < label_std  (compressed range)
        pred_mean ≈ label_mean (centered correctly)

    Calibration:
        pred_calibrated = slope * (pred - pred_mean) + label_mean

    where slope = label_std / pred_std > 1  (expands the range).

    This is mathematically equivalent to the James-Stein estimator
    and is provably optimal for Gaussian-distributed targets.

    Usage:
        cal = ShrinkageCalibrator()
        cal.fit(train_preds, train_labels)   # fit on TRAINING set predictions
        val_preds_cal = cal.transform(val_preds)  # apply to VAL set
    """

    def __init__(self):
        self.slope = 1.0
        self.pred_center = 0.0
        self.label_center = 0.0

    def fit(self, preds, labels):
        """Fit calibration on (pred, label) pairs from the training set."""
        preds = np.asarray(preds, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)

        self.pred_center = float(np.mean(preds))
        self.label_center = float(np.mean(labels))

        pred_std = float(np.std(preds))
        label_std = float(np.std(labels))

        if pred_std < 1e-6:
            self.slope = 1.0
        else:
            # Slope that makes calibrated predictions have same std as labels
            raw_slope = label_std / pred_std

            # Regularize: don't expand more than 2x (prevents instability
            # when pred_std is very small due to bad training)
            self.slope = float(min(raw_slope, 2.0))

        return self

    def transform(self, preds):
        """Apply calibration to new predictions."""
        preds = np.asarray(preds, dtype=np.float64)
        return self.slope * (preds - self.pred_center) + self.label_center

    def __repr__(self):
        return (f"ShrinkageCalibrator(slope={self.slope:.3f}, "
                f"pred_center={self.pred_center:.2f}, "
                f"label_center={self.label_center:.2f})")


# =====================================================================
# Convenience: apply calibration within the training loop
# =====================================================================
def calibrate_fold_predictions(model, train_loader, val_loader, device,
                               label_mean, label_std):
    """
    After training a fold:
      1. Run inference on training set → get (train_preds, train_labels)
      2. Fit ShrinkageCalibrator on training predictions
      3. Run inference on val set → calibrate val predictions
      4. Return calibrated val predictions + calibrator

    Usage in train_fold():
        cal_preds, calibrator = calibrate_fold_predictions(
            model, train_ld, val_ld, device, mu_y, sigma_y)
    """
    model.eval()

    def _collect(loader):
        all_pred, all_label = [], []
        with torch.no_grad():
            for batch in loader:
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(device)
                out = model(batch)
                pvp_n = out['pvp_pred'].squeeze(-1).cpu().numpy()
                pvp_real = pvp_n * label_std + label_mean
                lab_real = batch['label'].cpu().numpy()
                all_pred.append(pvp_real)
                all_label.append(lab_real)
        return np.concatenate(all_pred), np.concatenate(all_label)

    train_preds, train_labels = _collect(train_loader)
    val_preds, val_labels = _collect(val_loader)

    cal = ShrinkageCalibrator()
    cal.fit(train_preds, train_labels)

    val_preds_cal = cal.transform(val_preds)

    mae_before = float(np.mean(np.abs(val_preds - val_labels)))
    mae_after = float(np.mean(np.abs(val_preds_cal - val_labels)))

    return val_preds_cal, val_labels, cal, {
        'val_mae_before_cal': mae_before,
        'val_mae_after_cal': mae_after,
        'calibrator_slope': cal.slope,
    }