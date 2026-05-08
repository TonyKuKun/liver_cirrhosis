"""
Portal Vein Pressure Dataset
=============================
Loads geometric profile features, statistical features, and PVP labels.
Handles missing branches and normalizes features.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


# ── Profile feature keys (per centerline point) ──────────────────────
PROFILE_KEYS = ['area', 'perimeter', 'eq_diameter', 'circularity', 'curvature', 'inscribed_radius']
N_PROFILE_FEAT = len(PROFILE_KEYS)

# ── Branch names ─────────────────────────────────────────────────────
BRANCHES = ['mpv', 'sv', 'smv']
N_BRANCHES = len(BRANCHES)

# ── 28 statistical feature keys (order matters for consistency) ──────
STAT_KEYS = [
    # Length (6)
    'mpv_length', 'sv_length', 'smv_length', 'lpv_length', 'rpv_length',
    'total_centerline_length',
    # Tortuosity (5)
    'mpv_tortuosity', 'sv_tortuosity', 'smv_tortuosity', 'lpv_tortuosity', 'rpv_tortuosity',
    # Curvature (4)
    'mpv_mean_curvature', 'sv_mean_curvature', 'smv_mean_curvature', 'mpv_max_curvature',
    # Diameter (5)
    'mpv_mean_diameter', 'mpv_max_diameter', 'sv_mean_diameter', 'smv_mean_diameter',
    'sv_smv_diameter_ratio',
    # Area (4)
    'mpv_cross_section_area', 'sv_cross_section_area', 'smv_cross_section_area', 'mpv_area_cv',
    # Circularity (3)
    'mpv_mean_circularity', 'sv_mean_circularity', 'smv_mean_circularity',
    # Angle (1)
    'sv_smv_angle',
]
N_STAT_FEAT = len(STAT_KEYS)

# Default resample length for profiles
DEFAULT_N_POINTS = 100


class PortalVeinDataset(Dataset):
    """
    Loads geometric profile features, statistical features, and PVP labels.
    Handles missing branches and normalizes features.
    
    Parameters
    ----------
    root_dir : str
        Root folder containing patient_xxx/ subfolders.
    n_points : int
        Resample profile length (default 100, matching extraction pipeline).
    label_key : str
        'PVP' or 'PCG'.
    """

    def __init__(self, root_dir: str, n_points: int = DEFAULT_N_POINTS,
                 label_key: str = 'PVP'):
        super().__init__()
        self.root_dir = root_dir
        self.n_points = n_points
        self.label_key = label_key

        # ── Discover patients ────────────────────────────────────────
        self.patients = []
        for name in sorted(os.listdir(root_dir)):
            pdir = os.path.join(root_dir, name)
            label_file = os.path.join(pdir, 'label', f'{label_key}.txt')
            profile_file = os.path.join(pdir, 'centerline_profiles.json')
            stat_file = os.path.join(pdir, 'portal_vein_features.json')
            if os.path.isdir(pdir) and os.path.exists(label_file):
                self.patients.append({
                    'name': name,
                    'dir': pdir,
                    'label_file': label_file,
                    'profile_file': profile_file,
                    'stat_file': stat_file,
                })

        print(f"[Dataset] Found {len(self.patients)} patients with {label_key} labels.")

        # ── Pre-load all data for normalization ──────────────────────
        self.data = []
        for p in self.patients:
            item = self._load_one(p)
            if item is not None:
                self.data.append(item)

        print(f"[Dataset] Successfully loaded {len(self.data)} patients.")

        # ── Compute normalization stats ──────────────────────────────
        self._compute_normalization()

    def _load_one(self, p: dict) -> dict:
        """Load and validate a single patient."""
        try:
            # Label
            with open(p['label_file'], 'r') as f:
                label = float(f.read().strip())
            if not np.isfinite(label):
                print(f"[Dataset] Warning: {p['name']} has non-finite label, skipping.")
                return None

            # Statistical features
            stat_vec = np.zeros(N_STAT_FEAT, dtype=np.float32)
            if os.path.exists(p['stat_file']):
                with open(p['stat_file'], 'r') as f:
                    stat_json = json.load(f)
                for i, key in enumerate(STAT_KEYS):
                    val = stat_json.get(key, None)
                    if val is not None:
                        v = float(val)
                        stat_vec[i] = v if np.isfinite(v) else 0.0
                    else:
                        stat_vec[i] = 0.0

            # Profile features: (n_branches, n_points, n_profile_feat)
            profiles = np.zeros((N_BRANCHES, self.n_points, N_PROFILE_FEAT), dtype=np.float32)
            branch_mask = np.zeros(N_BRANCHES, dtype=np.float32)  # 1 if branch exists

            # Arc lengths for physics computation: (n_branches, n_points)
            arc_lengths = np.zeros((N_BRANCHES, self.n_points), dtype=np.float32)

            if os.path.exists(p['profile_file']):
                with open(p['profile_file'], 'r') as f:
                    prof_json = json.load(f)

                for bi, bname in enumerate(BRANCHES):
                    branch_data = prof_json.get(bname, None)
                    if branch_data is None:
                        continue

                    # Check we have actual data
                    area = branch_data.get('area', None)
                    if area is None or len(area) < 5:
                        continue

                    branch_mask[bi] = 1.0
                    n_raw = len(area)

                    for fi, fkey in enumerate(PROFILE_KEYS):
                        raw = np.array(branch_data.get(fkey, [0.0] * n_raw), dtype=np.float32)
                        # NaN / Inf → 0
                        raw = np.where(np.isfinite(raw), raw, 0.0)
                        # Resample to n_points if needed
                        if len(raw) != self.n_points:
                            xp = np.linspace(0, 1, len(raw))
                            x_new = np.linspace(0, 1, self.n_points)
                            raw = np.interp(x_new, xp, raw)
                        # Final NaN guard after interp
                        raw = np.where(np.isfinite(raw), raw, 0.0)
                        profiles[bi, :, fi] = raw

                    # Arc length
                    arc_raw = np.array(
                        branch_data.get('arc_length_mm', np.linspace(0, 1, n_raw)),
                        dtype=np.float32
                    )
                    arc_raw = np.where(np.isfinite(arc_raw), arc_raw, 0.0)
                    if len(arc_raw) != self.n_points:
                        xp = np.linspace(0, 1, len(arc_raw))
                        x_new = np.linspace(0, 1, self.n_points)
                        arc_raw = np.interp(x_new, xp, arc_raw)
                    arc_raw = np.where(np.isfinite(arc_raw), arc_raw, 0.0)
                    arc_lengths[bi] = arc_raw

            # ── Final NaN/Inf guard on all arrays ────────────────────
            stat_vec = np.where(np.isfinite(stat_vec), stat_vec, 0.0).astype(np.float32)
            profiles = np.where(np.isfinite(profiles), profiles, 0.0).astype(np.float32)
            arc_lengths = np.where(np.isfinite(arc_lengths), arc_lengths, 0.0).astype(np.float32)

            return {
                'name': p['name'],
                'profiles': profiles,           # (3, N, 6)
                'arc_lengths': arc_lengths,      # (3, N)
                'branch_mask': branch_mask,      # (3,)
                'stat_features': stat_vec,       # (28,)
                'label': np.float32(label),
            }

        except Exception as e:
            print(f"[Dataset] Warning: failed to load {p['name']}: {e}")
            return None

    def _compute_normalization(self):
        """Compute per-feature mean/std for profiles and statistics."""
        if len(self.data) == 0:
            return

        # ── Profile normalization (per branch, per feature) ──────────
        # Shape: (n_samples, 3, N, 6)
        all_profiles = np.stack([d['profiles'] for d in self.data])
        all_masks = np.stack([d['branch_mask'] for d in self.data])  # (n_samples, 3)

        self.profile_mean = np.zeros((N_BRANCHES, 1, N_PROFILE_FEAT), dtype=np.float32)
        self.profile_std = np.ones((N_BRANCHES, 1, N_PROFILE_FEAT), dtype=np.float32)

        for bi in range(N_BRANCHES):
            valid_idx = all_masks[:, bi] > 0
            if valid_idx.sum() > 1:
                branch_data = all_profiles[valid_idx, bi]  # (n_valid, N, 6)
                self.profile_mean[bi, 0] = branch_data.mean(axis=(0, 1))
                self.profile_std[bi, 0] = branch_data.std(axis=(0, 1)) + 1e-8

        # ── Arc length normalization ─────────────────────────────────
        all_arcs = np.stack([d['arc_lengths'] for d in self.data])
        self.arc_mean = np.zeros((N_BRANCHES, 1), dtype=np.float32)
        self.arc_std = np.ones((N_BRANCHES, 1), dtype=np.float32)

        for bi in range(N_BRANCHES):
            valid_idx = all_masks[:, bi] > 0
            if valid_idx.sum() > 1:
                arc_data = all_arcs[valid_idx, bi]
                self.arc_mean[bi, 0] = arc_data.mean()
                self.arc_std[bi, 0] = arc_data.std() + 1e-8

        # ── Stat feature normalization ───────────────────────────────
        all_stats = np.stack([d['stat_features'] for d in self.data])
        self.stat_mean = all_stats.mean(axis=0)
        self.stat_std = all_stats.std(axis=0) + 1e-8

        # ── Label normalization (for regression) ─────────────────────
        all_labels = np.array([d['label'] for d in self.data])
        self.label_mean = float(all_labels.mean())
        self.label_std = float(all_labels.std()) + 1e-8

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]

        # Normalize profiles (keep raw for physics layer, normalize for learning)
        profiles_raw = torch.from_numpy(d['profiles'].copy())          # (3, N, 6)
        profiles_norm = (profiles_raw - torch.from_numpy(self.profile_mean)) / \
                        torch.from_numpy(self.profile_std)
        profiles_norm = torch.nan_to_num(profiles_norm, nan=0.0, posinf=0.0, neginf=0.0)

        arc_lengths = torch.from_numpy(d['arc_lengths'].copy())         # (3, N)
        branch_mask = torch.from_numpy(d['branch_mask'].copy())         # (3,)

        # Normalize stat features
        stat_raw = torch.from_numpy(d['stat_features'].copy())
        stat_norm = (stat_raw - torch.from_numpy(self.stat_mean)) / \
                    torch.from_numpy(self.stat_std)
        stat_norm = torch.nan_to_num(stat_norm, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize label (for regression loss)
        label_raw = torch.tensor(d['label'], dtype=torch.float32)
        label_norm = (label_raw - torch.tensor(self.label_mean, dtype=torch.float32)) / \
                     torch.tensor(self.label_std, dtype=torch.float32)

        return {
            'profiles_raw': profiles_raw,      # (3, N, 6)
            'profiles_norm': profiles_norm,    # (3, N, 6)
            'arc_lengths': arc_lengths,         # (3, N)
            'branch_mask': branch_mask,         # (3,)
            'stat_features': stat_norm,         # (28,)
            'label': label_norm,                # scalar
            'label_raw': label_raw,             # scalar (unnormalized, for analysis)
            'patient_name': d['name'],
        }
