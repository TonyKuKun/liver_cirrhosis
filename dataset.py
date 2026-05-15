"""
Portal Vein Pressure Dataset — v4 (shape-aware, PVT-robust)
=============================================================

Adapted for the **v2 unified_features.json** schema which provides:
  • 11 pointwise geometry channels (incl. hydraulic_diameter, solidity, torsion …)
  • 52 system-level scalars (angles, clinical markers, hydraulic, topology …)
  • per-segment statistical summaries
  • vessel_presence dict with diagnostic info

Design principles unchanged from v3:
  1.  Feed the model FUNDAMENTAL quantities; let the physics layer derive the rest.
  2.  But now "fundamental" includes shape descriptors (solidity, circularity,
      n_components) that tell the physics layer how trustworthy its Poiseuille
      assumption is at each point.
  3.  Use hydraulic_diameter (= 4A/P) instead of eq_diameter for Poiseuille —
      correct for non-circular cross-sections (PVT crescent / annular lumens).
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


# ── Branch order ───────────────────────────────────────────────────────
SEGMENTS = ['mpv', 'sv', 'smv', 'lpv', 'rpv', 'tips', 'lgv', 'pgv']
N_SEGMENTS = len(SEGMENTS)
SEG_INDEX = {n: i for i, n in enumerate(SEGMENTS)}

# ── Junction topology (used by physics losses) ────────────────────────
JUNCTIONS = {
    'inflow':              {'children': ['sv', 'smv']},
    'confluence_outflow':  {'children': ['mpv', 'lgv', 'pgv']},
    'bifurcation':         {'parent': 'mpv', 'children': ['lpv', 'rpv', 'tips']},
}
JUNCTION_END_IDX = 0  # all branches: idx 0 is the junction-attached endpoint

# ── Per-point geometry channels (model input) ──────────────────────────
#
# 11 channels — split into two roles:
#   Physics channels (used in Poiseuille formulas):
#       area, hydraulic_diameter, curvature, inscribed_radius
#   Shape-awareness channels (encoder input, not directly in Poiseuille):
#       perimeter, torsion, solidity, r_insc_to_r_eq_ratio,
#       dA_ds_norm, circularity, n_components
#
PROFILE_KEYS = [
    'area',                  # 截面面积 mm²
    'hydraulic_diameter',    # 水力直径 4A/P mm (replaces eq_diameter for physics)
    'perimeter',             # 截面周长 mm
    'curvature',             # 中心线曲率 1/mm
    'torsion',               # 中心线挠率 1/mm
    'inscribed_radius',      # 最大内切球半径 mm
    'solidity',              # 实心度 A/A_convex ∈(0,1]
    'r_insc_to_r_eq_ratio',  # 瓶颈比 2·r_insc/D_eq ∈(0,~1.5]
    'dA_ds_norm',            # 归一化面积变化率 (dA/ds)/A
    'circularity',           # 圆形度 4πA/P² ∈(0,1]
    'n_components',          # 截面连通分量数 (0=无效, 1=正常, ≥2=分裂)
]
N_PROFILE_FEAT = len(PROFILE_KEYS)

# Index helpers (used by model's physics layer)
P_AREA  = 0
P_HDIAM = 1   # hydraulic diameter (was P_DIAM = eq_diameter in v3)
P_PERIM = 2
P_CURV  = 3
P_TORS  = 4
P_INSC  = 5
P_SOLID = 6
P_RRAT  = 7   # r_insc / r_eq ratio
P_DADS  = 8   # dA/ds normalized
P_CIRC  = 9
P_NCOMP = 10


# ── Auxiliary scalar features (model input) ────────────────────────────
#
# Selection logic:
#   Group A: angles — pure 3D info, lost in 1D profiles
#   Group F: clinical markers — strong priors, literature-validated
#   Group E: select topology — system-level descriptors
#   Flags: binary anatomy presence
#
AUX_KEYS = [
    # ─── A: Angles (7) ───
    'angle_sv_smv',
    'angle_mpv_lpv',
    'angle_mpv_rpv',
    'angle_lpv_rpv',
    'angle_mpv_bifurc_total',
    'mpv_bifurc_planarity_deg',
    'angle_mpv_tips',
    # ─── F: Clinical markers (11) ───
    'sv_max_to_mpv_max_diam_ratio',
    'mpv_trunk_length_mm',
    'max_tortuosity_index',
    'mean_tortuosity_index',
    'max_collateral_diameter_mm',
    'area_conservation_bifurc_deviation',
    'tips_stent_diameter_mm',
    'tips_stent_length_mm',
    'pvt_severity_grade',
    'min_lumen_area_to_max_ratio_mpv',
    'cavernous_transformation_flag',
    # ─── E: Topology (5) ───
    'branchpoint_density_per_cm',
    'mpv_taper_coefficient',
    'mpv_min_max_diameter_ratio',
    'splenic_dominance_index',
    'collateral_burden_score',
    # ─── G: Organ volumes from STL (3) — NEW ───
    # Spleen volume = proxy for splanchnic blood flow Q (the MISSING variable)
    # Liver volume = proxy for functional hepatic mass / intrahepatic resistance
    # Ratio = validated portal hypertension severity marker
    'spleen_volume_ml',
    'liver_volume_ml',
    'spleen_liver_ratio',
    # ─── Flags (3) ───
    'has_lgv',
    'has_pgv',
    'has_tips',
]
N_AUX = len(AUX_KEYS)  # 26

# Where to find each key in the JSON
AUX_LOOKUP = {}
_SYSTEM_KEYS = set()
_GLOBAL_KEYS = {'has_lgv', 'has_pgv', 'has_tips'}
_STL_KEYS = {'spleen_volume_ml', 'liver_volume_ml', 'spleen_liver_ratio'}
for k in AUX_KEYS:
    if k in _GLOBAL_KEYS:
        AUX_LOOKUP[k] = ('global', k)
    elif k in _STL_KEYS:
        AUX_LOOKUP[k] = ('stl', k)  # computed from STL files, not JSON
    else:
        AUX_LOOKUP[k] = ('system', k)
        _SYSTEM_KEYS.add(k)

# Indices of binary/ordinal flags (don't z-score normalize)
AUX_FLAG_INDICES = [
    AUX_KEYS.index(k) for k in [
        'has_lgv', 'has_pgv', 'has_tips',
        'cavernous_transformation_flag', 'pvt_severity_grade',
    ]
]

# Indices of organ volume features (for model's Q estimation)
AUX_SPLEEN_VOL_IDX = AUX_KEYS.index('spleen_volume_ml')
AUX_LIVER_VOL_IDX  = AUX_KEYS.index('liver_volume_ml')


# ── Sentinel keys kept for evaluation (NOT input) ─────────────────────
EVAL_SYSTEM_KEYS = [
    'confluence_murray3_ratio', 'confluence_murray3_deviation',
    'confluence_area_ratio',
    'mpv_bifurc_murray3_ratio', 'mpv_bifurc_murray3_deviation',
    'mpv_bifurc_area_ratio',
    'mpv_resistance_integral', 'sv_resistance_integral',
    'smv_resistance_integral', 'lpv_resistance_integral',
    'rpv_resistance_integral', 'tips_resistance_integral',
    'inflow_parallel_resistance', 'inflow_resistance_asymmetry',
    'mpv_effective_radius', 'tips_inflow_resistance_ratio',
    'sv_smv_diameter_asymmetry', 'sv_mpv_diameter_ratio',
    'smv_mpv_diameter_ratio', 'lpv_rpv_diameter_asymmetry',
    'lgv_mpv_diameter_ratio', 'pgv_mpv_diameter_ratio',
    'splenoportal_path_chord_ratio', 'collateral_length_mpv_ratio',
    'diameter_weighted_tortuosity',
    'tree_area_conservation_mean_dev',
    'mpv_proximal_diameter', 'mpv_distal_diameter',
    'n_collaterals_detected',
]

DEFAULT_N_POINTS = 100


# =====================================================================
# Helpers
# =====================================================================
def _safe_float(v, default=np.nan):
    if v is None:
        return default
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _compute_stl_volume_ml(stl_path):
    """
    Compute volume of a closed STL mesh in mL (= cm³).

    Uses the divergence theorem: for each triangle with vertices v0,v1,v2,
    the signed volume contribution is v0 · (v1 × v2) / 6.
    Sum over all faces, take absolute value → volume in mm³ → convert to mL.

    Returns np.nan if the file doesn't exist or parsing fails.
    """
    if not os.path.exists(stl_path):
        return np.nan
    try:
        with open(stl_path, 'rb') as f:
            header = f.read(80)
        is_ascii = header[:5] == b'solid' and b'\x00' not in header

        if is_ascii:
            verts = []
            with open(stl_path, 'r', errors='replace') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('vertex'):
                        parts = line.split()
                        verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            verts = np.array(verts, dtype=np.float64).reshape(-1, 3, 3)  # (n_tri, 3, 3)
        else:
            with open(stl_path, 'rb') as f:
                f.read(80)
                n_tri = int(np.frombuffer(f.read(4), dtype=np.uint32)[0])
                data = f.read(n_tri * 50)
            # Each facet: 12 bytes normal + 36 bytes vertices + 2 bytes attr
            dt = np.dtype([('normal', '<f4', 3), ('verts', '<f4', (3, 3)), ('attr', '<u2')])
            facets = np.frombuffer(data, dtype=dt, count=n_tri)
            verts = facets['verts'].astype(np.float64)  # (n_tri, 3, 3)

        if len(verts) < 4:
            return np.nan

        # Signed volume via divergence theorem
        v0 = verts[:, 0, :]
        v1 = verts[:, 1, :]
        v2 = verts[:, 2, :]
        cross = np.cross(v1, v2)
        signed_vol = np.sum(v0 * cross) / 6.0
        volume_mm3 = abs(signed_vol)
        volume_ml = volume_mm3 / 1000.0  # mm³ → cm³ = mL
        return float(volume_ml) if volume_ml > 0.1 else np.nan

    except Exception:
        return np.nan


def _resample(arr, n_target):
    """Plain linear resample for any 1-D array."""
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    if n == n_target:
        return arr.astype(np.float32)
    if n < 2:
        return np.zeros(n_target, np.float32)
    xp = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, n_target)
    return np.interp(x_new, xp, arr).astype(np.float32)


def _resample_with_mask(arr_raw, n_target):
    """
    Resample 1D array, treating None / NaN as missing.
    Returns (values, valid_mask).
    """
    arr = np.array([_safe_float(x, np.nan) for x in arr_raw], dtype=np.float64)
    n_raw = len(arr)
    if n_raw < 2:
        return np.zeros(n_target, np.float32), np.zeros(n_target, np.float32)

    valid_raw = np.isfinite(arr)
    if valid_raw.sum() < 2:
        return np.zeros(n_target, np.float32), np.zeros(n_target, np.float32)

    xp_full = np.linspace(0.0, 1.0, n_raw)
    x_new = np.linspace(0.0, 1.0, n_target)
    xp_v = xp_full[valid_raw]

    vals = np.interp(x_new, xp_v, arr[valid_raw]).astype(np.float32)
    first_v, last_v = xp_v[0], xp_v[-1]
    mask = ((x_new >= first_v) & (x_new <= last_v)).astype(np.float32)
    return vals, mask


# =====================================================================
# Dataset
# =====================================================================
class PortalVeinDataset(Dataset):

    def __init__(self, root_dir: str, n_points: int = DEFAULT_N_POINTS,
                 label_key: str = 'PVP', verbose: bool = True):
        super().__init__()
        self.root_dir = root_dir
        self.n_points = n_points
        self.label_key = label_key
        self.verbose = verbose

        # ── Discover patients ───────────────────────────────────
        self.patients = []
        for name in sorted(os.listdir(root_dir)):
            pdir = os.path.join(root_dir, name)
            if '@' in name or '!' in name:
                continue
            if not os.path.isdir(pdir):
                continue
            label_file = os.path.join(pdir, 'label', f'{label_key}.txt')
            if not os.path.exists(label_file):
                continue
            self.patients.append({
                'name': name,
                'dir': pdir,
                'label_file': label_file,
                'unified_file': os.path.join(pdir, 'unified_features.json'),
                'is_post_tips': '#' in name,
            })

        if verbose:
            n_post = sum(p['is_post_tips'] for p in self.patients)
            print(f"[Dataset] Discovered {len(self.patients)} patients "
                  f"(post-TIPS: {n_post}, pre-TIPS: {len(self.patients) - n_post}).")

        # ── Pre-load ────────────────────────────────────────────
        self.data = []
        for p in self.patients:
            item = self._load_one(p)
            if item is not None:
                self.data.append(item)

        if verbose:
            print(f"[Dataset] Loaded {len(self.data)} valid patients.")
            self._print_branch_diagnostics()

        self._compute_normalization()

    # -----------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------
    def _load_one(self, p):
        try:
            with open(p['label_file'], 'r', encoding='utf-8') as f:
                label = float(f.read().strip())
            if not np.isfinite(label):
                return None

            if not os.path.exists(p['unified_file']):
                return None
            with open(p['unified_file'], 'r', encoding='utf-8') as f:
                unified = json.load(f)

            meta = unified.get('_meta', {}) or {}
            is_post_tips = bool(meta.get('is_post_tips', p['is_post_tips']))

            # ── Vessel presence → segment_mask ─────────────────
            vp = unified.get('vessel_presence', {})
            segment_mask = np.zeros(N_SEGMENTS, dtype=np.float32)
            for si, sn in enumerate(SEGMENTS):
                info = vp.get(sn, {})
                if isinstance(info, dict) and info.get('present', False):
                    segment_mask[si] = 1.0
                elif isinstance(info, bool) and info:
                    segment_mask[si] = 1.0

            # ── Organ volumes from STL ──────────────────────────
            seg_dir = os.path.join(p['dir'], 'segmentation')
            spleen_vol = _compute_stl_volume_ml(os.path.join(seg_dir, 'spleen.stl'))
            liver_vol  = _compute_stl_volume_ml(os.path.join(seg_dir, 'liver.stl'))
            sl_ratio = np.nan
            if np.isfinite(spleen_vol) and np.isfinite(liver_vol) and liver_vol > 1.0:
                sl_ratio = spleen_vol / liver_vol
            stl_values = {
                'spleen_volume_ml': spleen_vol,
                'liver_volume_ml':  liver_vol,
                'spleen_liver_ratio': sl_ratio,
            }

            # ── Aux scalars ────────────────────────────────────
            aux, aux_mask = self._load_aux(unified, stl_values=stl_values)

            # Organ volumes as separate tensor (for model's Q estimation)
            organ_vols = np.array([
                spleen_vol if np.isfinite(spleen_vol) else 0.0,
                liver_vol  if np.isfinite(liver_vol)  else 0.0,
            ], dtype=np.float32)
            organ_valid = np.array([
                float(np.isfinite(spleen_vol)),
                float(np.isfinite(liver_vol)),
            ], dtype=np.float32)

            # ── 3D endpoints (for viz / junction physics) ──────
            endpoints_3d = self._load_endpoints_3d(unified)
            confluence_3d = self._load_confluence_3d(unified)

            # ── Pointwise profiles ─────────────────────────────
            pw_src = unified.get('pointwise', {}) or {}
            profiles, point_valid, arc_lengths = \
                self._load_pointwise(pw_src, segment_mask, patient_name=p['name'])

            # Update segment_mask: if pointwise has no valid points, mark absent
            for si in range(N_SEGMENTS):
                if point_valid[si].sum() < 5:
                    segment_mask[si] = 0.0

            # ── Extras for eval ────────────────────────────────
            extras_for_eval = self._load_extras(unified)

            return {
                'name': p['name'],
                'profiles':      profiles,
                'point_valid':   point_valid,
                'arc_lengths':   arc_lengths,
                'segment_mask':  segment_mask,
                'aux_scalars':   aux,
                'aux_mask':      aux_mask,
                'organ_volumes': organ_vols,    # (2,) [spleen_ml, liver_ml]
                'organ_valid':   organ_valid,   # (2,) [spleen_present, liver_present]
                'endpoints_3d':  endpoints_3d,
                'confluence_3d': confluence_3d,
                'label':         np.float32(label),
                'is_post_tips':  is_post_tips,
                'extras_for_eval': extras_for_eval,
            }
        except Exception as e:
            if self.verbose:
                print(f"[Dataset] Failed to load {p['name']}: "
                      f"{type(e).__name__}: {e}")
            return None

    def _load_aux(self, unified, stl_values=None):
        out = np.zeros(N_AUX, dtype=np.float32)
        mask = np.zeros(N_AUX, dtype=np.float32)
        stl_values = stl_values or {}

        # System features
        sys_all = {}
        sys_d = unified.get('system', {})
        if isinstance(sys_d, dict):
            sys_all = sys_d.get('all_values', {}) or {}
            avail = sys_d.get('available', {})
            if isinstance(avail, dict):
                for k, v in avail.items():
                    if k not in sys_all or sys_all[k] is None:
                        sys_all[k] = v

        glob = unified.get('global', {}) or {}

        for i, key in enumerate(AUX_KEYS):
            section, jkey = AUX_LOOKUP[key]
            if section == 'global':
                v = _safe_float(glob.get(jkey, None), np.nan)
            elif section == 'stl':
                v = _safe_float(stl_values.get(jkey, None), np.nan)
            else:  # system
                v = _safe_float(sys_all.get(jkey, None), np.nan)

            if np.isfinite(v):
                if i in AUX_FLAG_INDICES:
                    out[i] = float(v)  # keep as-is (0/1/2)
                else:
                    out[i] = float(v)
                mask[i] = 1.0
            else:
                out[i] = 0.0
                mask[i] = 0.0

        return out, mask

    def _load_endpoints_3d(self, unified):
        out = np.zeros((N_SEGMENTS, 2, 3), dtype=np.float32)
        seg_meta = unified.get('segments_meta', {}) or {}
        for si, sname in enumerate(SEGMENTS):
            sm = seg_meta.get(sname, None) or {}
            ep = sm.get('endpoints_coord', None)
            if ep is not None and len(ep) == 2:
                out[si] = np.asarray(ep, dtype=np.float32)
        return out

    def _load_confluence_3d(self, unified):
        sva = unified.get('sv_smv_angle', {}) or {}
        cp = sva.get('confluence_point_physical', None)
        if cp is not None:
            return np.asarray(cp, dtype=np.float32)
        return np.zeros(3, dtype=np.float32)

    def _load_pointwise(self, pw_json, segment_mask, patient_name=None):
        """
        Returns:
            profiles     (S, N, 11)
            point_valid  (S, N)
            arc_lengths  (S, N) mm
        """
        S, N = N_SEGMENTS, self.n_points
        profiles    = np.zeros((S, N, N_PROFILE_FEAT), dtype=np.float32)
        point_valid = np.zeros((S, N), dtype=np.float32)
        arc_lengths = np.zeros((S, N), dtype=np.float32)

        if not pw_json:
            return profiles, point_valid, arc_lengths

        # Case-insensitive key map
        key_map = {k.lower(): k for k in pw_json.keys() if isinstance(k, str)}

        for si, sname in enumerate(SEGMENTS):
            seg_data = pw_json.get(sname, None)
            if seg_data is None and sname.lower() in key_map:
                seg_data = pw_json[key_map[sname.lower()]]
            if not seg_data or not isinstance(seg_data, dict):
                continue

            # Check minimum data availability
            area_arr = seg_data.get('area', None)
            if area_arr is None or not hasattr(area_arr, '__len__') or len(area_arr) < 5:
                continue

            # Arc length
            arc_raw = seg_data.get('arc_length_mm', None)
            if arc_raw is not None and len(arc_raw) >= 2:
                arc_lengths[si] = _resample(arc_raw, N)
            else:
                arc_lengths[si] = np.linspace(0, 1, N).astype(np.float32)

            # Resample each profile channel
            per_pt_valid = np.ones(N, dtype=np.float32)
            for fi, fkey in enumerate(PROFILE_KEYS):
                raw = seg_data.get(fkey, None)
                if raw is None or not hasattr(raw, '__len__') or len(raw) < 2:
                    # Channel missing → fill with defaults
                    if fkey == 'solidity':
                        profiles[si, :, fi] = 1.0  # assume circular
                    elif fkey == 'r_insc_to_r_eq_ratio':
                        profiles[si, :, fi] = 1.0  # assume circular
                    elif fkey == 'circularity':
                        profiles[si, :, fi] = 1.0
                    elif fkey == 'n_components':
                        profiles[si, :, fi] = 1.0
                    else:
                        profiles[si, :, fi] = 0.0
                    continue

                vals, vmask = _resample_with_mask(raw, N)
                profiles[si, :, fi] = vals
                per_pt_valid = per_pt_valid * vmask

            # Additional validity: area must be positive
            area_valid = (profiles[si, :, P_AREA] > 1e-3).astype(np.float32)
            point_valid[si] = per_pt_valid * area_valid

        return profiles, point_valid, arc_lengths

    def _load_extras(self, unified):
        out = {}
        sys_all = {}
        sys_d = unified.get('system', {})
        if isinstance(sys_d, dict):
            sys_all = sys_d.get('all_values', {}) or {}
            avail = sys_d.get('available', {})
            if isinstance(avail, dict):
                for k, v in avail.items():
                    if k not in sys_all or sys_all[k] is None:
                        sys_all[k] = v
        for k in EVAL_SYSTEM_KEYS:
            out[k] = _safe_float(sys_all.get(k, None), np.nan)

        # Also store per-segment statistics
        stat = unified.get('statistical', {}) or {}
        for seg_name, seg_stats in stat.items():
            if isinstance(seg_stats, dict):
                for sk, sv in seg_stats.items():
                    out[f'{seg_name}_{sk}'] = _safe_float(sv, np.nan)
        return out

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------
    def _print_branch_diagnostics(self):
        if not self.verbose or not self.data:
            return
        n_per_branch = {sn: 0 for sn in SEGMENTS}
        for d in self.data:
            for si, sn in enumerate(SEGMENTS):
                if d['segment_mask'][si] > 0:
                    n_per_branch[sn] += 1
        print("[Dataset] Pointwise data availability per branch:")
        for sn in SEGMENTS:
            n = n_per_branch[sn]
            mark = "OK" if n > 0 else "ZERO"
            print(f"   {sn:>5s} : {n:3d} / {len(self.data)}  {mark}")

    # -----------------------------------------------------------------
    # Normalization
    # -----------------------------------------------------------------
    def _compute_normalization(self):
        # Per-channel stats over valid points
        per_channel_vals = [[] for _ in range(N_PROFILE_FEAT)]
        for d in self.data:
            valid_mask = d['point_valid'] * d['segment_mask'][:, None]
            for fi in range(N_PROFILE_FEAT):
                v = d['profiles'][:, :, fi][valid_mask > 0]
                if len(v):
                    per_channel_vals[fi].append(v)

        means = np.zeros(N_PROFILE_FEAT, dtype=np.float32)
        stds  = np.ones(N_PROFILE_FEAT, dtype=np.float32)
        for fi in range(N_PROFILE_FEAT):
            if per_channel_vals[fi]:
                arr = np.concatenate(per_channel_vals[fi])
                means[fi] = arr.mean()
                s = arr.std()
                stds[fi] = max(s, 1e-6)
        self.profile_mean = means
        self.profile_std  = stds

        # Aux normalization
        aux_vals = np.stack([d['aux_scalars'] for d in self.data])
        aux_pres = np.stack([d['aux_mask'] for d in self.data])
        aux_mean = np.zeros(N_AUX, dtype=np.float32)
        aux_std  = np.ones(N_AUX, dtype=np.float32)
        for i in range(N_AUX):
            if i in AUX_FLAG_INDICES:
                aux_mean[i] = 0.0
                aux_std[i]  = 1.0
            else:
                vals = aux_vals[:, i][aux_pres[:, i] > 0]
                if len(vals) >= 2:
                    aux_mean[i] = vals.mean()
                    aux_std[i]  = max(vals.std(), 1e-6)
        self.aux_mean = aux_mean
        self.aux_std  = aux_std

        # Label
        labels = np.array([d['label'] for d in self.data], dtype=np.float32)
        self.label_mean = float(labels.mean())
        self.label_std  = float(max(labels.std(), 1e-6))

        if self.verbose:
            print(f"[Dataset] Profile means: {self.profile_mean}")
            print(f"[Dataset] Profile stds : {self.profile_std}")
            print(f"[Dataset] Label mean/std: {self.label_mean:.3f} / {self.label_std:.3f}")

    # -----------------------------------------------------------------
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        profiles = d['profiles']
        pv = d['point_valid'][..., None]
        profiles_norm = (profiles - self.profile_mean) / self.profile_std
        profiles_norm = profiles_norm * pv

        aux = d['aux_scalars'].copy()
        aux_norm = (aux - self.aux_mean) / self.aux_std
        label_norm = (float(d['label']) - self.label_mean) / self.label_std

        return {
            'name':            d['name'],
            'profiles':        torch.from_numpy(profiles).float(),
            'profiles_norm':   torch.from_numpy(profiles_norm).float(),
            'arc_lengths':     torch.from_numpy(d['arc_lengths']).float(),
            'point_valid':     torch.from_numpy(d['point_valid']).float(),
            'segment_mask':    torch.from_numpy(d['segment_mask']).float(),
            'aux_scalars':     torch.from_numpy(aux).float(),
            'aux_norm':        torch.from_numpy(aux_norm).float(),
            'aux_mask':        torch.from_numpy(d['aux_mask']).float(),
            'organ_volumes':   torch.from_numpy(d['organ_volumes']).float(),
            'organ_valid':     torch.from_numpy(d['organ_valid']).float(),
            'endpoints_3d':    torch.from_numpy(d['endpoints_3d']).float(),
            'confluence_3d':   torch.from_numpy(d['confluence_3d']).float(),
            'is_post_tips':    torch.tensor(float(d['is_post_tips'])),
            'label':           torch.tensor(float(d['label'])).float(),
            'label_norm':      torch.tensor(float(label_norm)).float(),
            'extras_for_eval': d['extras_for_eval'],
        }


# =====================================================================
# Collate
# =====================================================================
def collate_fn(items):
    tensor_keys = ['profiles', 'profiles_norm', 'arc_lengths', 'point_valid',
                   'segment_mask', 'aux_scalars', 'aux_norm', 'aux_mask',
                   'organ_volumes', 'organ_valid',
                   'endpoints_3d', 'confluence_3d', 'is_post_tips',
                   'label', 'label_norm']
    out = {}
    for k in tensor_keys:
        out[k] = torch.stack([it[k] for it in items], dim=0)
    out['name'] = [it['name'] for it in items]
    out['extras_for_eval'] = [it['extras_for_eval'] for it in items]
    return out


# =====================================================================
# Quick test
# =====================================================================
if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else r'F:\PCG data\dataset\test4all_sample'
    ds = PortalVeinDataset(root, n_points=100, verbose=True)
    print(f"\nDataset size: {len(ds)}")
    if len(ds) == 0:
        print("Empty dataset.")
        sys.exit(0)
    s = ds[0]
    for k, v in s.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:>16s}: shape={tuple(v.shape)}, dtype={v.dtype}")
        elif isinstance(v, dict):
            present = sum(1 for vv in v.values() if np.isfinite(vv))
            print(f"  {k:>16s}: dict, {present}/{len(v)} present")
        else:
            print(f"  {k:>16s}: {v}")