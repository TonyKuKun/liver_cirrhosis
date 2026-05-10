"""
Portal Vein Pressure Dataset — v3 (selective, principled)
==========================================================

设计原则 (Design Principles)
─────────────────────────────────────────────────────────────────
1. **只输入"基本量",不输入"派生量"** —— Only feed the model fundamental
   geometric quantities. Ratios, asymmetries, deviations, pre-computed
   resistance integrals are all DROPPED from inputs (the model derives
   them via the physics layer). They are kept in `extras_for_eval` for
   sanity comparison after training.

2. **逐点几何 + 必要的拓扑标量** —— The model receives:
   • 4 per-point geometry channels: area, eq_diameter, curvature, inscribed_radius
     (perimeter and circularity dropped: derivable from area+diameter under
      circular assumption)
   • 11 truly non-derivable patient-level scalars:
       4 angles  + 1 planarity + 2 topology counts + 4 flags

3. **NaN-aware resampling** —— Per-point validity mask preserves the
   endpoint protection band from the JSON.

4. **3D 拓扑保留** —— endpoints_3d and sv_smv_confluence_3d are kept
   for STL/CFD overlay at inference time, NOT as model inputs.

输入到模型的特征清单 (Inputs to Model)
─────────────────────────────────────────────────────────────────
Per patient:
    profiles      (S=6, N=100, 4)   [area, eq_diameter, curvature, inscribed_radius]
    profiles_norm (S, N, 4)         globally z-scored across the dataset
    arc_lengths   (S, N)            in mm
    point_valid   (S, N)            ∈ {0, 1}
    segment_mask  (S,)              ∈ {0, 1} — which branches exist
    aux_scalars   (11,)             see AUX_KEYS below
    aux_mask      (11,)             missingness mask (1=present, 0=null)
    endpoints_3d  (S, 2, 3)         mm — for visualization / junction physics
    confluence_3d (3,)              SV-SMV physical confluence point
    label         scalar (PVP)
    label_norm    z-scored
    is_post_tips  bool
    extras_for_eval                 dict — JSON-precomputed quantities NOT used as
                                    input, but available for post-training comparison

Folder conventions
─────────────────────────────────────────────────────────────────
    20210909WuJinHeng    → pre-TIPS
    20210921WuJinHeng#   → post-TIPS  (contains '#')
    *@*  or  *!*         → invalid, skipped
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


# ── Branch order (model indexes branches by this order) ────────────────
# Six "core" portal branches + two collateral (compensation) vessels.
# Collaterals form in portal hypertension and decompress the system —
# their presence/size correlates strongly with PVP severity.
SEGMENTS = ['mpv', 'sv', 'smv', 'lpv', 'rpv', 'tips', 'lgv', 'pgv']
N_SEGMENTS = len(SEGMENTS)
SEG_INDEX = {n: i for i, n in enumerate(SEGMENTS)}

# ── Topology of junctions (used by physics losses) ─────────────────────
# Two-tier hierarchy:
#   1) Confluence node (where SV+SMV meet, possibly tapping LGV/PGV off):
#        Q_sv + Q_smv = Q_mpv + Q_lgv + Q_pgv      (mass conservation)
#   2) Bifurcation node (MPV splits into liver branches + optional TIPS):
#        Q_mpv = Q_lpv + Q_rpv + Q_tips            (mass conservation)
#
# Pre-TIPS patients MUST have MPV, SV, SMV, LPV, RPV; MAY have LGV/PGV.
# Post-TIPS patients MUST have MPV, SV, SMV, TIPS; MAY have LPV/RPV/LGV/PGV.
JUNCTIONS = {
    'inflow':            {'children': ['sv', 'smv']},                # → confluence
    'confluence_outflow': {'children': ['mpv', 'lgv', 'pgv']},        # confluence → ...
    'bifurcation':       {'parent': 'mpv', 'children': ['lpv', 'rpv', 'tips']},
}

# Flow-direction convention along the per-point arrays (proximal → distal):
#   MPV : index 0 = sv/smv confluence end (proximal)
#         index N-1 = lpv/rpv/tips bifurcation end (distal)
#   SV  : index 0 = confluence end with MPV (distal w.r.t. flow)
#   SMV : index 0 = confluence end with MPV (distal w.r.t. flow)
#   LPV : index 0 = bifurcation end (proximal)
#   RPV : index 0 = bifurcation end (proximal)
#   TIPS: index 0 = MPV-side end (proximal)
#
# So for ALL branches, "endpoint index 0" = junction-side. This convention
# makes junction physics simple: just sample idx=0.
JUNCTION_END_IDX = 0  # all branches: idx 0 is the junction-attached endpoint


# ── Per-point geometry channels (model input) ──────────────────────────
PROFILE_KEYS = ['area', 'eq_diameter', 'curvature', 'inscribed_radius']
N_PROFILE_FEAT = len(PROFILE_KEYS)
# Index helpers (used by the model's PoiseuilleHydrodynamics layer)
P_AREA = 0
P_DIAM = 1
P_CURV = 2
P_INSC = 3


# ── Auxiliary scalar features (model input) ────────────────────────────
# Selection criteria:
#   • Cannot be derived from per-point 1D profiles
#   • Not a ratio, asymmetry, or deviation (model computes those)
#   • Not a pre-computed resistance integral (physics layer computes that)
#
# 4 angles:        oriented branching geometry — pure 3D info, lost in 1D profiles
# 1 planarity:     describes whether MPV bifurcation is planar (3D info)
# 2 topo counts:   descriptive system-level statistics
# 4 flags:         binary anatomy presence indicators
AUX_KEYS = [
    # angles (degrees)
    'angle_sv_smv', 'angle_mpv_lpv', 'angle_mpv_rpv', 'angle_mpv_tips',
    # planarity
    'mpv_bifurc_planarity_deg',
    # topology counts
    'branchpoint_density_per_cm', 'n_collaterals_detected',
    # flags (binary)
    'has_lgv', 'has_pgv', 'has_compensation_vessel', 'has_tips',
]
N_AUX = len(AUX_KEYS)  # 11
AUX_FLAG_INDICES = [7, 8, 9, 10]  # indices of binary flags (don't normalize)

# Helper: where to look up each AUX key in the JSON
AUX_LOOKUP = {
    'angle_sv_smv':                ('system', 'angle_sv_smv'),
    'angle_mpv_lpv':               ('system', 'angle_mpv_lpv'),
    'angle_mpv_rpv':               ('system', 'angle_mpv_rpv'),
    'angle_mpv_tips':              ('system', 'angle_mpv_tips'),
    'mpv_bifurc_planarity_deg':    ('system', 'mpv_bifurc_planarity_deg'),
    'branchpoint_density_per_cm':  ('system', 'branchpoint_density_per_cm'),
    'n_collaterals_detected':      ('system', 'n_collaterals_detected'),
    'has_lgv':                     ('global', 'has_lgv'),
    'has_pgv':                     ('global', 'has_pgv'),
    'has_compensation_vessel':     ('global', 'has_compensation_vessel'),
    'has_tips':                    ('global', 'has_tips'),
}


# ── Sentinel JSON keys kept for evaluation (NOT used as input) ─────────
# After training, the model's derived resistances/Murray ratios can be
# compared against these for sanity / interpretability checks.
EVAL_SYSTEM_KEYS = [
    'confluence_murray3_ratio', 'confluence_murray3_deviation',
    'mpv_bifurc_murray3_ratio', 'mpv_bifurc_murray3_deviation',
    'mpv_resistance_integral', 'sv_resistance_integral',
    'smv_resistance_integral', 'lpv_resistance_integral',
    'rpv_resistance_integral', 'tips_resistance_integral',
    'inflow_parallel_resistance', 'inflow_resistance_asymmetry',
    'tips_inflow_resistance_ratio',
    'splenic_dominance_index', 'splenoportal_path_chord_ratio',
    'collateral_burden_score', 'tree_area_conservation_mean_dev',
]


DEFAULT_N_POINTS = 100


# =====================================================================
# Helpers
# =====================================================================
def _safe_float(v, default=np.nan):
    """Return float(v) if finite, else `default`. Distinguishes missing (nan) from zero."""
    if v is None:
        return default
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _resample_with_mask(arr_with_nan, n_target):
    """
    Resample 1D array preserving NaN-mask. Returns (values, valid_mask).

    A resampled point is "valid" iff it lies within [first_valid, last_valid]
    of the source array (i.e. inside the NaN-protection envelope).
    """
    arr = np.asarray(arr_with_nan, dtype=np.float64)
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


def _resample(arr, n_target):
    """Plain linear resample (no NaN-handling) — for arc_length only."""
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) == n_target:
        return arr.copy()
    if len(arr) < 2:
        return np.zeros(n_target, np.float32)
    xp = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, n_target)
    return np.interp(x_new, xp, arr).astype(np.float32)


# =====================================================================
# Dataset
# =====================================================================
class PortalVeinDataset(Dataset):
    """Portal vein geometry + PVP labels, with selective inputs."""

    def __init__(self, root_dir: str, n_points: int = DEFAULT_N_POINTS,
                 label_key: str = 'PVP', verbose: bool = True):
        super().__init__()
        self.root_dir = root_dir
        self.n_points = n_points
        self.label_key = label_key
        self.verbose = verbose

        # ── Discover patients ───────────────────────────────────────
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
                # Fallback for when unified_features.json has empty `pointwise: {}`
                # (the user's preprocessing pipeline writes per-point profiles
                # to a separate file in some cases).
                'pointwise_fallback_file': os.path.join(
                    pdir, 'centerline_pointwise_profiles.json'),
                'is_post_tips': '#' in name,
            })

        if verbose:
            n_post = sum(p['is_post_tips'] for p in self.patients)
            print(f"[Dataset] Discovered {len(self.patients)} patients "
                  f"(post-TIPS: {n_post}, pre-TIPS: {len(self.patients) - n_post}).")

        # ── Pre-load everything ─────────────────────────────────────
        self.data = []
        for p in self.patients:
            item = self._load_one(p)
            if item is None:
                continue
            self.data.append(item)

        if verbose:
            n_empty_pw = sum(d.get('_pw_is_empty', False) for d in self.data)
            n_with_any_pw = len(self.data) - n_empty_pw
            n_from_unified = sum(d.get('_pw_source') == 'unified.pointwise'
                                 for d in self.data)
            n_from_fallback = sum(d.get('_pw_source') == 'centerline_pointwise_profiles.json'
                                  for d in self.data)
            n_with_tips = sum(d['segment_mask'][SEG_INDEX['tips']] for d in self.data)
            n_with_lpvrpv = sum((d['segment_mask'][SEG_INDEX['lpv']] *
                                 d['segment_mask'][SEG_INDEX['rpv']])
                                for d in self.data)
            print(f"[Dataset] Loaded {len(self.data)} valid patients.")
            print(f"          - with non-empty pointwise:     {n_with_any_pw}")
            print(f"          - with empty pointwise {{}}:     {n_empty_pw}  "
                  f"(these contribute 0 to all losses!)")
            print(f"          - source = unified.pointwise:   {n_from_unified}")
            print(f"          - source = fallback file:       {n_from_fallback}")
            print(f"          - with tips_segment:            {int(n_with_tips)}")
            print(f"          - with both lpv+rpv:            {int(n_with_lpvrpv)}")

        # ── Per-branch availability diagnostics + unit auto-fix ─────
        self._print_branch_diagnostics()
        self._detect_and_fix_units()

        # ── Compute global normalization (over valid points only) ───
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

            unified = {}
            if os.path.exists(p['unified_file']):
                # CRITICAL: explicit utf-8 encoding (Chinese Windows defaults to GBK,
                # which crashes on degree symbols, μ, π, or non-ASCII characters
                # in the JSON output of the preprocessing pipeline).
                with open(p['unified_file'], 'r', encoding='utf-8') as f:
                    unified = json.load(f)
            else:
                # No unified_features.json → skip this patient entirely.
                # We do NOT fall back to centerline_profiles.json — the unified
                # file is the single source of truth.
                return None

            # Per-patient is_post_tips: trust both folder convention and JSON meta
            meta = unified.get('_meta', {}) or {}
            is_post_tips = bool(meta.get('is_post_tips', p['is_post_tips']))

            # ── 11-dim aux vector with mask ────────────────────────
            aux, aux_mask = self._load_aux(unified)

            # ── 3D topology (for viz, not input) ───────────────────
            endpoints_3d = self._load_endpoints_3d(unified)
            confluence_3d = self._load_confluence_3d(unified)

            # ── Per-point profiles (with NaN-mask) ─────────────────
            # PRIMARY: unified_features.json's `pointwise` field
            # FALLBACK: centerline_pointwise_profiles.json (same folder)
            #          — used when `pointwise` in unified is empty.
            pw_src = unified.get('pointwise', {}) or {}
            pw_source_str = 'unified.pointwise'

            if (not pw_src or len(pw_src) == 0) and os.path.exists(p['pointwise_fallback_file']):
                try:
                    with open(p['pointwise_fallback_file'], 'r', encoding='utf-8') as f:
                        fb = json.load(f)
                    # The fallback file has segments at top level (mpv, sv, …, _meta)
                    # — drop _meta and any non-dict entries.
                    pw_src = {k: v for k, v in fb.items()
                              if isinstance(v, dict) and not k.startswith('_')}
                    pw_source_str = 'centerline_pointwise_profiles.json'
                except UnicodeDecodeError as e:
                    if self.verbose:
                        print(f"[Dataset] ❌ Unicode error reading fallback for "
                              f"{p['name']}: {e}")
                except Exception as e:
                    if self.verbose:
                        print(f"[Dataset] ⚠️  Could not load pointwise fallback for "
                              f"{p['name']}: {type(e).__name__}: {e}")

            # Track empty-pointwise patients separately for diagnostics
            pw_is_empty = (not pw_src) or (len(pw_src) == 0)

            profiles, point_valid, arc_lengths, segment_mask = \
                self._load_pointwise(pw_src, patient_name=p['name'],
                                     source_label=pw_source_str)

            # ── Optional: JSON-precomputed quantities for evaluation ──
            extras_for_eval = self._load_extras(unified)

            return {
                'name': p['name'],
                'profiles':     profiles,        # (S, N, 4)
                'point_valid':  point_valid,     # (S, N)
                'arc_lengths':  arc_lengths,     # (S, N) mm
                'segment_mask': segment_mask,    # (S,)
                'aux_scalars':  aux,             # (11,)
                'aux_mask':     aux_mask,        # (11,)
                'endpoints_3d':  endpoints_3d,   # (S, 2, 3) mm
                'confluence_3d': confluence_3d,  # (3,) mm
                'label':         np.float32(label),
                'is_post_tips':  is_post_tips,
                'extras_for_eval': extras_for_eval,
                '_pw_is_empty':  pw_is_empty,        # diagnostic flag
                '_pw_source':    pw_source_str,      # 'unified.pointwise' or 'centerline_…'
            }
        except UnicodeDecodeError as e:
            if self.verbose:
                print(f"[Dataset] ❌ Unicode error on {p['name']}: {e}")
                print(f"           File: {p['unified_file']}")
                print(f"           Hint: forcing utf-8 didn't work — "
                      f"file may actually be GBK-encoded or corrupted.")
            return None
        except Exception as e:
            if self.verbose:
                print(f"[Dataset] failed to load {p['name']}: "
                      f"{type(e).__name__}: {e}")
            return None

    def _load_aux(self, unified):
        out = np.zeros(N_AUX, dtype=np.float32)
        mask = np.zeros(N_AUX, dtype=np.float32)
        for i, key in enumerate(AUX_KEYS):
            section, jkey = AUX_LOOKUP[key]
            sec = unified.get(section, {}) or {}
            v = _safe_float(sec.get(jkey, None), np.nan)
            if np.isfinite(v):
                # For flags, force to 0/1
                if i in AUX_FLAG_INDICES:
                    out[i] = 1.0 if v > 0.5 else 0.0
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
        cp = sva.get('confluence_point_physical', None) if sva else None
        if cp is not None:
            return np.asarray(cp, dtype=np.float32)
        return np.zeros(3, dtype=np.float32)

    def _load_pointwise(self, pw_json, patient_name=None, source_label='?'):
        """
        Returns:
            profiles     (S, N, 4)
            point_valid  (S, N)
            arc_lengths  (S, N) — in mm
            segment_mask (S,)
        """
        S, N = N_SEGMENTS, self.n_points
        profiles    = np.zeros((S, N, N_PROFILE_FEAT), dtype=np.float32)
        point_valid = np.zeros((S, N), dtype=np.float32)
        arc_lengths = np.zeros((S, N), dtype=np.float32)
        seg_mask    = np.zeros(S, dtype=np.float32)

        if not pw_json:
            return profiles, point_valid, arc_lengths, seg_mask

        # Build case-insensitive key map (handles 'MPV' vs 'mpv' vs 'Mpv')
        key_map = {k.lower(): k for k in pw_json.keys() if isinstance(k, str)}

        # First-time inspection: log full structure of the FIRST populated patient
        # so we can verify the JSON schema matches our assumptions.
        if not hasattr(self, '_inspected_pointwise'):
            print(f"\n[Dataset] === Inspecting first populated pointwise "
                  f"(patient: {patient_name}, source: {source_label}) ===")
            print(f"  Top-level pointwise keys: {sorted(pw_json.keys())}")
            for k, v in pw_json.items():
                if isinstance(v, dict):
                    sub = sorted(v.keys())
                    print(f"  pointwise['{k}']: subkeys = {sub}")
                    for fkey in PROFILE_KEYS + ['arc_length_mm']:
                        if fkey in v:
                            arr = v[fkey]
                            n = len(arr) if hasattr(arr, '__len__') else 'scalar'
                            sample = list(arr[:3]) if hasattr(arr, '__len__') and len(arr) > 0 else 'empty'
                            print(f"      '{fkey}': len={n}, first3={sample}")
                else:
                    print(f"  pointwise['{k}']: type={type(v).__name__} (not a dict)")
            print()
            self._inspected_pointwise = True

        for si, sname in enumerate(SEGMENTS):
            seg_data = pw_json.get(sname, None)
            if seg_data is None and sname.lower() in key_map:
                seg_data = pw_json[key_map[sname.lower()]]
            if not seg_data or not isinstance(seg_data, dict):
                continue
            area_arr = seg_data.get('area', None)
            if area_arr is None or not hasattr(area_arr, '__len__') or len(area_arr) < 5:
                continue

            seg_mask[si] = 1.0

            # Resample arc_length, use as ground truth s(p)
            arc_raw = seg_data.get('arc_length_mm', None)
            if arc_raw is None or len(arc_raw) < 2:
                arc_lengths[si] = np.linspace(0, 1, N).astype(np.float32)
            else:
                arc_lengths[si] = _resample(arc_raw, N)

            # Resample each profile feature with NaN mask
            per_pt_valid = np.ones(N, dtype=np.float32)
            for fi, fkey in enumerate(PROFILE_KEYS):
                raw = seg_data.get(fkey, None)
                if raw is None:
                    profiles[si, :, fi] = 0.0
                    continue
                vals, mask = _resample_with_mask(raw, N)
                profiles[si, :, fi] = vals
                per_pt_valid = per_pt_valid * mask

            point_valid[si] = per_pt_valid

        return profiles, point_valid, arc_lengths, seg_mask

    def _load_extras(self, unified):
        """JSON-precomputed quantities for post-training evaluation only."""
        out = {}
        sysd = unified.get('system', {}) or {}
        for k in EVAL_SYSTEM_KEYS:
            out[k] = _safe_float(sysd.get(k, None), np.nan)
        return out

    # -----------------------------------------------------------------
    # Unit auto-detection (CRITICAL: catches data preprocessing bugs)
    # -----------------------------------------------------------------
    def _detect_and_fix_units(self):
        """
        Compare area, eq_diameter, inscribed_radius across all loaded patients.
        Internal consistency for circular cross-sections demands:
            area_mm² ≈ π · (eq_diameter_mm / 2)²
            eq_diameter_mm ≈ 2 · inscribed_radius_mm

        If a ratio is far from 1, a unit mismatch is likely. We auto-detect
        common conversion factors (cm vs mm) and apply them uniformly.

        This catches the (very common) preprocessing mistake where vtkXMLPolyData
        scalar arrays are stored in cm while metadata is in mm.
        """
        if not self.data:
            return
        A_all, D_all, R_all = [], [], []
        for d in self.data:
            v = (d['point_valid'] * d['segment_mask'][:, None]) > 0
            prof = d['profiles']
            A_all.append(prof[..., P_AREA][v])
            D_all.append(prof[..., P_DIAM][v])
            R_all.append(prof[..., P_INSC][v])
        A = np.concatenate(A_all); D = np.concatenate(D_all); R = np.concatenate(R_all)
        ok = (A > 1e-9) & (D > 1e-9) & (R > 1e-9)
        if ok.sum() < 50:
            return  # not enough valid points
        A, D, R = A[ok], D[ok], R[ok]

        # Ratio 1: ratio of (π·(D/2)²) / area  -- should be ≈ 1
        # If ≈ 100 → area is in cm², D is in mm
        # If ≈ 0.01 → area is in mm², D is in cm (scaled differently)
        # If ≈ 1 → both in same units (good)
        ratio_AD = float(np.median(np.pi * (D / 2.0) ** 2 / A))

        # Ratio 2: D / (2·R_insc)  -- should be ≈ 1
        # If ≈ 0.1 → D is in cm, R_insc is in mm
        # If ≈ 10  → D is in mm, R_insc is in cm
        ratio_DR = float(np.median(D / (2.0 * R)))

        scale_A, scale_D, scale_R = 1.0, 1.0, 1.0  # multiplicative fixes

        # Detect: D is in cm (×10), R is in mm
        if 0.05 < ratio_DR < 0.2:
            scale_D = 10.0   # cm → mm
        elif 5.0 < ratio_DR < 20.0:
            scale_R = 10.0   # cm → mm

        # Detect: A is in cm² (×100), D is in mm  (after D fix)
        # After scaling D, recompute the area-vs-diameter consistency
        # Expected: π·(D_fixed/2)² / A  ≈ 1 if A is correct
        # If ≈ 100 → A in cm² → multiply A by 100
        eff_ratio_AD = ratio_AD * (scale_D ** 2)
        if 50.0 < eff_ratio_AD < 200.0:
            scale_A = 100.0  # cm² → mm²
        elif 0.005 < eff_ratio_AD < 0.02:
            scale_A = 0.01   # m² → mm² (very unlikely, but possible)

        if scale_A != 1.0 or scale_D != 1.0 or scale_R != 1.0:
            print("=" * 64)
            print("[Dataset] ⚠️  UNIT INCONSISTENCY DETECTED IN POINTWISE DATA")
            print(f"  median ratio  π·(D/2)² / area = {ratio_AD:.4f}  (expected ≈ 1)")
            print(f"  median ratio  D / (2·R_insc)   = {ratio_DR:.4f}  (expected ≈ 1)")
            print(f"  → Auto-correcting:  area  × {scale_A}")
            print(f"                      diam  × {scale_D}")
            print(f"                      R_insc× {scale_R}")
            print(f"  These are critical for Hagen-Poiseuille (WSS, Re, ΔP).")
            print(f"  Check your CT preprocessing pipeline for unit consistency.")
            print("=" * 64)

            for d in self.data:
                d['profiles'][..., P_AREA] *= scale_A
                d['profiles'][..., P_DIAM] *= scale_D
                d['profiles'][..., P_INSC] *= scale_R
        else:
            if self.verbose:
                print(f"[Dataset] Unit check OK: A/πr² ratio={ratio_AD:.3f}, "
                      f"D/2R ratio={ratio_DR:.3f} (both should be ≈ 1)")

    # -----------------------------------------------------------------
    # Per-branch loading diagnostics
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
            mark = "✓" if n > 0 else "❌  ZERO patients have this branch!"
            print(f"   {sn:>5s} : {n:3d} / {len(self.data)}  {mark}")
        # Anatomical sanity warnings
        if n_per_branch['lpv'] == 0 or n_per_branch['rpv'] == 0:
            print("[Dataset] ⚠️  WARNING: LPV or RPV missing from ALL patients.")
            print("           Anatomically, every patient should have both.")
            print("           Check your JSON's `pointwise` keys — possibly using")
            print("           different names (e.g. 'LPV', 'left_portal_vein').")
        n_post_tips_with_tips = sum(
            d['segment_mask'][SEG_INDEX['tips']] for d in self.data
            if d['is_post_tips']
        )
        n_post_tips_total = sum(d['is_post_tips'] for d in self.data)
        if n_post_tips_total > 0 and n_post_tips_with_tips == 0:
            print(f"[Dataset] ⚠️  WARNING: {n_post_tips_total} post-TIPS patients but"
                  f" 0 have a tips pointwise segment.")
            print(f"           Check `pointwise.tips` key in JSON.")
    def _compute_normalization(self):
        # Per-channel statistics over valid points across the entire dataset
        per_channel_vals = [[] for _ in range(N_PROFILE_FEAT)]
        for d in self.data:
            valid_mask = d['point_valid'] * d['segment_mask'][:, None]  # (S,N)
            for fi in range(N_PROFILE_FEAT):
                v = d['profiles'][:, :, fi][valid_mask > 0]
                if len(v):
                    per_channel_vals[fi].append(v)

        means = np.zeros(N_PROFILE_FEAT, dtype=np.float32)
        stds  = np.ones(N_PROFILE_FEAT,  dtype=np.float32)
        for fi in range(N_PROFILE_FEAT):
            if per_channel_vals[fi]:
                arr = np.concatenate(per_channel_vals[fi])
                means[fi] = arr.mean()
                s = arr.std()
                stds[fi] = max(s, 1e-6)
        self.profile_mean = means
        self.profile_std  = stds

        # Aux scalar normalization (only over present samples; flags untouched)
        aux_vals = []
        aux_pres = []
        for d in self.data:
            aux_vals.append(d['aux_scalars'])
            aux_pres.append(d['aux_mask'])
        aux_vals = np.stack(aux_vals)  # (P, 11)
        aux_pres = np.stack(aux_pres)
        aux_mean = np.zeros(N_AUX, dtype=np.float32)
        aux_std  = np.ones(N_AUX,  dtype=np.float32)
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

        # Label statistics
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
        profiles = d['profiles']                   # (S, N, 4)
        point_valid = d['point_valid'][..., None]  # (S, N, 1)
        # Normalize, mask-aware (set invalid points to 0 after norm)
        profiles_norm = (profiles - self.profile_mean) / self.profile_std
        profiles_norm = profiles_norm * point_valid

        # Normalize aux (skip flags)
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
            'endpoints_3d':    torch.from_numpy(d['endpoints_3d']).float(),
            'confluence_3d':   torch.from_numpy(d['confluence_3d']).float(),
            'is_post_tips':    torch.tensor(float(d['is_post_tips'])),
            'label':           torch.tensor(float(d['label'])).float(),
            'label_norm':      torch.tensor(float(label_norm)).float(),
            # Returned untouched for post-training evaluation
            'extras_for_eval': d['extras_for_eval'],
        }


# =====================================================================
# Collate (handles per-sample dict batches; numpy "extras_for_eval"
# stays as a list, not stacked)
# =====================================================================
def collate_fn(items):
    """Batch a list of dicts from PortalVeinDataset.__getitem__."""
    # Tensor keys to stack
    tensor_keys = ['profiles', 'profiles_norm', 'arc_lengths', 'point_valid',
                   'segment_mask', 'aux_scalars', 'aux_norm', 'aux_mask',
                   'endpoints_3d', 'confluence_3d', 'is_post_tips',
                   'label', 'label_norm']
    out = {}
    for k in tensor_keys:
        out[k] = torch.stack([it[k] for it in items], dim=0)
    out['name'] = [it['name'] for it in items]
    out['extras_for_eval'] = [it['extras_for_eval'] for it in items]
    return out


# =====================================================================
# Quick sanity test
# =====================================================================
if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else '/tmp/pvp_test'
    ds = PortalVeinDataset(root, n_points=100, verbose=True)
    print(f"\nDataset size: {len(ds)}")
    if len(ds) == 0:
        print("Empty dataset — nothing to inspect.")
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

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(loader))
    print("\nBatch shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:>16s}: {tuple(v.shape)}")
        else:
            print(f"  {k:>16s}: {type(v).__name__} (len {len(v)})")