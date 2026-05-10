"""
Physics-Informed Geometric Deep Learning for Portal Vein Pressure
==================================================================
v3 — Poiseuille-grounded, Q-parameterized, mass-conservation-by-construction

设计原则 (Design Principles)
─────────────────────────────────────────────────────────────────
1.  **One learned scalar per branch: Q (flow rate).**
    Every per-point hemodynamic field — velocity, WSS, Reynolds,
    local/cumulative resistance, pressure drop — is DERIVED from
    Q + geometry by Hagen-Poiseuille. The fields cannot drift apart
    from each other, because they share a common Q.

2.  **Mass conservation is enforced by parameterization, not by loss.**
    Inflow split  = softmax(logits)  → Q_sv + Q_smv = Q_mpv
    Outflow split = softmax(logits)  → Q_lpv + Q_rpv + Q_tips = Q_mpv
    Murray's law provides the *prior* on these logits (delta=0 ⇒ Murray-3
    split). The model learns DEVIATIONS from Murray.

3.  **Loss terms are real fluid mechanics, named with units.**
        L_main     : Huber on PVP                   [mmHg²]
        L_murray   : ‖logit deviations‖²            [dimensionless prior]
        L_pressure : pressure continuity at junctions [Pa²]
        L_smooth   : 2nd-derivative of radius profile [mm²/mm⁴]
        L_physio   : soft hinge on WSS, Re ranges     [Pa², ─]

4.  **Interpretable outputs in physical units.**
    Returned `hemodynamics` dict has named keys with SI units suffixed
    (`_pa`, `_m_per_s`, `_pa_s_per_m3`, dimensionless `reynolds`).
    These can be mapped back onto STL surfaces and compared with CFD
    quantitatively.

Architecture
─────────────────────────────────────────────────────────────────
                                 ┌─────────────────────────────┐
profiles, arc, masks ────────────►│ GeometryEncoder (shared)    │
                                 │   per-segment 1D-CNN        │
                                 │ ──► per-point hidden h(s)   │
                                 └────────────┬─────────────────┘
                                              │
                              ┌───────────────▼───────────────┐
                              │ AttentionPool (per branch)    │
                              │ ──► branch embedding e_i      │
                              └───────────────┬───────────────┘
                                              │
              aux_scalars ──────────────►┌────▼──────────────────────┐
                                         │ FlowRateEstimator         │
                                         │   inflow_split  (softmax) │
                                         │   outflow_split (softmax) │
                                         │ ──► Q_i for each branch   │
                                         └────┬──────────────────────┘
                                              │
                              ┌───────────────▼───────────────┐
                              │ PoiseuilleHydrodynamics       │
                              │ (no params; physical formulas)│
                              │ Q_i + geometry ──► v, P, τ_w, │
                              │   Re, R(s), ΔP(s) per point   │
                              └───────────────┬───────────────┘
                                              │
                                   ┌──────────▼──────────┐
                                   │ JunctionPhysics     │
                                   │ Murray dev / press. │
                                   │ continuity residuals│
                                   └──────────┬──────────┘
                                              │
                  ┌───────────────────────────▼─────────────────┐
                  │  PVPPredictor MLP                           │
                  │  inputs: branch_embeds, Q, ΔP@junctions,    │
                  │          residuals, aux_norm                │
                  └────────────────────┬────────────────────────┘
                                       │
                                  PVP_pred (B,1)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import (
    N_PROFILE_FEAT, N_SEGMENTS, SEGMENTS, SEG_INDEX, N_AUX,
    P_AREA, P_DIAM, P_CURV, P_INSC,
)


# =====================================================================
# Physical constants (blood at 37°C)
# =====================================================================
BLOOD_VISCOSITY_PA_S      = 3.5e-3        # μ ≈ 3.5 mPa·s = 3.5 cP
BLOOD_DENSITY_KG_M3       = 1060.0        # ρ
BLOOD_KIN_VISCOSITY_M2_S  = BLOOD_VISCOSITY_PA_S / BLOOD_DENSITY_KG_M3  # ν ≈ 3.3e-6
# Reference portal vein flow: ~800 mL/min = 1.33e-5 m³/s
Q_REF_M3_PER_S            = 800.0 * 1e-6 / 60.0   # ≈ 1.33e-5 m³/s

# Physiological ranges (used in soft-hinge loss)
WSS_PHYSIO_LO_PA   = 0.05    # below this is "stagnant", elevated thrombosis risk
WSS_PHYSIO_HI_PA   = 5.0     # above this is "high shear"
RE_PHYSIO_HI       = 1500.0  # turbulence threshold (laminar Poiseuille assumption)


# =====================================================================
# Module 1 — Poiseuille Hydrodynamics (deterministic, no parameters)
# =====================================================================
class PoiseuilleHydrodynamics(nn.Module):
    """
    Given:
        profiles (B, N, 4)   [area_mm², diameter_mm, curvature_1/mm, inscribed_radius_mm]
        arc      (B, N)      arc length in mm (monotonic from idx 0 → N-1)
        valid    (B, N)      ∈ {0, 1}
        Q_rel    (B,)        relative flow rate (Q_mpv = 1)

    Returns dict of per-point fields, all in physical units:
        radius_m            (B, N)
        area_m2             (B, N)
        velocity_m_per_s    (B, N)        v = Q / A
        wss_pa              (B, N)        τ = 4μQ / (π r³)
        reynolds            (B, N)        Re = vD/ν
        local_R_pa_s_per_m4 (B, N)        R'(s) = 8μ / (π r⁴)        [per metre length]
        cum_R_pa_s_per_m3   (B, N)        ∫R'(s) ds                  [Pa·s/m³]
        pressure_drop_pa    (B, N)        ΔP(s) = Q · cum_R(s)        [Pa, from idx 0]
        dean                (B, N)        De = Re · √(D / R_curv)     dimensionless
        area_gradient       (B, N)        dA/ds                       mm

    Note: The convention is that idx 0 is the "reference end" w.r.t. flow.
    Cumulative pressure drop at idx 0 is 0 by construction; ΔP grows along s.
    For MPV, idx 0 = confluence end (where flow ENTERS), so ΔP grows toward
    the bifurcation. This matches the physiological direction.
    """

    def __init__(self,
                 mu: float = BLOOD_VISCOSITY_PA_S,
                 rho: float = BLOOD_DENSITY_KG_M3,
                 q_ref: float = Q_REF_M3_PER_S):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.nu = mu / rho
        self.q_ref = q_ref

    @staticmethod
    def _safe_seg_lengths(arc, valid, eps=1e-6):
        """Compute per-segment ds (in mm) from arc array, with valid-mask aware."""
        ds = torch.zeros_like(arc)
        ds[..., 1:] = arc[..., 1:] - arc[..., :-1]
        ds[..., 0]  = ds[..., 1]
        ds = ds.clamp(min=eps) * (valid > 0).float()  # zero out invalid
        return ds

    def forward(self, profiles, arc, valid, Q_rel):
        """
        profiles: (B, N, 4),  arc: (B, N),  valid: (B, N),  Q_rel: (B,)
        """
        eps = 1e-9

        # Geometry (convert to SI)
        area_mm2  = profiles[..., P_AREA].clamp(min=eps)
        diam_mm   = profiles[..., P_DIAM].clamp(min=eps)
        curv_inv_mm = profiles[..., P_CURV].abs()
        # circular inscribed radius is in mm, sometimes useful as conservative r
        # we use eq_diameter / 2 as the hydraulic radius for Poiseuille
        radius_mm = 0.5 * diam_mm

        area_m2  = area_mm2  * 1e-6                   # mm² → m²
        diam_m   = diam_mm   * 1e-3
        radius_m = radius_mm * 1e-3
        curv_inv_m = curv_inv_mm * 1e3                # 1/mm → 1/m

        # Absolute flow per point: Q (m³/s) = Q_rel × Q_ref. (B, 1)
        Q_abs = Q_rel.unsqueeze(-1) * self.q_ref

        # 1. Velocity:                v = Q / A
        velocity = Q_abs / (area_m2 + eps)

        # 2. Wall shear stress:       τ_w = 4 μ Q / (π r³)   [Pa]
        wss = (4.0 * self.mu * Q_abs) / (math.pi * radius_m.pow(3) + eps)

        # 3. Reynolds:                Re = v D / ν
        reynolds = velocity * diam_m / (self.nu + eps)

        # 4. Local resistance per unit length:  R'(s) = 8μ/(π r⁴) [Pa·s/m⁴]
        local_R = (8.0 * self.mu) / (math.pi * radius_m.pow(4) + eps)

        # 5. Cumulative resistance:   ∫ R'(s) ds  [Pa·s/m³]
        # Use trapezoidal integration along arc; ds in metres.
        ds_m = self._safe_seg_lengths(arc, valid) * 1e-3   # mm → m
        # Trapezoidal: mean of adjacent local_R times ds, cumsum
        # We use the simpler left-Riemann (R'(s) at point i × ds[i])
        local_R_per_seg = local_R * ds_m
        cum_R = torch.cumsum(local_R_per_seg, dim=-1)

        # 6. Pressure drop (Pa) cumulative from idx 0
        pressure_drop = Q_abs * cum_R

        # 7. Dean number proxy:       De = Re × √(D × κ)
        dean = reynolds * torch.sqrt(diam_m * curv_inv_m + eps)

        # 8. Area gradient (mm² / mm)
        area_grad = torch.zeros_like(area_mm2)
        ds_mm = self._safe_seg_lengths(arc, valid)
        area_grad[..., 1:] = (area_mm2[..., 1:] - area_mm2[..., :-1]) / ds_mm[..., 1:]
        area_grad[..., 0] = area_grad[..., 1]

        # Mask out invalid points (for visualization / loss).
        # Note: cum_R was integrated using ds_m which is already zeroed at
        # invalid points, so cum_R is monotonic and correct up through the
        # last valid point. After masking, max-along-arc gives the total drop.
        v = (valid > 0).float()
        out = {
            'radius_m':           radius_m,
            'area_m2':            area_m2,
            'velocity_m_per_s':   velocity * v,
            'wss_pa':             wss * v,
            'reynolds':           reynolds * v,
            'local_R_pa_s_per_m4': local_R * v,
            'cum_R_pa_s_per_m3':  cum_R * v,
            'pressure_drop_pa':   pressure_drop * v,
            'dean':               dean * v,
            'area_gradient':      area_grad * v,
            # Convenient scalar summaries (used by JunctionPhysics & predictor)
            #   "_total" = value at the LAST VALID point (== max for monotonic cum_R)
            'cum_R_total':        (cum_R * v).max(dim=-1).values,         # (B,)
            'pressure_drop_total':(pressure_drop * v).max(dim=-1).values, # (B,)
            'valid_count':        v.sum(dim=-1),                          # (B,)
        }
        # Sanitize any NaN / inf
        for k in out:
            out[k] = torch.nan_to_num(out[k], nan=0.0, posinf=0.0, neginf=0.0)
        return out


# =====================================================================
# Module 2 — Geometry Encoder (shared across branches)
# =====================================================================
class GeometryEncoder(nn.Module):
    """
    Per-segment 1D-CNN encoder. Shared weights across all 6 segments
    (the same fluid mechanics applies to all vessels — better
    generalization on small datasets).

    Input:  (B, N, 4) profiles (already normalized)
    Output: (B, N, H) per-point hidden context
    """

    def __init__(self, d_in: int = N_PROFILE_FEAT, d_hidden: int = 32,
                 n_blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        chans = [d_in] + [d_hidden] * n_blocks
        ks = [7, 5, 3]

        layers = []
        for i in range(n_blocks):
            layers += [
                nn.Conv1d(chans[i], chans[i + 1],
                          kernel_size=ks[i % len(ks)],
                          padding=ks[i % len(ks)] // 2),
                nn.BatchNorm1d(chans[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        self.encoder = nn.Sequential(*layers)
        self.d_hidden = d_hidden

    def forward(self, profiles_norm):
        """profiles_norm: (B, N, 4)  →  hidden: (B, N, H)"""
        x = profiles_norm.transpose(1, 2)  # (B, 4, N)
        h = self.encoder(x)                # (B, H, N)
        return h.transpose(1, 2)           # (B, N, H)


# =====================================================================
# Module 3 — Mask-aware Attention Pooling
# =====================================================================
class AttentionPool(nn.Module):
    """Pools per-point hidden into a per-segment embedding, respecting valid mask."""

    def __init__(self, d_hidden: int, d_attn: int = 16):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_hidden, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, 1),
        )

    def forward(self, hidden, valid_mask):
        """
        hidden    : (B, N, H)
        valid_mask: (B, N)
        returns   : pooled (B, H), attn_w (B, N)
        """
        scores = self.attn(hidden).squeeze(-1)            # (B, N)
        scores = scores.masked_fill(valid_mask < 0.5, float('-inf'))
        # Avoid all-mask rows producing NaN
        all_masked = (valid_mask.sum(dim=-1) < 0.5)       # (B,)
        scores = torch.where(
            all_masked.unsqueeze(-1).expand_as(scores),
            torch.zeros_like(scores), scores
        )
        attn_w = F.softmax(scores, dim=-1)
        pooled = torch.einsum('bn,bnh->bh', attn_w, hidden)
        # If everything was masked, return zeros
        pooled = pooled * (~all_masked).float().unsqueeze(-1)
        return pooled, attn_w


# =====================================================================
# Module 4 — Flow Rate Estimator (Murray-prior + learned residual)
# =====================================================================
class FlowRateEstimator(nn.Module):
    """
    Predicts per-segment relative flow rates Q_i (B, S) such that:

      Inflow            : Q_sv + Q_smv                    = 1                (reference)
      Confluence outflow: Q_mpv + Q_lgv + Q_pgv           = 1                (mass cons.)
      Bifurcation       : Q_lpv + Q_rpv + Q_tips          = Q_mpv            (mass cons.)

    Q_mpv ≤ 1 — strictly less than 1 when collateral vessels (LGV/PGV)
    siphon flow off the portal system. The "collateral burden"
    1 − Q_mpv = Q_lgv + Q_pgv is directly meaningful for portal hypertension.

    Implementation per junction:
        target_logit_i = 3 · log(d_junction_i)        (Murray-3 prior)
        actual_logit   = target_logit_i + delta(MLP)  (learned correction)
        split          = softmax over present branches
    Murray's law is the prior; the model learns deviations (`delta`)
    that capture biologically meaningful departures from energy-optimal
    flow distribution (e.g., elevated splenic flow in cirrhosis).
    """

    def __init__(self, d_branch: int, d_aux: int = N_AUX, d_hidden: int = 32):
        super().__init__()
        # Inflow split (sv, smv): 2 logits
        self.inflow_head = nn.Sequential(
            nn.Linear(2 * d_branch + d_aux, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 2),
        )
        # NEW: Confluence-outflow split (mpv, lgv, pgv): 3 logits.
        # This captures the collateral burden — how much flow is being
        # siphoned off via porto-systemic shunts.
        self.conf_outflow_head = nn.Sequential(
            nn.Linear(3 * d_branch + d_aux, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 3),
        )
        # Bifurcation outflow split (lpv, rpv, tips): 3 logits
        self.bif_outflow_head = nn.Sequential(
            nn.Linear(4 * d_branch + d_aux, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 3),
        )
        # Init: deltas at zero so model starts at Murray prior at every junction
        for m in [self.inflow_head[-1], self.conf_outflow_head[-1],
                  self.bif_outflow_head[-1]]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    @staticmethod
    def _murray_logits(diameters_mm, eps=1e-3):
        """target_logit_i = 3 · log(d_i),  so softmax → d_i³ / Σ d_j³"""
        return 3.0 * torch.log(diameters_mm.clamp(min=eps))

    def forward(self, branch_embeds, aux_norm, segment_mask, junction_diameters):
        """
        branch_embeds       : (B, S, H)  per-segment embedding (zero where missing)
        aux_norm            : (B, A)
        segment_mask        : (B, S)
        junction_diameters  : (B, S)  diameter at each segment's junction-facing endpoint, mm

        Returns Q_per_branch: (B, S) and per-junction split logits/deltas.
        """
        B = branch_embeds.size(0)
        device = branch_embeds.device
        i_mpv  = SEG_INDEX['mpv']
        i_sv   = SEG_INDEX['sv']
        i_smv  = SEG_INDEX['smv']
        i_lpv  = SEG_INDEX['lpv']
        i_rpv  = SEG_INDEX['rpv']
        i_tips = SEG_INDEX['tips']
        i_lgv  = SEG_INDEX['lgv']
        i_pgv  = SEG_INDEX['pgv']

        # ── Inflow split (sv, smv) ──────────────────────────────
        ctx_in = torch.cat([
            branch_embeds[:, i_sv],
            branch_embeds[:, i_smv],
            aux_norm,
        ], dim=-1)
        inflow_delta = self.inflow_head(ctx_in)             # (B, 2)
        d_sv  = junction_diameters[:, i_sv]
        d_smv = junction_diameters[:, i_smv]
        prior_in = torch.stack([
            self._murray_logits(d_sv), self._murray_logits(d_smv)
        ], dim=-1)
        mask_in = torch.stack([segment_mask[:, i_sv], segment_mask[:, i_smv]], dim=-1)
        logits_in = prior_in + inflow_delta
        logits_in = logits_in.masked_fill(mask_in < 0.5, float('-1e9'))
        inflow_frac = F.softmax(logits_in, dim=-1)          # (B, 2)
        no_inflow = (mask_in.sum(dim=-1) < 0.5).unsqueeze(-1)
        inflow_frac = torch.where(
            no_inflow, torch.full_like(inflow_frac, 0.5), inflow_frac
        )

        # ── NEW: Confluence-outflow split (mpv, lgv, pgv) ──────
        ctx_co = torch.cat([
            branch_embeds[:, i_mpv],
            branch_embeds[:, i_lgv],
            branch_embeds[:, i_pgv],
            aux_norm,
        ], dim=-1)
        conf_outflow_delta = self.conf_outflow_head(ctx_co)  # (B, 3)
        d_mpv  = junction_diameters[:, i_mpv]
        d_lgv  = junction_diameters[:, i_lgv]
        d_pgv  = junction_diameters[:, i_pgv]
        prior_co = torch.stack([
            self._murray_logits(d_mpv),
            self._murray_logits(d_lgv),
            self._murray_logits(d_pgv),
        ], dim=-1)
        mask_co = torch.stack([
            segment_mask[:, i_mpv],
            segment_mask[:, i_lgv],
            segment_mask[:, i_pgv],
        ], dim=-1)
        logits_co = prior_co + conf_outflow_delta
        logits_co = logits_co.masked_fill(mask_co < 0.5, float('-1e9'))
        conf_outflow_frac = F.softmax(logits_co, dim=-1)     # (B, 3)
        # If MPV is absent (impossible normally) fall back to all-mpv = 1
        no_co = (mask_co.sum(dim=-1) < 0.5).unsqueeze(-1)
        conf_outflow_frac = torch.where(
            no_co, torch.tensor([[1.0, 0.0, 0.0]], device=device).expand_as(conf_outflow_frac),
            conf_outflow_frac
        )

        # ── Bifurcation outflow split (lpv, rpv, tips) ─────────
        ctx_bo = torch.cat([
            branch_embeds[:, i_mpv],
            branch_embeds[:, i_lpv],
            branch_embeds[:, i_rpv],
            branch_embeds[:, i_tips],
            aux_norm,
        ], dim=-1)
        bif_outflow_delta = self.bif_outflow_head(ctx_bo)    # (B, 3)
        d_lpv  = junction_diameters[:, i_lpv]
        d_rpv  = junction_diameters[:, i_rpv]
        d_tips = junction_diameters[:, i_tips]
        prior_bo = torch.stack([
            self._murray_logits(d_lpv),
            self._murray_logits(d_rpv),
            self._murray_logits(d_tips),
        ], dim=-1)
        mask_bo = torch.stack([
            segment_mask[:, i_lpv], segment_mask[:, i_rpv], segment_mask[:, i_tips]
        ], dim=-1)
        logits_bo = prior_bo + bif_outflow_delta
        logits_bo = logits_bo.masked_fill(mask_bo < 0.5, float('-1e9'))
        bif_outflow_frac = F.softmax(logits_bo, dim=-1)      # (B, 3)
        no_bo = (mask_bo.sum(dim=-1) < 0.5).unsqueeze(-1)
        bif_outflow_frac = torch.where(
            no_bo, torch.full_like(bif_outflow_frac, 1.0/3.0), bif_outflow_frac
        )

        # ── Assemble Q per segment via torch.stack (no in-place writes
        #    to a fresh tensor — that breaks autograd version tracking) ──
        Q_mpv  = conf_outflow_frac[:, 0]
        Q_lgv  = conf_outflow_frac[:, 1]
        Q_pgv  = conf_outflow_frac[:, 2]
        Q_sv   = inflow_frac[:, 0]
        Q_smv  = inflow_frac[:, 1]
        Q_lpv  = bif_outflow_frac[:, 0] * Q_mpv
        Q_rpv  = bif_outflow_frac[:, 1] * Q_mpv
        Q_tips = bif_outflow_frac[:, 2] * Q_mpv

        Q_list = [None] * N_SEGMENTS
        Q_list[i_mpv]  = Q_mpv
        Q_list[i_sv]   = Q_sv
        Q_list[i_smv]  = Q_smv
        Q_list[i_lpv]  = Q_lpv
        Q_list[i_rpv]  = Q_rpv
        Q_list[i_tips] = Q_tips
        Q_list[i_lgv]  = Q_lgv
        Q_list[i_pgv]  = Q_pgv
        Q = torch.stack(Q_list, dim=-1)            # (B, N_SEGMENTS)
        # Zero out missing branches
        Q = Q * segment_mask

        # Collateral burden: portal flow fraction that bypasses the liver via
        # LGV/PGV at the confluence. ∈ [0, 1]. Important clinical biomarker.
        collateral_fraction = (Q_lgv * segment_mask[:, i_lgv]
                               + Q_pgv * segment_mask[:, i_pgv])  # (B,)

        return {
            'Q':                    Q,                       # (B, S)
            'inflow_frac':          inflow_frac,             # (B, 2)
            'conf_outflow_frac':    conf_outflow_frac,       # (B, 3)
            'bif_outflow_frac':     bif_outflow_frac,        # (B, 3)
            'inflow_delta':         inflow_delta,            # (B, 2)
            'conf_outflow_delta':   conf_outflow_delta,      # (B, 3)
            'bif_outflow_delta':    bif_outflow_delta,       # (B, 3)
            'inflow_mask':          mask_in,                 # (B, 2)
            'conf_outflow_mask':    mask_co,                 # (B, 3)
            'bif_outflow_mask':     mask_bo,                 # (B, 3)
            'collateral_fraction':  collateral_fraction,     # (B,)
        }


# =====================================================================
# Module 5 — Junction Physics (no parameters)
# =====================================================================
# Convention for endpoint indexing along each branch's centerline:
#     • idx 0  = junction-facing end for SV, SMV, LPV, RPV, TIPS
#     • For MPV, idx 0 = confluence end, idx N-1 = bifurcation end
#
# So the "junction-end" index per branch per junction is:
JUNCTION_END = {
    'confluence':  {'mpv': 0,   'sv': 0, 'smv': 0},
    'bifurcation': {'mpv': -1,  'lpv': 0, 'rpv': 0, 'tips': 0},
}


class JunctionPhysics(nn.Module):
    """
    Computes physically-meaningful residuals at each junction:

        murray_dev_X      : norm of the FlowRateEstimator's `delta` logits at
                            junction X. Zero ⇒ Murray-3 split exactly.

        pressure_residual : the predicted pressure at the junction differs
                            between branches that share that junction.
                            (Should be near 0 if the model is consistent.)

    These residuals are returned BOTH as model-input features (so the
    PVPPredictor can use them) AND as loss terms.
    """

    # Characteristic pressure scale (Pa) — used to normalize bifurcation residual.
    # Portal vein typical Δ P along a segment is O(10–100) Pa. Setting this
    # makes (ΔP_lpv − ΔP_rpv)² / scale² a dimensionless O(1) residual.
    P_SCALE_PA = 100.0

    def forward(self, hemo_per_seg, flow_out, segment_mask, has_tips):
        """
        Three Murray-deviation residuals (one per junction) and a bifurcation
        pressure-continuity residual. Collateral burden (Q_lgv + Q_pgv) is
        emitted as a feature so the predictor can use it directly.

        Physics philosophy
        ──────────────────
        ❌  No divisions by Q anywhere — the divisive R = ΔP/Q caused
            catastrophic numerics in earlier versions.

        Murray-3 prior at three junctions:
            inflow:      Q_sv : Q_smv             ∝ d_sv³  : d_smv³
            conf_out:    Q_mpv : Q_lgv : Q_pgv    ∝ d_mpv³ : d_lgv³ : d_pgv³
            bif_out:     Q_lpv : Q_rpv : Q_tips   ∝ d_lpv³ : d_rpv³ : d_tips³

        Pressure continuity at bifurcation: LPV and RPV both feed roughly
        equal hepatic-sinusoidal pressures, so ΔP_lpv ≈ ΔP_rpv. TIPS is
        excluded (its outlet is systemic vein pressure, structurally lower).
        """
        B = segment_mask.size(0)
        device = segment_mask.device

        i_mpv = SEG_INDEX['mpv']; i_sv = SEG_INDEX['sv']; i_smv = SEG_INDEX['smv']
        i_lpv = SEG_INDEX['lpv']; i_rpv = SEG_INDEX['rpv']; i_tips = SEG_INDEX['tips']
        i_lgv = SEG_INDEX['lgv']; i_pgv = SEG_INDEX['pgv']

        # Junction activity masks
        m_inflow = (segment_mask[:, i_sv] * segment_mask[:, i_smv])
        m_conf_out = segment_mask[:, i_mpv]   # MPV must exist for conf-outflow node
        m_bif      = (segment_mask[:, i_mpv] *
                      (segment_mask[:, i_lpv] + segment_mask[:, i_rpv] + segment_mask[:, i_tips] > 0).float())
        m_bif_lpvrpv = segment_mask[:, i_lpv] * segment_mask[:, i_rpv]   # both children present

        # ── Murray deviations: ‖delta‖² per junction ───────────
        murr_in   = flow_out['inflow_delta'].pow(2).sum(dim=-1)        * m_inflow
        murr_co   = flow_out['conf_outflow_delta'].pow(2).sum(dim=-1)  * m_conf_out
        murr_bo   = flow_out['bif_outflow_delta'].pow(2).sum(dim=-1)   * m_bif

        # ── ΔP per branch (Pa, ≥ 0, no division anywhere) ──────
        dP_mpv  = hemo_per_seg[i_mpv]['pressure_drop_total']
        dP_sv   = hemo_per_seg[i_sv]['pressure_drop_total']
        dP_smv  = hemo_per_seg[i_smv]['pressure_drop_total']
        dP_lpv  = hemo_per_seg[i_lpv]['pressure_drop_total']
        dP_rpv  = hemo_per_seg[i_rpv]['pressure_drop_total']
        dP_tips = hemo_per_seg[i_tips]['pressure_drop_total']
        dP_lgv  = hemo_per_seg[i_lgv]['pressure_drop_total']
        dP_pgv  = hemo_per_seg[i_pgv]['pressure_drop_total']

        # Bifurcation pressure residual: only when both LPV and RPV are present.
        # Normalized by P_SCALE_PA² → dimensionless, O(1).
        press_resid_bifurc = (((dP_lpv - dP_rpv) / self.P_SCALE_PA).pow(2)) * m_bif_lpvrpv

        # ── Interpretable feature vector for the predictor ─────
        # 12 dims: 3 Murray devs + 1 bifurcation residual + 1 collateral
        # fraction + 7 log1p(ΔP) per non-confluence-degenerate branch.
        # log1p(ΔP) is bounded for any finite non-negative input → numerically safe.
        collat_frac = flow_out['collateral_fraction']  # (B,)
        features = torch.stack([
            murr_in, murr_co, murr_bo,
            press_resid_bifurc,
            collat_frac,                                                       # NEW
            torch.log1p(dP_mpv.clamp(min=0))  * segment_mask[:, i_mpv],
            torch.log1p(dP_sv.clamp(min=0))   * segment_mask[:, i_sv],
            torch.log1p(dP_smv.clamp(min=0))  * segment_mask[:, i_smv],
            torch.log1p(dP_lpv.clamp(min=0))  * segment_mask[:, i_lpv],
            torch.log1p(dP_rpv.clamp(min=0))  * segment_mask[:, i_rpv],
            torch.log1p(dP_tips.clamp(min=0)) * segment_mask[:, i_tips],
            torch.log1p(dP_lgv.clamp(min=0))  * segment_mask[:, i_lgv],         # NEW
            torch.log1p(dP_pgv.clamp(min=0))  * segment_mask[:, i_pgv],         # NEW
        ], dim=-1)                                                             # (B, 13)
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            'murray_dev_inflow':  murr_in,
            'murray_dev_conf_out': murr_co,
            'murray_dev_bif_out': murr_bo,
            'press_resid_bifurc': press_resid_bifurc,
            'collateral_fraction': collat_frac,
            'inflow_active':      m_inflow,
            'confluence_outflow_active': m_conf_out,
            'bifurcation_active': m_bif,
            'dP_per_branch':      torch.stack([dP_mpv, dP_sv, dP_smv, dP_lpv,
                                               dP_rpv, dP_tips, dP_lgv, dP_pgv], dim=-1),
            'features':           features,                                    # (B, 13)
        }


# =====================================================================
# Module 6 — Full model
# =====================================================================
class PortalPressureNet(nn.Module):
    """
    End-to-end:
        encode geometry → pool to branch embeds → estimate Q's →
        compute Poiseuille fields → derive junction residuals →
        fuse everything → predict PVP.
    """

    def __init__(self, d_hidden: int = 32, dropout: float = 0.3):
        super().__init__()
        self.d_hidden = d_hidden

        # Shared geometry encoder
        self.geom_encoder = GeometryEncoder(
            d_in=N_PROFILE_FEAT, d_hidden=d_hidden,
            n_blocks=3, dropout=dropout * 0.3,
        )
        self.branch_pool = AttentionPool(d_hidden, d_attn=16)

        # Flow-rate estimator (Murray-prior + learned residual)
        self.flow_est = FlowRateEstimator(
            d_branch=d_hidden, d_aux=N_AUX, d_hidden=d_hidden,
        )

        # Hydrodynamics (deterministic, no parameters)
        self.hydro = PoiseuilleHydrodynamics()
        self.junction_phys = JunctionPhysics()

        # PVP predictor
        # Inputs: flat branch embeds + Q + junction features (incl. per-branch
        # log1p(ΔP), Murray/pressure residuals, collateral fraction) + aux scalars
        d_branches  = N_SEGMENTS * d_hidden            # 8 * 32 = 256
        d_q         = N_SEGMENTS                       # 8
        d_junction  = 13                               # from JunctionPhysics.features
        d_in = d_branches + d_q + d_junction + N_AUX

        self.predictor = nn.Sequential(
            nn.Linear(d_in, d_hidden * 2),
            nn.LayerNorm(d_hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden * 2, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_hidden, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, batch):
        """
        batch: dict from collate_fn (or single-sample dict from dataset)
            profiles      (B, S, N, 4)
            profiles_norm (B, S, N, 4)
            arc_lengths   (B, S, N)
            point_valid   (B, S, N)
            segment_mask  (B, S)
            aux_norm      (B, A)
            is_post_tips  (B,)
        """
        profiles      = batch['profiles']
        profiles_norm = batch['profiles_norm']
        arc_lengths   = batch['arc_lengths']
        point_valid   = batch['point_valid']
        segment_mask  = batch['segment_mask']
        aux_norm      = batch['aux_norm']
        has_tips      = batch.get('is_post_tips', segment_mask[:, SEG_INDEX['tips']])

        B, S, N, _ = profiles.shape

        # ── Per-branch geometry encoding (shared weights) ────────
        branch_hidden = []
        branch_embed  = torch.zeros(B, S, self.d_hidden, device=profiles.device)
        attn_weights  = torch.zeros(B, S, N, device=profiles.device)
        for si in range(S):
            h_si = self.geom_encoder(profiles_norm[:, si])  # (B, N, H)
            branch_hidden.append(h_si)
            pooled, aw = self.branch_pool(h_si, point_valid[:, si])
            seg_alive = segment_mask[:, si].unsqueeze(-1)
            branch_embed[:, si] = pooled * seg_alive
            attn_weights[:, si] = aw

        # ── Junction-end diameter (idx 0 for non-MPV; idx 0 for MPV's confluence
        #    end, but we provide both ends below) ─────────────────
        # For Murray prior at confluence/bifurcation, we use:
        #   • SV[0], SMV[0]      → confluence Murray
        #   • LPV[0], RPV[0], TIPS[0] → bifurcation Murray
        junction_diam = profiles[:, :, 0, P_DIAM]              # (B, S) — idx 0 of every branch

        # ── Flow rate estimation (mass-conservation by construction) ──
        flow_out = self.flow_est(branch_embed, aux_norm, segment_mask, junction_diam)
        Q = flow_out['Q']                                       # (B, S)

        # ── Per-branch Poiseuille hydrodynamics ─────────────────
        hemo_per_seg = []
        endpoint_dP = torch.zeros(B, S, device=profiles.device)  # kept for inspection only
        for si in range(S):
            h = self.hydro(profiles[:, si], arc_lengths[:, si],
                           point_valid[:, si], Q[:, si])
            hemo_per_seg.append(h)
            endpoint_dP[:, si] = h['pressure_drop_total'] * segment_mask[:, si]

        # ── Junction physics residuals ──────────────────────────
        jp = self.junction_phys(hemo_per_seg, flow_out, segment_mask, has_tips)

        # ── Fuse and predict ────────────────────────────────────
        # NOTE: per-branch log1p(ΔP) values are already inside jp['features']
        # so we don't add a separate endpoint_dP feature (would be redundant).
        branch_flat = branch_embed.reshape(B, -1)                # (B, S*H)
        fused = torch.cat([
            branch_flat,                  # (B, S*H)
            Q,                            # (B, S)
            jp['features'],               # (B, 9)
            aux_norm,                     # (B, A)
        ], dim=-1)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)
        pvp_pred = self.predictor(fused)  # (B, 1)

        return {
            'pvp_pred':     pvp_pred,                         # (B, 1) normalized
            'Q':            Q,                                # (B, S)
            'flow_out':     flow_out,                         # dict of split fractions
            'attn_weights': attn_weights,                     # (B, S, N)
            'hemo_per_seg': hemo_per_seg,                     # list of S dicts
            'junction':     jp,                               # dict
            'branch_embed': branch_embed,                     # (B, S, H)
            'endpoint_dP':  endpoint_dP,                      # (B, S) Pa
            'segment_mask': segment_mask,
        }


# =====================================================================
# Module 7 — Physics-Informed Loss
# =====================================================================
class PhysicsInformedLoss(nn.Module):
    """
    L_total = L_main
            + λ_murray   · L_murray       (deviation from Murray-3 prior)
            + λ_press    · L_pressure     (junction pressure consistency)
            + λ_smooth   · L_smooth       (radius profile smoothness)
            + λ_physio   · L_physio       (WSS, Re soft physiological hinge)
            + λ_mono     · L_mono         (monotonicity of cumulative ΔP — sanity)

    Mass conservation does NOT need a loss term — it's enforced by the
    softmax flow-split parameterization.
    """

    def __init__(self,
                 lambda_murray:  float = 0.10,
                 lambda_press:   float = 0.05,
                 lambda_smooth:  float = 0.01,
                 lambda_physio:  float = 0.01,
                 lambda_mono:    float = 0.05,
                 huber_delta:    float = 1.0):
        super().__init__()
        self.lambda_murray  = lambda_murray
        self.lambda_press   = lambda_press
        self.lambda_smooth  = lambda_smooth
        self.lambda_physio  = lambda_physio
        self.lambda_mono    = lambda_mono
        self.main = nn.HuberLoss(delta=huber_delta)

    @staticmethod
    def _hinge_outside_range(x, lo, hi, mask):
        """
        Soft hinge that's 0 inside [lo, hi] and grows quadratically outside,
        normalized by bound magnitude so contributions are O(1) and dimensionless.
        """
        scale_lo = max(abs(lo), 1.0)
        scale_hi = max(abs(hi), 1.0)
        below = F.relu((lo - x) / scale_lo)
        above = F.relu((x - hi) / scale_hi)
        v = (below.pow(2) + above.pow(2)) * mask
        denom = mask.sum().clamp(min=1.0)
        return v.sum() / denom

    def forward(self, model_out, label_norm, batch):
        pvp_pred = model_out['pvp_pred'].squeeze(-1)
        L_main = self.main(pvp_pred, label_norm)

        # ── L_murray: ‖logit deviations from Murray prior‖² ──────
        # Three junctions now: inflow (sv,smv), confluence-outflow (mpv,lgv,pgv),
        # bifurcation-outflow (lpv,rpv,tips). Each contributes ‖delta‖² weighted
        # by junction activity (so absent junctions don't pull the loss).
        flow = model_out['flow_out']
        m_in   = model_out['junction']['inflow_active']
        m_co   = model_out['junction']['confluence_outflow_active']
        m_bo   = model_out['junction']['bifurcation_active']

        d_in = flow['inflow_delta']        * flow['inflow_mask']
        d_co = flow['conf_outflow_delta']  * flow['conf_outflow_mask']
        d_bo = flow['bif_outflow_delta']   * flow['bif_outflow_mask']

        L_murr_in = (d_in.pow(2).sum(dim=-1) * m_in).sum() / m_in.sum().clamp(min=1.0)
        L_murr_co = (d_co.pow(2).sum(dim=-1) * m_co).sum() / m_co.sum().clamp(min=1.0)
        L_murr_bo = (d_bo.pow(2).sum(dim=-1) * m_bo).sum() / m_bo.sum().clamp(min=1.0)
        L_murray = L_murr_in + L_murr_co + L_murr_bo

        # ── L_pressure: bifurcation pressure-continuity residual ────
        # Only the bifurcation has a defendable pressure-equality constraint
        # (LPV ≈ RPV, both feeding sinusoids). Confluence-outflow residual would
        # require absolute upstream/downstream pressures we don't have.
        L_press = (model_out['junction']['press_resid_bifurc'].sum()
                   / m_bo.sum().clamp(min=1.0))

        # ── L_smooth: 2nd-derivative of radius profile (per branch) ──
        # Geometric prior: radius shouldn't oscillate point-to-point.
        # Mostly stabilizes Poiseuille's r⁴ in the denominator.
        L_smooth = 0.0
        n_seg_used = 0
        profiles    = batch['profiles']
        point_valid = batch['point_valid']
        seg_mask    = batch['segment_mask']
        for si in range(N_SEGMENTS):
            alive = seg_mask[:, si] > 0
            if alive.sum() == 0:
                continue
            r = profiles[alive, si, :, P_DIAM] * 0.5
            v = point_valid[alive, si]
            d2r = r[:, 2:] - 2.0 * r[:, 1:-1] + r[:, :-2]
            mw  = v[:, 2:] * v[:, 1:-1] * v[:, :-2]
            L_smooth = L_smooth + (d2r.pow(2) * mw).sum() / mw.sum().clamp(min=1.0)
            n_seg_used += 1
        if n_seg_used > 0:
            L_smooth = L_smooth / n_seg_used
        else:
            L_smooth = torch.tensor(0.0, device=pvp_pred.device)

        # ── L_physio: soft hinge on WSS and Re ──────────────────
        L_physio = 0.0
        n_seg_used = 0
        for si in range(N_SEGMENTS):
            alive = seg_mask[:, si] > 0
            if alive.sum() == 0:
                continue
            h = model_out['hemo_per_seg'][si]
            v = point_valid[alive, si]
            wss = h['wss_pa'][alive]
            re_ = h['reynolds'][alive]
            # Hinge: outside [WSS_LO, WSS_HI] (already normalized by bound scale)
            L_physio = L_physio + self._hinge_outside_range(wss, WSS_PHYSIO_LO_PA, WSS_PHYSIO_HI_PA, v)
            # Re upper bound (laminar regime); normalized by RE_PHYSIO_HI inside hinge
            L_physio = L_physio + self._hinge_outside_range(re_, 0.0, RE_PHYSIO_HI, v)
            n_seg_used += 1
        if n_seg_used > 0:
            L_physio = L_physio / n_seg_used
        else:
            L_physio = torch.tensor(0.0, device=pvp_pred.device)

        # ── L_mono: ΔP(s) should be non-decreasing (sanity) ─────
        # Built-in by Poiseuille (cum_R is monotonic), but check on
        # any noisy edge cases.
        L_mono = 0.0
        n_seg_used = 0
        for si in range(N_SEGMENTS):
            alive = seg_mask[:, si] > 0
            if alive.sum() == 0:
                continue
            h = model_out['hemo_per_seg'][si]
            v = point_valid[alive, si]
            dp = h['pressure_drop_pa'][alive]
            ddp = dp[:, 1:] - dp[:, :-1]
            mw = v[:, 1:] * v[:, :-1]
            L_mono = L_mono + (F.relu(-ddp).pow(2) * mw).sum() / mw.sum().clamp(min=1.0)
            n_seg_used += 1
        if n_seg_used > 0:
            L_mono = L_mono / n_seg_used
        else:
            L_mono = torch.tensor(0.0, device=pvp_pred.device)

        # ── Total ────────────────────────────────────────────────
        # Safety: clamp each component to a sane upper bound so that no
        # single term can produce NaN/inf or dominate the gradient
        # catastrophically. Bounds are *very* permissive — they only kick
        # in for genuinely-broken numerics, never for healthy training.
        def _safe(x, cap=1e6):
            if not torch.is_tensor(x):
                x = torch.tensor(float(x), device=pvp_pred.device)
            return torch.nan_to_num(x, nan=0.0, posinf=cap, neginf=cap).clamp(max=cap)

        L_main_s    = _safe(L_main,    cap=1e3)
        L_murray_s  = _safe(L_murray,  cap=1e3)
        L_press_s   = _safe(L_press,   cap=1e3)
        L_smooth_s  = _safe(L_smooth,  cap=1e3)
        L_physio_s  = _safe(L_physio,  cap=1e3)
        L_mono_s    = _safe(L_mono,    cap=1e3)

        L_total = (L_main_s
                   + self.lambda_murray * L_murray_s
                   + self.lambda_press  * L_press_s
                   + self.lambda_smooth * L_smooth_s
                   + self.lambda_physio * L_physio_s
                   + self.lambda_mono   * L_mono_s)

        log = {
            'main':    float(L_main_s.item()),
            'murray':  float(L_murray_s.item()),
            'press':   float(L_press_s.item()),
            'smooth':  float(L_smooth_s.item()),
            'physio':  float(L_physio_s.item()),
            'mono':    float(L_mono_s.item()),
            'total':   float(L_total.item()),
        }
        return L_total, log


# =====================================================================
# Utility
# =====================================================================
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# =====================================================================
# Smoke test
# =====================================================================
if __name__ == '__main__':
    from dataset import collate_fn

    B, S, N = 4, N_SEGMENTS, 100
    device = 'cpu'

    # Mock a batch
    profiles_raw = torch.rand(B, S, N, 4) * 5 + 1  # positive geometry
    profiles_raw[..., P_DIAM] = profiles_raw[..., P_DIAM].clamp(min=0.5)
    profiles_raw[..., P_AREA] = math.pi * (profiles_raw[..., P_DIAM] / 2).pow(2)
    profiles_raw[..., P_INSC] = profiles_raw[..., P_DIAM] / 2 * 0.95
    profiles_raw[..., P_CURV] = torch.rand(B, S, N) * 0.05  # 1/mm

    profiles_norm = (profiles_raw - profiles_raw.mean(dim=(0, 1, 2))) / (profiles_raw.std(dim=(0, 1, 2)) + 1e-6)
    arc_lengths   = torch.linspace(0, 50, N).view(1, 1, N).expand(B, S, N).contiguous()
    point_valid   = torch.ones(B, S, N)
    point_valid[..., :3] = 0; point_valid[..., -3:] = 0  # endpoint protection
    segment_mask  = torch.ones(B, S)
    # Mix of presence patterns (some have TIPS, some have LGV/PGV)
    segment_mask[:, SEG_INDEX['tips']] = torch.tensor([1.0, 0.0, 1.0, 0.0])
    segment_mask[:, SEG_INDEX['lgv']]  = torch.tensor([0.0, 1.0, 1.0, 0.0])
    segment_mask[:, SEG_INDEX['pgv']]  = torch.tensor([0.0, 0.0, 1.0, 1.0])
    aux_scalars   = torch.rand(B, N_AUX)
    aux_norm      = aux_scalars
    aux_mask      = torch.ones(B, N_AUX)
    is_post_tips  = segment_mask[:, SEG_INDEX['tips']].clone()
    label_norm    = torch.randn(B)

    batch = {
        'profiles':      profiles_raw,
        'profiles_norm': profiles_norm,
        'arc_lengths':   arc_lengths,
        'point_valid':   point_valid,
        'segment_mask':  segment_mask,
        'aux_scalars':   aux_scalars,
        'aux_norm':      aux_norm,
        'aux_mask':      aux_mask,
        'is_post_tips':  is_post_tips,
        'label_norm':    label_norm,
    }

    model = PortalPressureNet(d_hidden=32, dropout=0.3).to(device)
    out = model(batch)
    print("Forward pass OK")
    print(f"  pvp_pred:       {tuple(out['pvp_pred'].shape)}")
    print(f"  Q:              {tuple(out['Q'].shape)}")
    Q = out['Q']
    print(f"  Q[0] by branch: {dict(zip(SEGMENTS, [round(v,3) for v in Q[0].tolist()]))}")
    print(f"  Inflow sum   (sv+smv) = {(Q[:, SEG_INDEX['sv']] + Q[:, SEG_INDEX['smv']]).tolist()}")
    print(f"  Conf-out sum (mpv+lgv+pgv) = "
          f"{(Q[:, SEG_INDEX['mpv']] + Q[:, SEG_INDEX['lgv']] + Q[:, SEG_INDEX['pgv']]).tolist()}")
    print(f"  Bif-out sum  (lpv+rpv+tips) vs Q_mpv = "
          f"{(Q[:, SEG_INDEX['lpv']] + Q[:, SEG_INDEX['rpv']] + Q[:, SEG_INDEX['tips']]).tolist()} "
          f"vs {Q[:, SEG_INDEX['mpv']].tolist()}")
    print(f"  Collateral fraction (Q_lgv + Q_pgv) = {out['flow_out']['collateral_fraction'].tolist()}")
    print(f"  attn_weights:   {tuple(out['attn_weights'].shape)}")
    print(f"  hemo[0] keys:   {list(out['hemo_per_seg'][0].keys())}")
    print(f"  velocity range: [{out['hemo_per_seg'][0]['velocity_m_per_s'].min():.4f}, "
          f"{out['hemo_per_seg'][0]['velocity_m_per_s'].max():.4f}] m/s")
    print(f"  WSS range:      [{out['hemo_per_seg'][0]['wss_pa'].min():.4f}, "
          f"{out['hemo_per_seg'][0]['wss_pa'].max():.4f}] Pa")
    print(f"  Re range:       [{out['hemo_per_seg'][0]['reynolds'].min():.1f}, "
          f"{out['hemo_per_seg'][0]['reynolds'].max():.1f}]")
    print(f"  ΔP@far end:     {out['endpoint_dP'][0].tolist()} Pa")
    print(f"  junction features: {out['junction']['features'][0].tolist()}")

    crit = PhysicsInformedLoss()
    L, log = crit(out, label_norm, batch)
    print("\nLoss components:")
    for k, v in log.items():
        print(f"  {k:>10s}: {v:.6f}")

    L.backward()
    print("\nBackward OK.")
    total, train = count_params(model)
    print(f"\nTotal params: {total:,} | Trainable: {train:,}")