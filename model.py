"""
Physics-Informed Geometric Deep Learning for Portal Vein Pressure — v5
=======================================================================
Bugfixes over v4 (see model.py review):

  1.  GeometryEncoder: BatchNorm1d → GroupNorm. BatchNorm computes
      statistics across the padded batch, polluted by zero-padding when
      segments have variable valid lengths. GroupNorm normalizes per-
      sample per-channel and is unaffected by other samples / padding.

  2.  PortalPressureNet: explicit physics baseline anchor. Final
      prediction is now:

          pvp_pred = baseline_norm + predictor_residual + residual_net

      where baseline_norm is the normalized Poiseuille pressure drop
      from the hepatic sinusoids back to the PVP measurement point
      (dP_mpv + dP_liver). The MLP only learns the *correction*. Anchors
      the prediction to the physics instead of letting MLP ignore the
      8-dim log1p(dP) features inside a 300-dim input.

  3.  PoiseuilleHydrodynamics: r_eff extracted as a static method
      `compute_r_eff` so the smoothness loss and resistance prior use
      EXACTLY the same effective radius as the forward physics. v4 had
      smoothness loss on hdiam/2 while forward used shape-aware r_eff.

  4.  PhysicsInformedLoss: TIPS patients now have bifurcation Murray
      loss FULLY disabled (was 0.35×). TIPS is an artificial stent with
      a clinically chosen diameter; it does not obey Murray's biological
      optimization law.

  5.  PoiseuilleHydrodynamics: clamp instead of nan_to_num(posinf=0)
      for resistance and pressure-drop. Setting r→0 cases to "zero
      resistance" was *inverting* the correct physics.

  6.  SplenicFlowEstimator: input features now standardized. Raw
      log(spleen) ≈ 5–8 was dominating the first hidden layer despite
      the zero-init final layer.

  7.  VesselGraphNet: removed (sv, smv) and (lpv, rpv) edges. These
      pairs are *siblings* sharing a common parent (MPV), not directly
      connected anatomically. The 2-layer GNN can still relate them via
      MPV in one extra hop.

  8.  Default dropout 0.30 → 0.15. The high rate + (Batch)Norm combo
      was a known stability hazard.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import (
    N_PROFILE_FEAT, N_SEGMENTS, SEGMENTS, SEG_INDEX, N_AUX,
    P_AREA, P_HDIAM, P_PERIM, P_CURV, P_TORS, P_INSC,
    P_SOLID, P_RRAT, P_DADS, P_CIRC, P_NCOMP,
    AUX_SPLEEN_VOL_IDX, AUX_LIVER_VOL_IDX,
)


# =====================================================================
# Physical constants (blood at 37°C)
# =====================================================================
BLOOD_VISCOSITY_PA_S      = 3.5e-3
BLOOD_DENSITY_KG_M3       = 1060.0
BLOOD_KIN_VISCOSITY_M2_S  = BLOOD_VISCOSITY_PA_S / BLOOD_DENSITY_KG_M3
Q_REF_M3_PER_S            = 800.0 * 1e-6 / 60.0   # ≈ 1.33e-5 m³/s

WSS_PHYSIO_LO_PA   = 0.05
WSS_PHYSIO_HI_PA   = 5.0
RE_PHYSIO_HI       = 1500.0

# Defaults for PVP normalization (override at model init from your dataset
# statistics: clinically, PVP ~ 5–25 mmHg = ~666–3333 Pa).
# These are used to put the physics baseline on the same scale as the
# normalized label space.
PVP_MEAN_PA_DEFAULT = 1600.0   # ≈ 12 mmHg
PVP_STD_PA_DEFAULT  = 800.0    # ≈ 6 mmHg

# Defaults for log-organ-volume standardization in SplenicFlowEstimator.
# Override at model init from your training-set statistics.
# log(spleen ml) for normal ≈ log(200) ≈ 5.3; cirrhotic ≈ log(800) ≈ 6.7.
# log(liver  ml) ≈ log(1500) ≈ 7.3.
LOG_VOL_MEAN_DEFAULT = 6.5
LOG_VOL_STD_DEFAULT  = 0.7


# =====================================================================
# Module 0 — Splenic Flow Estimator (patient-specific Q)
# =====================================================================
class SplenicFlowEstimator(nn.Module):
    """
    Converts organ volumes → patient-specific portal flow scale factor.

    v5: inputs are standardized (z-scored log volumes) so no single
    feature dominates the first hidden layer's gradient.
    """

    def __init__(self, d_hidden: int = 8,
                 log_vol_mean: float = LOG_VOL_MEAN_DEFAULT,
                 log_vol_std:  float = LOG_VOL_STD_DEFAULT):
        super().__init__()
        # Input: [spleen_log_z, liver_log_z, ratio_centered, n_valid_centered]
        self.net = nn.Sequential(
            nn.Linear(4, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
        )
        # Init final layer to 0 so q_scale starts at exp(0) = 1.0.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # Standardization constants (registered so they save with state_dict).
        self.register_buffer('log_vol_mean', torch.tensor(float(log_vol_mean)))
        self.register_buffer('log_vol_std',  torch.tensor(float(log_vol_std)))

    def forward(self, organ_volumes, organ_valid):
        """
        organ_volumes: (B, 2) — [spleen_ml, liver_ml]
        organ_valid:   (B, 2) — [spleen_present, liver_present]
        Returns: q_scale (B,) — multiplicative factor on Q_REF
        """
        spleen = organ_volumes[:, 0].clamp(min=1.0)
        liver  = organ_volumes[:, 1].clamp(min=1.0)

        # z-scored log volumes (only when valid, else 0)
        spleen_z = ((torch.log(spleen) - self.log_vol_mean) / self.log_vol_std
                    ) * organ_valid[:, 0]
        liver_z  = ((torch.log(liver)  - self.log_vol_mean) / self.log_vol_std
                    ) * organ_valid[:, 1]

        # spleen/liver ratio centered around 1 (normal ≈ 0.13–0.2; cirrhotic ↑)
        # Use log-ratio so it's symmetric around 0.
        ratio_log = (torch.log(spleen) - torch.log(liver)
                     ) * (organ_valid[:, 0] * organ_valid[:, 1])
        # Center: log(0.15) ≈ -1.9 for normal, so subtract that.
        ratio_z = (ratio_log + 1.9) * (organ_valid[:, 0] * organ_valid[:, 1])

        # How many organs are observed, centered around 1 (out of max 2).
        n_valid_centered = organ_valid.sum(dim=-1) - 1.0

        feats = torch.stack([spleen_z, liver_z, ratio_z, n_valid_centered], dim=-1)

        log_scale = self.net(feats).squeeze(-1)
        # Clamp so q_scale ∈ [exp(-1.2), exp(1.1)] ≈ [0.30, 3.0]
        q_scale = torch.exp(log_scale.clamp(-1.2, 1.1))
        return q_scale


# =====================================================================
# Module 1 — Poiseuille Hydrodynamics (shape-aware, no learnable params)
# =====================================================================
class PoiseuilleHydrodynamics(nn.Module):
    """
    Given per-point geometry (11 channels) + Q_rel:
      → velocity, WSS, Re, R', cumR, ΔP, Dean, area_gradient

    v5: r_eff extracted as static method so loss + resistance prior +
    forward path all use the SAME effective radius. Also: clamps replace
    nan_to_num(posinf=0) for physically meaningful inf-handling.
    """

    # Physical caps to keep numerics stable without inverting physics.
    # 1e15 Pa·s/m^4 is well above any anatomical resistance.
    LOCAL_R_CAP = 1.0e15
    CUM_R_CAP   = 1.0e16
    DP_CAP_PA   = 1.0e7    # 10 MPa is absurd; just guards against r→0 blowups

    def __init__(self, mu=BLOOD_VISCOSITY_PA_S, rho=BLOOD_DENSITY_KG_M3,
                 q_ref=Q_REF_M3_PER_S):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.nu = mu / rho
        self.q_ref = q_ref

    # ------------------------------------------------------------------
    @staticmethod
    def compute_r_eff_mm(profiles_slice, eps: float = 1e-9):
        """
        Shape-aware effective hydraulic radius (mm).

        Works on any leading batch shape; last dim must be N_PROFILE_FEAT.
        Used by:
          • this module's forward (Poiseuille)
          • _branch_resistance_prior (flow allocation)
          • PhysicsInformedLoss.smoothness  (regularization)
        Keeping them in sync avoids the v4 inconsistency where the loss
        smoothed hdiam/2 but the forward physics used r_eff.
        """
        hdiam_mm = profiles_slice[..., P_HDIAM].clamp(min=eps)
        insc_mm  = profiles_slice[..., P_INSC].clamp(min=eps)
        solid    = profiles_slice[..., P_SOLID].clamp(min=0.01, max=1.0)
        alpha    = (1.0 - solid).clamp(min=0.0, max=1.0)
        r_eff_mm = (1.0 - alpha) * (0.5 * hdiam_mm) + alpha * insc_mm
        return r_eff_mm, alpha

    # ------------------------------------------------------------------
    @staticmethod
    def _safe_seg_lengths(arc, valid, eps=1e-6):
        ds = torch.zeros_like(arc)
        ds[..., 1:] = arc[..., 1:] - arc[..., :-1]
        ds[..., 0]  = ds[..., 1]
        ds = ds.clamp(min=eps) * (valid > 0).float()
        return ds

    # ------------------------------------------------------------------
    def forward(self, profiles, arc, valid, Q_rel):
        """
        profiles: (B, N, 11)  arc: (B, N)  valid: (B, N)  Q_rel: (B,)
        """
        eps = 1e-9

        # ── Geometry extraction ──────────────────────────────
        area_mm2    = profiles[..., P_AREA].clamp(min=eps)
        hdiam_mm    = profiles[..., P_HDIAM].clamp(min=eps)
        curv_inv_mm = profiles[..., P_CURV].abs()

        r_eff_mm, alpha = self.compute_r_eff_mm(profiles, eps=eps)

        # Convert to SI
        area_m2  = area_mm2  * 1e-6
        diam_m   = hdiam_mm  * 1e-3
        r_eff_m  = r_eff_mm  * 1e-3
        curv_1_m = curv_inv_mm * 1e3

        # Absolute flow
        Q_abs = Q_rel.unsqueeze(-1) * self.q_ref

        # 1. Velocity
        velocity = Q_abs / (area_m2 + eps)

        # 2. WSS (using shape-aware effective radius)
        wss = (4.0 * self.mu * Q_abs) / (math.pi * r_eff_m.pow(3) + eps)

        # 3. Reynolds
        reynolds = velocity * diam_m / (self.nu + eps)

        # 4. Local resistance per unit length — CLAMP, do not zero out
        local_R = (8.0 * self.mu) / (math.pi * r_eff_m.pow(4) + eps)
        local_R = local_R.clamp(max=self.LOCAL_R_CAP)

        # 5. Cumulative resistance
        ds_m = self._safe_seg_lengths(arc, valid) * 1e-3
        cum_R = torch.cumsum(local_R * ds_m, dim=-1).clamp(max=self.CUM_R_CAP)

        # 6. Pressure drop — clamp instead of nan→0
        pressure_drop = (Q_abs * cum_R).clamp(max=self.DP_CAP_PA)

        # 7. Dean number
        dean = reynolds * torch.sqrt(diam_m * curv_1_m + eps)

        # 8. Area gradient
        area_grad = torch.zeros_like(area_mm2)
        ds_mm = self._safe_seg_lengths(arc, valid)
        area_grad[..., 1:] = (area_mm2[..., 1:] - area_mm2[..., :-1]) / ds_mm[..., 1:]
        area_grad[..., 0] = area_grad[..., 1]

        v = (valid > 0).float()
        out = {
            'radius_m':             r_eff_m,
            'area_m2':              area_m2,
            'velocity_m_per_s':     velocity * v,
            'wss_pa':               wss * v,
            'reynolds':             reynolds * v,
            'local_R_pa_s_per_m4':  local_R * v,
            'cum_R_pa_s_per_m3':    cum_R * v,
            'pressure_drop_pa':     pressure_drop * v,
            'dean':                 dean * v,
            'area_gradient':        area_grad * v,
            'shape_alpha':          alpha * v,
            'cum_R_total':          (cum_R * v).max(dim=-1).values,
            'pressure_drop_total':  (pressure_drop * v).max(dim=-1).values,
            'valid_count':          v.sum(dim=-1),
        }
        # Only NaN-protect (inf is already clamped). This way an
        # impossibly large pressure stays large, not silently zeroed.
        for k in out:
            out[k] = torch.nan_to_num(
                out[k], nan=0.0,
                posinf=self.DP_CAP_PA, neginf=-self.DP_CAP_PA,
            )
        return out


# =====================================================================
# Module 2 — Geometry Encoder (shared across branches)
# =====================================================================
class GeometryEncoder(nn.Module):
    """
    v5: BatchNorm1d → GroupNorm. BN's running stats are corrupted when
    different samples have different valid lengths (the padding zeros
    skew the per-channel mean/variance). GroupNorm is computed per-
    sample, per-group, and is mask-agnostic.
    """

    def __init__(self, d_in: int = N_PROFILE_FEAT, d_hidden: int = 32,
                 n_blocks: int = 3, dropout: float = 0.1,
                 gn_groups: int = 8):
        super().__init__()
        chans = [d_in] + [d_hidden] * n_blocks
        ks = [7, 5, 3]
        layers = []
        for i in range(n_blocks):
            out_c = chans[i + 1]
            # GroupNorm group count must divide channels
            groups = math.gcd(gn_groups, out_c) if out_c >= gn_groups else 1
            layers += [
                nn.Conv1d(chans[i], out_c,
                          kernel_size=ks[i % len(ks)],
                          padding=ks[i % len(ks)] // 2),
                nn.GroupNorm(num_groups=max(1, groups), num_channels=out_c),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        self.encoder = nn.Sequential(*layers)
        self.d_hidden = d_hidden

    def forward(self, profiles_norm):
        """profiles_norm: (B, N, C) → hidden: (B, N, H)"""
        x = profiles_norm.transpose(1, 2)
        h = self.encoder(x)
        return h.transpose(1, 2)


# =====================================================================
# Module 3 — Attention Pooling
# =====================================================================
class AttentionPool(nn.Module):

    def __init__(self, d_hidden: int, d_attn: int = 16):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_hidden, d_attn), nn.Tanh(), nn.Linear(d_attn, 1),
        )

    def forward(self, hidden, valid_mask):
        scores = self.attn(hidden).squeeze(-1)
        scores = scores.masked_fill(valid_mask < 0.5, float('-inf'))
        all_masked = (valid_mask.sum(dim=-1) < 0.5)
        scores = torch.where(all_masked.unsqueeze(-1).expand_as(scores),
                             torch.zeros_like(scores), scores)
        attn_w = F.softmax(scores, dim=-1)
        pooled = torch.einsum('bn,bnh->bh', attn_w, hidden)
        pooled = pooled * (~all_masked).float().unsqueeze(-1)
        return pooled, attn_w


# =====================================================================
# Module 3.5 — Vessel Topology GNN (corrected anatomy)
# =====================================================================
class ProfileTransformerEncoder(nn.Module):
    """
    Lightweight token encoder for each vessel profile.

    The architecture benchmark showed numeric Transformer tokens were the
    strongest simple baseline. This module brings that inductive bias into the
    main model while keeping the existing CNN/GNN path.
    """

    def __init__(self, d_in: int = N_PROFILE_FEAT, d_hidden: int = 32,
                 n_layers: int = 2, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        n_heads = max(1, math.gcd(n_heads, d_hidden))
        self.in_proj = nn.Linear(d_in, d_hidden)
        self.branch_embed = nn.Embedding(N_SEGMENTS, d_hidden)
        self.pos_mlp = nn.Sequential(
            nn.Linear(1, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_hidden,
            nhead=n_heads,
            dim_feedforward=d_hidden * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_hidden)

    def forward(self, profiles_norm, valid_mask, branch_idx: int):
        B, N, _ = profiles_norm.shape
        device = profiles_norm.device
        pos = torch.linspace(0.0, 1.0, N, device=device).view(1, N, 1).expand(B, -1, -1)
        branch_ids = torch.full((B,), int(branch_idx), dtype=torch.long, device=device)
        h = self.in_proj(profiles_norm)
        h = h + self.pos_mlp(pos) + self.branch_embed(branch_ids).unsqueeze(1)
        key_padding_mask = valid_mask < 0.5
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked, 0] = False
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)
        return h * valid_mask.unsqueeze(-1)


class VesselGraphNet(nn.Module):
    """
    v5: edge list cleaned up. SV-SMV and LPV-RPV pairs are *siblings*
    sharing a common parent (MPV) — they are not directly connected
    anatomically. The 2-layer GNN can still relate them in 2 hops
    through MPV.

    Remaining edges (each direction):
      SV  → MPV       (splenic into confluence)
      SMV → MPV       (mesenteric into confluence)
      MPV → LPV       (portal bifurcation, left)
      MPV → RPV       (portal bifurcation, right)
      MPV → TIPS      (shunt off the trunk)
      LGV → MPV       (collateral, gastric)
      PGV → MPV       (collateral, gastric)
    """

    _EDGE_PAIRS = [
        ('sv',  'mpv'),
        ('smv', 'mpv'),
        ('mpv', 'lpv'),
        ('mpv', 'rpv'),
        ('mpv', 'tips'),
        ('lgv', 'mpv'),
        ('pgv', 'mpv'),
    ]

    def __init__(self, d_hidden: int, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        adj = torch.zeros(N_SEGMENTS, N_SEGMENTS)
        for s_name, d_name in self._EDGE_PAIRS:
            si, di = SEG_INDEX[s_name], SEG_INDEX[d_name]
            adj[si, di] = 1.0
            adj[di, si] = 1.0
        adj = adj + torch.eye(N_SEGMENTS)
        self.register_buffer('adj', adj)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'Wq': nn.Linear(d_hidden, d_hidden, bias=False),
                'Wk': nn.Linear(d_hidden, d_hidden, bias=False),
                'Wv': nn.Linear(d_hidden, d_hidden, bias=False),
                'ffn': nn.Sequential(
                    nn.Linear(d_hidden, d_hidden * 2), nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_hidden * 2, d_hidden),
                ),
                'norm1': nn.LayerNorm(d_hidden),
                'norm2': nn.LayerNorm(d_hidden),
                'drop':  nn.Dropout(dropout),
            }))

    def forward(self, node_embed, segment_mask):
        """node_embed: (B, S, H), segment_mask: (B, S) → (B, S, H)"""
        h = node_embed
        adj = self.adj.unsqueeze(0)
        pair_mask = segment_mask.unsqueeze(1) * segment_mask.unsqueeze(2)
        edge_mask = adj * pair_mask

        for layer in self.layers:
            Q = layer['Wq'](h)
            K = layer['Wk'](h)
            V = layer['Wv'](h)
            attn = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(h.size(-1))
            attn = attn.masked_fill(edge_mask < 0.5, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0)
            attn = layer['drop'](attn)
            msg = torch.bmm(attn, V)
            h = layer['norm1'](h + msg)
            h = layer['norm2'](h + layer['ffn'](h))
            h = h * segment_mask.unsqueeze(-1)
        return h


# =====================================================================
# Module 3.6 — Physics Residual Net (non-Poiseuille correction)
# =====================================================================
class PhysicsResidualNet(nn.Module):
    """
    Parallel learnable path that captures what Poiseuille cannot:
    turbulence, entrance effects, vortices in PVT lumens.

    Initialized to output **zero** → at epoch 0 the prediction equals
    the physics path. Residual gradually learns corrections.
    """

    def __init__(self, d_in: int, d_hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        return self.net(features)

    @staticmethod
    def extract_features(hemo_per_seg, profiles, point_valid, segment_mask):
        """
        Per-branch non-Poiseuille indicators: max_Re, max_WSS,
        min_solidity, mean_solidity, max|dA/ds|.
        Returns (B, S * 5 = 40).
        """
        feats = []
        for si in range(N_SEGMENTS):
            h = hemo_per_seg[si]
            v = point_valid[:, si]
            alive = segment_mask[:, si].unsqueeze(-1)
            big_neg = torch.full_like(v, -1e9)
            big_pos = torch.full_like(v, 1e9)

            re_v  = torch.where(v > 0.5, h['reynolds'],  big_neg)
            wss_v = torch.where(v > 0.5, h['wss_pa'],    big_neg)
            sol   = profiles[:, si, :, P_SOLID]
            sol_mn = torch.where(v > 0.5, sol, big_pos)
            dads  = torch.where(v > 0.5, profiles[:, si, :, P_DADS].abs(), big_neg)

            max_re   = re_v.max(-1).values  * alive.squeeze(-1)
            max_wss  = wss_v.max(-1).values * alive.squeeze(-1)
            min_sol  = sol_mn.min(-1).values
            min_sol  = torch.where(alive.squeeze(-1) > 0.5, min_sol,
                                   torch.ones_like(min_sol))
            mean_sol = (sol * v).sum(-1) / v.sum(-1).clamp(1) * alive.squeeze(-1)
            max_dads = dads.max(-1).values  * alive.squeeze(-1)

            feats.extend([max_re, max_wss, min_sol, mean_sol, max_dads])
        return torch.stack(feats, dim=-1)


# =====================================================================
# Module 4 — Flow Rate Estimator
# =====================================================================
class FlowRateEstimator(nn.Module):

    def __init__(self, d_branch: int, d_aux: int = N_AUX, d_hidden: int = 32):
        super().__init__()
        self.inflow_head = nn.Sequential(
            nn.Linear(2 * d_branch + d_aux, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 2),
        )
        self.conf_outflow_head = nn.Sequential(
            nn.Linear(3 * d_branch + d_aux, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 3),
        )
        self.bif_outflow_head = nn.Sequential(
            nn.Linear(4 * d_branch + d_aux, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 3),
        )
        for m in [self.inflow_head[-1], self.conf_outflow_head[-1],
                  self.bif_outflow_head[-1]]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    @staticmethod
    def _murray_logits(diameters_mm, eps=1e-3):
        return 3.0 * torch.log(diameters_mm.clamp(min=eps))

    @staticmethod
    def _masked_center(logits, mask):
        present = (mask > 0.5).float()
        denom = present.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mean = (logits * present).sum(dim=-1, keepdim=True) / denom
        return (logits - mean).clamp(min=-6.0, max=6.0)

    @classmethod
    def _conductance_logits(cls, branch_resistance, branch_indices, mask):
        if branch_resistance is None:
            return torch.zeros_like(mask)
        r = branch_resistance[:, branch_indices].clamp(min=1e-6)
        logits = -torch.log(r)
        return cls._masked_center(logits, mask)

    def forward(self, branch_embeds, aux_norm, segment_mask, junction_diameters,
                branch_resistance=None):
        B = branch_embeds.size(0)
        device = branch_embeds.device
        i = SEG_INDEX

        # ── Inflow (sv, smv) ────────────────────────────────
        ctx_in = torch.cat([branch_embeds[:, i['sv']],
                            branch_embeds[:, i['smv']], aux_norm], dim=-1)
        delta_in = self.inflow_head(ctx_in)
        d_sv  = junction_diameters[:, i['sv']]
        d_smv = junction_diameters[:, i['smv']]
        prior_in = torch.stack([self._murray_logits(d_sv),
                                self._murray_logits(d_smv)], dim=-1)
        mask_in = torch.stack([segment_mask[:, i['sv']],
                               segment_mask[:, i['smv']]], dim=-1)
        logits_in = (prior_in + delta_in).masked_fill(mask_in < 0.5, -1e9)
        inflow_frac = F.softmax(logits_in, dim=-1)
        no_in = (mask_in.sum(-1) < 0.5).unsqueeze(-1)
        inflow_frac = torch.where(no_in, torch.full_like(inflow_frac, 0.5), inflow_frac)

        # ── Confluence outflow (mpv, lgv, pgv) ──────────────
        ctx_co = torch.cat([branch_embeds[:, i['mpv']], branch_embeds[:, i['lgv']],
                            branch_embeds[:, i['pgv']], aux_norm], dim=-1)
        delta_co = self.conf_outflow_head(ctx_co)
        prior_co = torch.stack([self._murray_logits(junction_diameters[:, i['mpv']]),
                                self._murray_logits(junction_diameters[:, i['lgv']]),
                                self._murray_logits(junction_diameters[:, i['pgv']])], dim=-1)
        mask_co = torch.stack([segment_mask[:, i['mpv']], segment_mask[:, i['lgv']],
                               segment_mask[:, i['pgv']]], dim=-1)
        conduct_co = self._conductance_logits(branch_resistance,
                                              [i['mpv'], i['lgv'], i['pgv']], mask_co)
        w_co = torch.tensor([0.5, 1.5, 1.5], device=device).view(1, 3)
        logits_co = (prior_co + w_co * conduct_co + delta_co).masked_fill(mask_co < 0.5, -1e9)
        conf_frac = F.softmax(logits_co, dim=-1)
        no_co = (mask_co.sum(-1) < 0.5).unsqueeze(-1)
        conf_frac = torch.where(no_co,
                                torch.tensor([[1., 0., 0.]], device=device).expand_as(conf_frac),
                                conf_frac)

        # ── Bifurcation outflow (lpv, rpv, tips) ────────────
        ctx_bo = torch.cat([branch_embeds[:, i['mpv']], branch_embeds[:, i['lpv']],
                            branch_embeds[:, i['rpv']], branch_embeds[:, i['tips']],
                            aux_norm], dim=-1)
        delta_bo = self.bif_outflow_head(ctx_bo)
        prior_bo = torch.stack([self._murray_logits(junction_diameters[:, i['lpv']]),
                                self._murray_logits(junction_diameters[:, i['rpv']]),
                                self._murray_logits(junction_diameters[:, i['tips']])], dim=-1)
        mask_bo = torch.stack([segment_mask[:, i['lpv']], segment_mask[:, i['rpv']],
                               segment_mask[:, i['tips']]], dim=-1)
        conduct_bo = self._conductance_logits(branch_resistance,
                                              [i['lpv'], i['rpv'], i['tips']], mask_bo)
        w_bo = torch.tensor([0.5, 0.5, 2.0], device=device).view(1, 3)
        logits_bo = (prior_bo + w_bo * conduct_bo + delta_bo).masked_fill(mask_bo < 0.5, -1e9)
        bif_frac = F.softmax(logits_bo, dim=-1)
        no_bo = (mask_bo.sum(-1) < 0.5).unsqueeze(-1)
        bif_frac = torch.where(no_bo, torch.full_like(bif_frac, 1./3.), bif_frac)

        # ── Assemble Q ──────────────────────────────────────
        Q_mpv  = conf_frac[:, 0]
        Q_lgv  = conf_frac[:, 1]
        Q_pgv  = conf_frac[:, 2]
        Q_sv   = inflow_frac[:, 0]
        Q_smv  = inflow_frac[:, 1]
        Q_lpv  = bif_frac[:, 0] * Q_mpv
        Q_rpv  = bif_frac[:, 1] * Q_mpv
        Q_tips = bif_frac[:, 2] * Q_mpv

        Q_list = [None] * N_SEGMENTS
        Q_list[i['mpv']]  = Q_mpv;  Q_list[i['sv']]   = Q_sv
        Q_list[i['smv']]  = Q_smv;  Q_list[i['lpv']]  = Q_lpv
        Q_list[i['rpv']]  = Q_rpv;  Q_list[i['tips']] = Q_tips
        Q_list[i['lgv']]  = Q_lgv;  Q_list[i['pgv']]  = Q_pgv
        Q = torch.stack(Q_list, dim=-1) * segment_mask

        collateral_frac = (Q_lgv * segment_mask[:, i['lgv']]
                           + Q_pgv * segment_mask[:, i['pgv']])
        tips_frac = Q_tips * segment_mask[:, i['tips']]
        liver_frac = Q_lpv + Q_rpv

        return {
            'Q': Q, 'inflow_frac': inflow_frac,
            'conf_outflow_frac': conf_frac, 'bif_outflow_frac': bif_frac,
            'inflow_delta': delta_in, 'conf_outflow_delta': delta_co,
            'bif_outflow_delta': delta_bo,
            'inflow_mask': mask_in, 'conf_outflow_mask': mask_co,
            'bif_outflow_mask': mask_bo,
            'collateral_fraction': collateral_frac,
            'tips_fraction': tips_frac, 'liver_fraction': liver_frac,
            'branch_resistance': branch_resistance,
        }


# =====================================================================
# Module 5 — Junction Physics
# =====================================================================
class JunctionPhysics(nn.Module):
    P_SCALE_PA = 100.0

    def forward(self, hemo_per_seg, flow_out, segment_mask, has_tips):
        B = segment_mask.size(0)
        device = segment_mask.device
        ix = SEG_INDEX

        m_inflow = segment_mask[:, ix['sv']] * segment_mask[:, ix['smv']]
        m_conf   = segment_mask[:, ix['mpv']]
        m_bif    = segment_mask[:, ix['mpv']] * (
            segment_mask[:, ix['lpv']] + segment_mask[:, ix['rpv']]
            + segment_mask[:, ix['tips']] > 0).float()
        m_bif_lr = segment_mask[:, ix['lpv']] * segment_mask[:, ix['rpv']]

        # Murray deviations
        murr_in = flow_out['inflow_delta'].pow(2).sum(-1) * m_inflow
        murr_co = flow_out['conf_outflow_delta'].pow(2).sum(-1) * m_conf
        murr_bo = flow_out['bif_outflow_delta'].pow(2).sum(-1) * m_bif

        # Pressure drops
        dP = {}
        for sn in SEGMENTS:
            dP[sn] = hemo_per_seg[SEG_INDEX[sn]]['pressure_drop_total']

        press_resid = ((dP['lpv'] - dP['rpv']) / self.P_SCALE_PA).pow(2) * m_bif_lr

        collat_frac = flow_out['collateral_fraction']
        tips_frac = flow_out['tips_fraction']
        liver_frac = flow_out['liver_fraction']

        features = torch.stack([
            murr_in, murr_co, murr_bo, press_resid,
            collat_frac, tips_frac, liver_frac,
        ] + [torch.log1p(dP[sn].clamp(min=0)) * segment_mask[:, SEG_INDEX[sn]]
             for sn in SEGMENTS], dim=-1)  # 7 + 8 = 15
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            'murray_dev_inflow': murr_in, 'murray_dev_conf_out': murr_co,
            'murray_dev_bif_out': murr_bo, 'press_resid_bifurc': press_resid,
            'collateral_fraction': collat_frac,
            'inflow_active': m_inflow, 'confluence_outflow_active': m_conf,
            'bifurcation_active': m_bif,
            'dP_per_branch': torch.stack([dP[sn] for sn in SEGMENTS], dim=-1),
            'features': features,
        }


# =====================================================================
# Module 6 — Full model
# =====================================================================
class PortalPressureNet(nn.Module):

    def __init__(self, d_hidden: int = 32, dropout: float = 0.15,
                 gnn_layers: int = 2, use_residual: bool = True,
                 use_q_scale: bool = True,
                 use_physics_baseline: bool = True,
                 use_aux: bool = True,
                 use_flow_features: bool = True,
                 use_branch_embed: bool = True,
                 use_profile_transformer: bool = True,
                 use_tips_head: bool = True,
                 use_aux_mask: bool = True,
                 pvp_mean_pa: float = PVP_MEAN_PA_DEFAULT,
                 pvp_std_pa:  float = PVP_STD_PA_DEFAULT,
                 log_vol_mean: float = LOG_VOL_MEAN_DEFAULT,
                 log_vol_std:  float = LOG_VOL_STD_DEFAULT):
        super().__init__()
        self.d_hidden = d_hidden
        self.use_residual = use_residual
        self.use_q_scale = use_q_scale
        self.use_physics_baseline = use_physics_baseline
        self.use_aux = use_aux
        self.use_flow_features = use_flow_features
        self.use_branch_embed = use_branch_embed
        self.use_profile_transformer = use_profile_transformer
        self.use_tips_head = use_tips_head
        self.use_aux_mask = use_aux_mask

        # ── Existing modules ─────────────────────────────────
        self.geom_encoder = GeometryEncoder(
            d_in=N_PROFILE_FEAT, d_hidden=d_hidden,
            n_blocks=3, dropout=dropout * 0.3,
        )
        self.profile_transformer = ProfileTransformerEncoder(
            d_in=N_PROFILE_FEAT, d_hidden=d_hidden,
            n_layers=2, n_heads=4, dropout=dropout * 0.3,
        )
        self.profile_fuse = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
        )
        self.branch_pool = AttentionPool(d_hidden, d_attn=16)

        self.vessel_gnn = VesselGraphNet(
            d_hidden=d_hidden, n_layers=gnn_layers, dropout=dropout * 0.3,
        )

        self.flow_est = FlowRateEstimator(
            d_branch=d_hidden, d_aux=N_AUX, d_hidden=d_hidden,
        )
        self.hydro = PoiseuilleHydrodynamics()
        self.junction_phys = JunctionPhysics()

        self.q_estimator = SplenicFlowEstimator(
            d_hidden=8, log_vol_mean=log_vol_mean, log_vol_std=log_vol_std,
        )

        # ── Physics-based predictor (residual on top of baseline) ─
        d_branches = N_SEGMENTS * d_hidden
        d_q        = N_SEGMENTS
        d_junction = 15
        d_qscale   = 1
        d_baseline = 1                          # ← NEW: pass baseline as a feature too
        d_aux_fused = N_AUX * (2 if use_aux_mask else 1)
        d_fused = d_branches + d_q + d_junction + d_qscale + d_baseline + d_aux_fused

        self.predictor = self._make_predictor(d_fused, d_hidden, dropout)
        self.predictor_pre = self._make_predictor(d_fused, d_hidden, dropout)
        self.predictor_post = self._make_predictor(d_fused, d_hidden, dropout)

        # Residual correction path
        d_residual_feats = N_SEGMENTS * 5
        self.residual_net = PhysicsResidualNet(
            d_in=d_fused + d_residual_feats, d_hidden=d_hidden,
        ) if use_residual else None

        # PVP normalization constants for the physics baseline.
        # These should be set to the training-set PVP mean and std (Pa).
        self.register_buffer('pvp_mean_pa', torch.tensor(float(pvp_mean_pa)))
        self.register_buffer('pvp_std_pa',  torch.tensor(float(pvp_std_pa)))

        self._init_weights()

        # ── Re-apply special inits AFTER _init_weights (which uses kaiming) ──

        # q_estimator: TRUE zero → starts at exp(0) = 1.0 (identity on Q).
        # Safe to fully zero: q_scale=1 is a perfectly valid runtime state
        # and gradient still flows because the input feature 'q_scale' enters
        # the predictor via a non-zero-init path.
        self.q_estimator.net[-1].weight.data.zero_()
        self.q_estimator.net[-1].bias.data.zero_()

        # Predictor final layers: small init, not zero, so gradients flow
        # into the encoders from the first step.
        for head in [self.predictor, self.predictor_pre, self.predictor_post]:
            head[-1].weight.data.mul_(0.01)
            head[-1].bias.data.zero_()

        # residual_net: TRUE zero → its job is purely corrective, no need
        # for it to inject random noise initially.
        if self.residual_net is not None:
            self.residual_net.net[-1].weight.data.zero_()
            self.residual_net.net[-1].bias.data.zero_()

        # flow_est: zero deltas → fall back to pure Murray prior at start.
        # Same reasoning as predictor would apply, EXCEPT that gradients
        # from `Q → hydro → baseline_pa → loss` flow directly through the
        # final layer's weights (via the input GELU activations), so the
        # head itself receives a useful signal. The first-layer encoders
        # get their signal through the `branch_embed → predictor` path
        # which we now keep open.
        for head in [self.flow_est.inflow_head[-1],
                     self.flow_est.conf_outflow_head[-1],
                     self.flow_est.bif_outflow_head[-1]]:
            head.weight.data.zero_()
            head.bias.data.zero_()

    @staticmethod
    def _make_predictor(d_fused: int, d_hidden: int, dropout: float):
        return nn.Sequential(
            nn.Linear(d_fused, d_hidden * 2),
            nn.LayerNorm(d_hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden * 2, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_hidden, 1),
        )

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

    # ------------------------------------------------------------------
    @staticmethod
    def _branch_resistance_prior(profiles, arc_lengths, point_valid, segment_mask):
        """
        Geometry-only resistance proxy. v5 uses the SAME compute_r_eff_mm
        helper as the forward Poiseuille path.
        """
        eps = 1e-6
        r_mm, _ = PoiseuilleHydrodynamics.compute_r_eff_mm(profiles, eps=eps)
        r_mm = r_mm.clamp(min=eps)

        ds = torch.zeros_like(arc_lengths)
        ds[..., 1:] = arc_lengths[..., 1:] - arc_lengths[..., :-1]
        ds[..., 0] = ds[..., 1]
        ds = ds.clamp(min=eps) * (point_valid > 0.5).float()

        resistance = (ds / r_mm.pow(4)).sum(dim=-1)
        valid = ((point_valid > 0.5).float().sum(dim=-1) > 1).float() * segment_mask
        high = torch.full_like(resistance, 1e6)
        resistance = torch.where(valid > 0.5, resistance.clamp(min=eps), high)
        return torch.nan_to_num(resistance, nan=1e6, posinf=1e6, neginf=1e6)

    # ------------------------------------------------------------------
    @staticmethod
    def _physics_baseline_pa(hemo_per_seg, segment_mask):
        """
        Cumulative ΔP from the hepatic sinusoids back to the PVP probe.

        Path traversed by blood in reverse (probe ← liver):
            probe-in-MPV ← MPV trunk ← bifurcation ← LPV/RPV ← sinusoids
        So baseline_pa = ΔP_mpv  +  mean(ΔP_lpv, ΔP_rpv).

        This represents what Poiseuille alone predicts for PVP minus the
        (essentially constant) hepatic-vein pressure, which is absorbed
        into the normalization mean.
        """
        ix = SEG_INDEX

        dP_mpv = hemo_per_seg[ix['mpv']]['pressure_drop_total']
        m_mpv  = segment_mask[:, ix['mpv']]

        dP_lpv = hemo_per_seg[ix['lpv']]['pressure_drop_total']
        dP_rpv = hemo_per_seg[ix['rpv']]['pressure_drop_total']
        m_lpv  = segment_mask[:, ix['lpv']]
        m_rpv  = segment_mask[:, ix['rpv']]

        n_liver  = (m_lpv + m_rpv).clamp(min=1.0)
        dP_liver = (dP_lpv * m_lpv + dP_rpv * m_rpv) / n_liver

        return dP_mpv * m_mpv + dP_liver  # (B,) Pa

    # ------------------------------------------------------------------
    def forward(self, batch):
        profiles      = batch['profiles']
        profiles_norm = batch['profiles_norm']
        arc_lengths   = batch['arc_lengths']
        point_valid   = batch['point_valid']
        segment_mask  = batch['segment_mask']
        aux_norm      = batch['aux_norm']
        aux_mask      = batch.get('aux_mask', torch.ones_like(aux_norm))
        organ_volumes = batch.get('organ_volumes',
                                  torch.zeros(profiles.size(0), 2, device=profiles.device))
        organ_valid   = batch.get('organ_valid',
                                  torch.zeros(profiles.size(0), 2, device=profiles.device))
        has_tips      = batch.get('is_post_tips', segment_mask[:, SEG_INDEX['tips']])

        B, S, N, _ = profiles.shape

        # ── Patient-specific Q scale from organ volumes ──────
        q_scale = self.q_estimator(organ_volumes, organ_valid)  # (B,)
        if not self.use_q_scale:
            q_scale = torch.ones_like(q_scale)

        # ── Per-branch encoding ──────────────────────────────
        branch_embed  = torch.zeros(B, S, self.d_hidden, device=profiles.device)
        attn_weights  = torch.zeros(B, S, N, device=profiles.device)
        for si in range(S):
            h_si = self.geom_encoder(profiles_norm[:, si])
            if self.use_profile_transformer:
                h_tx = self.profile_transformer(profiles_norm[:, si], point_valid[:, si], si)
                h_si = self.profile_fuse(torch.cat([h_si, h_tx], dim=-1))
            pooled, aw = self.branch_pool(h_si, point_valid[:, si])
            seg_alive = segment_mask[:, si].unsqueeze(-1)
            branch_embed[:, si] = pooled * seg_alive
            attn_weights[:, si] = aw

        # Graph message passing
        branch_embed = self.vessel_gnn(branch_embed, segment_mask)

        # Junction-end diameters
        junction_diam = profiles[:, :, 0, P_HDIAM]

        # Flow estimation (relative Q: fractions summing to 1)
        branch_resistance = self._branch_resistance_prior(
            profiles, arc_lengths, point_valid, segment_mask)
        aux_for_flow = aux_norm if self.use_aux else torch.zeros_like(aux_norm)
        flow_out = self.flow_est(
            branch_embed, aux_for_flow, segment_mask, junction_diam, branch_resistance)
        Q = flow_out['Q']  # (B, S) relative fractions

        # ── Per-branch Poiseuille with patient-specific Q ────
        Q_scaled = Q * q_scale.unsqueeze(-1)  # (B, S)
        hemo_per_seg = []
        for si in range(S):
            h = self.hydro(profiles[:, si], arc_lengths[:, si],
                           point_valid[:, si], Q_scaled[:, si])
            hemo_per_seg.append(h)

        # Junction physics
        jp = self.junction_phys(hemo_per_seg, flow_out, segment_mask, has_tips)

        # ── Physics baseline (v5) ────────────────────────────
        # Cumulative ΔP along the portal system → normalize to label space.
        baseline_pa   = self._physics_baseline_pa(hemo_per_seg, segment_mask)
        baseline_norm = ((baseline_pa - self.pvp_mean_pa) / self.pvp_std_pa
                         ).clamp(min=-5.0, max=5.0)
        if not self.use_physics_baseline:
            baseline_norm = torch.zeros_like(baseline_norm)

        # ── Predictor sees baseline as a feature too ─────────
        branch_for_fused = branch_embed if self.use_branch_embed else torch.zeros_like(branch_embed)
        q_for_fused = Q if self.use_flow_features else torch.zeros_like(Q)
        junction_for_fused = jp['features'] if self.use_flow_features else torch.zeros_like(jp['features'])
        aux_for_fused = aux_norm if self.use_aux else torch.zeros_like(aux_norm)
        aux_mask_for_fused = aux_mask if (self.use_aux and self.use_aux_mask) else torch.zeros_like(aux_mask)
        if self.use_aux_mask:
            aux_fused = torch.cat([aux_for_fused, aux_mask_for_fused], dim=-1)
        else:
            aux_fused = aux_for_fused
        branch_flat = branch_for_fused.reshape(B, -1)
        fused = torch.cat([
            branch_flat, q_for_fused, junction_for_fused,
            q_scale.unsqueeze(-1),
            baseline_norm.unsqueeze(-1),
            aux_fused,
        ], dim=-1)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1e3, neginf=-1e3)

        # Predictor's final layer is zero-init → at start, predictor outputs 0.
        # So initial prediction = baseline_norm (pure physics).
        if self.use_tips_head:
            pvp_pre = self.predictor_pre(fused).squeeze(-1)
            pvp_post = self.predictor_post(fused).squeeze(-1)
            pvp_correction = torch.where(has_tips.float() > 0.5, pvp_post, pvp_pre)
        else:
            pvp_correction = self.predictor(fused).squeeze(-1)
            pvp_pre = pvp_correction
            pvp_post = pvp_correction

        # Residual correction
        pvp_residual = torch.zeros_like(pvp_correction)
        if self.use_residual and self.residual_net is not None:
            resid_feats = PhysicsResidualNet.extract_features(
                hemo_per_seg, profiles, point_valid, segment_mask)
            resid_feats = torch.nan_to_num(resid_feats, nan=0.0, posinf=0.0, neginf=0.0)
            resid_input = torch.cat([fused, resid_feats], dim=-1)
            pvp_residual = self.residual_net(resid_input).squeeze(-1)

        pvp_pred = (baseline_norm + pvp_correction + pvp_residual).unsqueeze(-1)

        return {
            'pvp_pred': pvp_pred, 'Q': Q, 'flow_out': flow_out,
            # The "physics" path includes the explicit baseline + MLP correction.
            'pvp_physics': (baseline_norm + pvp_correction).unsqueeze(-1),
            'pvp_pre_head': (baseline_norm + pvp_pre).unsqueeze(-1),
            'pvp_post_head': (baseline_norm + pvp_post).unsqueeze(-1),
            'pvp_residual': pvp_residual.unsqueeze(-1),
            'pvp_baseline_norm': baseline_norm.unsqueeze(-1),  # ← interpretability
            'pvp_baseline_pa':   baseline_pa.unsqueeze(-1),
            'q_scale': q_scale,
            'attn_weights': attn_weights, 'hemo_per_seg': hemo_per_seg,
            'junction': jp, 'branch_embed': branch_embed,
            'endpoint_dP': torch.stack(
                [h['pressure_drop_total'] * segment_mask[:, si]
                 for si, h in enumerate(hemo_per_seg)], dim=-1),
            'segment_mask': segment_mask,
        }


# =====================================================================
# Module 7 — Physics-Informed Loss
# =====================================================================
class PhysicsInformedLoss(nn.Module):

    def __init__(self, lambda_murray=0.10, lambda_press=0.05,
                 lambda_smooth=0.01, lambda_physio=0.01,
                 lambda_mono=0.05, lambda_residual=0.05,
                 lambda_spread=0.50, extremity_alpha=1.5,
                 post_tips_high_alpha=0.0,
                 post_tips_high_threshold=0.5,
                 huber_delta=1.0):
        super().__init__()
        self.lambda_murray   = lambda_murray
        self.lambda_press    = lambda_press
        self.lambda_smooth   = lambda_smooth
        self.lambda_physio   = lambda_physio
        self.lambda_mono     = lambda_mono
        self.lambda_residual = lambda_residual
        self.lambda_spread   = lambda_spread
        self.extremity_alpha = extremity_alpha
        self.post_tips_high_alpha = post_tips_high_alpha
        self.post_tips_high_threshold = post_tips_high_threshold
        self.huber_delta     = huber_delta

    @staticmethod
    def _hinge_outside_range(x, lo, hi, mask):
        scale_lo = max(abs(lo), 1.0)
        scale_hi = max(abs(hi), 1.0)
        below = F.relu((lo - x) / scale_lo)
        above = F.relu((x - hi) / scale_hi)
        v = (below.pow(2) + above.pow(2)) * mask
        return v.sum() / mask.sum().clamp(min=1.0)

    def forward(self, model_out, label_norm, batch):
        pvp_pred = model_out['pvp_pred'].squeeze(-1)

        # ── Extremity-weighted Huber + asym ──────────────────
        extremity = label_norm.abs()
        extremity_weight = 1.0 + self.extremity_alpha * extremity.pow(2)
        has_tips = batch.get('is_post_tips', batch['segment_mask'][:, SEG_INDEX['tips']]).float()
        post_high = has_tips * (label_norm > self.post_tips_high_threshold).float()
        tail_weight = 1.0 + self.post_tips_high_alpha * post_high

        err = pvp_pred - label_norm
        asym = torch.where(
            (err < 0) & (label_norm > 0.5),
            torch.full_like(err, 1.5),
            torch.ones_like(err),
        )
        abs_err = err.abs()
        huber_elem = torch.where(
            abs_err <= self.huber_delta,
            0.5 * err.pow(2),
            self.huber_delta * (abs_err - 0.5 * self.huber_delta),
        )
        L_main = (huber_elem * extremity_weight * asym * tail_weight).mean()

        # ── Anti-shrinkage spread loss ───────────────────────
        if pvp_pred.numel() >= 4:
            pred_var = pvp_pred.var()
            label_var = label_norm.var().clamp(min=1e-3)
            L_spread = F.relu(1.0 - pred_var / label_var).pow(2)
        else:
            L_spread = torch.tensor(0.0, device=pvp_pred.device)

        # ── Residual size regularizer ────────────────────────
        pvp_residual = model_out.get('pvp_residual', torch.zeros_like(pvp_pred))
        L_residual = pvp_residual.squeeze(-1).pow(2).mean()

        flow = model_out['flow_out']
        jp   = model_out['junction']
        m_in = jp['inflow_active']
        m_co = jp['confluence_outflow_active']
        m_bo = jp['bifurcation_active']

        d_in = flow['inflow_delta']       * flow['inflow_mask']
        d_co = flow['conf_outflow_delta'] * flow['conf_outflow_mask']
        d_bo = flow['bif_outflow_delta']  * flow['bif_outflow_mask']

        L_murr_in = (d_in.pow(2).sum(-1) * m_in).sum() / m_in.sum().clamp(1)
        L_murr_co = (d_co.pow(2).sum(-1) * m_co).sum() / m_co.sum().clamp(1)

        # ── v5: TIPS patients fully disable bifurcation Murray loss ──
        # Murray's law describes biological vessel optimization; it does
        # NOT apply to a man-made stent whose diameter was chosen by the
        # interventionist.
        tips_mask_out = torch.where(has_tips > 0.5,
                                    torch.zeros_like(m_bo),
                                    torch.ones_like(m_bo))
        m_bo_w = m_bo * tips_mask_out
        L_murr_bo = (d_bo.pow(2).sum(-1) * m_bo_w).sum() / m_bo_w.sum().clamp(1)
        L_murray = L_murr_in + L_murr_co + L_murr_bo

        L_press = jp['press_resid_bifurc'].sum() / m_bo.sum().clamp(1)

        # ── v5: smoothness on the SAME r_eff used by forward physics ──
        profiles    = batch['profiles']
        point_valid = batch['point_valid']
        seg_mask    = batch['segment_mask']
        L_smooth = torch.tensor(0.0, device=pvp_pred.device)
        n_seg = 0
        for si in range(N_SEGMENTS):
            alive = seg_mask[:, si] > 0
            if alive.sum() == 0:
                continue
            # Same effective radius the Poiseuille layer actually uses.
            r_mm, _ = PoiseuilleHydrodynamics.compute_r_eff_mm(
                profiles[alive, si])  # (B', N)
            v = point_valid[alive, si]
            d2r = r_mm[:, 2:] - 2.0 * r_mm[:, 1:-1] + r_mm[:, :-2]
            mw = v[:, 2:] * v[:, 1:-1] * v[:, :-2]
            L_smooth = L_smooth + (d2r.pow(2) * mw).sum() / mw.sum().clamp(1)
            n_seg += 1
        if n_seg > 0:
            L_smooth = L_smooth / n_seg

        # ── Physio hinge on WSS and Re ───────────────────────
        L_physio = torch.tensor(0.0, device=pvp_pred.device)
        n_seg = 0
        for si in range(N_SEGMENTS):
            alive = seg_mask[:, si] > 0
            if alive.sum() == 0:
                continue
            h = model_out['hemo_per_seg'][si]
            v = point_valid[alive, si]
            L_physio = L_physio + self._hinge_outside_range(
                h['wss_pa'][alive], WSS_PHYSIO_LO_PA, WSS_PHYSIO_HI_PA, v)
            L_physio = L_physio + self._hinge_outside_range(
                h['reynolds'][alive], 0.0, RE_PHYSIO_HI, v)
            n_seg += 1
        if n_seg > 0:
            L_physio = L_physio / n_seg

        # ── Monotonicity of ΔP ───────────────────────────────
        L_mono = torch.tensor(0.0, device=pvp_pred.device)
        n_seg = 0
        for si in range(N_SEGMENTS):
            alive = seg_mask[:, si] > 0
            if alive.sum() == 0:
                continue
            dp = model_out['hemo_per_seg'][si]['pressure_drop_pa'][alive]
            v = point_valid[alive, si]
            ddp = dp[:, 1:] - dp[:, :-1]
            mw = v[:, 1:] * v[:, :-1]
            L_mono = L_mono + (F.relu(-ddp).pow(2) * mw).sum() / mw.sum().clamp(1)
            n_seg += 1
        if n_seg > 0:
            L_mono = L_mono / n_seg

        def _safe(x, cap=1e3):
            if not torch.is_tensor(x):
                x = torch.tensor(float(x), device=pvp_pred.device)
            return torch.nan_to_num(x, nan=0.0, posinf=cap, neginf=cap).clamp(max=cap)

        L_main_s   = _safe(L_main)
        L_murray_s = _safe(L_murray)
        L_press_s  = _safe(L_press)
        L_smooth_s = _safe(L_smooth)
        L_physio_s = _safe(L_physio)
        L_mono_s   = _safe(L_mono)

        L_total = (L_main_s
                   + self.lambda_murray * L_murray_s
                   + self.lambda_press  * L_press_s
                   + self.lambda_smooth * L_smooth_s
                   + self.lambda_physio * L_physio_s
                   + self.lambda_mono   * L_mono_s
                   + self.lambda_residual * _safe(L_residual)
                   + self.lambda_spread  * _safe(L_spread))

        log = {
            'main': float(L_main_s.detach()), 'murray': float(L_murray_s.detach()),
            'press': float(L_press_s.detach()), 'smooth': float(L_smooth_s.detach()),
            'physio': float(L_physio_s.detach()), 'mono': float(L_mono_s.detach()),
            'residual': float(_safe(L_residual).detach()),
            'spread': float(_safe(L_spread).detach()),
            'total': float(L_total.detach()),
        }
        return L_total, log


# =====================================================================
# Utility
# =====================================================================
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
