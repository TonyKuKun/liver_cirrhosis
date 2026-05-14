"""
Physics-Informed Geometric Deep Learning for Portal Vein Pressure — v4
=======================================================================
Shape-aware, PVT-robust, Poiseuille-grounded.

Key changes from v3:
  1.  11 pointwise channels (was 4): adds hydraulic_diameter, perimeter,
      torsion, solidity, r_insc_to_r_eq_ratio, dA_ds_norm, circularity,
      n_components.
  2.  Physics layer uses hydraulic_diameter (D_h = 4A/P) instead of
      eq_diameter for Poiseuille — correct for non-circular cross-sections.
  3.  Shape-aware resistance: interpolates between D_h/2 and inscribed_radius
      based on solidity, so PVT crescent lumens use the bottleneck radius.
  4.  26 aux scalars (was 11): adds clinical markers, PVT severity,
      tortuosity indices, TIPS stent params, area conservation, etc.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import (
    N_PROFILE_FEAT, N_SEGMENTS, SEGMENTS, SEG_INDEX, N_AUX,
    P_AREA, P_HDIAM, P_PERIM, P_CURV, P_TORS, P_INSC,
    P_SOLID, P_RRAT, P_DADS, P_CIRC, P_NCOMP,
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


# =====================================================================
# Module 1 — Poiseuille Hydrodynamics (shape-aware, no learnable params)
# =====================================================================
class PoiseuilleHydrodynamics(nn.Module):
    """
    Given per-point geometry (11 channels) + Q_rel:
      → velocity, WSS, Re, R', cumR, ΔP, Dean, area_gradient

    Key improvement: uses **hydraulic_diameter** for r in Poiseuille,
    with **shape-aware** blending toward inscribed_radius when solidity
    is low (= non-circular PVT crescent lumen).
    """

    def __init__(self, mu=BLOOD_VISCOSITY_PA_S, rho=BLOOD_DENSITY_KG_M3,
                 q_ref=Q_REF_M3_PER_S):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.nu = mu / rho
        self.q_ref = q_ref

    @staticmethod
    def _safe_seg_lengths(arc, valid, eps=1e-6):
        ds = torch.zeros_like(arc)
        ds[..., 1:] = arc[..., 1:] - arc[..., :-1]
        ds[..., 0]  = ds[..., 1]
        ds = ds.clamp(min=eps) * (valid > 0).float()
        return ds

    def forward(self, profiles, arc, valid, Q_rel):
        """
        profiles: (B, N, 11)  arc: (B, N)  valid: (B, N)  Q_rel: (B,)
        """
        eps = 1e-9

        # ── Geometry extraction ──────────────────────────────
        area_mm2    = profiles[..., P_AREA].clamp(min=eps)
        hdiam_mm    = profiles[..., P_HDIAM].clamp(min=eps)
        curv_inv_mm = profiles[..., P_CURV].abs()
        insc_mm     = profiles[..., P_INSC].clamp(min=eps)
        solidity    = profiles[..., P_SOLID].clamp(min=0.01, max=1.0)

        # Shape-aware effective radius:
        #   When solidity ≈ 1 (circular): use D_h/2 (= standard hydraulic radius)
        #   When solidity << 1 (crescent/PVT): blend toward inscribed_radius
        #   alpha = 0 for circle, ~0.5 for moderate PVT, ~1 for severe
        alpha = (1.0 - solidity).clamp(min=0.0, max=1.0)
        r_hdiam_mm  = 0.5 * hdiam_mm
        r_eff_mm    = (1.0 - alpha) * r_hdiam_mm + alpha * insc_mm

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

        # 4. Local resistance per unit length
        local_R = (8.0 * self.mu) / (math.pi * r_eff_m.pow(4) + eps)

        # 5. Cumulative resistance
        ds_m = self._safe_seg_lengths(arc, valid) * 1e-3
        cum_R = torch.cumsum(local_R * ds_m, dim=-1)

        # 6. Pressure drop
        pressure_drop = Q_abs * cum_R

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
            'shape_alpha':          alpha * v,  # for interpretability
            'cum_R_total':          (cum_R * v).max(dim=-1).values,
            'pressure_drop_total':  (pressure_drop * v).max(dim=-1).values,
            'valid_count':          v.sum(dim=-1),
        }
        for k in out:
            out[k] = torch.nan_to_num(out[k], nan=0.0, posinf=0.0, neginf=0.0)
        return out


# =====================================================================
# Module 2 — Geometry Encoder (shared across branches)
# =====================================================================
class GeometryEncoder(nn.Module):

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
# Module 3.5 — Vessel Topology GNN (dense attention, 8-node graph)
# =====================================================================
class VesselGraphNet(nn.Module):
    """
    Graph Attention on the 8-node vessel topology.

    AttentionPool compresses each branch independently — it doesn't know
    that SV dilation + MPV narrowing *together* mean high confluence pressure.
    GNN lets cross-branch reasoning happen BEFORE the predictor sees them.

    Dense masked self-attention (8 nodes is tiny; no need for sparse ops).
    """

    _EDGE_PAIRS = [
        ('sv',  'mpv'), ('smv', 'mpv'),
        ('mpv', 'lpv'), ('mpv', 'rpv'), ('mpv', 'tips'),
        ('lgv', 'mpv'), ('pgv', 'mpv'),
        ('sv',  'smv'), ('lpv', 'rpv'),
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

    Initialized to output **zero** → at epoch 0 prediction equals pure
    physics path. Residual gradually learns corrections during training.
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
        B = segment_mask.size(0)
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
            sol_mx = torch.where(v > 0.5, sol, big_neg)
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

    def __init__(self, d_hidden: int = 32, dropout: float = 0.3,
                 gnn_layers: int = 2, use_residual: bool = True):
        super().__init__()
        self.d_hidden = d_hidden
        self.use_residual = use_residual

        # ── Existing modules ─────────────────────────────────
        self.geom_encoder = GeometryEncoder(
            d_in=N_PROFILE_FEAT, d_hidden=d_hidden,
            n_blocks=3, dropout=dropout * 0.3,
        )
        self.branch_pool = AttentionPool(d_hidden, d_attn=16)

        # ── NEW: Graph message passing on vessel topology ────
        self.vessel_gnn = VesselGraphNet(
            d_hidden=d_hidden, n_layers=gnn_layers, dropout=dropout * 0.3,
        )

        self.flow_est = FlowRateEstimator(
            d_branch=d_hidden, d_aux=N_AUX, d_hidden=d_hidden,
        )
        self.hydro = PoiseuilleHydrodynamics()
        self.junction_phys = JunctionPhysics()

        # ── Physics-based predictor (main path) ──────────────
        d_branches = N_SEGMENTS * d_hidden
        d_q        = N_SEGMENTS
        d_junction = 15
        d_fused = d_branches + d_q + d_junction + N_AUX

        self.predictor = nn.Sequential(
            nn.Linear(d_fused, d_hidden * 2),
            nn.LayerNorm(d_hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden * 2, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_hidden, 1),
        )

        # ── NEW: Residual correction path ────────────────────
        # Captures non-Poiseuille effects (turbulence, vortices, entrance
        # effects) that the deterministic physics layer cannot model.
        # Initialized to output 0 → starts from pure physics prediction.
        d_residual_feats = N_SEGMENTS * 5   # 40 (from extract_features)
        self.residual_net = PhysicsResidualNet(
            d_in=d_fused + d_residual_feats, d_hidden=d_hidden,
        ) if use_residual else None

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

    @staticmethod
    def _branch_resistance_prior(profiles, arc_lengths, point_valid, segment_mask):
        """Geometry-only resistance proxy using shape-aware effective radius."""
        eps = 1e-6
        hdiam_mm = profiles[..., P_HDIAM].clamp(min=eps)
        insc_mm  = profiles[..., P_INSC].clamp(min=eps)
        solid    = profiles[..., P_SOLID].clamp(min=0.01, max=1.0)
        alpha    = (1.0 - solid).clamp(0, 1)
        r_mm = (1.0 - alpha) * (0.5 * hdiam_mm) + alpha * insc_mm
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

    def forward(self, batch):
        profiles      = batch['profiles']
        profiles_norm = batch['profiles_norm']
        arc_lengths   = batch['arc_lengths']
        point_valid   = batch['point_valid']
        segment_mask  = batch['segment_mask']
        aux_norm      = batch['aux_norm']
        has_tips      = batch.get('is_post_tips', segment_mask[:, SEG_INDEX['tips']])

        B, S, N, _ = profiles.shape

        # ── Per-branch encoding ──────────────────────────────
        branch_embed  = torch.zeros(B, S, self.d_hidden, device=profiles.device)
        attn_weights  = torch.zeros(B, S, N, device=profiles.device)
        for si in range(S):
            h_si = self.geom_encoder(profiles_norm[:, si])
            pooled, aw = self.branch_pool(h_si, point_valid[:, si])
            seg_alive = segment_mask[:, si].unsqueeze(-1)
            branch_embed[:, si] = pooled * seg_alive
            attn_weights[:, si] = aw

        # ── NEW: Graph message passing ───────────────────────
        # Each branch embedding now "knows" about its topological neighbors
        branch_embed = self.vessel_gnn(branch_embed, segment_mask)

        # Junction-end diameters (hydraulic diameter at idx 0)
        junction_diam = profiles[:, :, 0, P_HDIAM]

        # Flow estimation (uses GNN-enriched embeddings)
        branch_resistance = self._branch_resistance_prior(
            profiles, arc_lengths, point_valid, segment_mask)
        flow_out = self.flow_est(
            branch_embed, aux_norm, segment_mask, junction_diam, branch_resistance)
        Q = flow_out['Q']

        # Per-branch Poiseuille hydrodynamics
        hemo_per_seg = []
        for si in range(S):
            h = self.hydro(profiles[:, si], arc_lengths[:, si],
                           point_valid[:, si], Q[:, si])
            hemo_per_seg.append(h)

        # Junction physics
        jp = self.junction_phys(hemo_per_seg, flow_out, segment_mask, has_tips)

        # ── Physics-based prediction (main path) ─────────────
        branch_flat = branch_embed.reshape(B, -1)
        fused = torch.cat([branch_flat, Q, jp['features'], aux_norm], dim=-1)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)
        pvp_physics = self.predictor(fused)

        # ── NEW: Residual correction path ────────────────────
        pvp_residual = torch.zeros_like(pvp_physics)
        if self.use_residual and self.residual_net is not None:
            resid_feats = PhysicsResidualNet.extract_features(
                hemo_per_seg, profiles, point_valid, segment_mask)
            resid_feats = torch.nan_to_num(resid_feats, nan=0.0, posinf=0.0, neginf=0.0)
            resid_input = torch.cat([fused, resid_feats], dim=-1)
            pvp_residual = self.residual_net(resid_input)

        pvp_pred = pvp_physics + pvp_residual

        return {
            'pvp_pred': pvp_pred, 'Q': Q, 'flow_out': flow_out,
            'pvp_physics': pvp_physics,      # interpretable physics-only prediction
            'pvp_residual': pvp_residual,    # non-Poiseuille correction
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
                 severity_alpha=0.5, huber_delta=1.0):
        super().__init__()
        self.lambda_murray   = lambda_murray
        self.lambda_press    = lambda_press
        self.lambda_smooth   = lambda_smooth
        self.lambda_physio   = lambda_physio
        self.lambda_mono     = lambda_mono
        self.lambda_residual = lambda_residual
        self.severity_alpha  = severity_alpha
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

        # ── Severity-aware main loss ─────────────────────────
        # High PVP samples (label_norm >> 0) get more weight, preventing
        # the model from "playing it safe" by regressing toward the mean.
        #
        # weight_i = 1 + α · max(0, label_norm_i)
        #   → label at mean: weight = 1 (normal)
        #   → label 2σ above mean: weight = 1 + 2α (boosted)
        #
        # Also: asymmetric — underpredicting high PVP is penalized MORE
        # than overpredicting, because clinical consequence of missing
        # severe portal hypertension is much worse.
        severity_weight = 1.0 + self.severity_alpha * F.relu(label_norm)

        err = pvp_pred - label_norm
        # Asymmetric Huber: underprediction (err < 0) of high-label gets ×1.5
        under_penalty = torch.where(
            (err < 0) & (label_norm > 0.5),   # underpredicting high PVP
            torch.full_like(err, 1.5),
            torch.ones_like(err),
        )
        abs_err = err.abs()
        huber_elem = torch.where(
            abs_err <= self.huber_delta,
            0.5 * err.pow(2),
            self.huber_delta * (abs_err - 0.5 * self.huber_delta),
        )
        L_main = (huber_elem * severity_weight * under_penalty).mean()

        # ── L_residual: keep the residual small ─────────────
        # The physics path should do most of the work; the residual
        # only corrects what physics genuinely cannot capture.
        pvp_residual = model_out.get('pvp_residual', torch.zeros_like(pvp_pred))
        L_residual = pvp_residual.squeeze(-1).pow(2).mean()

        flow = model_out['flow_out']
        jp = model_out['junction']
        m_in = jp['inflow_active']
        m_co = jp['confluence_outflow_active']
        m_bo = jp['bifurcation_active']

        d_in = flow['inflow_delta']       * flow['inflow_mask']
        d_co = flow['conf_outflow_delta'] * flow['conf_outflow_mask']
        d_bo = flow['bif_outflow_delta']  * flow['bif_outflow_mask']

        L_murr_in = (d_in.pow(2).sum(-1) * m_in).sum() / m_in.sum().clamp(1)
        L_murr_co = (d_co.pow(2).sum(-1) * m_co).sum() / m_co.sum().clamp(1)

        has_tips = batch.get('is_post_tips', batch['segment_mask'][:, SEG_INDEX['tips']])
        tips_relax = torch.where(has_tips > 0.5,
                                 torch.full_like(m_bo, 0.35), torch.ones_like(m_bo))
        m_bo_w = m_bo * tips_relax
        L_murr_bo = (d_bo.pow(2).sum(-1) * m_bo_w).sum() / m_bo_w.sum().clamp(1)
        L_murray = L_murr_in + L_murr_co + L_murr_bo

        L_press = jp['press_resid_bifurc'].sum() / m_bo.sum().clamp(1)

        # Smoothness on hydraulic_diameter profile (was eq_diameter)
        profiles    = batch['profiles']
        point_valid = batch['point_valid']
        seg_mask    = batch['segment_mask']
        L_smooth = torch.tensor(0.0, device=pvp_pred.device)
        n_seg = 0
        for si in range(N_SEGMENTS):
            alive = seg_mask[:, si] > 0
            if alive.sum() == 0:
                continue
            r = profiles[alive, si, :, P_HDIAM] * 0.5
            v = point_valid[alive, si]
            d2r = r[:, 2:] - 2.0 * r[:, 1:-1] + r[:, :-2]
            mw = v[:, 2:] * v[:, 1:-1] * v[:, :-2]
            L_smooth = L_smooth + (d2r.pow(2) * mw).sum() / mw.sum().clamp(1)
            n_seg += 1
        if n_seg > 0:
            L_smooth = L_smooth / n_seg

        # Physio hinge on WSS and Re
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

        # Monotonicity of ΔP
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
                   + self.lambda_residual * _safe(L_residual))

        log = {
            'main': float(L_main_s.detach()), 'murray': float(L_murray_s.detach()),
            'press': float(L_press_s.detach()), 'smooth': float(L_smooth_s.detach()),
            'physio': float(L_physio_s.detach()), 'mono': float(L_mono_s.detach()),
            'residual': float(_safe(L_residual).detach()),
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