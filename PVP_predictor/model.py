"""Physics-constrained flow model for portal vein pressure prediction.

This model is intentionally separate from the root-level segmentation ``model.py``.
It keeps the data loader unchanged, but changes the modeling order to:

    selected profile geometry -> learnable physics layer -> flow features
    -> organ-volume flow scale -> global flow correction -> PVP prediction.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import (
    AUX_KEYS,
    N_AUX,
    N_PROFILE_FEAT,
    N_SEGMENTS,
    P_AREA,
    P_CIRC,
    P_CURV,
    P_DADS,
    P_HDIAM,
    P_INSC,
    P_NCOMP,
    P_PERIM,
    P_RRAT,
    P_SOLID,
    P_TORS,
    PROFILE_KEYS,
    SEGMENTS,
    SEG_INDEX,
)


BLOOD_VISCOSITY_PA_S = 3.5e-3
BLOOD_DENSITY_KG_M3 = 1060.0
BLOOD_KIN_VISCOSITY_M2_S = BLOOD_VISCOSITY_PA_S / BLOOD_DENSITY_KG_M3
Q_REF_M3_PER_S = 800.0 * 1e-6 / 60.0
MMHG_TO_PA = 133.322

WSS_PHYSIO_LO_PA = 0.05
WSS_PHYSIO_HI_PA = 5.0
RE_PHYSIO_HI = 1500.0

CORE_PROFILE_KEYS = (
    "area",
    "hydraulic_diameter",
    "curvature",
    "inscribed_radius",
    "solidity",
    "circularity",
    "dA_ds_norm",
)
OPTIONAL_PROFILE_KEYS = (
    "perimeter",
    "torsion",
    "r_insc_to_r_eq_ratio",
    "n_components",
)
EXCLUDED_AUX_KEYS = {"has_lgv", "has_pgv", "has_tips"}
UNRELIABLE_LENGTH_SEGMENTS = ("smv", "lpv", "rpv")
SIX_VESSEL_SEGMENTS = ("collateral", "sv", "mpv", "smv", "lpv", "rpv")
SIX_VESSEL_INDEX = {name: i for i, name in enumerate(SIX_VESSEL_SEGMENTS)}
THREE_VESSEL_SEGMENTS = ("collateral", "mpv", "sv")
THREE_VESSEL_INDEX = {name: i for i, name in enumerate(THREE_VESSEL_SEGMENTS)}
THREE_VESSEL_HELPER_SEGMENTS = ("smv", "lpv", "rpv")
COLLATERAL_SOURCE_SEGMENTS = ("lgv", "pgv", "tips")
COLLATERAL_TYPE_NAMES = ("none", "lgv", "pgv", "tips")


def _indices_for(keys: Sequence[str]) -> List[int]:
    return [PROFILE_KEYS.index(k) for k in keys]


CORE_PROFILE_INDICES = _indices_for(CORE_PROFILE_KEYS)
ALL_PROFILE_INDICES = list(range(N_PROFILE_FEAT))
ORGAN_AUX_INDICES = [
    AUX_KEYS.index("spleen_volume_ml"),
    AUX_KEYS.index("liver_volume_ml"),
    AUX_KEYS.index("spleen_liver_ratio"),
]
GLOBAL_AUX_INDICES = [
    i for i, key in enumerate(AUX_KEYS)
    if key not in EXCLUDED_AUX_KEYS and i not in ORGAN_AUX_INDICES
]


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int, eps: float = 1e-6) -> torch.Tensor:
    w = mask.unsqueeze(-1)
    return (x * w).sum(dim=dim) / w.sum(dim=dim).clamp(min=eps)


def masked_max(x: torch.Tensor, mask: torch.Tensor, dim: int, fill: float = -1e9) -> torch.Tensor:
    return torch.where(mask.unsqueeze(-1) > 0.5, x, torch.full_like(x, fill)).max(dim=dim).values


def masked_min(x: torch.Tensor, mask: torch.Tensor, dim: int, fill: float = 1e9) -> torch.Tensor:
    return torch.where(mask.unsqueeze(-1) > 0.5, x, torch.full_like(x, fill)).min(dim=dim).values


def _collateral_source_weights(segment_mask: torch.Tensor):
    """Return one-hot source weights for merged collateral branch.

    Priority is TIPS > LGV > PGV. The data should rarely have more than one,
    but priority keeps the merged branch deterministic if it happens.
    """
    lgv = segment_mask[:, SEG_INDEX["lgv"]] > 0.5
    pgv = segment_mask[:, SEG_INDEX["pgv"]] > 0.5
    tips = segment_mask[:, SEG_INDEX["tips"]] > 0.5
    source_id = torch.where(
        tips,
        torch.full_like(segment_mask[:, 0], 3, dtype=torch.long),
        torch.where(
            lgv,
            torch.full_like(segment_mask[:, 0], 1, dtype=torch.long),
            torch.where(
                pgv,
                torch.full_like(segment_mask[:, 0], 2, dtype=torch.long),
                torch.zeros_like(segment_mask[:, 0], dtype=torch.long),
            ),
        ),
    )
    weights = torch.stack([
        (source_id == 1).float(),
        (source_id == 2).float(),
        (source_id == 3).float(),
    ], dim=-1)
    return weights, source_id


def _merge_collateral_tensor(tensor: torch.Tensor, source_weights: torch.Tensor) -> torch.Tensor:
    idx = [SEG_INDEX[name] for name in COLLATERAL_SOURCE_SEGMENTS]
    selected = tensor[:, idx]
    view_shape = [tensor.size(0), len(idx)] + [1] * (tensor.dim() - 2)
    return (selected * source_weights.view(*view_shape)).sum(dim=1)


def _apply_batch_weight(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    view_shape = [weight.size(0)] + [1] * (value.dim() - 1)
    return value * weight.view(*view_shape)


class GeometryFeatureSelector(nn.Module):
    """Select only the trusted profile channels by default."""

    def __init__(self, use_all_profile_channels: bool = False):
        super().__init__()
        indices = ALL_PROFILE_INDICES if use_all_profile_channels else CORE_PROFILE_INDICES
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long), persistent=False)
        self.feature_names = [PROFILE_KEYS[i] for i in indices]

    @property
    def n_features(self) -> int:
        return int(self.indices.numel())

    def forward(self, profiles: torch.Tensor) -> torch.Tensor:
        return profiles.index_select(-1, self.indices.to(profiles.device))


class BranchProfileEncoder(nn.Module):
    """Compact encoder over selected pointwise geometry channels."""

    def __init__(self, d_in: int, d_hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        d_stats = d_in * 4
        self.net = nn.Sequential(
            nn.Linear(d_stats, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
        )

    def forward(self, selected_profiles_norm: torch.Tensor, point_valid: torch.Tensor, segment_mask: torch.Tensor):
        means = masked_mean(selected_profiles_norm, point_valid, dim=2)
        maxs = masked_max(selected_profiles_norm, point_valid, dim=2)
        mins = masked_min(selected_profiles_norm, point_valid, dim=2)
        valid_frac = point_valid.mean(dim=-1, keepdim=True).expand_as(means)
        stats = torch.cat([means, maxs, mins, valid_frac], dim=-1)
        stats = torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
        emb = self.net(stats)
        return emb * segment_mask.unsqueeze(-1), stats * segment_mask.unsqueeze(-1)


class OrganFlowScaleNet(nn.Module):
    """Estimate patient-specific portal inflow scale from spleen/liver volumes."""

    def __init__(self, d_hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, organ_volumes: torch.Tensor, organ_valid: torch.Tensor, enabled: bool = True) -> torch.Tensor:
        if not enabled:
            return torch.ones(organ_volumes.size(0), device=organ_volumes.device)
        vols = organ_volumes.clamp(min=0.0)
        log_vols = torch.log1p(vols)
        ratio = torch.zeros(vols.size(0), 1, device=vols.device)
        valid_ratio = (organ_valid[:, :1] * organ_valid[:, 1:2]).clamp(0.0, 1.0)
        ratio = torch.where(
            valid_ratio > 0.5,
            vols[:, :1] / vols[:, 1:2].clamp(min=1.0),
            ratio,
        )
        x = torch.cat([log_vols, ratio, organ_valid, valid_ratio], dim=-1)
        log_scale = self.net(torch.nan_to_num(x, nan=0.0)).squeeze(-1)
        return torch.exp(log_scale.clamp(-1.2, 1.1))


def organ_volume_features(organ_volumes: torch.Tensor, organ_valid: torch.Tensor) -> torch.Tensor:
    vols = organ_volumes.clamp(min=0.0)
    log_spleen = torch.log1p(vols[:, :1]) / 8.0
    log_liver = torch.log1p(vols[:, 1:2]) / 8.0
    ratio_valid = (organ_valid[:, :1] * organ_valid[:, 1:2]).clamp(0.0, 1.0)
    ratio = torch.where(
        ratio_valid > 0.5,
        vols[:, :1] / vols[:, 1:2].clamp(min=1.0),
        torch.zeros_like(log_spleen),
    ).clamp(0.0, 5.0)
    return torch.cat([log_spleen, log_liver, ratio, organ_valid[:, :1], organ_valid[:, 1:2]], dim=-1)


class OrganBranchScaleNet(nn.Module):
    """Organ-volume boundary scales for inlet branches.

    Liver volume only scales the MPV inlet boundary, and spleen volume only
    scales the SV inlet boundary. Other branches keep scale 1.0 so organ
    volumes do not globally inflate all internal flows.
    """

    def __init__(self, branch_names: Sequence[str], d_hidden: int = 16):
        super().__init__()
        self.branch_names = tuple(branch_names)
        self.n_branches = len(self.branch_names)
        self.sv_net = nn.Sequential(
            nn.Linear(2, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
        )
        self.mpv_net = nn.Sequential(
            nn.Linear(2, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
        )
        for net in (self.sv_net, self.mpv_net):
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

    def forward(
        self,
        organ_volumes: torch.Tensor,
        organ_valid: torch.Tensor,
        segment_mask: torch.Tensor,
        collateral_type_onehot: torch.Tensor,
        enabled: bool = True,
    ) -> torch.Tensor:
        B = segment_mask.size(0)
        if not enabled:
            return torch.ones(B, self.n_branches, device=segment_mask.device)
        organ = organ_volume_features(organ_volumes, organ_valid)
        scale = torch.ones(B, self.n_branches, device=segment_mask.device)

        if "sv" in self.branch_names:
            sv_ix = self.branch_names.index("sv")
            sv_x = torch.cat([organ[:, 0:1], organ[:, 3:4]], dim=-1)
            sv_scale = torch.exp(self.sv_net(torch.nan_to_num(sv_x, nan=0.0)).squeeze(-1).clamp(-0.9, 0.9))
            scale[:, sv_ix] = sv_scale

        if "mpv" in self.branch_names:
            mpv_ix = self.branch_names.index("mpv")
            mpv_x = torch.cat([organ[:, 1:2], organ[:, 4:5]], dim=-1)
            mpv_scale = torch.exp(self.mpv_net(torch.nan_to_num(mpv_x, nan=0.0)).squeeze(-1).clamp(-0.9, 0.9))
            scale[:, mpv_ix] = mpv_scale

        return scale * segment_mask + (1.0 - segment_mask)


class GlobalFlowCorrector(nn.Module):
    """Use global non-flag features to correct flow logits and flow features."""

    def __init__(
        self,
        d_branch: int,
        d_flow: int,
        d_hidden: int = 32,
        dropout: float = 0.1,
        use_organ_global_features: bool = True,
        n_branches: int = N_SEGMENTS,
        flow_logit_dim: int = 8,
        d_extra_global: int = 0,
    ):
        super().__init__()
        self.n_branches = int(n_branches)
        self.flow_logit_dim = int(flow_logit_dim)
        self.d_extra_global = int(d_extra_global)
        indices = list(GLOBAL_AUX_INDICES)
        if use_organ_global_features:
            indices.extend(i for i in ORGAN_AUX_INDICES if i not in indices)
        self.register_buffer("aux_indices", torch.tensor(indices, dtype=torch.long), persistent=False)
        self.aux_names = [AUX_KEYS[i] for i in indices]
        self.aux_encoder = nn.Sequential(
            nn.Linear(len(indices) * 2 + self.d_extra_global, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
        )
        self.flow_delta = nn.Sequential(
            nn.Linear(d_flow + d_branch + d_hidden + 1, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_flow),
        )
        self.logit_delta = nn.Sequential(
            nn.Linear(d_branch * self.n_branches + d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, self.flow_logit_dim),
        )
        nn.init.zeros_(self.flow_delta[-1].weight)
        nn.init.zeros_(self.flow_delta[-1].bias)
        nn.init.zeros_(self.logit_delta[-1].weight)
        nn.init.zeros_(self.logit_delta[-1].bias)

    def select_aux(
        self,
        aux_norm: torch.Tensor,
        aux_mask: torch.Tensor,
        extra_global_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        idx = self.aux_indices.to(aux_norm.device)
        x = aux_norm.index_select(-1, idx)
        m = aux_mask.index_select(-1, idx)
        parts = [x, m]
        if self.d_extra_global > 0:
            if extra_global_features is None:
                extra_global_features = torch.zeros(
                    aux_norm.size(0), self.d_extra_global, device=aux_norm.device
                )
            parts.append(extra_global_features)
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        branch_embed: torch.Tensor,
        flow_features: torch.Tensor,
        aux_norm: torch.Tensor,
        aux_mask: torch.Tensor,
        segment_mask: torch.Tensor,
        extra_global_features: torch.Tensor | None = None,
        enabled: bool = True,
    ):
        B = branch_embed.size(0)
        if not enabled:
            context = torch.zeros(B, branch_embed.size(-1), device=branch_embed.device)
            return flow_features, torch.zeros(B, self.flow_logit_dim, device=branch_embed.device), context
        aux_context = self.aux_encoder(self.select_aux(aux_norm, aux_mask, extra_global_features))
        aux_rep = aux_context.unsqueeze(1).expand(-1, self.n_branches, -1)
        flow_in = torch.cat([flow_features, branch_embed, aux_rep, segment_mask.unsqueeze(-1)], dim=-1)
        delta = torch.tanh(self.flow_delta(flow_in)) * 0.25
        corrected = (flow_features + delta) * segment_mask.unsqueeze(-1)
        logit_delta = torch.tanh(self.logit_delta(torch.cat([branch_embed.reshape(B, -1), aux_context], dim=-1)))
        return corrected, logit_delta, aux_context


class LearnablePhysicsLayer(nn.Module):
    """Differentiable hemodynamics layer with learnable physical calibration."""

    def __init__(
        self,
        fixed_physics_params: bool = False,
        use_unreliable_raw_lengths: bool = False,
        branch_names: Sequence[str] = SEGMENTS,
        q_ref: float = Q_REF_M3_PER_S,
    ):
        super().__init__()
        self.q_ref = q_ref
        self.use_unreliable_raw_lengths = use_unreliable_raw_lengths
        self.branch_names = tuple(branch_names)
        raw_mask = torch.ones(len(self.branch_names), dtype=torch.float32)
        if not use_unreliable_raw_lengths:
            for name in UNRELIABLE_LENGTH_SEGMENTS:
                if name in self.branch_names:
                    raw_mask[self.branch_names.index(name)] = 0.0
        self.register_buffer("used_raw_length_mask", raw_mask, persistent=False)
        self.log_mu_scale = nn.Parameter(torch.zeros(()))
        self.radius_power_delta = nn.Parameter(torch.zeros(()))
        self.log_pressure_scale = nn.Parameter(torch.zeros(()))
        self.log_length_scale_mm = nn.Parameter(torch.log(torch.full((len(self.branch_names),), 70.0)))
        if fixed_physics_params:
            for p in self.parameters():
                p.requires_grad_(False)

    def physical_parameters(self) -> Mapping[str, torch.Tensor]:
        return {
            "mu_scale": torch.exp(self.log_mu_scale.clamp(-1.0, 1.0)),
            "radius_power": 4.0 + self.radius_power_delta.clamp(-0.8, 0.8),
            "pressure_scale": torch.exp(self.log_pressure_scale.clamp(-1.0, 1.0)),
            "learned_length_scale_mm": torch.exp(self.log_length_scale_mm.clamp(math.log(10.0), math.log(250.0))),
        }

    @staticmethod
    def effective_radius_mm(profiles: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        hdiam = profiles[..., P_HDIAM].clamp(min=eps)
        insc = profiles[..., P_INSC].clamp(min=eps)
        solid = profiles[..., P_SOLID].clamp(min=0.02, max=1.0)
        alpha = (1.0 - solid).clamp(0.0, 1.0)
        return ((1.0 - alpha) * 0.5 * hdiam + alpha * insc).clamp(min=eps)

    def _ds_mm(self, arc_lengths: torch.Tensor, point_valid: torch.Tensor) -> torch.Tensor:
        B, S, N = arc_lengths.shape
        raw = torch.zeros_like(arc_lengths)
        raw[..., 1:] = arc_lengths[..., 1:] - arc_lengths[..., :-1]
        raw[..., 0] = raw[..., 1]
        raw = raw.clamp(min=1e-4)

        valid_count = point_valid.sum(dim=-1, keepdim=True).clamp(min=2.0)
        learned_total = torch.exp(
            self.log_length_scale_mm.clamp(math.log(10.0), math.log(250.0))
        ).view(1, S, 1)
        normalized = learned_total / (valid_count - 1.0).clamp(min=1.0)
        raw_mask = self.used_raw_length_mask.view(1, S, 1).to(arc_lengths.device)
        ds = raw * raw_mask + normalized * (1.0 - raw_mask)
        return ds * (point_valid > 0.5).float()

    def resistance_proxy(self, profiles: torch.Tensor, arc_lengths: torch.Tensor, point_valid: torch.Tensor, segment_mask: torch.Tensor):
        r_mm = self.effective_radius_mm(profiles)
        ds_mm = self._ds_mm(arc_lengths, point_valid)
        power = self.physical_parameters()["radius_power"]
        resistance = (ds_mm / r_mm.pow(power)).sum(dim=-1)
        valid = ((point_valid > 0.5).float().sum(dim=-1) > 1).float() * segment_mask
        high = torch.full_like(resistance, 1e6)
        return torch.where(valid > 0.5, resistance.clamp(min=1e-6), high)

    def forward(
        self,
        profiles: torch.Tensor,
        arc_lengths: torch.Tensor,
        point_valid: torch.Tensor,
        segment_mask: torch.Tensor,
        q_rel: torch.Tensor,
        q_scale: torch.Tensor,
    ):
        params = self.physical_parameters()
        mu = BLOOD_VISCOSITY_PA_S * params["mu_scale"]
        nu = BLOOD_KIN_VISCOSITY_M2_S * params["mu_scale"]
        pressure_scale = params["pressure_scale"]
        radius_power = params["radius_power"]

        area_mm2 = profiles[..., P_AREA].clamp(min=1e-6)
        hdiam_mm = profiles[..., P_HDIAM].clamp(min=1e-6)
        curv_inv_mm = profiles[..., P_CURV].abs()
        r_eff_mm = self.effective_radius_mm(profiles)
        ds_m = self._ds_mm(arc_lengths, point_valid) * 1e-3

        area_m2 = area_mm2 * 1e-6
        hdiam_m = hdiam_mm * 1e-3
        r_eff_m = r_eff_mm * 1e-3
        curv_1_m = curv_inv_mm * 1e3
        if q_scale.dim() == 1:
            q_scale_branch = q_scale.unsqueeze(-1).expand_as(q_rel)
        else:
            q_scale_branch = q_scale
        q_abs = (q_rel * q_scale_branch).unsqueeze(-1) * self.q_ref

        velocity = q_abs / area_m2.clamp(min=1e-9)
        wss = (4.0 * mu * q_abs) / (math.pi * r_eff_m.pow(3.0).clamp(min=1e-12))
        reynolds = velocity * hdiam_m / nu.clamp(min=1e-9)
        local_R = (8.0 * mu) / (math.pi * r_eff_m.pow(radius_power).clamp(min=1e-12))
        local_R = local_R.clamp(max=1e13)
        cum_R = torch.cumsum(local_R * ds_m, dim=-1).clamp(max=2e12)
        pressure_drop = (q_abs * cum_R * pressure_scale).clamp(max=1e5)
        dean = reynolds * torch.sqrt((hdiam_m * curv_1_m).clamp(min=1e-12))

        area_grad = torch.zeros_like(area_mm2)
        ds_mm = self._ds_mm(arc_lengths, point_valid).clamp(min=1e-4)
        area_grad[..., 1:] = (area_mm2[..., 1:] - area_mm2[..., :-1]) / ds_mm[..., 1:]
        area_grad[..., 0] = area_grad[..., 1]

        v = (point_valid > 0.5).float()
        hemo = []
        for si in range(profiles.size(1)):
            vv = v[:, si]
            hemo.append({
                "radius_m": r_eff_m[:, si] * vv,
                "area_m2": area_m2[:, si] * vv,
                "velocity_m_per_s": velocity[:, si] * vv,
                "wss_pa": wss[:, si] * vv,
                "reynolds": reynolds[:, si] * vv,
                "local_R_pa_s_per_m4": local_R[:, si] * vv,
                "cum_R_pa_s_per_m3": cum_R[:, si] * vv,
                "pressure_drop_pa": pressure_drop[:, si] * vv,
                "pressure_drop_total": pressure_drop[:, si, -1] * segment_mask[:, si],
                "dean": dean[:, si] * vv,
                "area_gradient": area_grad[:, si] * vv,
            })
        return hemo

    @staticmethod
    def summarize_hemodynamics(hemo_per_seg: Sequence[Mapping[str, torch.Tensor]], point_valid: torch.Tensor, segment_mask: torch.Tensor):
        feats = []
        for si, h in enumerate(hemo_per_seg):
            v = point_valid[:, si]
            alive = segment_mask[:, si].unsqueeze(-1)
            vel = masked_mean(torch.log1p(h["velocity_m_per_s"].abs()).unsqueeze(-1), v, dim=1)
            wss = masked_mean(torch.log1p(h["wss_pa"].abs()).unsqueeze(-1), v, dim=1)
            re_max = masked_max(torch.log1p(h["reynolds"].abs()).unsqueeze(-1), v, dim=1)
            dean_max = masked_max(torch.log1p(h["dean"].abs()).unsqueeze(-1), v, dim=1)
            dp = torch.log1p(h["pressure_drop_total"].abs()).unsqueeze(-1)
            ag = masked_max(torch.log1p(h["area_gradient"].abs()).unsqueeze(-1), v, dim=1)
            feats.append(torch.cat([vel, wss, re_max, dean_max, dp, ag], dim=-1) * alive)
        return torch.stack(feats, dim=1)


class FlowAllocator(nn.Module):
    """Allocate relative flow through portal-system junctions."""

    def __init__(
        self,
        d_branch: int,
        d_hidden: int = 32,
        branch_names: Sequence[str] = SEGMENTS,
    ):
        super().__init__()
        self.branch_names = tuple(branch_names)
        self.branch_index = {name: i for i, name in enumerate(self.branch_names)}
        self.n_branches = len(self.branch_names)
        if self.branch_names == THREE_VESSEL_SEGMENTS:
            self.flow_logit_dim = 6
        elif self.branch_names == SIX_VESSEL_SEGMENTS:
            self.flow_logit_dim = 7
        else:
            self.flow_logit_dim = 9
        self.ctx = nn.Sequential(
            nn.Linear(d_branch * self.n_branches + self.flow_logit_dim, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, self.flow_logit_dim),
        )
        nn.init.zeros_(self.ctx[-1].weight)
        nn.init.zeros_(self.ctx[-1].bias)

    @staticmethod
    def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = logits.masked_fill(mask <= 0.5, -1e9)
        out = F.softmax(masked, dim=-1)
        fallback = torch.full_like(out, 1.0 / out.size(-1))
        return torch.where(mask.sum(dim=-1, keepdim=True) > 0.5, out, fallback) * mask

    @staticmethod
    def _prior(diam_mm: torch.Tensor, resistance: torch.Tensor) -> torch.Tensor:
        diam_prior = 3.0 * torch.log(diam_mm.clamp(min=1e-3))
        conduct_prior = -0.35 * torch.log(resistance.clamp(min=1e-6))
        return diam_prior + conduct_prior

    def forward(
        self,
        branch_embed: torch.Tensor,
        segment_mask: torch.Tensor,
        junction_diameters: torch.Tensor,
        branch_resistance: torch.Tensor,
        global_logit_delta: torch.Tensor,
        collateral_source_weights: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        B = branch_embed.size(0)
        base_delta = self.ctx(torch.cat([branch_embed.reshape(B, -1), global_logit_delta], dim=-1))
        delta = base_delta + global_logit_delta
        if self.branch_names == THREE_VESSEL_SEGMENTS:
            return self._forward_three(
                segment_mask,
                junction_diameters,
                branch_resistance,
                delta,
                collateral_source_weights,
            )
        if self.branch_names == SIX_VESSEL_SEGMENTS:
            return self._forward_six(
                segment_mask,
                junction_diameters,
                branch_resistance,
                delta,
                collateral_source_weights,
            )
        return self._forward_eight(segment_mask, junction_diameters, branch_resistance, delta)

    def _forward_eight(
        self,
        segment_mask: torch.Tensor,
        junction_diameters: torch.Tensor,
        branch_resistance: torch.Tensor,
        delta: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        ix = SEG_INDEX

        inflow_idx = [ix["sv"], ix["smv"]]
        sv_split_idx = [ix["mpv"], ix["pgv"]]
        conf_idx = [ix["mpv"], ix["lgv"]]
        bif_idx = [ix["lpv"], ix["rpv"], ix["tips"]]

        inflow_prior = self._prior(junction_diameters[:, inflow_idx], branch_resistance[:, inflow_idx])
        sv_split_prior = self._prior(junction_diameters[:, sv_split_idx], branch_resistance[:, sv_split_idx])
        conf_prior = self._prior(junction_diameters[:, conf_idx], branch_resistance[:, conf_idx])
        bif_prior = self._prior(junction_diameters[:, bif_idx], branch_resistance[:, bif_idx])

        inflow_mask = segment_mask[:, inflow_idx]
        sv_split_mask = torch.stack([
            segment_mask[:, ix["sv"]] * segment_mask[:, ix["mpv"]],
            segment_mask[:, ix["sv"]] * segment_mask[:, ix["pgv"]],
        ], dim=-1)
        conf_mask = segment_mask[:, conf_idx]
        bif_mask = segment_mask[:, bif_idx]

        inflow_frac = self._masked_softmax(inflow_prior + delta[:, 0:2], inflow_mask)
        sv_split_frac = self._masked_softmax(sv_split_prior + delta[:, 2:4], sv_split_mask)
        conf_frac = self._masked_softmax(conf_prior + delta[:, 4:6], conf_mask)
        bif_frac = self._masked_softmax(bif_prior + delta[:, 6:9], bif_mask)

        q_sv_total = inflow_frac[:, 0]
        q_smv = inflow_frac[:, 1]
        q_pgv = q_sv_total * sv_split_frac[:, 1] * segment_mask[:, ix["pgv"]]
        q_sv_to_mpv = q_sv_total * sv_split_frac[:, 0]
        q_portal_pool = (q_sv_to_mpv + q_smv).clamp(min=0.0)
        q_lgv = q_portal_pool * conf_frac[:, 1] * segment_mask[:, ix["lgv"]]
        q_mpv = q_portal_pool * conf_frac[:, 0]
        q_cols = []
        for name in SEGMENTS:
            if name == "mpv":
                q_cols.append(q_mpv)
            elif name == "sv":
                q_cols.append(q_sv_total)
            elif name == "smv":
                q_cols.append(q_smv)
            elif name == "lpv":
                q_cols.append(q_mpv * bif_frac[:, 0])
            elif name == "rpv":
                q_cols.append(q_mpv * bif_frac[:, 1])
            elif name == "tips":
                q_cols.append(q_mpv * bif_frac[:, 2])
            elif name == "lgv":
                q_cols.append(q_lgv)
            elif name == "pgv":
                q_cols.append(q_pgv)
        q = torch.stack(q_cols, dim=-1) * segment_mask
        split_resid = (
            (q[:, ix["mpv"]] + q[:, ix["lgv"]] - q_smv - q_sv_to_mpv).pow(2)
            + (q_sv_to_mpv + q[:, ix["pgv"]] - q[:, ix["sv"]]).pow(2)
            + (q[:, ix["lpv"]] + q[:, ix["rpv"]] + q[:, ix["tips"]] - q[:, ix["mpv"]]).pow(2)
        )

        return {
            "Q": q,
            "inflow_frac": inflow_frac,
            "sv_outflow_frac": sv_split_frac,
            "conf_outflow_frac": conf_frac,
            "bif_outflow_frac": bif_frac,
            "inflow_delta": delta[:, 0:2],
            "sv_outflow_delta": delta[:, 2:4],
            "conf_outflow_delta": delta[:, 4:6],
            "bif_outflow_delta": delta[:, 6:9],
            "inflow_mask": inflow_mask,
            "sv_outflow_mask": sv_split_mask,
            "conf_outflow_mask": conf_mask,
            "bif_outflow_mask": bif_mask,
            "tips_fraction": bif_frac[:, 2] * segment_mask[:, ix["tips"]],
            "collateral_fraction": q[:, ix["lgv"]] + q[:, ix["pgv"]],
            "liver_fraction": q[:, ix["lpv"]] + q[:, ix["rpv"]],
            "split_residual": split_resid,
            "branch_resistance": branch_resistance,
            "Q_model": q,
        }

    def _forward_six(
        self,
        segment_mask: torch.Tensor,
        junction_diameters: torch.Tensor,
        branch_resistance: torch.Tensor,
        delta: torch.Tensor,
        collateral_source_weights: torch.Tensor | None,
    ) -> Dict[str, torch.Tensor]:
        ix = self.branch_index
        inflow_idx = [ix["sv"], ix["smv"]]
        conf_idx = [ix["mpv"], ix["collateral"]]
        bif_idx = [ix["lpv"], ix["rpv"], ix["collateral"]]

        inflow_prior = self._prior(junction_diameters[:, inflow_idx], branch_resistance[:, inflow_idx])
        conf_prior = self._prior(junction_diameters[:, conf_idx], branch_resistance[:, conf_idx])
        bif_prior = self._prior(junction_diameters[:, bif_idx], branch_resistance[:, bif_idx])

        inflow_mask = segment_mask[:, inflow_idx]
        if collateral_source_weights is None:
            collateral_source_weights = torch.zeros(segment_mask.size(0), 3, device=segment_mask.device)
        is_tips = collateral_source_weights[:, 2]
        is_natural_collateral = (collateral_source_weights[:, 0] + collateral_source_weights[:, 1]).clamp(max=1.0)
        conf_mask = torch.stack([
            segment_mask[:, ix["mpv"]],
            segment_mask[:, ix["collateral"]] * is_natural_collateral,
        ], dim=-1)
        bif_mask = torch.stack([
            segment_mask[:, ix["lpv"]],
            segment_mask[:, ix["rpv"]],
            segment_mask[:, ix["collateral"]] * is_tips,
        ], dim=-1)

        inflow_frac = self._masked_softmax(inflow_prior + delta[:, 0:2], inflow_mask)
        conf_frac = self._masked_softmax(conf_prior + delta[:, 2:4], conf_mask)
        bif_frac = self._masked_softmax(bif_prior + delta[:, 4:7], bif_mask)

        q_mpv = conf_frac[:, 0]
        q_collateral = conf_frac[:, 1] + q_mpv * bif_frac[:, 2]
        q6 = torch.stack([
            q_collateral,
            inflow_frac[:, 0],
            q_mpv,
            inflow_frac[:, 1],
            q_mpv * bif_frac[:, 0],
            q_mpv * bif_frac[:, 1],
        ], dim=-1) * segment_mask

        q8_cols = []
        for name in SEGMENTS:
            if name == "mpv":
                q8_cols.append(q6[:, ix["mpv"]])
            elif name == "sv":
                q8_cols.append(q6[:, ix["sv"]])
            elif name == "smv":
                q8_cols.append(q6[:, ix["smv"]])
            elif name == "lpv":
                q8_cols.append(q6[:, ix["lpv"]])
            elif name == "rpv":
                q8_cols.append(q6[:, ix["rpv"]])
            elif name == "lgv":
                q8_cols.append(q_collateral * collateral_source_weights[:, 0])
            elif name == "pgv":
                q8_cols.append(q_collateral * collateral_source_weights[:, 1])
            elif name == "tips":
                q8_cols.append(q_collateral * collateral_source_weights[:, 2])
        q8 = torch.stack(q8_cols, dim=-1)

        return {
            "Q": q8,
            "Q_model": q6,
            "inflow_frac": inflow_frac,
            "conf_outflow_frac": conf_frac,
            "bif_outflow_frac": bif_frac,
            "inflow_delta": delta[:, 0:2],
            "conf_outflow_delta": delta[:, 2:4],
            "bif_outflow_delta": delta[:, 4:7],
            "inflow_mask": inflow_mask,
            "conf_outflow_mask": conf_mask,
            "bif_outflow_mask": bif_mask,
            "tips_fraction": q_collateral * collateral_source_weights[:, 2],
            "collateral_fraction": q_collateral,
            "liver_fraction": q6[:, ix["lpv"]] + q6[:, ix["rpv"]],
            "branch_resistance": branch_resistance,
        }

    def _forward_three(
        self,
        segment_mask: torch.Tensor,
        junction_diameters: torch.Tensor,
        branch_resistance: torch.Tensor,
        delta: torch.Tensor,
        collateral_source_weights: torch.Tensor | None,
    ) -> Dict[str, torch.Tensor]:
        ix = self.branch_index
        base_idx = [ix["sv"], ix["mpv"]]
        mpv_split_idx = [ix["mpv"], ix["collateral"]]
        sv_split_idx = [ix["sv"], ix["collateral"]]

        base_prior = self._prior(junction_diameters[:, base_idx], branch_resistance[:, base_idx])
        mpv_prior = self._prior(junction_diameters[:, mpv_split_idx], branch_resistance[:, mpv_split_idx])
        sv_prior = self._prior(junction_diameters[:, sv_split_idx], branch_resistance[:, sv_split_idx])

        if collateral_source_weights is None:
            collateral_source_weights = torch.zeros(segment_mask.size(0), 3, device=segment_mask.device)
        is_lgv = collateral_source_weights[:, 0]
        is_pgv = collateral_source_weights[:, 1]
        is_tips = collateral_source_weights[:, 2]
        is_mpv_side = (is_lgv + is_tips).clamp(max=1.0)
        is_sv_side = is_pgv

        base_mask = segment_mask[:, base_idx]
        mpv_split_mask = torch.stack([
            segment_mask[:, ix["mpv"]],
            segment_mask[:, ix["collateral"]] * is_mpv_side,
        ], dim=-1)
        sv_split_mask = torch.stack([
            segment_mask[:, ix["sv"]],
            segment_mask[:, ix["collateral"]] * is_sv_side,
        ], dim=-1)

        base_frac = self._masked_softmax(base_prior + delta[:, 0:2], base_mask)
        mpv_split = self._masked_softmax(mpv_prior + delta[:, 2:4], mpv_split_mask)
        sv_split = self._masked_softmax(sv_prior + delta[:, 4:6], sv_split_mask)

        q_sv_base = base_frac[:, 0]
        q_mpv_base = base_frac[:, 1]
        q_mpv = q_mpv_base * mpv_split[:, 0]
        q_sv = q_sv_base * sv_split[:, 0]
        q_collateral = q_mpv_base * mpv_split[:, 1] + q_sv_base * sv_split[:, 1]
        q3 = torch.stack([q_collateral, q_mpv, q_sv], dim=-1) * segment_mask

        q8_cols = []
        zero = torch.zeros_like(q_mpv)
        for name in SEGMENTS:
            if name == "mpv":
                q8_cols.append(q_mpv)
            elif name == "sv":
                q8_cols.append(q_sv)
            elif name in ("smv", "lpv", "rpv"):
                q8_cols.append(zero)
            elif name == "lgv":
                q8_cols.append(q_collateral * collateral_source_weights[:, 0])
            elif name == "pgv":
                q8_cols.append(q_collateral * collateral_source_weights[:, 1])
            elif name == "tips":
                q8_cols.append(q_collateral * collateral_source_weights[:, 2])
        q8 = torch.stack(q8_cols, dim=-1)

        return {
            "Q": q8,
            "Q_model": q3,
            "inflow_frac": base_frac,
            "conf_outflow_frac": mpv_split,
            "bif_outflow_frac": sv_split,
            "inflow_delta": delta[:, 0:2],
            "conf_outflow_delta": delta[:, 2:4],
            "bif_outflow_delta": delta[:, 4:6],
            "inflow_mask": base_mask,
            "conf_outflow_mask": mpv_split_mask,
            "bif_outflow_mask": sv_split_mask,
            "tips_fraction": q_collateral * collateral_source_weights[:, 2],
            "collateral_fraction": q_collateral,
            "liver_fraction": q_mpv,
            "branch_resistance": branch_resistance,
            "collateral_mpv_side": is_mpv_side,
            "collateral_sv_side": is_sv_side,
        }


class FlowGraphRefiner(nn.Module):
    """Small graph network over branch flow states."""

    def __init__(
        self,
        d_in: int,
        d_hidden: int = 32,
        n_layers: int = 2,
        dropout: float = 0.1,
        branch_names: Sequence[str] = SEGMENTS,
    ):
        super().__init__()
        self.branch_names = tuple(branch_names)
        self.branch_index = {name: i for i, name in enumerate(self.branch_names)}
        n_branches = len(self.branch_names)
        self.in_proj = nn.Linear(d_in, d_hidden)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_hidden * 2, d_hidden),
                nn.LayerNorm(d_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])
        adj = torch.eye(n_branches)
        edges = self._anatomical_edges()
        for a, b in edges:
            ia, ib = self.branch_index[a], self.branch_index[b]
            adj[ia, ib] = 1.0
            adj[ib, ia] = 1.0
        adj = adj / adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        self.register_buffer("adj", adj, persistent=False)

    def _anatomical_edges(self):
        if self.branch_names == THREE_VESSEL_SEGMENTS:
            return [("sv", "mpv")]
        if self.branch_names == SIX_VESSEL_SEGMENTS:
            return [
                ("sv", "mpv"), ("smv", "mpv"), ("mpv", "lpv"),
                ("mpv", "rpv"), ("mpv", "collateral"),
            ]
        return [
            ("sv", "mpv"), ("smv", "mpv"), ("mpv", "lpv"), ("mpv", "rpv"),
            ("mpv", "tips"), ("mpv", "lgv"), ("sv", "pgv"),
        ]

    def _three_vessel_adj(self, segment_mask: torch.Tensor, collateral_source_weights: torch.Tensor | None):
        B = segment_mask.size(0)
        device = segment_mask.device
        adj = torch.zeros(B, 3, 3, device=device)
        ix = self.branch_index
        sv, mpv, collateral = ix["sv"], ix["mpv"], ix["collateral"]
        eye = torch.eye(3, device=device).unsqueeze(0)
        adj = adj + eye * segment_mask.unsqueeze(-1)
        adj[:, sv, mpv] = 1.0
        adj[:, mpv, sv] = 1.0
        if collateral_source_weights is None:
            collateral_source_weights = torch.zeros(B, 3, device=device)
        mpv_side = (collateral_source_weights[:, 0] + collateral_source_weights[:, 2]).clamp(max=1.0)
        sv_side = collateral_source_weights[:, 1]
        active_collateral = segment_mask[:, collateral]
        mpv_edge = mpv_side * active_collateral
        sv_edge = sv_side * active_collateral
        adj[:, collateral, mpv] = mpv_edge
        adj[:, mpv, collateral] = mpv_edge
        adj[:, collateral, sv] = sv_edge
        adj[:, sv, collateral] = sv_edge
        active = segment_mask.unsqueeze(1) * segment_mask.unsqueeze(2)
        adj = adj * active
        return adj / adj.sum(dim=-1, keepdim=True).clamp(min=1.0)

    @staticmethod
    def _endpoints_from_centerline(centerline_points: torch.Tensor, centerline_valid: torch.Tensor):
        valid = centerline_valid > 0.5
        first_idx = valid.float().argmax(dim=-1)
        rev_idx = valid.flip(-1).float().argmax(dim=-1)
        last_idx = centerline_valid.size(-1) - 1 - rev_idx
        gather_first = first_idx[..., None, None].expand(-1, -1, 1, 3)
        gather_last = last_idx[..., None, None].expand(-1, -1, 1, 3)
        first = centerline_points.gather(2, gather_first).squeeze(2)
        last = centerline_points.gather(2, gather_last).squeeze(2)
        branch_valid = valid.any(dim=-1).float()
        return torch.stack([first, last], dim=2), branch_valid

    def _position_weighted_adj(
        self,
        segment_mask: torch.Tensor,
        centerline_points: torch.Tensor | None,
        centerline_valid: torch.Tensor | None,
    ):
        if centerline_points is None or centerline_valid is None:
            return self.adj.to(segment_mask.device)
        B = segment_mask.size(0)
        device = segment_mask.device
        adj = torch.eye(len(self.branch_names), device=device).unsqueeze(0).repeat(B, 1, 1)
        endpoints, branch_valid = self._endpoints_from_centerline(centerline_points, centerline_valid)
        for a, b in self._anatomical_edges():
            ia, ib = self.branch_index[a], self.branch_index[b]
            pa = endpoints[:, ia]
            pb = endpoints[:, ib]
            dist = torch.cdist(pa, pb).amin(dim=(1, 2))
            has_pos = branch_valid[:, ia] * branch_valid[:, ib]
            weight = torch.where(has_pos > 0.5, torch.exp(-dist / 50.0).clamp(min=0.05), torch.ones_like(dist))
            adj[:, ia, ib] = weight
            adj[:, ib, ia] = weight
        active = segment_mask.unsqueeze(1) * segment_mask.unsqueeze(2)
        adj = adj * active
        return adj / adj.sum(dim=-1, keepdim=True).clamp(min=1.0)

    def forward(
        self,
        flow_features: torch.Tensor,
        segment_mask: torch.Tensor,
        enabled: bool = True,
        collateral_source_weights: torch.Tensor | None = None,
        centerline_points: torch.Tensor | None = None,
        centerline_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.in_proj(flow_features) * segment_mask.unsqueeze(-1)
        if not enabled:
            return h
        adj = (
            self._three_vessel_adj(segment_mask, collateral_source_weights)
            if self.branch_names == THREE_VESSEL_SEGMENTS
            else self._position_weighted_adj(segment_mask, centerline_points, centerline_valid)
        )
        for layer in self.layers:
            if adj.dim() == 3:
                neigh = torch.einsum("bij,bjh->bih", adj, h)
            else:
                neigh = torch.einsum("ij,bjh->bih", adj, h)
            h = layer(torch.cat([h, neigh], dim=-1)) * segment_mask.unsqueeze(-1)
        return h


class NewPortalPressureNet(nn.Module):
    def __init__(
        self,
        d_hidden: int = 32,
        dropout: float = 0.15,
        flow_gnn_layers: int = 2,
        use_organ_flow_scale: bool = False,
        use_global_flow_corrector: bool = True,
        use_flow_graph: bool = True,
        fixed_physics_params: bool = False,
        use_all_profile_channels: bool = False,
        use_unreliable_raw_lengths: bool = False,
        use_organ_global_features: bool = False,
        disable_organ_features: bool = False,
        use_six_vessel_layout: bool = False,
        use_three_vessel_layout: bool = False,
        use_organ_branch_scales: bool = True,
        label_mean: float = 25.0,
        label_std: float = 6.0,
    ):
        super().__init__()
        if use_six_vessel_layout and use_three_vessel_layout:
            raise ValueError("Choose at most one compact layout: six-vessel or three-vessel.")
        self.use_six_vessel_layout = use_six_vessel_layout
        self.use_three_vessel_layout = use_three_vessel_layout
        if use_three_vessel_layout:
            self.branch_names = THREE_VESSEL_SEGMENTS
        elif use_six_vessel_layout:
            self.branch_names = SIX_VESSEL_SEGMENTS
        else:
            self.branch_names = tuple(SEGMENTS)
        self.n_model_segments = len(self.branch_names)
        if use_three_vessel_layout:
            self.flow_logit_dim = 6
        elif use_six_vessel_layout:
            self.flow_logit_dim = 7
        else:
            self.flow_logit_dim = 9
        self.use_organ_flow_scale = use_organ_flow_scale
        self.use_organ_branch_scales = use_organ_flow_scale and use_organ_branch_scales
        self.use_global_flow_corrector = use_global_flow_corrector
        self.use_flow_graph = use_flow_graph
        self.disable_organ_features = bool(disable_organ_features)
        self.label_mean = float(label_mean)
        self.label_std = float(max(label_std, 1e-3))
        self.d_flow = 1

        self.selector = GeometryFeatureSelector(use_all_profile_channels=use_all_profile_channels)
        self.branch_encoder = BranchProfileEncoder(self.selector.n_features, d_hidden, dropout)
        self.organ_flow = OrganFlowScaleNet(d_hidden=max(8, d_hidden // 2))
        self.organ_branch_scale = OrganBranchScaleNet(
            self.branch_names, d_hidden=max(8, d_hidden // 2)
        )
        self.physics = LearnablePhysicsLayer(
            fixed_physics_params=fixed_physics_params,
            use_unreliable_raw_lengths=use_unreliable_raw_lengths,
            branch_names=self.branch_names,
        )
        self.flow_allocator = FlowAllocator(
            d_branch=d_hidden, d_hidden=d_hidden, branch_names=self.branch_names
        )
        self.global_corrector = GlobalFlowCorrector(
            d_hidden,
            d_flow=self.d_flow,
            d_hidden=d_hidden,
            dropout=dropout,
            use_organ_global_features=use_organ_global_features,
            n_branches=self.n_model_segments,
            flow_logit_dim=self.flow_logit_dim,
            d_extra_global=self._extra_global_dim(),
        )
        self.graph_refiner = FlowGraphRefiner(
            d_in=self.d_flow,
            d_hidden=d_hidden,
            n_layers=flow_gnn_layers,
            dropout=dropout,
            branch_names=self.branch_names,
        )
        self.baseline_gate_logit = nn.Parameter(torch.tensor(-0.2))

        d_fused = (
            self.n_model_segments * d_hidden
            + self.n_model_segments * self.d_flow
            + self.n_model_segments
            + (self.n_model_segments if self.use_organ_branch_scales else 1)
            + 1
            + self.n_model_segments
            + d_hidden
        )
        self.predictor = nn.Sequential(
            nn.Linear(d_fused, d_hidden * 2),
            nn.LayerNorm(d_hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden * 2, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
        )

    @property
    def selected_profile_names(self) -> List[str]:
        return list(self.selector.feature_names)

    @property
    def global_aux_names(self) -> List[str]:
        return list(self.global_corrector.aux_names)

    def _extra_global_dim(self) -> int:
        if self.use_three_vessel_layout:
            helper_dim = len(THREE_VESSEL_HELPER_SEGMENTS) * (self.selector.n_features + 1)
            return 4 + helper_dim + 5
        if self.use_six_vessel_layout:
            return 4
        return 0

    def _extra_global_features(
        self,
        profiles_norm: torch.Tensor,
        point_valid: torch.Tensor,
        segment_mask: torch.Tensor,
        organ_volumes: torch.Tensor,
        organ_valid: torch.Tensor,
        collateral_type_onehot: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.use_six_vessel_layout:
            return collateral_type_onehot
        if not self.use_three_vessel_layout:
            return None

        selected = self.selector(profiles_norm)
        helper_parts = []
        for name in THREE_VESSEL_HELPER_SEGMENTS:
            si = SEG_INDEX[name]
            mask = point_valid[:, si]
            branch_alive = segment_mask[:, si:si + 1]
            means = masked_mean(selected[:, si], mask, dim=1) * branch_alive
            valid_frac = mask.mean(dim=-1, keepdim=True) * branch_alive
            helper_parts.extend([means, valid_frac])
        helper = torch.cat(helper_parts, dim=-1)
        organ = organ_volume_features(organ_volumes, organ_valid)
        return torch.cat([collateral_type_onehot, helper, organ], dim=-1)

    def _layout_inputs(
        self,
        profiles: torch.Tensor,
        profiles_norm: torch.Tensor,
        arc_lengths: torch.Tensor,
        centerline_points: torch.Tensor,
        centerline_valid: torch.Tensor,
        point_valid: torch.Tensor,
        segment_mask: torch.Tensor,
    ):
        if not (self.use_six_vessel_layout or self.use_three_vessel_layout):
            zeros = torch.zeros(segment_mask.size(0), 3, device=segment_mask.device)
            source_id = torch.zeros(segment_mask.size(0), dtype=torch.long, device=segment_mask.device)
            return {
                "profiles": profiles,
                "profiles_norm": profiles_norm,
                "arc_lengths": arc_lengths,
                "centerline_points": centerline_points,
                "centerline_valid": centerline_valid,
                "point_valid": point_valid,
                "segment_mask": segment_mask,
                "source_weights": zeros,
                "source_id": source_id,
                "type_onehot": F.one_hot(source_id, num_classes=4).float(),
            }

        source_weights, source_id = _collateral_source_weights(segment_mask)
        collat_mask = source_weights.sum(dim=-1).clamp(max=1.0)
        order = (
            ["collateral", "mpv", "sv"]
            if self.use_three_vessel_layout
            else ["collateral", "sv", "mpv", "smv", "lpv", "rpv"]
        )

        def build(tensor: torch.Tensor) -> torch.Tensor:
            parts = []
            for name in order:
                if name == "collateral":
                    parts.append(_merge_collateral_tensor(tensor, source_weights))
                else:
                    parts.append(tensor[:, SEG_INDEX[name]])
            return torch.stack(parts, dim=1)

        segment_parts = []
        for name in order:
            if name == "collateral":
                segment_parts.append(collat_mask)
            else:
                segment_parts.append(segment_mask[:, SEG_INDEX[name]])

        return {
            "profiles": build(profiles),
            "profiles_norm": build(profiles_norm),
            "arc_lengths": build(arc_lengths),
            "centerline_points": build(centerline_points),
            "centerline_valid": build(centerline_valid),
            "point_valid": build(point_valid),
            "segment_mask": torch.stack(segment_parts, dim=-1),
            "source_weights": source_weights,
            "source_id": source_id,
            "type_onehot": F.one_hot(source_id, num_classes=4).float(),
        }

    def _expand_hemo_to_original(
        self,
        hemo_per_seg: Sequence[Mapping[str, torch.Tensor]],
        source_weights: torch.Tensor,
    ):
        if not (self.use_six_vessel_layout or self.use_three_vessel_layout):
            return hemo_per_seg
        if self.use_three_vessel_layout:
            three_ix = THREE_VESSEL_INDEX
            mapped = {
                "mpv": hemo_per_seg[three_ix["mpv"]],
                "sv": hemo_per_seg[three_ix["sv"]],
            }
            template = hemo_per_seg[three_ix["mpv"]]
            zeros = {k: torch.zeros_like(v) for k, v in template.items()}
            for name in THREE_VESSEL_HELPER_SEGMENTS:
                mapped[name] = zeros
            collat = hemo_per_seg[three_ix["collateral"]]
            for wi, name in enumerate(COLLATERAL_SOURCE_SEGMENTS):
                mapped[name] = {k: _apply_batch_weight(v, source_weights[:, wi]) for k, v in collat.items()}
            return [mapped[name] for name in SEGMENTS]

        six_ix = SIX_VESSEL_INDEX
        mapped = {}
        for name in ("sv", "mpv", "smv", "lpv", "rpv"):
            mapped[name] = hemo_per_seg[six_ix[name]]
        collat = hemo_per_seg[six_ix["collateral"]]
        for wi, name in enumerate(COLLATERAL_SOURCE_SEGMENTS):
            mapped[name] = {k: _apply_batch_weight(v, source_weights[:, wi]) for k, v in collat.items()}
        return [mapped[name] for name in SEGMENTS]

    def _expanded_length_mask(self, device: torch.device):
        raw = self.physics.used_raw_length_mask.to(device)
        if not (self.use_six_vessel_layout or self.use_three_vessel_layout):
            return raw
        if self.use_three_vessel_layout:
            out = torch.zeros(N_SEGMENTS, device=device)
            three_ix = THREE_VESSEL_INDEX
            for name in ("mpv", "sv"):
                out[SEG_INDEX[name]] = raw[three_ix[name]]
            for name in COLLATERAL_SOURCE_SEGMENTS:
                out[SEG_INDEX[name]] = raw[three_ix["collateral"]]
            return out

        out = torch.ones(N_SEGMENTS, device=device)
        six_ix = SIX_VESSEL_INDEX
        for name in ("sv", "mpv", "smv", "lpv", "rpv"):
            out[SEG_INDEX[name]] = raw[six_ix[name]]
        for name in COLLATERAL_SOURCE_SEGMENTS:
            out[SEG_INDEX[name]] = raw[six_ix["collateral"]]
        return out

    def _physics_loss_segment_mask(self, segment_mask: torch.Tensor, source_weights: torch.Tensor) -> torch.Tensor:
        if not self.use_three_vessel_layout:
            return segment_mask
        out = torch.zeros_like(segment_mask)
        out[:, SEG_INDEX["mpv"]] = segment_mask[:, SEG_INDEX["mpv"]]
        out[:, SEG_INDEX["sv"]] = segment_mask[:, SEG_INDEX["sv"]]
        for wi, name in enumerate(COLLATERAL_SOURCE_SEGMENTS):
            out[:, SEG_INDEX[name]] = segment_mask[:, SEG_INDEX[name]] * source_weights[:, wi]
        return out

    def _physics_baseline_norm(self, hemo_per_seg: Sequence[Mapping[str, torch.Tensor]], segment_mask: torch.Tensor):
        ix = SEG_INDEX
        dP_mpv = hemo_per_seg[ix["mpv"]]["pressure_drop_total"] * segment_mask[:, ix["mpv"]]
        liver_terms = []
        for name in ("lpv", "rpv"):
            si = ix[name]
            liver_terms.append(hemo_per_seg[si]["pressure_drop_total"] * segment_mask[:, si])
        liver_dP = torch.stack(liver_terms, dim=-1)
        denom = segment_mask[:, [ix["lpv"], ix["rpv"]]].sum(dim=-1).clamp(min=1.0)
        liver_mean = liver_dP.sum(dim=-1) / denom
        baseline_mmHg = (dP_mpv + liver_mean) / MMHG_TO_PA
        raw_norm = (baseline_mmHg - self.label_mean) / self.label_std
        return raw_norm.clamp(-5.0, 5.0)

    @staticmethod
    def _junction_features(flow_out: Mapping[str, torch.Tensor], hemo_per_seg: Sequence[Mapping[str, torch.Tensor]], segment_mask: torch.Tensor):
        ix = SEG_INDEX
        dP = torch.stack([h["pressure_drop_total"] for h in hemo_per_seg], dim=-1)
        lpv_rpv_gap = (dP[:, ix["lpv"]] - dP[:, ix["rpv"]]).abs()
        collateral = flow_out["collateral_fraction"]
        tips = flow_out["tips_fraction"]
        liver = flow_out["liver_fraction"]
        return {
            "features": torch.stack([
                collateral,
                tips,
                liver,
                lpv_rpv_gap / 1000.0,
                dP[:, ix["mpv"]] / 1000.0,
            ], dim=-1),
            "inflow_active": segment_mask[:, ix["sv"]] * segment_mask[:, ix["smv"]],
            "confluence_outflow_active": segment_mask[:, ix["mpv"]],
            "bifurcation_active": segment_mask[:, ix["mpv"]] * (
                segment_mask[:, ix["lpv"]] + segment_mask[:, ix["rpv"]]
            ).clamp(max=1.0),
            "press_resid_bifurc": (lpv_rpv_gap / 1000.0).pow(2),
        }

    def _three_vessel_auxiliary_loss(
        self,
        flow_out: Mapping[str, torch.Tensor],
        profiles: torch.Tensor,
        segment_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_three_vessel_layout:
            return torch.zeros(profiles.size(0), device=profiles.device)
        ix = SEG_INDEX
        q = flow_out["Q_model"]
        three_ix = THREE_VESSEL_INDEX
        q_mpv = q[:, three_ix["mpv"]]
        q_sv = q[:, three_ix["sv"]]
        implied_smv = F.relu(q_mpv - q_sv) / q_mpv.clamp(min=1e-3)

        sv_d = profiles[:, ix["sv"], 0, P_HDIAM].clamp(min=1e-3)
        smv_d = profiles[:, ix["smv"], 0, P_HDIAM].clamp(min=1e-3)
        expected_smv = smv_d.pow(3.0) / (sv_d.pow(3.0) + smv_d.pow(3.0)).clamp(min=1e-3)
        active = segment_mask[:, ix["sv"]] * segment_mask[:, ix["smv"]] * segment_mask[:, ix["mpv"]]
        return (implied_smv - expected_smv.detach()).pow(2) * active

    def forward(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, object]:
        profiles = batch["profiles"]
        profiles_norm = batch["profiles_norm"]
        arc_lengths = batch["arc_lengths"]
        point_valid = batch["point_valid"]
        centerline_points = batch.get(
            "centerline_points",
            torch.zeros(*profiles.shape[:3], 3, device=profiles.device),
        )
        centerline_valid = batch.get(
            "centerline_valid",
            torch.zeros_like(point_valid),
        )
        segment_mask = batch["segment_mask"]
        aux_norm = batch["aux_norm"]
        aux_mask = batch.get("aux_mask", torch.ones_like(aux_norm))
        organ_volumes = batch.get("organ_volumes", torch.zeros(profiles.size(0), 2, device=profiles.device))
        organ_valid = batch.get("organ_valid", torch.zeros(profiles.size(0), 2, device=profiles.device))
        if self.disable_organ_features:
            organ_volumes = torch.zeros_like(organ_volumes)
            organ_valid = torch.zeros_like(organ_valid)
            organ_idx = torch.tensor(ORGAN_AUX_INDICES, device=aux_norm.device, dtype=torch.long)
            aux_norm = aux_norm.clone()
            aux_mask = aux_mask.clone()
            aux_norm[:, organ_idx] = 0.0
            aux_mask[:, organ_idx] = 0.0

        layout = self._layout_inputs(
            profiles,
            profiles_norm,
            arc_lengths,
            centerline_points,
            centerline_valid,
            point_valid,
            segment_mask,
        )
        model_profiles = layout["profiles"]
        model_profiles_norm = layout["profiles_norm"]
        model_arc_lengths = layout["arc_lengths"]
        model_centerline_points = layout["centerline_points"]
        model_centerline_valid = layout["centerline_valid"]
        model_point_valid = layout["point_valid"]
        model_segment_mask = layout["segment_mask"]
        source_weights = layout["source_weights"]
        collateral_type_onehot = layout["type_onehot"]
        extra_global_features = self._extra_global_features(
            profiles_norm,
            point_valid,
            segment_mask,
            organ_volumes,
            organ_valid,
            collateral_type_onehot,
        )

        selected_norm = self.selector(model_profiles_norm)
        branch_embed, branch_stats = self.branch_encoder(selected_norm, model_point_valid, model_segment_mask)

        q_scale_global = torch.ones(profiles.size(0), device=profiles.device)
        branch_q_scale = self.organ_branch_scale(
            organ_volumes,
            organ_valid,
            model_segment_mask,
            collateral_type_onehot,
            enabled=self.use_organ_branch_scales,
        )
        q_scale = branch_q_scale
        branch_resistance = self.physics.resistance_proxy(
            model_profiles, model_arc_lengths, model_point_valid, model_segment_mask
        )
        junction_diam = model_profiles[:, :, 0, P_HDIAM].clamp(min=1e-3)

        zero_flow_features = torch.zeros(profiles.size(0), self.n_model_segments, self.d_flow, device=profiles.device)
        pre_corrected, global_delta, global_context = self.global_corrector(
            branch_embed,
            zero_flow_features,
            aux_norm,
            aux_mask,
            model_segment_mask,
            extra_global_features,
            enabled=self.use_global_flow_corrector,
        )
        flow_out = self.flow_allocator(
            branch_embed,
            model_segment_mask,
            junction_diam,
            branch_resistance,
            global_delta,
            collateral_source_weights=source_weights,
        )
        Q = flow_out["Q"]
        Q_model = flow_out.get("Q_model", Q)

        hemo_model = self.physics(
            model_profiles, model_arc_lengths, model_point_valid, model_segment_mask, Q_model, q_scale
        )
        hemo_per_seg = self._expand_hemo_to_original(hemo_model, source_weights)
        flow_features = Q_model.unsqueeze(-1) * model_segment_mask.unsqueeze(-1)
        corrected_flow, _, global_context = self.global_corrector(
            branch_embed,
            flow_features,
            aux_norm,
            aux_mask,
            model_segment_mask,
            extra_global_features,
            enabled=self.use_global_flow_corrector,
        )
        refined = self.graph_refiner(
            corrected_flow,
            model_segment_mask,
            enabled=self.use_flow_graph,
            collateral_source_weights=source_weights,
            centerline_points=model_centerline_points,
            centerline_valid=model_centerline_valid,
        )

        baseline_raw_norm = self._physics_baseline_norm(hemo_per_seg, segment_mask)
        gate = torch.sigmoid(self.baseline_gate_logit)
        baseline_norm = baseline_raw_norm * gate

        q_scale_fused = q_scale if self.use_organ_branch_scales else q_scale_global.unsqueeze(-1)
        fused = torch.cat([
            refined.reshape(profiles.size(0), -1),
            corrected_flow.reshape(profiles.size(0), -1),
            Q_model,
            q_scale_fused,
            baseline_norm.unsqueeze(-1),
            model_segment_mask,
            global_context,
        ], dim=-1)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1e3, neginf=-1e3)

        correction = self.predictor(fused).squeeze(-1)
        pvp_pred = (baseline_norm + correction).unsqueeze(-1)
        jp = self._junction_features(flow_out, hemo_per_seg, segment_mask)
        jp["helper_aux_loss"] = self._three_vessel_auxiliary_loss(flow_out, profiles, segment_mask)

        return {
            "pvp_pred": pvp_pred,
            "Q": Q,
            "flow_out": flow_out,
            "junction": jp,
            "hemo_per_seg": hemo_per_seg,
            "flow_features": corrected_flow,
            "raw_flow_features": flow_features,
            "branch_embed": branch_embed,
            "branch_stats": branch_stats,
            "pvp_baseline_norm": baseline_norm.unsqueeze(-1),
            "pvp_baseline_raw_norm": baseline_raw_norm.unsqueeze(-1),
            "pvp_physics_calibrated_norm": baseline_norm.unsqueeze(-1),
            "pvp_physics_delta_norm": (baseline_norm - baseline_raw_norm).unsqueeze(-1),
            "pvp_physics_gate": gate.expand_as(baseline_norm).unsqueeze(-1),
            "q_scale": q_scale,
            "segment_mask": segment_mask,
            "model_segment_mask": model_segment_mask,
            "collateral_type": layout["source_id"],
            "collateral_type_onehot": collateral_type_onehot,
            "used_raw_length_mask": self._expanded_length_mask(profiles.device),
            "physics_loss_segment_mask": self._physics_loss_segment_mask(segment_mask, source_weights),
            "selected_profile_names": self.selected_profile_names,
            "global_aux_names": self.global_aux_names,
        }


class NewPhysicsLoss(nn.Module):
    def __init__(
        self,
        lambda_shunt: float = 0.03,
        split_loss_mode: str = "core_confluence",
    ):
        super().__init__()
        if split_loss_mode not in {"full", "core_confluence"}:
            raise ValueError(f"Unknown split_loss_mode: {split_loss_mode}")
        self.split_loss_mode = split_loss_mode
        self.lambda_shunt = float(lambda_shunt)

    @staticmethod
    def _safe(x: torch.Tensor, cap: float = 1e3) -> torch.Tensor:
        return torch.nan_to_num(x, nan=0.0, posinf=cap, neginf=cap).clamp(max=cap)

    @staticmethod
    def _core_confluence_split_loss(model_out: Mapping[str, object], batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        q = model_out["Q"]
        segment_mask = model_out.get("segment_mask", batch["segment_mask"])
        ix = SEG_INDEX
        active = segment_mask[:, ix["mpv"]] * segment_mask[:, ix["smv"]] * segment_mask[:, ix["sv"]]
        if active.sum() == 0:
            pred = model_out["pvp_pred"].squeeze(-1)
            return torch.tensor(0.0, device=pred.device)
        resid = (q[:, ix["mpv"]] - q[:, ix["smv"]] - q[:, ix["sv"]]).pow(2)
        return (resid * active).sum() / active.sum().clamp(min=1.0)

    @staticmethod
    def _full_split_loss(model_out: Mapping[str, object], batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        flow = model_out["flow_out"]
        if "split_residual" in flow:
            return flow["split_residual"].mean()
        q = model_out["Q"]
        ix = SEG_INDEX
        resid = (
            (q[:, ix["mpv"]] + q[:, ix["lgv"]] + q[:, ix["pgv"]] - q[:, ix["sv"]] - q[:, ix["smv"]]).pow(2)
            + (q[:, ix["lpv"]] + q[:, ix["rpv"]] + q[:, ix["tips"]] - q[:, ix["mpv"]]).pow(2)
        )
        return resid.mean()

    def _split_loss(self, model_out: Mapping[str, object], batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.split_loss_mode == "core_confluence":
            return self._core_confluence_split_loss(model_out, batch)
        return self._full_split_loss(model_out, batch)

    def forward(self, model_out: Mapping[str, object], label_norm: torch.Tensor, batch: Mapping[str, torch.Tensor]):
        pred = model_out["pvp_pred"].squeeze(-1)
        err = pred - label_norm
        L_main = err.pow(2).mean()
        L_shunt = self._split_loss(model_out, batch)
        L_main = self._safe(L_main)
        L_shunt = self._safe(L_shunt)
        total = L_main + self.lambda_shunt * L_shunt
        return total, {
            "main": float(L_main.detach()),
            "shunt": float(L_shunt.detach()),
            "total": float(total.detach()),
        }


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
