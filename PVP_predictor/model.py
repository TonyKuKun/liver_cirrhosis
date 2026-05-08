"""
Physics-Informed Geometric Deep Learning for Portal Vein Pressure Prediction
=============================================================================

Architecture overview:
    ┌──────────────────────────────────────────────────────────────────┐
    │  For each branch (MPV / SV / SMV):                              │
    │                                                                  │
    │  Raw Geometric Profile  ──→  PhysicsPriorLayer (no params)       │
    │       (N, D_geo)              → 6 physics-derived features       │
    │                                                                  │
    │  [Geo_norm ∥ Physics]   ──→  PhysicsResidualModule (learnable)   │
    │       (N, D_geo+D_phy)        → corrected hemodynamic features   │
    │                                                                  │
    │  Corrected features     ──→  AttentionPooling                    │
    │       (N, D_phy)              → branch embedding (D_branch)      │
    │                                                                  │
    ├──────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │  [Branch embeddings]    ──→  Cross-Branch Attention              │
    │  + Statistical features ──→  Fusion MLP                          │
    │                         ──→  PVP Prediction                      │
    └──────────────────────────────────────────────────────────────────┘

Key design principles:
    1. Physics Prior Layer has ZERO learnable parameters — pure physics
    2. Residual module learns CORRECTIONS on top of physics, not from scratch
    3. alpha parameter controls physics-vs-learned balance (starts physics-heavy)
    4. Continuity loss regularizes hemodynamic consistency
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import N_PROFILE_FEAT, N_STAT_FEAT, N_BRANCHES, BRANCHES

# ── Number of physics-derived features ───────────────────────────────
N_PHYSICS_FEAT = 7


# =====================================================================
# Module 1: Deterministic Physics Prior Layer (无可学习参数)
# =====================================================================
class PhysicsPriorLayer(nn.Module):
    """
    Computes hemodynamic proxy features from raw geometry using
    established fluid mechanics relationships.

    All outputs are RELATIVE values (no absolute flow rate needed).

    Input per branch:
        profiles_raw : (B, N, 6)  [area, perimeter, eq_diameter, circularity, curvature, inscribed_radius]
        arc_lengths  : (B, N)

    Output:
        physics_features : (B, N, 7)
            [0] segment_resistance   — Poiseuille: R ∝ ΔL / r⁴
            [1] relative_velocity    — Continuity: v_rel = A_ref / A
            [2] relative_wss         — WSS ∝ 1 / r³
            [3] dean_number_proxy    — Secondary flow: De ∝ v·r·√(r·κ)
            [4] cumulative_resistance — ΣR from inlet
            [5] normalized_pressure_drop — ΣR / R_total
            [6] area_gradient        — dA/ds (cross-section expansion/contraction rate)
    """

    def __init__(self):
        super().__init__()
        # No learnable parameters

    def forward(self, profiles_raw, arc_lengths):
        """
        profiles_raw : (B, N, 6)  — raw (unnormalized) geometric features
        arc_lengths  : (B, N)     — cumulative arc length in mm
        """
        eps = 1e-8
        B, N, _ = profiles_raw.shape

        area        = profiles_raw[..., 0]  # (B, N) mm²
        perimeter   = profiles_raw[..., 1]  # (B, N) mm
        eq_diameter = profiles_raw[..., 2]  # (B, N) mm
        circularity = profiles_raw[..., 3]  # (B, N)
        curvature   = profiles_raw[..., 4]  # (B, N) 1/mm
        radius      = eq_diameter / 2.0 + eps  # (B, N) mm

        # ── Segment length (difference of arc_lengths) ───────────────
        # Pad first segment with same value as second
        seg_length = torch.zeros_like(arc_lengths)
        seg_length[:, 1:] = arc_lengths[:, 1:] - arc_lengths[:, :-1]
        seg_length[:, 0] = seg_length[:, 1]
        seg_length = seg_length.clamp(min=eps)

        # ── 1. Segment resistance (Poiseuille law) ───────────────────
        #    R_seg = 8μL / (πr⁴)  →  proportional to L / r⁴
        r4 = radius.pow(4) + eps
        R_seg = seg_length / r4  # (B, N)

        # ── 2. Relative velocity (continuity equation) ───────────────
        #    A₁v₁ = A₂v₂  →  v_i / v_ref = A_ref / A_i
        A_ref = area[:, 0:1] + eps  # reference: inlet (confluence end)
        v_relative = A_ref / (area + eps)  # (B, N)

        # ── 3. Relative wall shear stress ────────────────────────────
        #    WSS = 4μQ / (πr³)  →  at constant Q:  WSS ∝ 1/r³
        WSS_relative = 1.0 / (radius.pow(3) + eps)  # (B, N)

        # ── 4. Dean number proxy (secondary flow intensity) ──────────
        #    De = Re · √(d / 2R_curv) ∝ v · r · √(r · κ)
        kappa_abs = curvature.abs() + eps
        Dean_proxy = v_relative * radius * torch.sqrt(radius * kappa_abs)

        # ── 5. Cumulative resistance ─────────────────────────────────
        R_cumulative = torch.cumsum(R_seg, dim=1)  # (B, N)

        # ── 6. Normalized pressure drop ──────────────────────────────
        R_total = R_cumulative[:, -1:] + eps
        P_drop_norm = R_cumulative / R_total  # (B, N), range [0, 1]

        # ── 7. Area gradient (expansion / contraction rate) ──────────
        #    dA/ds: positive = expanding, negative = contracting
        area_grad = torch.zeros_like(area)
        area_grad[:, 1:] = (area[:, 1:] - area[:, :-1]) / seg_length[:, 1:]
        area_grad[:, 0] = area_grad[:, 1]

        # ── Stack all physics features ───────────────────────────────
        physics = torch.stack([
            R_seg,              # [0] segment resistance
            v_relative,         # [1] relative velocity
            WSS_relative,       # [2] relative WSS
            Dean_proxy,         # [3] Dean number proxy
            R_cumulative,       # [4] cumulative resistance
            P_drop_norm,        # [5] normalized pressure drop
            area_grad,          # [6] area gradient
        ], dim=-1)  # (B, N, 7)

        # ── Guard against NaN/Inf from divisions ────────────────────
        physics = torch.nan_to_num(physics, nan=0.0, posinf=0.0, neginf=0.0)

        return physics


# =====================================================================
# Module 2: Physics Residual Learning Module (可学习参数)
# =====================================================================
class PhysicsResidualModule(nn.Module):
    """
    Learns corrections on physics-derived features using the raw
    geometric context. Key design:
        corrected = physics_normalized + alpha * residual

    Uses 1D-CNN for spatial modeling along centerline (lighter than
    Transformer, better for small datasets).

    The 1D-CNN captures spatial dependencies: upstream geometry
    affects downstream hemodynamics.
    """

    def __init__(self, d_geo: int = N_PROFILE_FEAT, d_physics: int = N_PHYSICS_FEAT,
                 d_hidden: int = 32):
        super().__init__()
        d_in = d_geo + d_physics

        # ── Spatial encoder (1D-CNN along centerline) ────────────────
        self.encoder = nn.Sequential(
            # Conv1d expects (B, C, L)
            nn.Conv1d(d_in, d_hidden, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_hidden),
            nn.GELU(),
            nn.Conv1d(d_hidden, d_hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_hidden),
            nn.GELU(),
            nn.Conv1d(d_hidden, d_hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_hidden),
            nn.GELU(),
        )

        # ── Residual head: maps back to physics feature space ────────
        self.residual_head = nn.Sequential(
            nn.Conv1d(d_hidden, d_physics, kernel_size=1),
            nn.Tanh(),  # bound residual to [-1, 1]
        )

        # ── Learnable residual scaling (starts small → physics-dominated) ──
        self.alpha = nn.Parameter(torch.tensor(0.1))

        # ── Hidden projection for downstream use ─────────────────────
        self.hidden_proj = nn.Conv1d(d_hidden, d_hidden, kernel_size=1)

        self.d_hidden = d_hidden

    def forward(self, geo_norm, physics_raw):
        """
        geo_norm    : (B, N, D_geo)     — normalized geometric features
        physics_raw : (B, N, D_physics) — raw physics features from PhysicsPriorLayer

        Returns:
            corrected : (B, N, D_physics) — corrected hemodynamic features
            hidden    : (B, N, D_hidden)  — hidden representations
        """
        # ── Normalize physics features internally ────────────────────
        # Use running stats (per feature) to normalize physics features
        p_mean = physics_raw.mean(dim=1, keepdim=True)
        p_std = physics_raw.std(dim=1, keepdim=True) + 1e-8
        physics_norm = (physics_raw - p_mean) / p_std
        physics_norm = torch.nan_to_num(physics_norm, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Concatenate geo + physics ────────────────────────────────
        x = torch.cat([geo_norm, physics_norm], dim=-1)  # (B, N, D_geo+D_phy)

        # ── 1D-CNN encoding (transpose for Conv1d) ───────────────────
        x = x.transpose(1, 2)  # (B, C, N)
        h = self.encoder(x)    # (B, d_hidden, N)

        # ── Compute residual correction ──────────────────────────────
        residual = self.residual_head(h)  # (B, D_physics, N)
        residual = residual.transpose(1, 2)  # (B, N, D_physics)

        # ── Correct physics features ─────────────────────────────────
        corrected = physics_norm + self.alpha * residual  # (B, N, D_physics)

        # ── Hidden features ──────────────────────────────────────────
        hidden = self.hidden_proj(h)       # (B, d_hidden, N)
        hidden = hidden.transpose(1, 2)    # (B, N, d_hidden)

        return corrected, hidden, physics_norm


# =====================================================================
# Attention Pooling
# =====================================================================
class AttentionPooling(nn.Module):
    """
    Learns which centerline positions matter most for prediction.
    Outputs attention weights for visualization.
    """

    def __init__(self, d_in: int, d_attn: int = 16):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, 1),
        )

    def forward(self, x, mask=None):
        """
        x    : (B, N, D)
        mask : (B, N) optional, 1 = valid, 0 = pad

        Returns:
            pooled : (B, D)
            weights: (B, N) — attention weights for visualization
        """
        scores = self.attn_net(x).squeeze(-1)  # (B, N)

        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float('-inf'))

        weights = F.softmax(scores, dim=1)  # (B, N)
        pooled = torch.einsum('bn,bnd->bd', weights, x)  # (B, D)

        return pooled, weights


# =====================================================================
# Branch Encoder (processes one branch)
# =====================================================================
class BranchEncoder(nn.Module):
    """Encodes a single branch: physics prior → residual → attention pool."""

    def __init__(self, d_hidden: int = 32):
        super().__init__()
        self.physics_prior = PhysicsPriorLayer()
        self.residual_module = PhysicsResidualModule(
            d_geo=N_PROFILE_FEAT, d_physics=N_PHYSICS_FEAT, d_hidden=d_hidden,
        )
        # Pool from both corrected physics and hidden features
        self.attn_pool_physics = AttentionPooling(N_PHYSICS_FEAT, d_attn=8)
        self.attn_pool_hidden = AttentionPooling(d_hidden, d_attn=8)

        self.d_out = N_PHYSICS_FEAT + d_hidden

    def forward(self, profiles_raw, profiles_norm, arc_lengths):
        """
        profiles_raw  : (B, N, 6)
        profiles_norm : (B, N, 6)
        arc_lengths   : (B, N)

        Returns:
            branch_embed    : (B, D_out)
            attn_weights    : (B, N) — physics attention weights
            corrected       : (B, N, D_physics) — for continuity loss
            physics_normed  : (B, N, D_physics) — for analysis
        """
        # Step 1: Deterministic physics computation
        physics_raw = self.physics_prior(profiles_raw, arc_lengths)  # (B, N, 7)

        # Step 2: Learnable residual correction
        corrected, hidden, physics_normed = self.residual_module(
            profiles_norm, physics_raw
        )

        # Step 3: Attention pooling
        pooled_phy, attn_w = self.attn_pool_physics(corrected)  # (B, 7), (B, N)
        pooled_hid, _      = self.attn_pool_hidden(hidden)      # (B, d_hidden)

        branch_embed = torch.cat([pooled_phy, pooled_hid], dim=-1)  # (B, D_out)

        return branch_embed, attn_w, corrected, physics_normed


# =====================================================================
# Full Model: Portal Vein Pressure Predictor
# =====================================================================
class PortalPressureNet(nn.Module):
    """
    Physics-Informed Geometric Deep Learning for PVP Prediction.

    Inputs:
        - profiles_raw  : (B, 3, N, 6)  — raw geometric profiles per branch
        - profiles_norm : (B, 3, N, 6)  — normalized profiles
        - arc_lengths   : (B, 3, N)     — arc lengths per branch
        - branch_mask   : (B, 3)        — 1 if branch exists
        - stat_features : (B, 28)       — normalized statistical features

    Outputs:
        - pvp_pred          : (B, 1)         — predicted PVP
        - attn_weights      : (B, 3, N)      — per-branch attention maps
        - branch_corrected  : list of (B, N, 7) — for continuity loss
    """

    def __init__(self, d_hidden: int = 32, dropout: float = 0.3):
        super().__init__()

        # ── Per-branch encoders (shared architecture, separate weights) ──
        self.branch_encoders = nn.ModuleDict({
            name: BranchEncoder(d_hidden=d_hidden) for name in BRANCHES
        })

        d_branch = self.branch_encoders['mpv'].d_out  # same for all

        # ── Cross-branch attention ───────────────────────────────────
        self.cross_branch_attn = nn.MultiheadAttention(
            embed_dim=d_branch, num_heads=1, batch_first=True, dropout=0.1,
        )
        self.branch_norm = nn.LayerNorm(d_branch)

        # ── Statistical feature encoder ──────────────────────────────
        self.stat_encoder = nn.Sequential(
            nn.Linear(N_STAT_FEAT, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
        )

        # ── Fusion and prediction head ───────────────────────────────
        d_fused = d_branch * N_BRANCHES + d_hidden
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

    def forward(self, profiles_raw, profiles_norm, arc_lengths,
                branch_mask, stat_features):
        B = profiles_raw.shape[0]
        N = profiles_raw.shape[2]

        # ── Encode each branch ───────────────────────────────────────
        branch_embeds = []
        all_attn_weights = torch.zeros(B, N_BRANCHES, N, device=profiles_raw.device)
        all_corrected = []

        for bi, bname in enumerate(BRANCHES):
            # Extract this branch's data
            p_raw  = profiles_raw[:, bi]   # (B, N, 6)
            p_norm = profiles_norm[:, bi]  # (B, N, 6)
            arcs   = arc_lengths[:, bi]    # (B, N)
            mask_b = branch_mask[:, bi]    # (B,)

            embed, attn_w, corrected, _ = self.branch_encoders[bname](
                p_raw, p_norm, arcs
            )

            # Zero out missing branches
            embed = embed * mask_b.unsqueeze(-1)
            branch_embeds.append(embed)
            all_attn_weights[:, bi] = attn_w
            all_corrected.append(corrected)

        # ── Cross-branch attention ───────────────────────────────────
        # Stack branch embeddings: (B, 3, D_branch)
        branch_stack = torch.stack(branch_embeds, dim=1)

        # Create key_padding_mask for missing branches
        key_padding_mask = (branch_mask == 0)  # True = ignore

        # Self-attention across branches
        branch_attn_out, _ = self.cross_branch_attn(
            branch_stack, branch_stack, branch_stack,
            key_padding_mask=key_padding_mask,
        )
        branch_attn_out = self.branch_norm(branch_stack + branch_attn_out)

        # Flatten branch embeddings
        branch_flat = branch_attn_out.reshape(B, -1)  # (B, 3*D_branch)

        # ── Encode statistical features ──────────────────────────────
        stat_embed = self.stat_encoder(stat_features)  # (B, d_hidden)

        # ── Fuse and predict ─────────────────────────────────────────
        fused = torch.cat([branch_flat, stat_embed], dim=-1)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)
        pvp_pred = self.predictor(fused)  # (B, 1)

        return pvp_pred, all_attn_weights, all_corrected


# =====================================================================
# Physics-Informed Loss Function
# =====================================================================
class PhysicsInformedLoss(nn.Module):
    """
    Combined loss:
        L = L_main + λ_cont * L_continuity + λ_mono * L_monotonicity

    L_main       : MSE / Huber on PVP prediction
    L_continuity : Flow conservation — A_i * v_i ≈ const along non-branching segments
    L_monotonicity: Cumulative resistance should be non-decreasing
    """

    def __init__(self, lambda_cont: float = 0.1, lambda_mono: float = 0.05,
                 use_huber: bool = True):
        super().__init__()
        self.lambda_cont = lambda_cont
        self.lambda_mono = lambda_mono

        if use_huber:
            self.main_loss = nn.HuberLoss(delta=3.0)
        else:
            self.main_loss = nn.MSELoss()

    def forward(self, pvp_pred, pvp_true, corrected_list, branch_mask,
                profiles_raw):
        """
        pvp_pred       : (B, 1)
        pvp_true       : (B,)
        corrected_list : list of (B, N, 7) per branch
        branch_mask    : (B, 3)
        profiles_raw   : (B, 3, N, 6)
        """
        # ── Main prediction loss ─────────────────────────────────────
        L_main = self.main_loss(pvp_pred.squeeze(-1), pvp_true)

        # ── Physics consistency losses ───────────────────────────────
        L_cont = torch.tensor(0.0, device=pvp_pred.device)
        L_mono = torch.tensor(0.0, device=pvp_pred.device)
        n_valid = 0

        for bi in range(N_BRANCHES):
            mask_b = branch_mask[:, bi]  # (B,)
            valid = mask_b > 0
            if valid.sum() == 0:
                continue

            corrected = corrected_list[bi][valid]      # (B', N, 7)
            area = profiles_raw[valid, bi, :, 0]       # (B', N)

            # Feature indices from PhysicsPriorLayer:
            # [0]=R_seg, [1]=v_rel, [2]=WSS, [3]=Dean, [4]=R_cum, [5]=P_drop, [6]=area_grad
            v_rel = corrected[..., 1]    # (B', N)
            R_cum = corrected[..., 4]    # (B', N)

            # Continuity: A * v_relative should be approximately constant
            # (since Q = A * v = const along unbranched segment)
            flow_rate = area * v_rel           # proxy for Q
            flow_rate = torch.nan_to_num(flow_rate, nan=0.0, posinf=0.0, neginf=0.0)
            flow_diff = flow_rate[:, 1:] - flow_rate[:, :-1]
            L_cont_branch = (flow_diff ** 2).mean()

            # Monotonicity: cumulative resistance should increase
            R_diff = R_cum[:, 1:] - R_cum[:, :-1]
            L_mono_branch = F.relu(-R_diff).mean()

            L_cont = L_cont + L_cont_branch
            L_mono = L_mono + L_mono_branch
            n_valid += 1

        if n_valid > 0:
            L_cont = L_cont / n_valid
            L_mono = L_mono / n_valid

        # ── Total loss ───────────────────────────────────────────────
        L_total = L_main + self.lambda_cont * L_cont + self.lambda_mono * L_mono

        return L_total, {
            'main': L_main.item(),
            'continuity': L_cont.item(),
            'monotonicity': L_mono.item(),
            'total': L_total.item(),
            'alpha_mpv': self.get_alpha(0),
            'alpha_sv': self.get_alpha(1),
            'alpha_smv': self.get_alpha(2),
        }

    def get_alpha(self, branch_idx):
        """Helper placeholder — alpha is in the model, not loss."""
        return 0.0  # will be filled by trainer


# =====================================================================
# Utility: count parameters
# =====================================================================
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == '__main__':
    # ── Quick test ───────────────────────────────────────────────────
    B, N = 4, 100
    model = PortalPressureNet(d_hidden=32, dropout=0.3)

    profiles_raw  = torch.randn(B, 3, N, 6).abs() + 1.0  # positive values
    profiles_norm = torch.randn(B, 3, N, 6)
    arc_lengths   = torch.linspace(0, 50, N).unsqueeze(0).unsqueeze(0).expand(B, 3, N)
    branch_mask   = torch.ones(B, 3)
    stat_features = torch.randn(B, 28)

    pvp_pred, attn_w, corrected = model(
        profiles_raw, profiles_norm, arc_lengths, branch_mask, stat_features
    )

    print(f"PVP prediction shape: {pvp_pred.shape}")
    print(f"Attention weights shape: {attn_w.shape}")
    print(f"Corrected features: {len(corrected)} branches, each {corrected[0].shape}")

    total, trainable = count_params(model)
    print(f"Total params: {total:,} | Trainable: {trainable:,}")

    # Print alpha values
    for name in BRANCHES:
        alpha = model.branch_encoders[name].residual_module.alpha
        print(f"  {name} alpha: {alpha.item():.4f}")