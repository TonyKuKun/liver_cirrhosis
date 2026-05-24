"""Pure architecture baselines for PVP prediction."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import N_AUX, N_PROFILE_FEAT, N_SEGMENTS, SEG_INDEX, SEGMENTS
from architecture_benchmark.datasets import N_STL_GLOBAL


MODEL_NAMES = (
    "numeric_mlp",
    "numeric_cnn",
    "numeric_transformer",
    "numeric_gnn",
    "numeric_cnn_gnn",
    "stl_pointnet",
    "stl_centerline_gnn",
    "stl_pointnet_centerline_gnn",
    "fusion_numeric_stl",
)


def masked_mean(x, mask, dim, keepdim=False):
    mask = mask.float()
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    return (x * mask).sum(dim=dim, keepdim=keepdim) / mask.sum(dim=dim, keepdim=keepdim).clamp(min=1.0)


def masked_max(x, mask, dim):
    mask = mask.float()
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    return x.masked_fill(mask < 0.5, -1e6).max(dim=dim).values


def branch_adjacency(device):
    edges = [
        ("sv", "mpv"), ("smv", "mpv"), ("mpv", "lpv"), ("mpv", "rpv"), ("mpv", "tips"),
        ("lgv", "mpv"), ("pgv", "mpv"),
    ]
    adj = torch.eye(N_SEGMENTS, device=device)
    for a, b in edges:
        ai, bi = SEG_INDEX[a], SEG_INDEX[b]
        adj[ai, bi] = 1.0
        adj[bi, ai] = 1.0
    return adj


class MLPHead(nn.Module):
    def __init__(self, d_in, d_hidden=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x):
        return self.net(torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3))


class DenseGraphBlock(nn.Module):
    def __init__(self, d_hidden, dropout=0.1):
        super().__init__()
        self.self_proj = nn.Linear(d_hidden, d_hidden)
        self.nei_proj = nn.Linear(d_hidden, d_hidden)
        self.ffn = nn.Sequential(nn.Linear(d_hidden, d_hidden * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_hidden * 2, d_hidden))
        self.norm1 = nn.LayerNorm(d_hidden)
        self.norm2 = nn.LayerNorm(d_hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, adj, mask):
        adj = adj * mask.unsqueeze(1) * mask.unsqueeze(2)
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        msg = torch.bmm(adj / deg, h)
        h = self.norm1(h + self.drop(self.self_proj(h) + self.nei_proj(msg)))
        h = self.norm2(h + self.drop(self.ffn(h)))
        return h * mask.unsqueeze(-1)


class BranchGNN(nn.Module):
    def __init__(self, d_hidden, n_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DenseGraphBlock(d_hidden, dropout) for _ in range(n_layers)])

    def forward(self, h, segment_mask):
        adj = branch_adjacency(h.device).unsqueeze(0).repeat(h.size(0), 1, 1)
        for layer in self.layers:
            h = layer(h, adj, segment_mask)
        return h


class BranchCNNEncoder(nn.Module):
    def __init__(self, d_hidden=64, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(N_PROFILE_FEAT, d_hidden, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_hidden, d_hidden, 5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, profiles_norm, point_valid):
        b, s, n, f = profiles_norm.shape
        x = profiles_norm.reshape(b * s, n, f).transpose(1, 2)
        h = self.conv(x).transpose(1, 2).reshape(b, s, n, -1)
        mean = masked_mean(h, point_valid, dim=2)
        mx = masked_max(h, point_valid, dim=2)
        return 0.5 * (mean + mx)


class NumericStatsEncoder(nn.Module):
    def __init__(self, d_hidden=64):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(N_PROFILE_FEAT * 4 + 1, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_hidden))

    def forward(self, batch):
        x = batch["profiles_norm"]
        mask = batch["point_valid"]
        mean = masked_mean(x, mask, dim=2)
        centered = (x - mean.unsqueeze(2)) * mask.unsqueeze(-1)
        std = torch.sqrt(masked_mean(centered.pow(2), mask, dim=2).clamp(min=0.0))
        mn = x.masked_fill(mask.unsqueeze(-1) < 0.5, 1e6).min(dim=2).values
        mx = x.masked_fill(mask.unsqueeze(-1) < 0.5, -1e6).max(dim=2).values
        stats = torch.cat([mean, std, mn, mx, batch["segment_mask"].unsqueeze(-1)], dim=-1)
        return self.proj(torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)) * batch["segment_mask"].unsqueeze(-1)


class NumericMLP(nn.Module):
    def __init__(self, d_hidden=64, dropout=0.2):
        super().__init__()
        self.stats = NumericStatsEncoder(d_hidden=d_hidden)
        self.head = MLPHead(N_SEGMENTS * d_hidden + N_AUX, d_hidden=d_hidden, dropout=dropout)

    def forward(self, batch):
        h = self.stats(batch).reshape(batch["profiles"].size(0), -1)
        return self.head(torch.cat([h, batch["aux_norm"]], dim=-1))


class NumericCNN(nn.Module):
    def __init__(self, d_hidden=64, dropout=0.2):
        super().__init__()
        self.encoder = BranchCNNEncoder(d_hidden, dropout)
        self.head = MLPHead(N_SEGMENTS * d_hidden + N_AUX, d_hidden, dropout)

    def forward(self, batch):
        h = self.encoder(batch["profiles_norm"], batch["point_valid"]) * batch["segment_mask"].unsqueeze(-1)
        return self.head(torch.cat([h.reshape(h.size(0), -1), batch["aux_norm"]], dim=-1))


class NumericGNN(nn.Module):
    def __init__(self, d_hidden=64, dropout=0.2):
        super().__init__()
        self.stats = NumericStatsEncoder(d_hidden)
        self.gnn = BranchGNN(d_hidden, n_layers=2, dropout=dropout * 0.5)
        self.head = MLPHead(d_hidden + N_AUX, d_hidden, dropout)

    def forward(self, batch):
        h = self.gnn(self.stats(batch), batch["segment_mask"])
        pooled = masked_mean(h, batch["segment_mask"], dim=1)
        return self.head(torch.cat([pooled, batch["aux_norm"]], dim=-1))


class NumericCNNGNN(nn.Module):
    def __init__(self, d_hidden=64, dropout=0.2):
        super().__init__()
        self.encoder = BranchCNNEncoder(d_hidden, dropout)
        self.gnn = BranchGNN(d_hidden, n_layers=2, dropout=dropout * 0.5)
        self.head = MLPHead(d_hidden + N_AUX, d_hidden, dropout)

    def forward(self, batch):
        h = self.encoder(batch["profiles_norm"], batch["point_valid"]) * batch["segment_mask"].unsqueeze(-1)
        h = self.gnn(h, batch["segment_mask"])
        pooled = masked_mean(h, batch["segment_mask"], dim=1)
        return self.head(torch.cat([pooled, batch["aux_norm"]], dim=-1))


class NumericTransformer(nn.Module):
    def __init__(self, d_hidden=64, dropout=0.2, n_heads=4, n_layers=2):
        super().__init__()
        self.proj = nn.Linear(N_PROFILE_FEAT, d_hidden)
        self.seg_emb = nn.Embedding(N_SEGMENTS, d_hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_hidden, nhead=n_heads, dim_feedforward=d_hidden * 2,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = MLPHead(d_hidden + N_AUX, d_hidden, dropout)

    def forward(self, batch):
        x = batch["profiles_norm"]
        b, s, n, _ = x.shape
        h = self.proj(x)
        seg_ids = torch.arange(s, device=x.device).view(1, s, 1).expand(b, s, n)
        h = h + self.seg_emb(seg_ids)
        h = h.reshape(b, s * n, -1)
        token_mask = (batch["point_valid"] * batch["segment_mask"].unsqueeze(-1)).reshape(b, s * n)
        h = self.encoder(h, src_key_padding_mask=token_mask < 0.5)
        pooled = masked_mean(h, token_mask, dim=1)
        return self.head(torch.cat([pooled, batch["aux_norm"]], dim=-1))


class PointNetEncoder(nn.Module):
    def __init__(self, d_hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
        )

    def forward(self, points, valid):
        h = self.net(points)
        pooled = h.max(dim=1).values
        return pooled * valid.view(-1, 1)


class CenterlineGNNEncoder(nn.Module):
    def __init__(self, d_hidden=64, n_layers=3, dropout=0.1):
        super().__init__()
        self.input = nn.Linear(3 + N_SEGMENTS + 1, d_hidden)
        self.layers = nn.ModuleList([DenseGraphBlock(d_hidden, dropout) for _ in range(n_layers)])

    def _adjacency(self, batch):
        valid = batch["centerline_valid"]
        b, s, m = valid.shape
        t = s * m
        adj = torch.eye(t, device=valid.device).unsqueeze(0).repeat(b, 1, 1)
        for si in range(s):
            base = si * m
            idx = torch.arange(base, base + m - 1, device=valid.device)
            adj[:, idx, idx + 1] = 1.0
            adj[:, idx + 1, idx] = 1.0
        pairs = [
            ("sv", "mpv", "start"), ("smv", "mpv", "start"),
            ("lgv", "mpv", "start"), ("pgv", "mpv", "start"),
            ("mpv", "lpv", "mpv_end"), ("mpv", "rpv", "mpv_end"), ("mpv", "tips", "mpv_end"),
        ]
        for a, c, mode in pairs:
            ai, ci = SEG_INDEX[a], SEG_INDEX[c]
            a_node = ai * m
            c_node = ci * m
            if mode == "mpv_end":
                a_node = ai * m + (m - 1)
            adj[:, a_node, c_node] = 1.0
            adj[:, c_node, a_node] = 1.0
        mask = valid.reshape(b, t)
        return adj, mask

    def forward(self, batch):
        pos = batch["centerline_pos"]
        b, s, m, _ = pos.shape
        seg_onehot = F.one_hot(torch.arange(s, device=pos.device), num_classes=s).float().view(1, s, 1, s).expand(b, s, m, s)
        u = torch.linspace(0, 1, m, device=pos.device).view(1, 1, m, 1).expand(b, s, m, 1)
        h = self.input(torch.cat([pos, seg_onehot, u], dim=-1)).reshape(b, s * m, -1)
        adj, mask = self._adjacency(batch)
        for layer in self.layers:
            h = layer(h, adj, mask)
        return masked_mean(h, mask, dim=1)


class STLPointNet(nn.Module):
    def __init__(self, d_hidden=64, dropout=0.2):
        super().__init__()
        self.vessel = PointNetEncoder(d_hidden)
        self.spleen = PointNetEncoder(d_hidden)
        self.liver = PointNetEncoder(d_hidden)
        self.head = MLPHead(d_hidden * 3 + N_STL_GLOBAL, d_hidden, dropout)

    def encode(self, batch):
        return torch.cat([
            self.vessel(batch["vessel_points"], batch["vessel_valid"]),
            self.spleen(batch["spleen_points"], batch["spleen_valid"]),
            self.liver(batch["liver_points"], batch["liver_valid"]),
            batch["stl_global_norm"],
        ], dim=-1)

    def forward(self, batch):
        return self.head(self.encode(batch))


class STLCenterlineGNN(nn.Module):
    def __init__(self, d_hidden=64, dropout=0.2):
        super().__init__()
        self.centerline = CenterlineGNNEncoder(d_hidden, dropout=dropout * 0.5)
        self.head = MLPHead(d_hidden, d_hidden, dropout)

    def encode(self, batch):
        return self.centerline(batch)

    def forward(self, batch):
        return self.head(self.encode(batch))


class STLPointNetCenterlineGNN(nn.Module):
    def __init__(self, d_hidden=64, dropout=0.2):
        super().__init__()
        self.pointnet = STLPointNet(d_hidden, dropout)
        self.centerline = CenterlineGNNEncoder(d_hidden, dropout=dropout * 0.5)
        self.head = MLPHead(d_hidden * 4 + N_STL_GLOBAL, d_hidden, dropout)

    def encode(self, batch):
        return torch.cat([self.pointnet.encode(batch), self.centerline(batch)], dim=-1)

    def forward(self, batch):
        return self.head(self.encode(batch))


class FusionNumericSTL(nn.Module):
    def __init__(self, d_hidden=64, dropout=0.2):
        super().__init__()
        self.numeric = NumericCNNGNN(d_hidden, dropout)
        self.numeric_headless = self.numeric
        self.stl = STLPointNetCenterlineGNN(d_hidden, dropout)
        self.head = MLPHead(d_hidden + N_AUX + d_hidden * 4 + N_STL_GLOBAL, d_hidden, dropout)

    def forward(self, batch):
        h_num = self.numeric.encoder(batch["profiles_norm"], batch["point_valid"]) * batch["segment_mask"].unsqueeze(-1)
        h_num = self.numeric.gnn(h_num, batch["segment_mask"])
        num = torch.cat([masked_mean(h_num, batch["segment_mask"], dim=1), batch["aux_norm"]], dim=-1)
        stl = self.stl.encode(batch)
        return self.head(torch.cat([num, stl], dim=-1))


def build_model(model_name: str, d_hidden=64, dropout=0.2) -> nn.Module:
    if model_name == "numeric_mlp":
        return NumericMLP(d_hidden, dropout)
    if model_name == "numeric_cnn":
        return NumericCNN(d_hidden, dropout)
    if model_name == "numeric_transformer":
        return NumericTransformer(d_hidden, dropout)
    if model_name == "numeric_gnn":
        return NumericGNN(d_hidden, dropout)
    if model_name == "numeric_cnn_gnn":
        return NumericCNNGNN(d_hidden, dropout)
    if model_name == "stl_pointnet":
        return STLPointNet(d_hidden, dropout)
    if model_name == "stl_centerline_gnn":
        return STLCenterlineGNN(d_hidden, dropout)
    if model_name == "stl_pointnet_centerline_gnn":
        return STLPointNetCenterlineGNN(d_hidden, dropout)
    if model_name == "fusion_numeric_stl":
        return FusionNumericSTL(d_hidden, dropout)
    raise ValueError(f"Unknown model_name: {model_name}")
