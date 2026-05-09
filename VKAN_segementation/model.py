from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANGate3D(nn.Module):
    """Lightweight spline-like channel gate used in the VKAN refinement blocks."""

    def __init__(self, channels: int, knots: int = 8) -> None:
        super().__init__()
        self.knots = nn.Parameter(torch.linspace(-1.0, 1.0, knots), requires_grad=False)
        self.weight = nn.Parameter(torch.zeros(channels, knots))
        self.bias = nn.Parameter(torch.ones(channels))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=(2, 3, 4))
        basis = torch.relu(1.0 - torch.abs(pooled.unsqueeze(-1) - self.knots.view(1, 1, -1)))
        gate = (basis * self.weight.unsqueeze(0)).sum(dim=-1) + self.bias.unsqueeze(0)
        return x * torch.sigmoid(gate).view(x.shape[0], x.shape[1], 1, 1, 1)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.GELU(),
            KANGate3D(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VesselVKAN(nn.Module):
    """Small 3D U-Net with KAN gates for STL occupancy refinement."""

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(1, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.pool = nn.MaxPool3d(2)
        self.mid = ConvBlock(c * 4, c * 8)
        self.up3 = nn.ConvTranspose3d(c * 8, c * 4, 2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4)
        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, 2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose3d(c * 2, c, 2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)
        self.out = nn.Conv3d(c, 1, 1)

    @staticmethod
    def _match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[2:] == ref.shape[2:]:
            return x
        return F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        m = self.mid(self.pool(e3))
        d3 = self._match(self.up3(m), e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self._match(self.up2(d3), e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self._match(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.6, bce_weight: float = 0.4) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target)
        prob = torch.sigmoid(logits)
        dims = tuple(range(1, prob.ndim))
        inter = (prob * target).sum(dim=dims)
        denom = prob.sum(dim=dims) + target.sum(dim=dims)
        dice = 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean()
        return self.bce_weight * bce + self.dice_weight * dice


def dice_score(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred = (torch.sigmoid(logits) >= threshold).float()
    inter = (pred * target).sum().item()
    denom = pred.sum().item() + target.sum().item()
    return float((2.0 * inter + 1.0) / (denom + 1.0))

