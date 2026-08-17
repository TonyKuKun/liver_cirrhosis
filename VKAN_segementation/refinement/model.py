from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi

MODEL_NAMES = ("vkan", "nnvnet")


class KANGate3D(nn.Module):
    """Lightweight spline-like channel gate used in refinement blocks."""

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
        d3 = self.dec3(torch.cat([self._match(self.up3(m), e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._match(self.up2(d3), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._match(self.up1(d2), e1), e1], dim=1))
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


def _soft_erode_3d(volume: torch.Tensor) -> torch.Tensor:
    depth = -F.max_pool3d(-volume, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
    height = -F.max_pool3d(-volume, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0))
    width = -F.max_pool3d(-volume, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))
    return torch.minimum(torch.minimum(depth, height), width)


def _soft_open_3d(volume: torch.Tensor) -> torch.Tensor:
    eroded = _soft_erode_3d(volume)
    return F.max_pool3d(eroded, kernel_size=3, stride=1, padding=1)


def _soft_skeletonize_3d(volume: torch.Tensor, iterations: int) -> torch.Tensor:
    opened = _soft_open_3d(volume)
    skeleton = F.relu(volume - opened)
    for _ in range(iterations):
        volume = _soft_erode_3d(volume)
        opened = _soft_open_3d(volume)
        delta = F.relu(volume - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


class SoftCLDiceLoss(nn.Module):
    """Differentiable 3D centerline Dice loss for tubular structures."""

    def __init__(self, iterations: int = 5, smooth: float = 1e-6) -> None:
        super().__init__()
        if iterations < 0:
            raise ValueError("iterations must be greater than or equal to 0")
        self.iterations = int(iterations)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(logits)
        predicted_skeleton = _soft_skeletonize_3d(probability, self.iterations)
        target_skeleton = _soft_skeletonize_3d(target, self.iterations)
        dims = tuple(range(1, probability.ndim))

        topology_precision = (
            (predicted_skeleton * target).sum(dim=dims) + self.smooth
        ) / (predicted_skeleton.sum(dim=dims) + self.smooth)
        topology_sensitivity = (
            (target_skeleton * probability).sum(dim=dims) + self.smooth
        ) / (target_skeleton.sum(dim=dims) + self.smooth)
        cldice = (2.0 * topology_precision * topology_sensitivity + self.smooth) / (
            topology_precision + topology_sensitivity + self.smooth
        )
        return 1.0 - cldice.mean()


class DiceBCECLDiceLoss(nn.Module):
    """Blend the existing region loss with a topology-aware clDice term."""

    def __init__(
        self,
        cldice_weight: float = 0.2,
        skeleton_iterations: int = 5,
        dice_weight: float = 0.6,
        bce_weight: float = 0.4,
    ) -> None:
        super().__init__()
        self.base_loss = DiceBCELoss(dice_weight=dice_weight, bce_weight=bce_weight)
        self.cldice_loss = SoftCLDiceLoss(iterations=skeleton_iterations)
        self.set_cldice_weight(cldice_weight)

    def set_cldice_weight(self, weight: float) -> None:
        if not 0.0 <= weight <= 1.0:
            raise ValueError("cldice_weight must be between 0 and 1")
        self.cldice_weight = float(weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base = self.base_loss(logits, target)
        if self.cldice_weight == 0.0:
            return base
        topology = self.cldice_loss(logits, target)
        return (1.0 - self.cldice_weight) * base + self.cldice_weight * topology


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Return the largest 26-connected component of a binary 3D mask."""
    if not mask.any():
        return mask

    labels, component_count = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    counts = np.bincount(labels.ravel(), minlength=component_count + 1)
    counts[0] = 0
    return labels == int(np.argmax(counts))


def dice_score(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    largest_component: bool = False,
) -> float:
    """Dice for a prediction batch, optionally filtering each case to one component."""
    pred = torch.sigmoid(logits) >= threshold
    if largest_component:
        pred_array = pred.detach().cpu().numpy()
        filtered = np.zeros_like(pred_array, dtype=bool)
        for batch_index in range(pred_array.shape[0]):
            for channel_index in range(pred_array.shape[1]):
                filtered[batch_index, channel_index] = _largest_connected_component(pred_array[batch_index, channel_index])
        filtered_pred = torch.from_numpy(filtered).to(device=target.device, dtype=target.dtype)
    else:
        filtered_pred = pred.to(dtype=target.dtype)
    inter = (filtered_pred * target).sum().item()
    denom = filtered_pred.sum().item() + target.sum().item()
    return float((2.0 * inter + 1.0) / (denom + 1.0))


def create_refinement_model(model_name: str = "vkan", base_channels: int = 16) -> nn.Module:
    name = model_name.lower().replace("-", "").replace("_", "")
    if name in {"vkan", "vesselvkan"}:
        return VesselVKAN(base_channels=base_channels)
    if name in {"nnvnet", "nnvnet3d"}:
        try:
            from .nnvnet import NNVNet
        except ImportError:
            try:
                from VKAN_segementation.refinement.nnvnet import NNVNet
            except ImportError:
                from refinement.nnvnet import NNVNet
        return NNVNet(base_channels=base_channels)
    raise ValueError(f"Unknown refinement model '{model_name}'. Available: {', '.join(MODEL_NAMES)}")

