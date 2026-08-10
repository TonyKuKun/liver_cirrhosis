"""Two-channel 3D nnVNet and segmentation losses for refinement2."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.projection = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
        )
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(self.layers(value) + self.projection(value))


class DownBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.block = ResidualBlock3D(out_channels, out_channels, dropout=dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(self.downsample(value))


class UpBlock3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ResidualBlock3D(out_channels + skip_channels, out_channels, dropout=dropout)

    @staticmethod
    def _match_shape(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if value.shape[2:] == reference.shape[2:]:
            return value
        return F.interpolate(value, size=reference.shape[2:], mode="trilinear", align_corners=False)

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        upsampled = self._match_shape(self.upsample(value), skip)
        return self.block(torch.cat([upsampled, skip], dim=1))


class CTPretrainNNVNet(nn.Module):
    """nnVNet-style network with an optional coarse-mask residual prior.

    A positive prior strength makes the initial prediction follow the coarse
    pretrain mask. The network then learns corrections instead of rebuilding
    the whole vessel mask from a small dataset.
    """

    input_channels = 2

    def __init__(self, base_channels: int = 24, prior_strength: float = 0.0) -> None:
        super().__init__()
        channels = int(base_channels)
        self.prior_strength = float(prior_strength)
        self.encoder1 = ResidualBlock3D(self.input_channels, channels)
        self.encoder2 = DownBlock3D(channels, channels * 2)
        self.encoder3 = DownBlock3D(channels * 2, channels * 4)
        self.encoder4 = DownBlock3D(channels * 4, channels * 8, dropout=0.15)
        self.bottleneck = ResidualBlock3D(channels * 8, channels * 8, dropout=0.25)
        self.decoder3 = UpBlock3D(channels * 8, channels * 4, channels * 4, dropout=0.1)
        self.decoder2 = UpBlock3D(channels * 4, channels * 2, channels * 2)
        self.decoder1 = UpBlock3D(channels * 2, channels, channels)
        self.output = nn.Conv3d(channels, 1, kernel_size=1)
        if self.prior_strength > 0.0:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[1] != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, got {value.shape[1]}")
        encoder1 = self.encoder1(value)
        encoder2 = self.encoder2(encoder1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)
        bottleneck = self.bottleneck(encoder4)
        decoder3 = self.decoder3(bottleneck, encoder3)
        decoder2 = self.decoder2(decoder3, encoder2)
        decoder1 = self.decoder1(decoder2, encoder1)
        logits = self.output(decoder1)
        if self.prior_strength > 0.0:
            prior = torch.where(value[:, 1:2] >= 0.5, self.prior_strength, -self.prior_strength)
            logits = logits + prior
        return logits


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.6, bce_weight: float = 0.4) -> None:
        super().__init__()
        if dice_weight < 0 or bce_weight < 0 or dice_weight + bce_weight == 0:
            raise ValueError("dice_weight and bce_weight must be non-negative with a positive sum")
        self.dice_weight = float(dice_weight)
        self.bce_weight = float(bce_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(logits)
        dimensions = tuple(range(1, probability.ndim))
        intersection = (probability * target).sum(dim=dimensions)
        denominator = probability.sum(dim=dimensions) + target.sum(dim=dimensions)
        dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        bce_loss = F.binary_cross_entropy_with_logits(logits, target)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


class FocalTverskyLoss(nn.Module):
    """Tversky loss with alpha weighting false positives and beta false negatives."""

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 1.0) -> None:
        super().__init__()
        if alpha < 0 or beta < 0 or alpha + beta == 0 or gamma <= 0:
            raise ValueError("alpha, beta, and gamma must be positive-compatible values")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(logits)
        dimensions = tuple(range(1, probability.ndim))
        true_positive = (probability * target).sum(dim=dimensions)
        false_positive = (probability * (1.0 - target)).sum(dim=dimensions)
        false_negative = ((1.0 - probability) * target).sum(dim=dimensions)
        score = (true_positive + 1.0) / (
            true_positive + self.alpha * false_positive + self.beta * false_negative + 1.0
        )
        return torch.pow(1.0 - score, self.gamma).mean()


class DiceFocalTverskyLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 0.5,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= dice_weight <= 1.0:
            raise ValueError("dice_weight must be within [0, 1]")
        self.dice_weight = float(dice_weight)
        self.dice = DiceBCELoss(dice_weight=1.0, bce_weight=0.0)
        self.tversky = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, target) + (1.0 - self.dice_weight) * self.tversky(logits, target)


def create_loss(
    name: str,
    dice_weight: float = 0.6,
    alpha: float = 0.7,
    beta: float = 0.3,
    gamma: float = 1.0,
) -> nn.Module:
    normalized = name.lower().replace("-", "_")
    if normalized == "dice_bce":
        return DiceBCELoss(dice_weight=dice_weight, bce_weight=1.0 - dice_weight)
    if normalized in {"tversky", "focal_tversky"}:
        return FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)
    if normalized == "dice_focal_tversky":
        return DiceFocalTverskyLoss(dice_weight=dice_weight, alpha=alpha, beta=beta, gamma=gamma)
    raise ValueError(f"Unknown loss '{name}'")


def dice_per_case(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    prediction = torch.sigmoid(logits) >= threshold
    dimensions = tuple(range(1, target.ndim))
    intersection = (prediction * target.bool()).sum(dim=dimensions)
    denominator = prediction.sum(dim=dimensions) + target.bool().sum(dim=dimensions)
    return (2.0 * intersection + 1.0) / (denominator + 1.0)
