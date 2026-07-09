"""3D U-Net with bottleneck embedding extraction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """Two 3x3x3 convolutions with instance normalization and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3D(nn.Module):
    """
    Standard 3D U-Net for BraTS segmentation.

    The bottleneck tensor is globally average-pooled to produce a fixed-size
    embedding vector used in later Failure Cartography experiments.
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base_features: int = 32,
        embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        f = base_features

        self.enc1 = ConvBlock3D(in_channels, f)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(f, f * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock3D(f * 2, f * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = ConvBlock3D(f * 4, f * 8)
        self.pool4 = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock3D(f * 8, f * 16)
        self.embedding_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(f * 16, embedding_dim),
        )

        self.up4 = nn.ConvTranspose3d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock3D(f * 16, f * 8)
        self.up3 = nn.ConvTranspose3d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(f * 8, f * 4)
        self.up2 = nn.ConvTranspose3d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(f * 4, f * 2)
        self.up1 = nn.ConvTranspose3d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(f * 2, f)

        self.seg_head = nn.Conv3d(f, num_classes, kernel_size=1)

    def encode(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Run the encoder and return skip features plus bottleneck activations."""
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool1(s1))
        s3 = self.enc3(self.pool2(s2))
        s4 = self.enc4(self.pool3(s3))
        bottleneck = self.bottleneck(self.pool4(s4))
        return [s1, s2, s3, s4], bottleneck

    def decode(self, skips: list[torch.Tensor], bottleneck: torch.Tensor) -> torch.Tensor:
        """Decode from bottleneck using skip connections."""
        s1, s2, s3, s4 = skips
        x = self.up4(bottleneck)
        x = self.dec4(torch.cat([x, s4], dim=1))
        x = self.up3(x)
        x = self.dec3(torch.cat([x, s3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, s2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, s1], dim=1))
        return self.seg_head(x)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            logits: (B, num_classes, D, H, W)
            embedding: (B, embedding_dim) bottleneck feature after GAP + linear
            bottleneck: (B, C, d, h, w) raw bottleneck feature map
        """
        skips, bottleneck = self.encode(x)
        logits = self.decode(skips, bottleneck)
        embedding = self.embedding_head(bottleneck)
        return logits, embedding, bottleneck


class DiceCrossEntropyLoss(nn.Module):
    """Combined multi-class Dice and cross-entropy loss."""

    def __init__(self, dice_weight: float = 0.5, num_classes: int = 4) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets)
        dice_loss = self._multiclass_dice_loss(logits, targets)
        return self.dice_weight * dice_loss + (1.0 - self.dice_weight) * ce_loss

    def _multiclass_dice_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()

        dims = (0, 2, 3, 4)
        intersection = (probs * targets_one_hot).sum(dims)
        cardinality = probs.sum(dims) + targets_one_hot.sum(dims)
        dice = (2.0 * intersection + 1e-5) / (cardinality + 1e-5)
        # Exclude background (class 0) from the mean, as is common in BraTS.
        return 1.0 - dice[1:].mean()


def validate_patch_size(patch_size: list[int] | tuple[int, ...], downsample_factor: int = 16) -> None:
    """
    Ensure patch spatial dims are divisible by the U-Net downsampling factor.

    This U-Net has 4× MaxPool3d(2), so each spatial axis must be divisible by 16.
    """
    for axis, size in enumerate(patch_size):
        if int(size) % downsample_factor != 0:
            raise ValueError(
                f"patch_size[{axis}]={size} is not divisible by {downsample_factor}. "
                "U-Net skip connections will break for this config."
            )


def build_model(config: dict) -> UNet3D:
    """Instantiate the 3D U-Net from configuration."""
    validate_patch_size(config["data"]["patch_size"])
    model_cfg = config["model"]
    return UNet3D(
        in_channels=model_cfg["in_channels"],
        num_classes=model_cfg["num_classes"],
        base_features=model_cfg["base_features"],
        embedding_dim=model_cfg["embedding_dim"],
    )
