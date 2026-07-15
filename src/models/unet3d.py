"""3D U-Net with bottleneck embedding and optional intermediate-layer hooks.

The nine named stages in ``LAYER_NAMES`` are shared by training, TTA, layer
embedding export, interventions, and consistency analysis. Forward returns
``(logits, embedding, bottleneck)``; hookable activations are exposed for
ablation / embedding export via helper methods further below.
"""

from __future__ import annotations

from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F

# Spatial stages in shallow→deep→shallow order (analysis configs use this list).
LAYER_NAMES: Final[tuple[str, ...]] = (
    "encoder1",
    "encoder2",
    "encoder3",
    "encoder4",
    "bottleneck",
    "decoder4",
    "decoder3",
    "decoder2",
    "decoder1",
)


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
        # Channel width doubles each encoder stage (f, 2f, 4f, 8f, 16f at bottleneck).
        width = base_features

        self.enc1 = ConvBlock3D(in_channels, width)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(width, width * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock3D(width * 2, width * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = ConvBlock3D(width * 4, width * 8)
        self.pool4 = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock3D(width * 8, width * 16)
        self.embedding_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(width * 16, embedding_dim),
        )

        self.up4 = nn.ConvTranspose3d(width * 16, width * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock3D(width * 16, width * 8)
        self.up3 = nn.ConvTranspose3d(width * 8, width * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(width * 8, width * 4)
        self.up2 = nn.ConvTranspose3d(width * 4, width * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(width * 4, width * 2)
        self.up1 = nn.ConvTranspose3d(width * 2, width, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(width * 2, width)

        self.seg_head = nn.Conv3d(width, num_classes, kernel_size=1)

    def encode(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Run the encoder and return skip features plus bottleneck activations."""
        skip1 = self.enc1(x)
        skip2 = self.enc2(self.pool1(skip1))
        skip3 = self.enc3(self.pool2(skip2))
        skip4 = self.enc4(self.pool3(skip3))
        bottleneck = self.bottleneck(self.pool4(skip4))
        return [skip1, skip2, skip3, skip4], bottleneck

    def decode(self, skips: list[torch.Tensor], bottleneck: torch.Tensor) -> torch.Tensor:
        """Decode from bottleneck using skip connections; return logits."""
        skip1, skip2, skip3, skip4 = skips
        x = self.up4(bottleneck)
        x = self.dec4(torch.cat([x, skip4], dim=1))
        x = self.up3(x)
        x = self.dec3(torch.cat([x, skip3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, skip2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, skip1], dim=1))
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

    @staticmethod
    def _global_avg_pool(features: torch.Tensor) -> torch.Tensor:
        """Global average pool a (B, C, D, H, W) map to (B, C)."""
        return F.adaptive_avg_pool3d(features, 1).flatten(1)

    @staticmethod
    def apply_activation_intervention(
        tensor: torch.Tensor,
        mode: str,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Replace a layer activation map for ablation studies.

        Modes:
          - zero: all zeros
          - mean: per-channel spatial mean broadcast (removes spatial detail)
          - noise: Gaussian noise with per-channel matched variance
        """
        if mode == "zero":
            return torch.zeros_like(tensor)
        if mode == "mean":
            channel_mean = tensor.mean(dim=(2, 3, 4), keepdim=True)
            return channel_mean.expand_as(tensor)
        if mode == "noise":
            std = tensor.std(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
            noise = torch.randn(
                tensor.shape,
                device=tensor.device,
                dtype=tensor.dtype,
                generator=generator,
            )
            return noise * std
        raise ValueError(f"Unknown intervention mode: {mode}")

    @staticmethod
    def apply_channel_perturbation(
        tensor: torch.Tensor,
        channel_delta: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """
        Add alpha * per-channel spatially constant perturbation to an activation map.

        ``channel_delta`` is a unit direction in channel space (C,) or (1,C,1,1,1).
        This implements A' = A + alpha * GAP*(delta_v) with spatially constant lift.
        """
        if channel_delta.dim() == 1:
            delta = channel_delta.view(1, -1, 1, 1, 1)
        else:
            delta = channel_delta
        if delta.shape[1] != tensor.shape[1]:
            raise ValueError(
                f"channel_delta has {delta.shape[1]} channels but tensor has {tensor.shape[1]}"
            )
        return tensor + float(alpha) * delta.to(device=tensor.device, dtype=tensor.dtype)

    @staticmethod
    def activation_rms(tensor: torch.Tensor) -> torch.Tensor:
        """Root-mean-square over all elements of an activation map."""
        return torch.sqrt(torch.mean(tensor.float().pow(2)))

    @staticmethod
    def rms_perturbation_ratio(
        original: torch.Tensor, perturbed: torch.Tensor
    ) -> torch.Tensor:
        """RMS(A' - A) / RMS(A)."""
        diff = perturbed.float() - original.float()
        num = torch.sqrt(torch.mean(diff.pow(2)))
        den = UNet3D.activation_rms(original).clamp_min(1e-8)
        return num / den

    def forward_to_decoder1(self, x: torch.Tensor) -> torch.Tensor:
        """Run encoder + decoder up to (and including) decoder1; return d1 activations."""
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool1(s1))
        s3 = self.enc3(self.pool2(s2))
        s4 = self.enc4(self.pool3(s3))
        bottleneck_map = self.bottleneck(self.pool4(s4))

        d = self.up4(bottleneck_map)
        d4 = self.dec4(torch.cat([d, s4], dim=1))
        d = self.up3(d4)
        d3 = self.dec3(torch.cat([d, s3], dim=1))
        d = self.up2(d3)
        d2 = self.dec2(torch.cat([d, s2], dim=1))
        d = self.up1(d2)
        return self.dec1(torch.cat([d, s1], dim=1))

    def forward_from_decoder1(self, d1: torch.Tensor) -> torch.Tensor:
        """Decode from decoder1 activations through the segmentation head only."""
        return self.seg_head(d1)

    def forward_to_decoder2(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run encoder + decoder through decoder2; return (d2, s1 skip for dec1)."""
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool1(s1))
        s3 = self.enc3(self.pool2(s2))
        s4 = self.enc4(self.pool3(s3))
        bottleneck_map = self.bottleneck(self.pool4(s4))

        d = self.up4(bottleneck_map)
        d4 = self.dec4(torch.cat([d, s4], dim=1))
        d = self.up3(d4)
        d3 = self.dec3(torch.cat([d, s3], dim=1))
        d = self.up2(d3)
        d2 = self.dec2(torch.cat([d, s2], dim=1))
        return d2, s1

    def forward_from_decoder2(self, d2: torch.Tensor, s1_skip: torch.Tensor) -> torch.Tensor:
        """Decode from decoder2 activations through decoder1 and seg_head."""
        d = self.up1(d2)
        d1 = self.dec1(torch.cat([d, s1_skip], dim=1))
        return self.seg_head(d1)

    def forward_with_representation_edit(
        self,
        x: torch.Tensor,
        edit_layer: str,
        channel_delta: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """
        Forward pass with additive semantic editing on one layer's activation tensor.

        Editing sites mirror causal ablation placement:
          - encoder1–4: **skip-path editing** — the encoder runs normally; only the
            skip tensor fused at decode is perturbed. This matches probe readouts
            (GAP of encoder block output) but does NOT alter deeper encoder states.
          - bottleneck: perturb bottleneck feature map, then decode.
          - decoder4–1: perturb decoder block output, then continue downstream.
        """
        if edit_layer not in LAYER_NAMES:
            raise ValueError(f"Unknown layer: {edit_layer}")

        edit = lambda t: self.apply_channel_perturbation(t, channel_delta, alpha)

        s1 = self.enc1(x)
        s2 = self.enc2(self.pool1(s1))
        s3 = self.enc3(self.pool2(s2))
        s4 = self.enc4(self.pool3(s3))
        bottleneck_map = self.bottleneck(self.pool4(s4))

        s1_skip = edit(s1) if edit_layer == "encoder1" else s1
        s2_skip = edit(s2) if edit_layer == "encoder2" else s2
        s3_skip = edit(s3) if edit_layer == "encoder3" else s3
        s4_skip = edit(s4) if edit_layer == "encoder4" else s4

        if edit_layer == "bottleneck":
            bottleneck_map = edit(bottleneck_map)

        d = self.up4(bottleneck_map)
        d4 = self.dec4(torch.cat([d, s4_skip], dim=1))
        if edit_layer == "decoder4":
            d4 = edit(d4)

        d = self.up3(d4)
        d3 = self.dec3(torch.cat([d, s3_skip], dim=1))
        if edit_layer == "decoder3":
            d3 = edit(d3)

        d = self.up2(d3)
        d2 = self.dec2(torch.cat([d, s2_skip], dim=1))
        if edit_layer == "decoder2":
            d2 = edit(d2)

        d = self.up1(d2)
        d1 = self.dec1(torch.cat([d, s1_skip], dim=1))
        if edit_layer == "decoder1":
            d1 = edit(d1)

        return self.seg_head(d1)

    @staticmethod
    def activation_rho(original: torch.Tensor, ablated: torch.Tensor) -> float:
        """Relative L2 perturbation: ||A - A'|| / ||A|| at the ablation site."""
        num = torch.norm(original.float() - ablated.float())
        den = torch.norm(original.float()).clamp_min(1e-8)
        return float((num / den).item())

    def forward_with_intervention(
        self,
        x: torch.Tensor,
        ablate_layer: str,
        intervention: str,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, float]:
        """
        Forward pass with a single-layer ablation.

        Ablation semantics (U-Net pathway):
          - encoder1–encoder4: **skip ablation** — full encoder runs normally;
            only the skip tensor fused during decoding is replaced. Tests skip
            contribution without corrupting deeper encoder states.
          - bottleneck: replace bottleneck map, then decode.
          - decoder4–decoder1: replace decoder block output, then continue
            downstream decoding. Tests causal dependence on that decoder stage.
        """
        if ablate_layer not in LAYER_NAMES:
            raise ValueError(f"Unknown layer: {ablate_layer}")

        intervene = lambda t: self.apply_activation_intervention(
            t, intervention, generator
        )
        rho = 0.0

        # Full encoder path (always computed from real activations).
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool1(s1))
        s3 = self.enc3(self.pool2(s2))
        s4 = self.enc4(self.pool3(s3))
        bottleneck_map = self.bottleneck(self.pool4(s4))

        # Skip tensors used at decode fusion (encoder ablation = skip-only).
        if ablate_layer == "encoder1":
            s1_skip = intervene(s1)
            rho = self.activation_rho(s1, s1_skip)
        else:
            s1_skip = s1
        if ablate_layer == "encoder2":
            s2_skip = intervene(s2)
            rho = self.activation_rho(s2, s2_skip)
        else:
            s2_skip = s2
        if ablate_layer == "encoder3":
            s3_skip = intervene(s3)
            rho = self.activation_rho(s3, s3_skip)
        else:
            s3_skip = s3
        if ablate_layer == "encoder4":
            s4_skip = intervene(s4)
            rho = self.activation_rho(s4, s4_skip)
        else:
            s4_skip = s4

        if ablate_layer == "bottleneck":
            bn_ablated = intervene(bottleneck_map)
            rho = self.activation_rho(bottleneck_map, bn_ablated)
            bottleneck_map = bn_ablated

        d = self.up4(bottleneck_map)
        d4 = self.dec4(torch.cat([d, s4_skip], dim=1))
        if ablate_layer == "decoder4":
            d4_ablated = intervene(d4)
            rho = self.activation_rho(d4, d4_ablated)
            d4 = d4_ablated

        d = self.up3(d4)
        d3 = self.dec3(torch.cat([d, s3_skip], dim=1))
        if ablate_layer == "decoder3":
            d3_ablated = intervene(d3)
            rho = self.activation_rho(d3, d3_ablated)
            d3 = d3_ablated

        d = self.up2(d3)
        d2 = self.dec2(torch.cat([d, s2_skip], dim=1))
        if ablate_layer == "decoder2":
            d2_ablated = intervene(d2)
            rho = self.activation_rho(d2, d2_ablated)
            d2 = d2_ablated

        d = self.up1(d2)
        d1 = self.dec1(torch.cat([d, s1_skip], dim=1))
        if ablate_layer == "decoder1":
            d1_ablated = intervene(d1)
            rho = self.activation_rho(d1, d1_ablated)
            d1 = d1_ablated

        return self.seg_head(d1), rho

    def forward_with_layer_embeddings(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward pass returning globally pooled vectors for each encoder/decoder block.

        The bottleneck vector uses the same embedding_head as the standard forward
        pass (GAP + linear), so it matches existing bottleneck embeddings.
        """
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool1(s1))
        s3 = self.enc3(self.pool2(s2))
        s4 = self.enc4(self.pool3(s3))
        bottleneck_map = self.bottleneck(self.pool4(s4))

        d = self.up4(bottleneck_map)
        d4 = self.dec4(torch.cat([d, s4], dim=1))
        d = self.up3(d4)
        d3 = self.dec3(torch.cat([d, s3], dim=1))
        d = self.up2(d3)
        d2 = self.dec2(torch.cat([d, s2], dim=1))
        d = self.up1(d2)
        d1 = self.dec1(torch.cat([d, s1], dim=1))
        logits = self.seg_head(d1)

        layer_embeddings = {
            "encoder1": self._global_avg_pool(s1),
            "encoder2": self._global_avg_pool(s2),
            "encoder3": self._global_avg_pool(s3),
            "encoder4": self._global_avg_pool(s4),
            "bottleneck": self.embedding_head(bottleneck_map),
            "decoder4": self._global_avg_pool(d4),
            "decoder3": self._global_avg_pool(d3),
            "decoder2": self._global_avg_pool(d2),
            "decoder1": self._global_avg_pool(d1),
        }
        return logits, layer_embeddings


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
