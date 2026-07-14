"""Spatially gated decoder1 edema repair (frozen U-Net + tiny trainable module)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


CLS_BG = 0
CLS_NEC = 1
CLS_EDEMA = 2
CLS_ENH = 3


def _unit(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return vec / vec.norm().clamp_min(eps)


def build_d_initial(
    d_probe: torch.Tensor,
    seg_head_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Hybrid direction: probe + (W_edema - W_background) in decoder1 channel space.

    seg_head_weight: (num_classes, C, 1, 1, 1)
    """
    w = seg_head_weight.detach().float().reshape(seg_head_weight.shape[0], -1)
    d_output = _unit(w[CLS_EDEMA] - w[CLS_BG])
    d_probe_u = _unit(d_probe.detach().float().flatten())
    return _unit(d_probe_u + d_output)


def build_support_mask(
    baseline_logits: torch.Tensor,
    dilate_radius: int = 2,
    uncertainty_margin: float = 0.15,
) -> torch.Tensor:
    """
    Detached spatial support from frozen baseline logits.

    baseline_logits: (C, D, H, W) or (1, C, D, H, W)
    Returns float mask in {0,1} with same spatial shape, no grad.
    """
    if baseline_logits.dim() == 5:
        logits = baseline_logits[0]
    else:
        logits = baseline_logits
    with torch.no_grad():
        probs = torch.softmax(logits.float(), dim=0)
        p_wt = 1.0 - probs[CLS_BG]
        wt_mask = (torch.argmax(logits, dim=0) > 0).float()

        if dilate_radius > 0:
            k = 2 * dilate_radius + 1
            wt_mask = F.max_pool3d(
                wt_mask.view(1, 1, *wt_mask.shape),
                kernel_size=k,
                stride=1,
                padding=dilate_radius,
            ).view_as(wt_mask)

        top2 = torch.topk(probs, k=2, dim=0).values
        uncertain = (top2[0] - top2[1]) < uncertainty_margin
        # Uncertain voxels near tumor: intersect with dilated WT band (already dilated)
        # plus a slightly larger WT probability band.
        near = (p_wt > 0.05).float()
        support = torch.clamp(wt_mask + (uncertain.float() * near), max=1.0)
    return support.detach()


class SpatialEdemaRepair(nn.Module):
    """Trainable gate + scale + delta on top of a frozen decoder1 edema direction."""

    def __init__(
        self,
        d_initial: torch.Tensor,
        channels: int,
        init_raw_scale: float = 0.0,
    ) -> None:
        super().__init__()
        d0 = _unit(d_initial.float().flatten())
        if d0.numel() != channels:
            raise ValueError(f"d_initial has {d0.numel()} channels, expected {channels}")
        self.register_buffer("d_initial", d0)
        self.delta = nn.Parameter(torch.zeros(channels))
        self.raw_scale = nn.Parameter(torch.tensor(float(init_raw_scale)))
        self.gate = nn.Conv3d(channels, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    @property
    def scale(self) -> torch.Tensor:
        return F.softplus(self.raw_scale)

    def direction(self) -> torch.Tensor:
        return _unit(self.d_initial + self.delta)

    def cosine_to_initial(self) -> torch.Tensor:
        d = self.direction()
        return F.cosine_similarity(d.unsqueeze(0), self.d_initial.unsqueeze(0))

    def forward(
        self,
        h: torch.Tensor,
        support: torch.Tensor,
        enabled: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        h: (B, C, D, H, W)
        support: (B, 1, D, H, W) or (B, D, H, W) or (D, H, W)
        Returns edited h and gate map (B, 1, D, H, W).
        """
        if support.dim() == 3:
            support = support.unsqueeze(0).unsqueeze(0)
        elif support.dim() == 4:
            support = support.unsqueeze(1)
        support = support.to(device=h.device, dtype=h.dtype)

        gate = torch.tanh(self.gate(h))
        if not enabled:
            return h, gate * 0.0

        d = self.direction().view(1, -1, 1, 1, 1).to(dtype=h.dtype)
        edited = h + support * self.scale.to(dtype=h.dtype) * gate * d
        return edited, gate

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [self.delta, self.raw_scale, *self.gate.parameters()]
