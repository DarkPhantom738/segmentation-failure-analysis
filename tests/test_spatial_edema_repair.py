"""Unit tests for spatial edema repair module."""

from __future__ import annotations

import torch

from src.models.spatial_edema_repair import (
    SpatialEdemaRepair,
    build_d_initial,
    build_support_mask,
)
from src.models.unet3d import UNet3D


def test_build_d_initial_unit_norm():
    d_probe = torch.randn(32)
    w = torch.randn(4, 32, 1, 1, 1)
    d0 = build_d_initial(d_probe, w)
    assert d0.shape == (32,)
    assert torch.isclose(d0.norm(), torch.tensor(1.0), atol=1e-5)


def test_support_mask_excludes_distant_background():
    logits = torch.zeros(4, 16, 16, 16)
    logits[0] = 5.0  # background everywhere
    logits[2, 8, 8, 8] = 10.0  # one edema voxel center
    support = build_support_mask(logits, dilate_radius=2)
    assert support[8, 8, 8] > 0.5
    assert support[0, 0, 0] < 0.5


def test_repair_forward_and_grads():
    d0 = torch.randn(32)
    d0 = d0 / d0.norm()
    repair = SpatialEdemaRepair(d0, channels=32)
    h = torch.randn(1, 32, 8, 8, 8, requires_grad=False)
    support = torch.ones(1, 1, 8, 8, 8)
    edited, gate = repair(h, support, enabled=True)
    assert edited.shape == h.shape
    loss = edited.mean() + gate.mean()
    loss.backward()
    assert repair.delta.grad is not None
    assert repair.raw_scale.grad is not None
    assert repair.gate.weight.grad is not None


def test_disabled_repair_is_identity():
    d0 = torch.randn(32)
    d0 = d0 / d0.norm()
    repair = SpatialEdemaRepair(d0, channels=32)
    # non-zero params
    with torch.no_grad():
        repair.raw_scale.fill_(1.0)
        repair.gate.bias.fill_(0.5)
    h = torch.randn(1, 32, 8, 8, 8)
    support = torch.ones(1, 1, 8, 8, 8)
    edited, _ = repair(h, support, enabled=False)
    assert torch.allclose(edited, h)


def test_unet_decoder1_to_head_shapes():
    model = UNet3D(in_channels=4, num_classes=4, base_features=32, embedding_dim=128)
    x = torch.randn(1, 4, 64, 64, 64)
    d1 = model.forward_to_decoder1(x)
    assert d1.shape[1] == 32
    logits = model.forward_from_decoder1(d1)
    assert logits.shape == (1, 4, 64, 64, 64)
