"""Classifier-inverted decoder1 repair via predicted class-transition gates."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

CLS_BG = 0
CLS_NEC = 1
CLS_EDEMA = 2
CLS_ENH = 3

# Transition label IDs
T_NONE = 0
T_BG_TO_ED = 1
T_NEC_TO_ED = 2
T_ENH_TO_ED = 3
T_ED_TO_BG = 4
T_ED_TO_NEC = 5
T_ED_TO_ENH = 6
N_TRANSITIONS = 7

TRANSITION_SPECS: list[tuple[int, int | None, int | None, str]] = [
    (T_NONE, None, None, "none"),
    (T_BG_TO_ED, CLS_BG, CLS_EDEMA, "bg_to_edema"),
    (T_NEC_TO_ED, CLS_NEC, CLS_EDEMA, "nec_to_edema"),
    (T_ENH_TO_ED, CLS_ENH, CLS_EDEMA, "enh_to_edema"),
    (T_ED_TO_BG, CLS_EDEMA, CLS_BG, "edema_to_bg"),
    (T_ED_TO_NEC, CLS_EDEMA, CLS_NEC, "edema_to_nec"),
    (T_ED_TO_ENH, CLS_EDEMA, CLS_ENH, "edema_to_enh"),
]


def build_transition_directions(
    seg_head_weight: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, list[dict]]:
    """
    Derive six min-norm decoder1 directions d_(a->b) = pinv(W) @ (e_b - e_a).

    seg_head_weight: (num_classes, C, 1, 1, 1)
    Returns:
      dirs: (6, C) unit directions ordered as transitions 1..6
      meta: list of verification dicts
    """
    w = seg_head_weight.detach().float().reshape(seg_head_weight.shape[0], -1).cpu()
    return _build_transition_directions_from_w(w, eps)


def _build_transition_directions_from_w(
    w: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, list[dict]]:
    # W: (4, C)
    pinv = torch.linalg.pinv(w)  # (C, 4)
    dirs: list[torch.Tensor] = []
    meta: list[dict] = []
    for tid, src, tgt, name in TRANSITION_SPECS:
        if src is None or tgt is None:
            continue
        q = torch.zeros(w.shape[0], dtype=torch.float32, device=w.device)
        q[tgt] = 1.0
        q[src] = -1.0
        d = pinv @ q
        d = d / d.norm().clamp_min(eps)
        projected = w @ d
        dirs.append(d.cpu())
        meta.append(
            {
                "transition_id": tid,
                "name": name,
                "source": int(src),
                "target": int(tgt),
                "proj_source": float(projected[src].cpu()),
                "proj_target": float(projected[tgt].cpu()),
                "target_minus_source": float((projected[tgt] - projected[src]).cpu()),
                "ok": bool((projected[tgt] - projected[src]) > 0),
            }
        )
    return torch.stack(dirs, dim=0), meta


def make_transition_labels(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    pred, gt: integer tensors same shape.
    Returns long labels in {0..6}; non-edema transitions map to 0 (ignored as none).
    """
    labels = torch.zeros_like(gt, dtype=torch.long)
    # FN-style toward edema
    labels[(pred == CLS_BG) & (gt == CLS_EDEMA)] = T_BG_TO_ED
    labels[(pred == CLS_NEC) & (gt == CLS_EDEMA)] = T_NEC_TO_ED
    labels[(pred == CLS_ENH) & (gt == CLS_EDEMA)] = T_ENH_TO_ED
    # FP-style away from edema
    labels[(pred == CLS_EDEMA) & (gt == CLS_BG)] = T_ED_TO_BG
    labels[(pred == CLS_EDEMA) & (gt == CLS_NEC)] = T_ED_TO_NEC
    labels[(pred == CLS_EDEMA) & (gt == CLS_ENH)] = T_ED_TO_ENH
    return labels


def baseline_aux_channels(baseline_logits: torch.Tensor) -> torch.Tensor:
    """
    baseline_logits: (B, 4, D, H, W) or (4, D, H, W)
    Returns (B, 6, D, H, W): 4 probs + entropy + top1-top2 margin.
    """
    if baseline_logits.dim() == 4:
        baseline_logits = baseline_logits.unsqueeze(0)
    probs = torch.softmax(baseline_logits.float(), dim=1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1, keepdim=True)
    top2 = torch.topk(probs, k=2, dim=1).values
    margin = (top2[:, 0:1] - top2[:, 1:2])
    return torch.cat([probs, entropy, margin], dim=1)


class TransitionGate(nn.Module):
    """Lightweight voxelwise 7-way transition classifier."""

    def __init__(self, decoder_channels: int = 32, aux_channels: int = 6, hidden: int = 16):
        super().__init__()
        in_ch = decoder_channels + aux_channels
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, N_TRANSITIONS, kernel_size=1),
        )

    def forward(self, h: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h, aux], dim=1))


class ClassifierInvertedRepair(nn.Module):
    """Predict transitions; apply fixed classifier-inverted decoder1 directions."""

    def __init__(
        self,
        directions: torch.Tensor,
        decoder_channels: int = 32,
        init_raw_scale: float = -2.0,
    ) -> None:
        super().__init__()
        if directions.shape != (6, decoder_channels):
            raise ValueError(f"directions must be (6, {decoder_channels}), got {tuple(directions.shape)}")
        self.register_buffer("directions", directions.float())
        self.gate = TransitionGate(decoder_channels=decoder_channels)
        self.raw_scale = nn.Parameter(torch.tensor(float(init_raw_scale)))

    @property
    def scale(self) -> torch.Tensor:
        return F.softplus(self.raw_scale)

    def transition_logits(self, h: torch.Tensor, baseline_logits: torch.Tensor) -> torch.Tensor:
        aux = baseline_aux_channels(baseline_logits)
        if aux.shape[0] != h.shape[0]:
            aux = aux.expand(h.shape[0], -1, -1, -1, -1)
        return self.gate(h, aux.to(device=h.device, dtype=h.dtype))

    def forward(
        self,
        h: torch.Tensor,
        baseline_logits: torch.Tensor,
        abstain_threshold: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns repaired h and transition probabilities (B, 7, D, H, W).
        """
        t_logits = self.transition_logits(h, baseline_logits)
        probs = torch.softmax(t_logits, dim=1)
        corr = probs[:, 1:]  # B,6,...
        if abstain_threshold > 0:
            max_corr, _ = corr.max(dim=1, keepdim=True)
            corr = corr * (max_corr >= abstain_threshold).float()
        # sum_k g_k * d_k
        edit = torch.einsum("bkdhw,kc->bcdhw", corr, self.directions.to(dtype=h.dtype))
        repaired = h + self.scale.to(dtype=h.dtype) * edit
        return repaired, probs

    def trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.gate.parameters()) + [self.raw_scale]


class LogitRefinementHead(nn.Module):
    """Same gate backbone, but predicts 4-class logit residuals directly."""

    def __init__(self, decoder_channels: int = 32, hidden: int = 16):
        super().__init__()
        self.gate = TransitionGate(decoder_channels=decoder_channels, hidden=hidden)
        # Reuse first layers idea: map 7 transition logits -> 4 residual via 1x1
        self.to_residual = nn.Conv3d(N_TRANSITIONS, 4, kernel_size=1)

    def forward(self, h: torch.Tensor, baseline_logits: torch.Tensor) -> torch.Tensor:
        t_logits = self.gate(h, baseline_aux_channels(baseline_logits).to(h.device))
        return baseline_logits + self.to_residual(t_logits)
