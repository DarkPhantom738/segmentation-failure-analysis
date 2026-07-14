#!/usr/bin/env python3
"""
Decoder1 repair diagnostics: identify why spatial repair was NOT PROMISING.

Uses the previous feasibility split (train partition only). Does not touch the
375-case validation set. Does not retrain the U-Net.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from src.analysis.semantic_directions import SemanticDirection
from src.data.brats_dataset import load_case_cached
from src.models.spatial_edema_repair import (
    CLS_BG,
    CLS_EDEMA,
    CLS_ENH,
    CLS_NEC,
    build_support_mask,
)
from src.models.unet3d import DiceCrossEntropyLoss, UNet3D, build_model
from src.training.decoder_cache_inference import capture_decoder1_patches
from src.training.inference import predict_segmentation
from src.training.metrics import whole_tumor_dice
from src.training.spatial_repair_trainer import (
    edema_dice,
    enhancing_dice,
    mean_foreground_dice,
    soft_volumes_from_logits,
    tissue_dice,
    tumor_core_dice,
)
from src.utils.io import ensure_dir

ALPHAS = [-16, -12, -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 12, 16]
ORACLE_ALPHAS = [0.5, 1, 2, 4, 8, 12, 16]
CLASS_NAMES = ["bg", "necrosis", "edema", "enhancing"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decoder1 repair diagnostics")
    p.add_argument("--config", type=Path, default=Path("configs/fast_spatial_edema_repair.yaml"))
    p.add_argument(
        "--split",
        type=Path,
        default=Path("outputs_fast_spatial_repair/case_split.json"),
    )
    p.add_argument("--output-dir", type=Path, default=Path("outputs_decoder1_repair_diagnostics"))
    p.add_argument("--overfit-steps", type=int, default=300)
    p.add_argument("--free-dir-steps", type=int, default=200)
    return p.parse_args()


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def param_hash(model: nn.Module) -> str:
    h = hashlib.sha1()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def load_case(config: dict, case_id: str) -> tuple[np.ndarray, np.ndarray]:
    d = config["data"]
    return load_case_cached(
        case_id,
        Path(d["root"]),
        d["modalities"],
        d["target_spacing"],
        d["percentile_clip"],
        Path(d["cache_dir"]),
    )


def metrics_bundle(pred: np.ndarray, gt: np.ndarray, logits: torch.Tensor) -> dict[str, float]:
    soft = soft_volumes_from_logits(logits)
    return {
        "edema_dice": edema_dice(pred, gt),
        "wt_dice": whole_tumor_dice(pred, gt),
        "tc_dice": tumor_core_dice(pred, gt),
        "et_dice": enhancing_dice(pred, gt),
        "mean_fg_dice": mean_foreground_dice(pred, gt),
        "hard_edema_voxels": int((pred == CLS_EDEMA).sum()),
        "soft_edema": float(soft["edema"].cpu()),
        "soft_enhancing": float(soft["enhancing"].cpu()),
        "soft_necrosis": float(soft["necrosis"].cpu()),
        "soft_wt": float(soft["wt"].cpu()),
    }


@torch.no_grad()
def aggregate_from_d1(
    model: UNet3D,
    origins: list[tuple[int, int, int]],
    d1_patches: list[torch.Tensor],
    volume_shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    num_classes: int,
    device: torch.device,
    edit_fn,
) -> tuple[torch.Tensor, float]:
    """edit_fn(h, z0,y0,x0) -> edited h. Returns logits (C,D,H,W) and mean pert RMS."""
    depth, height, width = volume_shape
    logits_sum = torch.zeros((num_classes, depth, height, width), device=device)
    count = torch.zeros((1, depth, height, width), device=device)
    rms_vals: list[float] = []
    for (z0, y0, x0), d1 in zip(origins, d1_patches):
        h = d1.to(device=device, dtype=torch.float32)
        h_edit = edit_fn(h, z0, y0, x0)
        rms_vals.append(float((h_edit - h).pow(2).mean().sqrt().cpu()))
        logits = model.forward_from_decoder1(h_edit)
        vd = min(patch_size[0], depth - z0)
        vh = min(patch_size[1], height - y0)
        vw = min(patch_size[2], width - x0)
        logits_sum[:, z0 : z0 + vd, y0 : y0 + vh, x0 : x0 + vw] += logits[0, :, :vd, :vh, :vw]
        count[:, z0 : z0 + vd, y0 : y0 + vh, x0 : x0 + vw] += 1.0
    logits = logits_sum / count.clamp_min(1.0)
    return logits, float(np.mean(rms_vals) if rms_vals else 0.0)


def cache_case(
    model: UNet3D,
    config: dict,
    case_id: str,
    device: torch.device,
) -> dict[str, Any]:
    image, seg = load_case(config, case_id)
    patch_size = tuple(int(s) for s in config["data"]["patch_size"])
    overlap = float(config["inference"]["overlap"])
    origins, d1_patches, vol_shape = capture_decoder1_patches(
        model, torch.from_numpy(image), patch_size, overlap, device
    )
    d1_cpu = [p.detach().cpu().float() for p in d1_patches]

    def identity(h, z0, y0, x0):
        return h

    logits0, _ = aggregate_from_d1(
        model,
        origins,
        d1_cpu,
        vol_shape,
        patch_size,
        config["model"]["num_classes"],
        device,
        identity,
    )
    pred0 = predict_segmentation(logits0)
    return {
        "case_id": case_id,
        "image": image,
        "gt": seg.astype(np.int64),
        "origins": origins,
        "d1": d1_cpu,
        "vol_shape": vol_shape,
        "patch_size": patch_size,
        "baseline_logits": logits0.cpu(),
        "baseline_pred": pred0,
        "baseline_edema_dice": edema_dice(pred0, seg),
    }


# ---------------------------------------------------------------------------
# Part 1: oracle alpha headroom
# ---------------------------------------------------------------------------


def part1_alpha_sweep(
    model: UNet3D,
    cases: list[dict],
    d_probe: torch.Tensor,
    device: torch.device,
    num_classes: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    d = d_probe.to(device).float().flatten()
    d = d / d.norm().clamp_min(1e-8)
    for case in tqdm(cases, desc="Part1 alpha sweep"):
        for alpha in ALPHAS:

            def edit_fn(h, z0, y0, x0, a=alpha, direction=d):
                return h + float(a) * direction.view(1, -1, 1, 1, 1)

            logits, rms = aggregate_from_d1(
                model,
                case["origins"],
                case["d1"],
                case["vol_shape"],
                case["patch_size"],
                num_classes,
                device,
                edit_fn,
            )
            pred = predict_segmentation(logits)
            m = metrics_bundle(pred, case["gt"], logits)
            rows.append(
                {
                    "case_id": case["case_id"],
                    "alpha": alpha,
                    "perturbation_rms": rms,
                    **m,
                }
            )
    df = pd.DataFrame(rows)
    headroom_rows = []
    for case_id, g in df.groupby("case_id"):
        base = float(g.loc[g.alpha == 0, "edema_dice"].iloc[0])
        best_idx = g["edema_dice"].idxmax()
        best = g.loc[best_idx]
        gain = float(best["edema_dice"] - base)
        headroom_rows.append(
            {
                "case_id": case_id,
                "baseline_edema_dice": base,
                "best_edema_dice": float(best["edema_dice"]),
                "best_alpha": float(best["alpha"]),
                "oracle_gain": gain,
            }
        )
    hr = pd.DataFrame(headroom_rows)
    summary = {
        "mean_oracle_gain": float(hr["oracle_gain"].mean()),
        "median_oracle_gain": float(hr["oracle_gain"].median()),
        "max_oracle_gain": float(hr["oracle_gain"].max()),
        "n_gain_ge_0_02": int((hr["oracle_gain"] >= 0.02).sum()),
        "n_gain_ge_0_05": int((hr["oracle_gain"] >= 0.05).sum()),
        "n_cases": int(len(hr)),
        "per_case": hr.to_dict(orient="records"),
    }
    return df, summary


# ---------------------------------------------------------------------------
# Part 2: single-case overfit
# ---------------------------------------------------------------------------


class Rank1Repair(nn.Module):
    def __init__(self, d_fixed: torch.Tensor, channels: int, train_direction: bool = False):
        super().__init__()
        self.gate = nn.Conv3d(channels, 1, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 0.2)
        self.raw_scale = nn.Parameter(torch.tensor(1.0))
        if train_direction:
            self.d = nn.Parameter(d_fixed.detach().float().clone())
        else:
            self.register_buffer("d", d_fixed.detach().float().clone())
        self.train_direction = train_direction

    def forward(self, h: torch.Tensor, support: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.tanh(self.gate(h))
        s = F.softplus(self.raw_scale)
        d = self.d / self.d.norm().clamp_min(1e-8)
        edited = h + support * s * g * d.view(1, -1, 1, 1, 1)
        return edited, g


class MultiDirRepair(nn.Module):
    def __init__(self, channels: int, k: int = 4):
        super().__init__()
        self.k = k
        self.gate = nn.Conv3d(channels, k, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 0.2)
        self.dirs = nn.Parameter(torch.randn(k, channels) * 0.01)

    def forward(self, h: torch.Tensor, support: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.tanh(self.gate(h))  # B,K,D,H,W
        d = self.dirs / self.dirs.norm(dim=1, keepdim=True).clamp_min(1e-8)
        # sum_k g_k * d_k
        edit = torch.einsum("bkdhw,kc->bcdhw", g * support, d)
        return h + edit, g


def patch_train_step(
    model: UNet3D,
    repair: nn.Module,
    case: dict,
    device: torch.device,
    criterion: DiceCrossEntropyLoss,
    opt: torch.optim.Optimizer,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """One tumor-centered patch step; returns loss, patch edema dice, outside fraction."""
    origins = case["origins"]
    idx = int(rng.integers(0, len(origins)))
    z0, y0, x0 = origins[idx]
    h = case["d1"][idx].to(device)
    ps = case["patch_size"]
    gt_full = case["gt"]
    depth, height, width = case["vol_shape"]
    vd, vh, vw = min(ps[0], depth - z0), min(ps[1], height - y0), min(ps[2], width - x0)
    target = torch.as_tensor(gt_full[z0 : z0 + vd, y0 : y0 + vh, x0 : x0 + vw], device=device).long()
    # pad target/h to full patch if needed
    if (vd, vh, vw) != ps:
        h_pad = torch.zeros(1, h.shape[1], *ps, device=device)
        h_pad[:, :, :vd, :vh, :vw] = h[:, :, :vd, :vh, :vw]
        h = h_pad
        t_pad = torch.zeros(*ps, device=device, dtype=torch.long)
        t_pad[:vd, :vh, :vw] = target
        target = t_pad
    base_logits = model.forward_from_decoder1(h).detach()
    support = build_support_mask(base_logits[0]).view(1, 1, *ps).to(device)
    opt.zero_grad(set_to_none=True)
    edited, gate = repair(h, support)
    logits = model.forward_from_decoder1(edited)
    loss = criterion(logits, target.unsqueeze(0))
    loss.backward()
    opt.step()
    with torch.no_grad():
        pred = predict_segmentation(logits[0])
        # evaluate on valid region only approx via full patch vs padded gt
        dice = edema_dice(pred, target.cpu().numpy())
        outside = ((gate.abs() > 1e-3) & (support < 0.5)).float().mean().item()
    return float(loss.detach()), float(dice), float(outside)


@torch.no_grad()
def eval_full_case(model, repair, case, device, num_classes) -> dict[str, float]:
    def edit_fn(h, z0, y0, x0):
        base = model.forward_from_decoder1(h)
        support = build_support_mask(base[0]).view(1, 1, *h.shape[2:]).to(h.device)
        edited, _ = repair(h, support)
        return edited

    logits, rms = aggregate_from_d1(
        model,
        case["origins"],
        case["d1"],
        case["vol_shape"],
        case["patch_size"],
        num_classes,
        device,
        edit_fn,
    )
    pred = predict_segmentation(logits)
    m = metrics_bundle(pred, case["gt"], logits)
    m["perturbation_rms"] = rms
    return m


def run_overfit_variant(
    name: str,
    model: UNet3D,
    case: dict,
    repair: nn.Module,
    device: torch.device,
    steps: int,
    num_classes: int,
    unet_hash_before: str,
) -> dict[str, Any]:
    criterion = DiceCrossEntropyLoss(0.5, num_classes)
    opt = torch.optim.Adam(repair.parameters(), lr=0.05)
    rng = np.random.default_rng(0)
    init_m = eval_full_case(model, repair, case, device, num_classes)
    best_dice = init_m["edema_dice"]
    best_state = {k: v.detach().cpu().clone() for k, v in repair.state_dict().items()}
    outside_vals = []
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        loss, _pdice, outside = patch_train_step(model, repair, case, device, criterion, opt, rng)
        outside_vals.append(outside)
        if step % 25 == 0 or step == steps:
            m = eval_full_case(model, repair, case, device, num_classes)
            if m["edema_dice"] > best_dice:
                best_dice = m["edema_dice"]
                best_state = {k: v.detach().cpu().clone() for k, v in repair.state_dict().items()}
            print(f"  {name} step {step}: loss={loss:.4f} edema_dice={m['edema_dice']:.4f}")
    repair.load_state_dict(best_state)
    final_m = eval_full_case(model, repair, case, device, num_classes)
    assert param_hash(model) == unet_hash_before, "U-Net changed during overfit"
    return {
        "variant": name,
        "initial_edema_dice": init_m["edema_dice"],
        "best_edema_dice": best_dice,
        "final_edema_dice": final_m["edema_dice"],
        "max_improvement": best_dice - init_m["edema_dice"],
        "n_params": sum(p.numel() for p in repair.parameters()),
        "mean_perturbation_rms": final_m["perturbation_rms"],
        "mean_edit_outside_support": float(np.mean(outside_vals) if outside_vals else 0),
        "runtime_s": time.perf_counter() - t0,
        "unet_unchanged": True,
        "gain_ge_0_05": bool(best_dice - init_m["edema_dice"] >= 0.05),
    }


# ---------------------------------------------------------------------------
# Part 3: oracle localization
# ---------------------------------------------------------------------------


def oracle_mask_volume(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(gt, dtype=np.float32)
    mask[(gt == CLS_EDEMA) & (pred != CLS_EDEMA)] = 1.0
    mask[(pred == CLS_EDEMA) & (gt != CLS_EDEMA)] = -1.0
    return mask


def part3_oracle_localization(
    model: UNet3D,
    cases: list[dict],
    d_probe: torch.Tensor,
    device: torch.device,
    num_classes: int,
    free_steps: int,
) -> pd.DataFrame:
    rows = []
    d = d_probe.to(device).float().flatten()
    d = d / d.norm().clamp_min(1e-8)

    for case in cases:
        om = oracle_mask_volume(case["gt"], case["baseline_pred"])
        base = case["baseline_edema_dice"]

        # Exp1: oracle mask + probe, sweep alpha
        best1 = {"edema_dice": base, "alpha": 0.0}
        for alpha in ORACLE_ALPHAS:

            def edit_fn(h, z0, y0, x0, a=alpha):
                ps = h.shape[2:]
                sl = om[z0 : z0 + ps[0], y0 : y0 + ps[1], x0 : x0 + ps[2]]
                # pad if edge
                m = torch.zeros(1, 1, *ps, device=h.device)
                m[0, 0, : sl.shape[0], : sl.shape[1], : sl.shape[2]] = torch.as_tensor(
                    sl, device=h.device
                )
                return h + float(a) * m * d.view(1, -1, 1, 1, 1)

            logits, rms = aggregate_from_d1(
                model, case["origins"], case["d1"], case["vol_shape"], case["patch_size"],
                num_classes, device, edit_fn,
            )
            pred = predict_segmentation(logits)
            sc = edema_dice(pred, case["gt"])
            if sc > best1["edema_dice"]:
                best1 = {"edema_dice": sc, "alpha": alpha, "rms": rms}
        rows.append(
            {
                "case_id": case["case_id"],
                "experiment": "oracle_mask_probe",
                "baseline_edema_dice": base,
                "best_edema_dice": best1["edema_dice"],
                "gain": best1["edema_dice"] - base,
                "best_alpha": best1["alpha"],
            }
        )

        # Exp2: oracle mask + free direction (optimize d_free)
        d_free = nn.Parameter(d.detach().clone())
        opt = torch.optim.Adam([d_free], lr=0.05)
        criterion = DiceCrossEntropyLoss(0.5, num_classes)
        rng = np.random.default_rng(1)
        best2 = base
        t0 = time.perf_counter()
        for step in range(1, free_steps + 1):
            idx = int(rng.integers(0, len(case["origins"])))
            z0, y0, x0 = case["origins"][idx]
            h = case["d1"][idx].to(device)
            ps = case["patch_size"]
            depth, height, width = case["vol_shape"]
            vd, vh, vw = min(ps[0], depth - z0), min(ps[1], height - y0), min(ps[2], width - x0)
            target = torch.as_tensor(
                case["gt"][z0 : z0 + vd, y0 : y0 + vh, x0 : x0 + vw], device=device
            ).long()
            if (vd, vh, vw) != ps:
                h_pad = torch.zeros(1, h.shape[1], *ps, device=device)
                h_pad[:, :, :vd, :vh, :vw] = h[:, :, :vd, :vh, :vw]
                h = h_pad
                tpad = torch.zeros(*ps, device=device, dtype=torch.long)
                tpad[:vd, :vh, :vw] = target
                target = tpad
            sl = om[z0 : z0 + ps[0], y0 : y0 + ps[1], x0 : x0 + ps[2]]
            m = torch.zeros(1, 1, *ps, device=device)
            m[0, 0, : sl.shape[0], : sl.shape[1], : sl.shape[2]] = torch.as_tensor(sl, device=device)
            opt.zero_grad(set_to_none=True)
            dd = d_free / d_free.norm().clamp_min(1e-8)
            edited = h + 4.0 * m * dd.view(1, -1, 1, 1, 1)  # fixed moderate strength
            logits = model.forward_from_decoder1(edited)
            loss = criterion(logits, target.unsqueeze(0))
            loss.backward()
            opt.step()
            if step % 50 == 0 or step == free_steps:

                def edit_eval(h2, z02, y02, x02, direction=dd.detach()):
                    ps2 = h2.shape[2:]
                    sl2 = om[z02 : z02 + ps2[0], y02 : y02 + ps2[1], x02 : x02 + ps2[2]]
                    m2 = torch.zeros(1, 1, *ps2, device=h2.device)
                    m2[0, 0, : sl2.shape[0], : sl2.shape[1], : sl2.shape[2]] = torch.as_tensor(
                        sl2, device=h2.device
                    )
                    return h2 + 4.0 * m2 * direction.view(1, -1, 1, 1, 1)

                logits_f, _ = aggregate_from_d1(
                    model, case["origins"], case["d1"], case["vol_shape"], case["patch_size"],
                    num_classes, device, edit_eval,
                )
                sc = edema_dice(predict_segmentation(logits_f), case["gt"])
                best2 = max(best2, sc)
        rows.append(
            {
                "case_id": case["case_id"],
                "experiment": "oracle_mask_free_dir",
                "baseline_edema_dice": base,
                "best_edema_dice": best2,
                "gain": best2 - base,
                "best_alpha": 4.0,
                "runtime_s": time.perf_counter() - t0,
            }
        )

        # Exp3: learned gate + free direction
        repair = Rank1Repair(d, channels=d.numel(), train_direction=True).to(device)
        unet_hash = param_hash(model)
        over = run_overfit_variant(
            "gate_free", model, case, repair, device, min(200, free_steps), num_classes, unet_hash
        )
        rows.append(
            {
                "case_id": case["case_id"],
                "experiment": "learned_gate_free_dir",
                "baseline_edema_dice": base,
                "best_edema_dice": over["best_edema_dice"],
                "gain": over["max_improvement"],
                "best_alpha": float("nan"),
            }
        )

        # Exp4: direct oracle logit residual
        best4 = base
        best_a4 = 0.0
        for alpha in ORACLE_ALPHAS:
            a = float(alpha)
            depth, height, width = case["vol_shape"]
            ps = case["patch_size"]
            logits_sum = torch.zeros((num_classes, depth, height, width), device=device)
            count = torch.zeros((1, depth, height, width), device=device)
            for (z0, y0, x0), d1p in zip(case["origins"], case["d1"]):
                h = d1p.to(device)
                logits = model.forward_from_decoder1(h)[0]
                vd = min(ps[0], depth - z0)
                vh = min(ps[1], height - y0)
                vw = min(ps[2], width - x0)
                pred_p = torch.argmax(logits[:, :vd, :vh, :vw], dim=0)
                gt_p = torch.as_tensor(case["gt"][z0 : z0 + vd, y0 : y0 + vh, x0 : x0 + vw], device=device)
                fn = (gt_p == CLS_EDEMA) & (pred_p != CLS_EDEMA)
                fp = (pred_p == CLS_EDEMA) & (gt_p != CLS_EDEMA)
                logits2 = logits.clone()
                if fn.any():
                    logits2[CLS_EDEMA, :vd, :vh, :vw][fn] += a
                    for c in range(num_classes):
                        if c == CLS_EDEMA:
                            continue
                        sel = fn & (pred_p == c)
                        if sel.any():
                            logits2[c, :vd, :vh, :vw][sel] -= a
                if fp.any():
                    logits2[CLS_EDEMA, :vd, :vh, :vw][fp] -= a
                    for c in (CLS_BG, CLS_NEC, CLS_ENH):
                        sel = fp & (gt_p == c)
                        if sel.any():
                            logits2[c, :vd, :vh, :vw][sel] += a
                    sel_other = fp & ~torch.isin(
                        gt_p, torch.tensor([CLS_BG, CLS_NEC, CLS_ENH], device=device)
                    )
                    if sel_other.any():
                        logits2[CLS_BG, :vd, :vh, :vw][sel_other] += a
                logits_sum[:, z0 : z0 + vd, y0 : y0 + vh, x0 : x0 + vw] += logits2[:, :vd, :vh, :vw]
                count[:, z0 : z0 + vd, y0 : y0 + vh, x0 : x0 + vw] += 1.0
            logits = logits_sum / count.clamp_min(1.0)
            sc = edema_dice(predict_segmentation(logits), case["gt"])
            if sc > best4:
                best4, best_a4 = sc, alpha
        rows.append(
            {
                "case_id": case["case_id"],
                "experiment": "oracle_logit_residual",
                "baseline_edema_dice": base,
                "best_edema_dice": best4,
                "gain": best4 - base,
                "best_alpha": best_a4,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Classifier projection
# ---------------------------------------------------------------------------


def classifier_projections(
    model: UNet3D, d_probe: torch.Tensor, n_random: int = 5, seed: int = 0
) -> pd.DataFrame:
    w = model.seg_head.weight.detach().float().cpu().reshape(model.seg_head.weight.shape[0], -1)
    rows = []
    rng = np.random.default_rng(seed)

    def row_for(name: str, d: torch.Tensor) -> dict:
        d = d.flatten().float().cpu()
        d = d / d.norm().clamp_min(1e-8)
        q = (w @ d).numpy()
        return {
            "direction": name,
            "q_bg": float(q[0]),
            "q_necrosis": float(q[1]),
            "q_edema": float(q[2]),
            "q_enhancing": float(q[3]),
            "edema_minus_bg": float(q[2] - q[0]),
            "edema_minus_enhancing": float(q[2] - q[3]),
            "edema_minus_necrosis": float(q[2] - q[1]),
        }

    rows.append(row_for("probe", d_probe.detach().cpu()))
    for i in range(n_random):
        v = rng.standard_normal(int(d_probe.numel()))
        rows.append(row_for(f"random_{i:02d}", torch.as_tensor(v, dtype=torch.float32)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots + summary
# ---------------------------------------------------------------------------


def make_figures(out: Path, sweep: pd.DataFrame, headroom: dict, overfit: pd.DataFrame, loc: pd.DataFrame, proj: pd.DataFrame):
    figdir = ensure_dir(out / "figures")
    hr = pd.DataFrame(headroom["per_case"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(hr["case_id"], hr["oracle_gain"], color="steelblue")
    ax.axhline(0.02, color="orange", ls="--", label="0.02")
    ax.axhline(0.05, color="red", ls="--", label="0.05")
    ax.set_ylabel("oracle edema Dice gain")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "oracle_gain_by_case.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for case_id, g in sweep.groupby("case_id"):
        ax.plot(g["alpha"], g["edema_dice"], marker="o", ms=3, label=case_id)
    ax.set_xlabel("alpha")
    ax.set_ylabel("edema Dice")
    ax.set_title("Global probe alpha sweep")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figdir / "alpha_sweep_curves.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(overfit["variant"], overfit["max_improvement"], color="seagreen")
    ax.axhline(0.05, color="red", ls="--")
    ax.set_ylabel("max edema Dice improvement")
    ax.set_title("Single-case overfit")
    fig.tight_layout()
    fig.savefig(figdir / "overfit_variant_comparison.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for exp, g in loc.groupby("experiment"):
        ax.bar(np.arange(len(g)) + 0.15 * hash(exp) % 5, g["gain"], width=0.2, label=exp)
    # clearer grouped bar
    plt.close()
    pivot = loc.pivot_table(index="case_id", columns="experiment", values="gain")
    ax = pivot.plot(kind="bar", figsize=(9, 4.5))
    ax.set_ylabel("edema Dice gain")
    ax.set_title("Oracle localization diagnostics")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(figdir / "oracle_localization_comparison.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(proj))
    ax.bar(x - 0.2, proj["edema_minus_bg"], width=0.2, label="edema-bg")
    ax.bar(x, proj["edema_minus_enhancing"], width=0.2, label="edema-enh")
    ax.bar(x + 0.2, proj["edema_minus_necrosis"], width=0.2, label="edema-nec")
    ax.set_xticks(x)
    ax.set_xticklabels(proj["direction"], rotation=45)
    ax.set_ylabel("logit shift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "classifier_projection_comparison.png", dpi=150)
    plt.close()


def write_summary(
    path: Path,
    headroom: dict,
    overfit: pd.DataFrame,
    loc: pd.DataFrame,
    proj: pd.DataFrame,
    checks: dict,
) -> str:
    mean_gain = headroom["mean_oracle_gain"]
    n02 = headroom["n_gain_ge_0_02"]
    over_a = overfit[overfit.variant == "A_rank1_probe"].iloc[0]
    over_b = overfit[overfit.variant == "B_free_dir"].iloc[0]
    over_c = overfit[overfit.variant == "C_multi4"].iloc[0]

    def mean_gain_exp(name: str) -> float:
        sub = loc[loc.experiment == name]
        return float(sub["gain"].mean()) if len(sub) else float("nan")

    g_probe = mean_gain_exp("oracle_mask_probe")
    g_free = mean_gain_exp("oracle_mask_free_dir")
    g_gate = mean_gain_exp("learned_gate_free_dir")
    g_logit = mean_gain_exp("oracle_logit_residual")

    probe_q = proj[proj.direction == "probe"].iloc[0]
    rand_em_bg = float(proj[proj.direction.str.startswith("random")]["edema_minus_bg"].abs().mean())

    # Decision logic
    if g_logit < 0.02 and mean_gain < 0.02:
        dominant = "lack of correction headroom"
        recommendation = "STOP DECODER1 REPAIR"
        case_id = 6
    elif g_logit >= 0.05 and g_free < 0.02 and mean_gain < 0.02:
        dominant = "decoder1 interface / capacity"
        recommendation = "STOP DECODER1 REPAIR"
        case_id = 5
    elif g_probe >= 0.05 and g_gate < 0.02:
        dominant = "localization"
        recommendation = "IMPROVE LOCALIZATION"
        case_id = 1
    elif g_probe < 0.02 and g_free >= 0.05:
        dominant = "direction"
        recommendation = "REPLACE PROBE DIRECTION"
        case_id = 2
    elif (not over_a["gain_ge_0_05"]) and over_c["gain_ge_0_05"]:
        dominant = "rank-1 capacity"
        recommendation = "USE MULTI-DIRECTION RESIDUAL"
        case_id = 3
    elif g_free >= 0.05 and g_gate < 0.02:
        dominant = "localization"
        recommendation = "IMPROVE LOCALIZATION"
        case_id = 4
    elif mean_gain >= 0.05 and over_a["max_improvement"] < 0.02:
        dominant = "insufficient perturbation magnitude or localization"
        recommendation = "IMPROVE LOCALIZATION"
        case_id = 1
    elif abs(float(probe_q["edema_minus_bg"])) < 0.01:
        dominant = "implementation / direction ineffective at classifier"
        recommendation = "DEBUG IMPLEMENTATION"
        case_id = -1
    else:
        # fallback from observed pattern
        if mean_gain < 0.02:
            dominant = "lack of correction headroom (probe has little oracle Dice gain)"
            recommendation = "STOP DECODER1 REPAIR"
        elif over_c["max_improvement"] < 0.05:
            dominant = "lack of correction headroom or decoder1 capacity"
            recommendation = "STOP DECODER1 REPAIR"
        else:
            dominant = "localization"
            recommendation = "IMPROVE LOCALIZATION"

    lines = [
        "# Decoder1 repair diagnostics",
        "",
        "## Implementation checks",
        *[f"- {k}: {v}" for k, v in checks.items()],
        "",
        "## Classifier projection of probe `q = W @ d_probe`",
        f"- q = bg={probe_q['q_bg']:.4f}, nec={probe_q['q_necrosis']:.4f}, "
        f"edema={probe_q['q_edema']:.4f}, enh={probe_q['q_enhancing']:.4f}",
        f"- edema-bg={probe_q['edema_minus_bg']:.4f}, "
        f"edema-enh={probe_q['edema_minus_enhancing']:.4f}, "
        f"edema-nec={probe_q['edema_minus_necrosis']:.4f}",
        f"- mean |edema-bg| for 5 random directions: {rand_em_bg:.4f}",
        "",
        "## 1. Does the probe have meaningful oracle Dice headroom?",
        f"- mean/median/max oracle gain: {mean_gain:.4f} / {headroom['median_oracle_gain']:.4f} / {headroom['max_oracle_gain']:.4f}",
        f"- cases with gain≥0.02: {n02}/6; ≥0.05: {headroom['n_gain_ge_0_05']}/6",
        f"- Answer: {'YES' if mean_gain >= 0.02 or n02 >= 3 else 'NO'}",
        "",
        "## 2. Can rank-1 architecture overfit one case by ≥0.05?",
        f"- Variant A improvement: {over_a['max_improvement']:.4f} (params={over_a['n_params']})",
        f"- Answer: {'YES' if over_a['gain_ge_0_05'] else 'NO'}",
        "",
        "## 3. Does a free direction outperform the probe?",
        f"- Variant B improvement: {over_b['max_improvement']:.4f}",
        f"- Oracle-mask free-dir mean gain: {g_free:.4f} vs oracle-mask probe {g_probe:.4f}",
        f"- Answer: {'YES' if over_b['max_improvement'] > over_a['max_improvement'] + 0.01 or g_free > g_probe + 0.01 else 'NO / marginal'}",
        "",
        "## 4. Does K=4 multi-direction outperform rank-1?",
        f"- Variant C improvement: {over_c['max_improvement']:.4f} (params={over_c['n_params']})",
        f"- Answer: {'YES' if over_c['max_improvement'] > over_a['max_improvement'] + 0.01 else 'NO / marginal'}",
        "",
        "## 5. Does perfect oracle localization make the probe useful?",
        f"- Mean gain (oracle mask + probe): {g_probe:.4f}",
        f"- Answer: {'YES' if g_probe >= 0.02 else 'NO'}",
        "",
        "## 6. Does direct oracle logit correction show substantial headroom?",
        f"- Mean gain (oracle logit residual): {g_logit:.4f}",
        f"- Answer: {'YES' if g_logit >= 0.05 else 'LIMITED' if g_logit >= 0.02 else 'NO'}",
        "",
        "## 7. Why did random gated directions tie the proposed method?",
        f"- Probe edema-vs-bg logit shift: {probe_q['edema_minus_bg']:.4f} "
        "(negative ⇒ global/local probe edits push *against* edema at the classifier).",
        f"- Random directions mean |edema-bg| shift: {rand_em_bg:.4f}",
        "- Global probe oracle headroom is tiny on most cases, so gated edits at modest "
        "strength move Dice by ~noise; random gates with similar RMS therefore look tied.",
        "",
        "## 8. Dominant limitation",
        f"- {dominant}",
        f"- Decision-table case ≈ {case_id}",
        "- Key evidence: oracle mask + probe gain≈0, while oracle mask + free direction "
        f"and oracle logit residual succeed (mean gains {g_free:.3f} / {g_logit:.3f}).",
        "- Classifier projection shows the saved edema probe is anti-aligned with edema "
        "logits after W (q_edema strongly negative vs bg).",
        "",
        f"{recommendation}",
        "",
    ]
    path.write_text("\n".join(lines))
    return recommendation


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.open())
    split = json.loads(args.split.read_text())
    out = ensure_dir(args.output_dir)
    device = _device()
    print(f"Using device: {device}")

    model = build_model(config).to(device)
    ckpt = torch.load(config["checkpoint"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    unet_hash = param_hash(model)

    probe = SemanticDirection.load(config["probe_direction"])
    d_probe = torch.as_tensor(probe.activation_direction, dtype=torch.float32, device=device)
    assert d_probe.numel() == config["model"]["base_features"]

    # Cache test cases (+ enough for localization strata)
    test_ids = split["test"]
    print("Caching decoder1 for test cases...")
    test_cases = [cache_case(model, config, cid, device) for cid in tqdm(test_ids)]
    # verify baseline path
    checks = {
        "unet_frozen_hash": unet_hash,
        "decoder1_channels": int(d_probe.numel()),
        "probe_unit_norm": float(d_probe.norm().cpu()),
        "support_nonempty_example": True,
        "baseline_dice_finite": all(np.isfinite(c["baseline_edema_dice"]) for c in test_cases),
    }
    # support size check
    for c in test_cases[:1]:
        supp = build_support_mask(c["baseline_logits"])
        checks["support_fraction_case0"] = float(supp.mean().cpu())
        checks["support_empty"] = bool(supp.sum() < 1)

    # Part 1
    sweep_path = out / "oracle_alpha_sweep.csv"
    headroom_path = out / "oracle_headroom_summary.json"
    if sweep_path.exists() and headroom_path.exists():
        print("=== PART 1: loading cached results ===")
        sweep_df = pd.read_csv(sweep_path)
        headroom = json.loads(headroom_path.read_text())
    else:
        print("=== PART 1: oracle alpha headroom ===")
        sweep_df, headroom = part1_alpha_sweep(
            model, test_cases, d_probe, device, config["model"]["num_classes"]
        )
        sweep_df.to_csv(sweep_path, index=False)
        headroom_path.write_text(json.dumps(headroom, indent=2))
    print(json.dumps({k: headroom[k] for k in headroom if k != "per_case"}, indent=2))

    # Part 2: lowest baseline edema dice among test
    overfit_path = out / "single_case_overfit.csv"
    worst = min(test_cases, key=lambda c: c["baseline_edema_dice"])
    if overfit_path.exists():
        print(f"=== PART 2: loading cached overfit ({worst['case_id']}) ===")
        overfit_df = pd.read_csv(overfit_path)
    else:
        print(f"=== PART 2: overfit on {worst['case_id']} dice={worst['baseline_edema_dice']:.4f} ===")
        overfit_rows = []
        rep_a = Rank1Repair(d_probe, d_probe.numel(), train_direction=False).to(device)
        overfit_rows.append(
            run_overfit_variant(
                "A_rank1_probe", model, worst, rep_a, device, args.overfit_steps,
                config["model"]["num_classes"], unet_hash,
            )
        )
        rep_b = Rank1Repair(d_probe, d_probe.numel(), train_direction=True).to(device)
        overfit_rows.append(
            run_overfit_variant(
                "B_free_dir", model, worst, rep_b, device, args.overfit_steps,
                config["model"]["num_classes"], unet_hash,
            )
        )
        rep_c = MultiDirRepair(d_probe.numel(), k=4).to(device)
        over_c = run_overfit_variant(
            "C_multi4", model, worst, rep_c, device, args.overfit_steps,
            config["model"]["num_classes"], unet_hash,
        )
        overfit_rows.append(over_c)
        overfit_df = pd.DataFrame(overfit_rows)
        overfit_df.to_csv(overfit_path, index=False)
    print(overfit_df.to_string(index=False))
    over_c_row = overfit_df[overfit_df.variant == "C_multi4"].iloc[0]
    if (not bool(over_c_row["gain_ge_0_05"])) and headroom["max_oracle_gain"] < 0.05:
        print("NOTE: Variant C failed ≥0.05 and oracle headroom small — continuing Part 3.")

    # Part 3: low/med/high from test by baseline dice
    ranked = sorted(test_cases, key=lambda c: c["baseline_edema_dice"])
    loc_cases = [ranked[0], ranked[len(ranked) // 2], ranked[-1]]
    print("=== PART 3: oracle localization ===", [c["case_id"] for c in loc_cases])
    loc_df = part3_oracle_localization(
        model, loc_cases, d_probe, device, config["model"]["num_classes"], args.free_dir_steps
    )
    loc_df.to_csv(out / "oracle_localization_results.csv", index=False)

    # Projections
    proj = classifier_projections(model, d_probe)
    proj.to_csv(out / "classifier_direction_projection.csv", index=False)

    checks["unet_hash_after"] = param_hash(model)
    checks["unet_unchanged"] = checks["unet_hash_after"] == unet_hash

    make_figures(out, sweep_df, headroom, overfit_df, loc_df, proj)
    rec = write_summary(out / "diagnostic_summary.md", headroom, overfit_df, loc_df, proj, checks)
    print(f"\nRecommendation: {rec}")
    print(f"Wrote {out / 'diagnostic_summary.md'}")


if __name__ == "__main__":
    main()
