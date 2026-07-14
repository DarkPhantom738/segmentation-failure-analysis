"""Training / evaluation helpers for spatially gated edema repair."""

from __future__ import annotations

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
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from src.analysis.semantic_directions import SemanticDirection
from src.data.brats_dataset import load_case_cached, split_cases
from src.models.spatial_edema_repair import (
    CLS_BG,
    CLS_EDEMA,
    CLS_ENH,
    CLS_NEC,
    SpatialEdemaRepair,
    build_d_initial,
    build_support_mask,
)
from src.models.unet3d import DiceCrossEntropyLoss, UNet3D, build_model
from src.training.inference import (
    _axis_steps,
    _extract_patch,
    _patch_indices,
    predict_segmentation,
    sliding_window_inference,
)
from src.training.metrics import dice_score, whole_tumor_dice, whole_tumor_mask
from src.utils.io import ensure_dir


def tissue_dice(pred: np.ndarray, gt: np.ndarray, classes: tuple[int, ...]) -> float:
    p = np.isin(pred, classes)
    g = np.isin(gt, classes)
    return dice_score(p, g)


def edema_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    return tissue_dice(pred, gt, (CLS_EDEMA,))


def tumor_core_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    return tissue_dice(pred, gt, (CLS_NEC, CLS_ENH))


def enhancing_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    return tissue_dice(pred, gt, (CLS_ENH,))


def mean_foreground_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(
        np.mean(
            [
                edema_dice(pred, gt),
                tumor_core_dice(pred, gt),
                enhancing_dice(pred, gt),
            ]
        )
    )


def soft_volumes_from_logits(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    probs = torch.softmax(logits, dim=0)
    return {
        "necrosis": probs[CLS_NEC].sum(),
        "edema": probs[CLS_EDEMA].sum(),
        "enhancing": probs[CLS_ENH].sum(),
        "wt": probs[1:].sum(),
    }


def tumor_centered_crop(
    image: torch.Tensor,
    seg: torch.Tensor,
    patch_size: tuple[int, int, int],
    center_mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int]]:
    """Crop 64^3 patch around a random voxel in center_mask (fallback: volume center)."""
    _, d, h, w = image.shape
    pd_, ph, pw = patch_size
    coords = np.argwhere(center_mask > 0)
    if len(coords) == 0:
        cz, cy, cx = d // 2, h // 2, w // 2
    else:
        cz, cy, cx = coords[rng.integers(0, len(coords))]
    z0 = int(np.clip(cz - pd_ // 2, 0, max(0, d - pd_)))
    y0 = int(np.clip(cy - ph // 2, 0, max(0, h - ph)))
    x0 = int(np.clip(cx - pw // 2, 0, max(0, w - pw)))
    # if volume smaller than patch, start at 0
    z0 = min(z0, max(0, d - pd_))
    y0 = min(y0, max(0, h - ph))
    x0 = min(x0, max(0, w - pw))
    z1, y1, x1 = z0 + pd_, y0 + ph, x0 + pw
    # pad if needed
    img = image[:, z0:z1, y0:y1, x0:x1]
    lab = seg[z0:z1, y0:y1, x0:x1]
    if img.shape[1:] != patch_size:
        pad = [0, max(0, pw - img.shape[3]), 0, max(0, ph - img.shape[2]), 0, max(0, pd_ - img.shape[1])]
        img = F.pad(img.unsqueeze(0), pad).squeeze(0)
        lab = F.pad(lab.unsqueeze(0).unsqueeze(0).float(), pad).squeeze().long()
    return img, lab, (z0, y0, x0)


def get_train_partition_case_ids(config: dict) -> list[str]:
    cache = Path(config["data"]["cache_dir"])
    ids = sorted({p.name.split("_v2_")[0] for p in cache.glob("*image.npy")})
    train_ids, _val_ids = split_cases(ids, config["data"]["val_fraction"], config["seed"])
    return train_ids


@torch.no_grad()
def baseline_volume_prediction(
    model: UNet3D,
    image: torch.Tensor,
    patch_size: tuple[int, int, int],
    num_classes: int,
    overlap: float,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    logits, _, _ = sliding_window_inference(
        model, image, patch_size, num_classes, overlap=overlap, device=device
    )
    return logits.cpu(), predict_segmentation(logits)


def score_edema_dice_pool(
    model: UNet3D,
    case_ids: list[str],
    config: dict,
    device: torch.device,
    max_pool: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pool = list(case_ids)
    rng.shuffle(pool)
    pool = pool[:max_pool]
    patch_size = tuple(int(s) for s in config["data"]["patch_size"])
    overlap = float(config.get("inference", {}).get("overlap", 0.25))
    rows = []
    data_cfg = config["data"]
    for case_id in tqdm(pool, desc="Score train edema Dice"):
        image, seg = load_case_cached(
            case_id,
            Path(data_cfg["root"]),
            data_cfg["modalities"],
            data_cfg["target_spacing"],
            data_cfg["percentile_clip"],
            Path(data_cfg["cache_dir"]),
        )
        if int((seg == CLS_EDEMA).sum()) < 50:
            continue
        logits, pred = baseline_volume_prediction(
            model,
            torch.from_numpy(image),
            patch_size,
            config["model"]["num_classes"],
            overlap,
            device,
        )
        rows.append(
            {
                "case_id": case_id,
                "edema_dice": edema_dice(pred, seg),
                "wt_dice": whole_tumor_dice(pred, seg),
                "edema_voxels_gt": int((seg == CLS_EDEMA).sum()),
            }
        )
    return pd.DataFrame(rows)


def select_feasibility_split(
    scored: pd.DataFrame,
    n_train: int,
    n_dev: int,
    n_test: int,
    seed: int,
) -> dict[str, list[str]]:
    ranked = scored.sort_values("edema_dice").reset_index(drop=True)
    n = len(ranked)
    if n < n_train + n_dev + n_test:
        raise ValueError(f"Need at least {n_train+n_dev+n_test} scored cases, got {n}")
    bins = [ranked.iloc[: n // 3], ranked.iloc[n // 3 : 2 * n // 3], ranked.iloc[2 * n // 3 :]]
    rng = np.random.default_rng(seed)
    # Distribute remainders so totals match exactly.
    targets = {"train": n_train, "dev": n_dev, "test": n_test}
    base = {k: v // 3 for k, v in targets.items()}
    rem = {k: v - 3 * base[k] for k, v in targets.items()}
    out = {"train": [], "dev": [], "test": []}
    for bidx, sub in enumerate(bins):
        need = base["train"] + base["dev"] + base["test"]
        # give remainder slots to first bins
        extra = []
        for name in ("train", "dev", "test"):
            if rem[name] > 0:
                extra.append(name)
                rem[name] -= 1
                need += 1
        idx = rng.choice(len(sub), size=min(need, len(sub)), replace=False)
        chosen = sub.iloc[idx]["case_id"].tolist()
        cursor = 0
        n_tr = base["train"] + (1 if "train" in extra else 0)
        n_dv = base["dev"] + (1 if "dev" in extra else 0)
        n_te = base["test"] + (1 if "test" in extra else 0)
        out["train"].extend(chosen[cursor : cursor + n_tr])
        cursor += n_tr
        out["dev"].extend(chosen[cursor : cursor + n_dv])
        cursor += n_dv
        out["test"].extend(chosen[cursor : cursor + n_te])
    return out


def repair_logits_from_decoder1(
    model: UNet3D,
    repair: SpatialEdemaRepair,
    image_batch: torch.Tensor,
    support: torch.Tensor,
    enabled: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward to decoder1, apply repair, seg_head. Returns logits, h, gate."""
    h = model.forward_to_decoder1(image_batch)
    h_edit, gate = repair(h, support, enabled=enabled)
    logits = model.forward_from_decoder1(h_edit)
    return logits, h, gate


def compute_train_loss(
    model: UNet3D,
    repair: SpatialEdemaRepair,
    image: torch.Tensor,
    target: torch.Tensor,
    baseline_logits: torch.Tensor,
    support: torch.Tensor,
    criterion: DiceCrossEntropyLoss,
    lambdas: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    logits, _h, gate = repair_logits_from_decoder1(model, repair, image, support, enabled=True)
    l_seg = criterion(logits, target)

    probs_e = torch.softmax(logits, dim=1)
    probs_b = torch.softmax(baseline_logits, dim=1).detach()
    outside = (1.0 - support).clamp(0, 1)
    # KL(edited || baseline) outside support, mean over voxels
    kl = (probs_e * (torch.log(probs_e.clamp_min(1e-8)) - torch.log(probs_b.clamp_min(1e-8)))).sum(dim=1)
    denom = outside.sum().clamp_min(1.0)
    l_fid = (kl * outside.squeeze(1)).sum() / denom

    l_sparse = (gate.abs() * support).mean()
    l_dir = (repair.delta ** 2).sum()

    soft_e = soft_volumes_from_logits(logits[0])
    soft_b = soft_volumes_from_logits(baseline_logits[0])
    wt_support = support.sum().clamp_min(1.0)
    l_off = (
        (soft_e["enhancing"] - soft_b["enhancing"]).abs()
        + (soft_e["necrosis"] - soft_b["necrosis"]).abs()
    ) / wt_support

    loss = (
        l_seg
        + lambdas["fidelity"] * l_fid
        + lambdas["sparse"] * l_sparse
        + lambdas["direction"] * l_dir
        + lambdas["off_target"] * l_off
    )
    stats = {
        "loss": float(loss.detach()),
        "l_seg": float(l_seg.detach()),
        "l_fid": float(l_fid.detach()),
        "l_sparse": float(l_sparse.detach()),
        "l_dir": float(l_dir.detach()),
        "l_off": float(l_off.detach()),
        "scale": float(repair.scale.detach()),
        "cos": float(repair.cosine_to_initial().detach()),
    }
    return loss, stats


@torch.no_grad()
def sliding_window_repair_inference(
    model: UNet3D,
    repair: SpatialEdemaRepair | None,
    image: torch.Tensor,
    patch_size: tuple[int, int, int],
    num_classes: int,
    overlap: float,
    device: torch.device,
    mode: str = "proposed",
    direction: torch.Tensor | None = None,
    alpha: float = 0.0,
    logit_bias: float = 0.0,
    match_rms: float | None = None,
) -> dict[str, Any]:
    """
    Full-volume inference for baselines and proposed repair.

    modes:
      baseline | proposed | global_probe | global_output | logit_bias | random_gated
    """
    patch_size = tuple(int(s) for s in patch_size)
    image_b = image.unsqueeze(0).to(device)
    _, _, depth, height, width = image_b.shape
    stride = tuple(max(1, int(p * (1.0 - overlap))) for p in patch_size)

    logits_sum = torch.zeros((num_classes, depth, height, width), device=device)
    count_map = torch.zeros((1, depth, height, width), device=device)
    gate_sum = torch.zeros((1, depth, height, width), device=device)
    support_sum = torch.zeros((1, depth, height, width), device=device)
    base_logits_sum = torch.zeros((num_classes, depth, height, width), device=device)

    origins = list(
        _patch_indices(
            _axis_steps(depth, patch_size[0], stride[0]),
            _axis_steps(height, patch_size[1], stride[1]),
            _axis_steps(width, patch_size[2], stride[2]),
        )
    )

    model.eval()
    if repair is not None:
        repair.eval()

    t0 = time.perf_counter()
    for z0, y0, x0 in origins:
        patch = _extract_patch(image_b, z0, y0, x0, patch_size)
        h = model.forward_to_decoder1(patch)
        base_logits = model.forward_from_decoder1(h)
        support = build_support_mask(base_logits[0], dilate_radius=2).view(1, 1, *base_logits.shape[2:])

        if mode == "baseline":
            logits = base_logits
            gate = torch.zeros_like(support)
        elif mode == "proposed":
            assert repair is not None
            h_edit, gate = repair(h, support, enabled=True)
            logits = model.forward_from_decoder1(h_edit)
        elif mode in ("global_probe", "global_output", "random_gated"):
            assert direction is not None
            d = direction.view(1, -1, 1, 1, 1).to(device=h.device, dtype=h.dtype)
            if mode == "random_gated":
                assert repair is not None
                gate = torch.tanh(repair.gate(h)).detach()
                s = repair.scale.detach()
                # match RMS of proposed perturbation on this patch
                prop_edit = (support * s * gate * repair.direction().view(1, -1, 1, 1, 1)).detach()
                rnd_edit = support * gate * d
                if match_rms is not None:
                    scale_r = match_rms
                else:
                    rms_p = prop_edit.pow(2).mean().sqrt().clamp_min(1e-8)
                    rms_r = rnd_edit.pow(2).mean().sqrt().clamp_min(1e-8)
                    scale_r = (rms_p / rms_r).item()
                h_edit = h + rnd_edit * scale_r
                logits = model.forward_from_decoder1(h_edit)
            else:
                gate = torch.zeros_like(support)
                h_edit = h + float(alpha) * d
                logits = model.forward_from_decoder1(h_edit)
        elif mode == "logit_bias":
            logits = base_logits.clone()
            logits[:, CLS_EDEMA] = logits[:, CLS_EDEMA] + float(logit_bias)
            gate = torch.zeros_like(support)
        else:
            raise ValueError(mode)

        vd = min(patch_size[0], depth - z0)
        vh = min(patch_size[1], height - y0)
        vw = min(patch_size[2], width - x0)
        z1, y1, x1 = z0 + vd, y0 + vh, x0 + vw
        logits_sum[:, z0:z1, y0:y1, x0:x1] += logits[0, :, :vd, :vh, :vw]
        base_logits_sum[:, z0:z1, y0:y1, x0:x1] += base_logits[0, :, :vd, :vh, :vw]
        count_map[:, z0:z1, y0:y1, x0:x1] += 1.0
        gate_sum[:, z0:z1, y0:y1, x0:x1] += gate[0, :, :vd, :vh, :vw].abs()
        support_sum[:, z0:z1, y0:y1, x0:x1] += support[0, :, :vd, :vh, :vw]

    logits = logits_sum / count_map.clamp_min(1.0)
    base_logits = base_logits_sum / count_map.clamp_min(1.0)
    gate_vol = gate_sum / count_map.clamp_min(1.0)
    support_vol = (support_sum / count_map.clamp_min(1.0) > 0.5).float()
    runtime = time.perf_counter() - t0

    pred = predict_segmentation(logits)
    base_pred = predict_segmentation(base_logits)
    probs = torch.softmax(logits, dim=0)
    base_probs = torch.softmax(base_logits, dim=0)
    edema_prob_diff = (probs[CLS_EDEMA] - base_probs[CLS_EDEMA]).cpu().numpy()

    # perturbation RMS relative to baseline decoder path approximated via logit change
    pert_rms = float((logits - base_logits).pow(2).mean().sqrt().cpu())
    n_edited = int(((gate_vol > 1e-3) & (support_vol > 0.5)).sum().cpu())
    mean_abs_gate = float(gate_vol.mean().cpu())

    return {
        "logits": logits.cpu(),
        "baseline_logits": base_logits.cpu(),
        "pred": pred,
        "baseline_pred": base_pred,
        "gate": gate_vol[0].cpu().numpy(),
        "support": support_vol[0].cpu().numpy(),
        "edema_prob_diff": edema_prob_diff,
        "runtime_s": runtime,
        "n_edited_voxels": n_edited,
        "mean_abs_gate": mean_abs_gate,
        "perturbation_rms": pert_rms,
        "soft": {k: float(v.cpu()) for k, v in soft_volumes_from_logits(logits).items()},
        "soft_baseline": {k: float(v.cpu()) for k, v in soft_volumes_from_logits(base_logits).items()},
    }


def metrics_for_prediction(pred: np.ndarray, gt: np.ndarray, soft: dict, soft_b: dict) -> dict[str, float]:
    return {
        "edema_dice": edema_dice(pred, gt),
        "wt_dice": whole_tumor_dice(pred, gt),
        "tc_dice": tumor_core_dice(pred, gt),
        "et_dice": enhancing_dice(pred, gt),
        "mean_fg_dice": mean_foreground_dice(pred, gt),
        "abs_edema_volume_error": abs(int((pred == CLS_EDEMA).sum()) - int((gt == CLS_EDEMA).sum())),
        "edema_fp": int(((pred == CLS_EDEMA) & (gt != CLS_EDEMA)).sum()),
        "edema_fn": int(((pred != CLS_EDEMA) & (gt == CLS_EDEMA)).sum()),
        "enhancing_volume_change": soft["enhancing"] - soft_b["enhancing"],
        "necrosis_volume_change": soft["necrosis"] - soft_b["necrosis"],
        "wt_volume_change": soft["wt"] - soft_b["wt"],
    }
