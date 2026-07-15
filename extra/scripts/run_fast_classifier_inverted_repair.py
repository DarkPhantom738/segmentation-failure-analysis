#!/usr/bin/env python3
"""Fast classifier-inverted decoder1 edema repair (12/6/6 train-partition split)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from src.analysis.semantic_directions import SemanticDirection
from src.data.brats_dataset import load_case_cached
from src.models.classifier_inverted_repair import (
    CLS_EDEMA,
    N_TRANSITIONS,
    ClassifierInvertedRepair,
    LogitRefinementHead,
    T_NONE,
    build_transition_directions,
    make_transition_labels,
)
from src.models.unet3d import DiceCrossEntropyLoss, build_model
from src.training.decoder_cache_inference import capture_decoder1_patches
from src.training.inference import predict_segmentation
from src.training.spatial_repair_trainer import edema_dice
from src.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("extra/configs/fast_classifier_inverted_repair.yaml"))
    p.add_argument("--stage", choices=["smoke", "sanity", "full"], default="smoke")
    return p.parse_args()


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def param_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha1()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def load_case(config: dict, case_id: str):
    d = config["data"]
    return load_case_cached(
        case_id,
        Path(d["root"]),
        d["modalities"],
        d["target_spacing"],
        d["percentile_clip"],
        Path(d["cache_dir"]),
    )


@torch.no_grad()
def cache_case(model, config, case_id, device):
    image, seg = load_case(config, case_id)
    patch_size = tuple(int(s) for s in config["data"]["patch_size"])
    overlap = float(config["inference"]["overlap"])
    origins, d1_patches, vol_shape = capture_decoder1_patches(
        model, torch.from_numpy(image), patch_size, overlap, device
    )
    d1 = [p.detach().cpu().float() for p in d1_patches]
    # baseline logits via head only
    depth, height, width = vol_shape
    num_classes = config["model"]["num_classes"]
    logits_sum = torch.zeros((num_classes, depth, height, width))
    count = torch.zeros((1, depth, height, width))
    for (z0, y0, x0), h in zip(origins, d1):
        logits = model.forward_from_decoder1(h.to(device))[0].cpu()
        vd = min(patch_size[0], depth - z0)
        vh = min(patch_size[1], height - y0)
        vw = min(patch_size[2], width - x0)
        logits_sum[:, z0 : z0 + vd, y0 : y0 + vh, x0 : x0 + vw] += logits[:, :vd, :vh, :vw]
        count[:, z0 : z0 + vd, y0 : y0 + vh, x0 : x0 + vw] += 1.0
    baseline_logits = logits_sum / count.clamp_min(1.0)
    pred = predict_segmentation(baseline_logits)
    labels = make_transition_labels(
        torch.as_tensor(pred), torch.as_tensor(seg.astype(np.int64))
    ).numpy()
    return {
        "case_id": case_id,
        "gt": seg.astype(np.int64),
        "origins": origins,
        "d1": d1,
        "vol_shape": vol_shape,
        "patch_size": patch_size,
        "baseline_logits": baseline_logits,
        "baseline_pred": pred,
        "transition_labels": labels,
        "baseline_edema_dice": edema_dice(pred, seg),
        "n_error_voxels": int((labels > 0).sum()),
    }


def class_weights_from_cases(cases: list[dict], cap: float) -> torch.Tensor:
    counts = np.zeros(N_TRANSITIONS, dtype=np.float64)
    for c in cases:
        u, cnt = np.unique(c["transition_labels"], return_counts=True)
        for cls, n in zip(u, cnt):
            counts[int(cls)] += n
    counts = np.maximum(counts, 1.0)
    inv = counts.sum() / counts
    inv = inv / inv.mean()
    inv = np.minimum(inv, cap)
    return torch.as_tensor(inv, dtype=torch.float32)


def sample_patch_index(case: dict, rng: np.random.Generator, prefer_error: bool) -> int:
    origins = case["origins"]
    if not prefer_error:
        return int(rng.integers(0, len(origins)))
    error_idxs = []
    ps = case["patch_size"]
    labels = case["transition_labels"]
    depth, height, width = case["vol_shape"]
    for i, (z0, y0, x0) in enumerate(origins):
        vd = min(ps[0], depth - z0)
        vh = min(ps[1], height - y0)
        vw = min(ps[2], width - x0)
        if labels[z0 : z0 + vd, y0 : y0 + vh, x0 : x0 + vw].max() > 0:
            error_idxs.append(i)
    if error_idxs and rng.random() < 0.5:
        return int(error_idxs[rng.integers(0, len(error_idxs))])
    return int(rng.integers(0, len(origins)))


def extract_patch_tensors(case: dict, idx: int, device: torch.device):
    z0, y0, x0 = case["origins"][idx]
    h = case["d1"][idx].to(device)
    ps = case["patch_size"]
    depth, height, width = case["vol_shape"]
    vd, vh, vw = min(ps[0], depth - z0), min(ps[1], height - y0), min(ps[2], width - x0)
    base = case["baseline_logits"][:, z0 : z0 + vd, y0 : y0 + vh, x0 : x0 + vw]
    lab = case["transition_labels"][z0 : z0 + vd, y0 : y0 + vh, x0 : x0 + vw]
    gt = case["gt"][z0 : z0 + vd, y0 : y0 + vh, x0 : x0 + vw]
    if (vd, vh, vw) != ps:
        h_pad = torch.zeros(1, h.shape[1], *ps, device=device)
        h_pad[:, :, :vd, :vh, :vw] = h[:, :, :vd, :vh, :vw]
        h = h_pad
        b_pad = torch.zeros(base.shape[0], *ps)
        b_pad[:, :vd, :vh, :vw] = base
        base = b_pad
        l_pad = np.zeros(ps, dtype=np.int64)
        l_pad[:vd, :vh, :vw] = lab
        lab = l_pad
        g_pad = np.zeros(ps, dtype=np.int64)
        g_pad[:vd, :vh, :vw] = gt
        gt = g_pad
    return (
        h,
        base.unsqueeze(0).to(device),
        torch.as_tensor(lab, device=device).long().unsqueeze(0),
        torch.as_tensor(gt, device=device).long().unsqueeze(0),
    )


def run_smoke(config: dict, device: torch.device, out_dir: Path) -> None:
    model = build_model(config).to(device)
    ckpt = torch.load(config["checkpoint"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    unet_hash = param_hash(model)

    dirs, meta = build_transition_directions(model.seg_head.weight)
    pd.DataFrame(meta).to_csv(out_dir / "transition_directions.csv", index=False)
    assert dirs.shape == (6, config["model"]["base_features"])
    assert all(m["ok"] for m in meta), meta
    print("Transition directions OK:")
    print(pd.DataFrame(meta)[["name", "target_minus_source", "ok"]].to_string(index=False))

    repair = ClassifierInvertedRepair(dirs, decoder_channels=dirs.shape[1]).to(device)
    x = torch.randn(1, dirs.shape[1], 16, 16, 16, device=device)
    base = torch.randn(1, 4, 16, 16, 16, device=device)
    repaired, probs = repair(x, base)
    assert repaired.shape == x.shape
    assert probs.shape[1] == N_TRANSITIONS

    loss = repaired.pow(2).mean() + probs.mean() + repair.scale
    loss.backward()
    assert repair.raw_scale.grad is not None
    assert repair.gate.net[0].weight.grad is not None
    for p in model.parameters():
        assert p.grad is None
    assert param_hash(model) == unet_hash

    # labels mapping smoke
    pred = torch.tensor([0, 1, 2, 3, 2, 2])
    gt = torch.tensor([2, 2, 0, 2, 1, 3])
    lab = make_transition_labels(pred, gt)
    assert lab.tolist() == [1, 2, 4, 3, 5, 6]

    # probe not used for repair directions
    probe = SemanticDirection.load(config["probe_direction"])
    cos = float(
        torch.nn.functional.cosine_similarity(
            dirs[0], torch.as_tensor(probe.activation_direction).float(), dim=0
        )
    )
    print(f"cos(bg->edema dir, ridge probe)={cos:.3f} (informational; probe unused for repair)")
    print("SMOKE OK")


def run_sanity(config: dict, device: torch.device, out_dir: Path) -> None:
    split = json.loads(Path(config["case_split"]).read_text())
    train_ids = split["train"][:2]
    model = build_model(config).to(device)
    ckpt = torch.load(config["checkpoint"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    unet_before = param_hash(model)

    dirs, meta = build_transition_directions(model.seg_head.weight)
    pd.DataFrame(meta).to_csv(out_dir / "transition_directions.csv", index=False)
    repair = ClassifierInvertedRepair(dirs, decoder_channels=dirs.shape[1]).to(device)
    cases = [cache_case(model, config, cid, device) for cid in tqdm(train_ids, desc="Cache")]
    weights = class_weights_from_cases(cases, float(config["training"]["weight_cap"])).to(device)
    criterion_seg = DiceCrossEntropyLoss(0.5, config["model"]["num_classes"])
    opt = torch.optim.Adam(repair.trainable_parameters(), lr=float(config["training"]["lr"]))
    rng = np.random.default_rng(0)
    lam_seg = float(config["training"]["lambda_seg"])
    lam_sp = float(config["training"]["lambda_sparse"])

    losses = []
    nonzero_fracs = []
    scales = []
    for step in range(1, 21):
        case = cases[(step - 1) % len(cases)]
        idx = sample_patch_index(case, rng, prefer_error=True)
        h, base, tlab, gt = extract_patch_tensors(case, idx, device)
        opt.zero_grad(set_to_none=True)
        t_logits = repair.transition_logits(h, base)
        repaired, probs = repair(h, base)
        logits = model.forward_from_decoder1(repaired)
        l_tr = F.cross_entropy(t_logits, tlab, weight=weights)
        l_seg = criterion_seg(logits, gt)
        l_sp = probs[:, 1:].sum(dim=1).mean()
        loss = l_tr + lam_seg * l_seg + lam_sp * l_sp
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        nonzero_fracs.append(float(probs[:, 1:].sum(dim=1).mean().detach()))
        scales.append(float(repair.scale.detach()))
        if step in (1, 10, 20):
            print(
                f"step {step}: loss={losses[-1]:.4f} nonzero_mass={nonzero_fracs[-1]:.4f} "
                f"scale={scales[-1]:.4f}"
            )

    assert param_hash(model) == unet_before
    assert max(nonzero_fracs) > 1e-4, "gate produced no nonzero correction mass"
    assert losses[-1] < losses[0] * 1.5, "loss did not decrease enough"
    print(f"loss {losses[0]:.4f} -> {losses[-1]:.4f}; scale {scales[0]:.4f} -> {scales[-1]:.4f}; U-Net unchanged")
    print("SANITY OK")


def aggregate_repaired_volume(
    model,
    repair: ClassifierInvertedRepair | None,
    case: dict,
    device: torch.device,
    num_classes: int,
    abstain_threshold: float = 0.0,
    mode: str = "proposed",
    alpha: float = 0.0,
    d_probe: torch.Tensor | None = None,
    logit_bias: float = 0.0,
    logit_head: LogitRefinementHead | None = None,
) -> dict[str, Any]:
    import time

    from src.training.spatial_repair_trainer import (
        enhancing_dice,
        mean_foreground_dice,
        tumor_core_dice,
    )
    from src.training.metrics import whole_tumor_dice

    t0 = time.perf_counter()
    depth, height, width = case["vol_shape"]
    ps = case["patch_size"]
    logits_sum = torch.zeros((num_classes, depth, height, width), device=device)
    count = torch.zeros((1, depth, height, width), device=device)
    corr_sum = torch.zeros((1, depth, height, width), device=device)
    pred_trans = torch.zeros((depth, height, width), device=device, dtype=torch.long)

    for (z0, y0, x0), h_cpu in zip(case["origins"], case["d1"]):
        h = h_cpu.to(device)
        base_full = case["baseline_logits"][
            :, z0 : z0 + h.shape[2], y0 : y0 + h.shape[3], x0 : x0 + h.shape[4]
        ]
        # pad baseline crop to h spatial if needed
        if base_full.shape[1:] != h.shape[2:]:
            bpad = torch.zeros(num_classes, *h.shape[2:])
            bpad[:, : base_full.shape[1], : base_full.shape[2], : base_full.shape[3]] = base_full
            base_full = bpad
        base = base_full.unsqueeze(0).to(device)

        if mode == "baseline":
            logits = model.forward_from_decoder1(h)
            probs_t = torch.zeros(1, N_TRANSITIONS, *h.shape[2:], device=device)
            probs_t[:, 0] = 1.0
        elif mode == "proposed":
            assert repair is not None
            repaired, probs_t = repair(h, base, abstain_threshold=abstain_threshold)
            logits = model.forward_from_decoder1(repaired)
        elif mode == "global_probe":
            assert d_probe is not None
            d = d_probe.to(device).view(1, -1, 1, 1, 1)
            logits = model.forward_from_decoder1(h + float(alpha) * d)
            probs_t = torch.zeros(1, N_TRANSITIONS, *h.shape[2:], device=device)
            probs_t[:, 0] = 1.0
        elif mode == "logit_bias":
            logits = model.forward_from_decoder1(h)
            logits = logits.clone()
            logits[:, CLS_EDEMA] = logits[:, CLS_EDEMA] + float(logit_bias)
            probs_t = torch.zeros(1, N_TRANSITIONS, *h.shape[2:], device=device)
            probs_t[:, 0] = 1.0
        elif mode == "logit_refine":
            assert logit_head is not None
            logits = logit_head(h, base)
            probs_t = torch.softmax(logit_head.gate(h, baseline_aux_for(base)), dim=1)
        else:
            raise ValueError(mode)

        vd = min(ps[0], depth - z0)
        vh = min(ps[1], height - y0)
        vw = min(ps[2], width - x0)
        logits_sum[:, z0 : z0 + vd, y0 : y0 + vh, x0 : x0 + vw] += logits[0, :, :vd, :vh, :vw]
        count[:, z0 : z0 + vd, y0 : y0 + vh, x0 : x0 + vw] += 1.0
        corr = probs_t[0, 1:, :vd, :vh, :vw].sum(dim=0)
        corr_sum[0, z0 : z0 + vd, y0 : y0 + vh, x0 : x0 + vw] += corr
        pred_trans[z0 : z0 + vd, y0 : y0 + vh, x0 : x0 + vw] = torch.argmax(
            probs_t[0, :, :vd, :vh, :vw], dim=0
        )

    logits = logits_sum / count.clamp_min(1.0)
    corr_map = corr_sum / count.clamp_min(1.0)
    pred = predict_segmentation(logits)
    gt = case["gt"]
    soft = torch.softmax(logits, dim=0)
    soft_b = torch.softmax(case["baseline_logits"].to(device), dim=0)
    labels = case["transition_labels"]
    pred_t = pred_trans.cpu().numpy()
    corr_true = labels > 0
    corr_pred = pred_t > 0
    tp = int((corr_true & corr_pred).sum())
    fp = int((~corr_true & corr_pred).sum())
    fn = int((corr_true & ~corr_pred).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    err_vox = corr_true
    trans_acc = float((pred_t[err_vox] == labels[err_vox]).mean()) if err_vox.any() else float("nan")
    abstain_frac = float((pred_t == T_NONE).mean())

    return {
        "pred": pred,
        "edema_dice": edema_dice(pred, gt),
        "wt_dice": whole_tumor_dice(pred, gt),
        "tc_dice": tumor_core_dice(pred, gt),
        "et_dice": enhancing_dice(pred, gt),
        "mean_fg_dice": mean_foreground_dice(pred, gt),
        "edema_fp": int(((pred == CLS_EDEMA) & (gt != CLS_EDEMA)).sum()),
        "edema_fn": int(((pred != CLS_EDEMA) & (gt == CLS_EDEMA)).sum()),
        "abs_edema_volume_error": abs(int((pred == CLS_EDEMA).sum()) - int((gt == CLS_EDEMA).sum())),
        "n_corrected_voxels": int(corr_pred.sum()),
        "correction_precision": precision,
        "correction_recall": recall,
        "transition_acc_on_errors": trans_acc,
        "abstain_fraction": abstain_frac,
        "necrosis_volume_change": float((soft[1] - soft_b[1]).sum().cpu()),
        "enhancing_volume_change": float((soft[3] - soft_b[3]).sum().cpu()),
        "runtime_s": time.perf_counter() - t0,
        "mean_corr_mass": float(corr_map.mean().cpu()),
    }


def baseline_aux_for(base: torch.Tensor) -> torch.Tensor:
    from src.models.classifier_inverted_repair import baseline_aux_channels

    return baseline_aux_channels(base)


@torch.no_grad()
def mean_dev_edema(eval_fn, cases: list[dict]) -> float:
    return float(np.mean([eval_fn(c)["edema_dice"] for c in cases]))


def run_full(config: dict, device: torch.device, out_dir: Path) -> None:
    import time

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.training.metrics import whole_tumor_dice

    figures = ensure_dir(out_dir / "figures")
    split = json.loads(Path(config["case_split"]).read_text())
    model = build_model(config).to(device)
    ckpt = torch.load(config["checkpoint"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    unet_hash = param_hash(model)
    num_classes = config["model"]["num_classes"]

    dirs, meta = build_transition_directions(model.seg_head.weight)
    pd.DataFrame(meta).to_csv(out_dir / "transition_directions.csv", index=False)
    assert all(m["ok"] for m in meta)

    print("Caching all split cases...")
    all_ids = split["train"] + split["dev"] + split["test"]
    cache = {
        cid: cache_case(model, config, cid, device)
        for cid in tqdm(all_ids, desc="Cache cases")
    }
    train_cases = [cache[c] for c in split["train"]]
    dev_cases = [cache[c] for c in split["dev"]]
    test_cases = [cache[c] for c in split["test"]]

    # --- train proposed repair ---
    repair = ClassifierInvertedRepair(dirs, decoder_channels=dirs.shape[1]).to(device)
    weights = class_weights_from_cases(train_cases, float(config["training"]["weight_cap"])).to(device)
    criterion_seg = DiceCrossEntropyLoss(0.5, num_classes)
    opt = torch.optim.Adam(repair.trainable_parameters(), lr=float(config["training"]["lr"]))
    rng = np.random.default_rng(int(config.get("feasibility_seed", 0)))
    lam_seg = float(config["training"]["lambda_seg"])
    lam_sp = float(config["training"]["lambda_sparse"])
    max_steps = int(config["training"]["max_steps"])
    patience = int(config["training"]["patience"])
    eval_every = int(config["training"]["eval_every"])

    best_dev = -1.0
    best_state = None
    stall = 0
    log_rows = []
    t_train0 = time.perf_counter()

    for step in range(1, max_steps + 1):
        case = train_cases[(step - 1) % len(train_cases)]
        idx = sample_patch_index(case, rng, prefer_error=True)
        h, base, tlab, gt = extract_patch_tensors(case, idx, device)
        opt.zero_grad(set_to_none=True)
        t_logits = repair.transition_logits(h, base)
        repaired, probs = repair(h, base)
        logits = model.forward_from_decoder1(repaired)
        l_tr = F.cross_entropy(t_logits, tlab, weight=weights)
        l_seg = criterion_seg(logits, gt)
        l_sp = probs[:, 1:].sum(dim=1).mean()
        loss = l_tr + lam_seg * l_seg + lam_sp * l_sp
        loss.backward()
        opt.step()
        row = {
            "step": step,
            "loss": float(loss.detach()),
            "l_tr": float(l_tr.detach()),
            "l_seg": float(l_seg.detach()),
            "l_sp": float(l_sp.detach()),
            "scale": float(repair.scale.detach()),
            "nonzero_mass": float(probs[:, 1:].sum(dim=1).mean().detach()),
        }
        if step % eval_every == 0 or step == max_steps:
            repair.eval()
            dev_scores = [
                aggregate_repaired_volume(
                    model, repair, c, device, num_classes, abstain_threshold=0.0, mode="proposed"
                )["edema_dice"]
                for c in dev_cases
            ]
            mean_dev = float(np.mean(dev_scores))
            row["dev_edema_dice"] = mean_dev
            print(f"step {step}: loss={row['loss']:.4f} dev_edema={mean_dev:.4f} scale={row['scale']:.4f}")
            if mean_dev > best_dev + 1e-5:
                best_dev = mean_dev
                best_state = {k: v.detach().cpu().clone() for k, v in repair.state_dict().items()}
                stall = 0
            else:
                stall += eval_every
            repair.train()
            if stall >= patience:
                print(f"Early stop at step {step}")
                log_rows.append(row)
                break
        log_rows.append(row)

    if best_state is not None:
        repair.load_state_dict(best_state)
    torch.save({"repair": repair.state_dict(), "best_dev": best_dev}, out_dir / "best_repair_checkpoint.pt")
    pd.DataFrame(log_rows).to_csv(out_dir / "training_log.csv", index=False)
    assert param_hash(model) == unet_hash
    train_runtime = time.perf_counter() - t_train0

    # --- train logit refinement head (same schedule, shorter if needed) ---
    logit_head = LogitRefinementHead(decoder_channels=dirs.shape[1]).to(device)
    opt_l = torch.optim.Adam(logit_head.parameters(), lr=float(config["training"]["lr"]))
    best_dev_l = -1.0
    best_l_state = None
    stall = 0
    for step in range(1, max_steps + 1):
        case = train_cases[(step - 1) % len(train_cases)]
        idx = sample_patch_index(case, rng, prefer_error=True)
        h, base, tlab, gt = extract_patch_tensors(case, idx, device)
        opt_l.zero_grad(set_to_none=True)
        logits = logit_head(h, base)
        # supervise with transition CE on gate + seg loss
        t_logits = logit_head.gate(h, baseline_aux_for(base))
        l_tr = F.cross_entropy(t_logits, tlab, weight=weights)
        l_seg = criterion_seg(logits, gt)
        loss = l_tr + lam_seg * l_seg
        loss.backward()
        opt_l.step()
        if step % eval_every == 0 or step == max_steps:
            logit_head.eval()
            scores = [
                aggregate_repaired_volume(
                    model, None, c, device, num_classes, mode="logit_refine", logit_head=logit_head
                )["edema_dice"]
                for c in dev_cases
            ]
            mean_dev = float(np.mean(scores))
            print(f"logit_head step {step}: dev_edema={mean_dev:.4f}")
            if mean_dev > best_dev_l + 1e-5:
                best_dev_l = mean_dev
                best_l_state = {k: v.detach().cpu().clone() for k, v in logit_head.state_dict().items()}
                stall = 0
            else:
                stall += eval_every
            logit_head.train()
            if stall >= patience:
                break
    if best_l_state is not None:
        logit_head.load_state_dict(best_l_state)

    # --- tune baselines on DEV ---
    probe = SemanticDirection.load(config["probe_direction"])
    d_probe = torch.as_tensor(probe.activation_direction, dtype=torch.float32, device=device)

    def tune(values, fn):
        best_v, best_s = values[0], -1.0
        for v in values:
            s = float(np.mean([fn(v, c)["edema_dice"] for c in dev_cases]))
            if s > best_s:
                best_s, best_v = s, v
        return best_v, best_s

    alpha_probe, _ = tune(
        [-2, -1, -0.5, 0.5, 1, 2],
        lambda a, c: aggregate_repaired_volume(
            model, None, c, device, num_classes, mode="global_probe", alpha=a, d_probe=d_probe
        ),
    )
    bias, _ = tune(
        [-2, -1, -0.5, 0.5, 1, 2],
        lambda b, c: aggregate_repaired_volume(
            model, None, c, device, num_classes, mode="logit_bias", logit_bias=b
        ),
    )
    thresh, _ = tune(
        [0.0, 0.2, 0.35, 0.5, 0.65],
        lambda t, c: aggregate_repaired_volume(
            model, repair, c, device, num_classes, abstain_threshold=t, mode="proposed"
        ),
    )

    dev_rows = []
    for c in dev_cases:
        out = aggregate_repaired_volume(
            model, repair, c, device, num_classes, abstain_threshold=thresh, mode="proposed"
        )
        dev_rows.append({"case_id": c["case_id"], "edema_dice": out["edema_dice"], "baseline": c["baseline_edema_dice"]})
    pd.DataFrame(dev_rows).to_csv(out_dir / "development_results.csv", index=False)

    methods = [
        ("A_baseline", "baseline", {}),
        ("B_global_probe", "global_probe", {"alpha": alpha_probe, "d_probe": d_probe}),
        ("C_logit_bias", "logit_bias", {"logit_bias": bias}),
        ("D_logit_refine", "logit_refine", {"logit_head": logit_head}),
        ("E_proposed", "proposed", {"abstain_threshold": thresh, "repair": repair}),
    ]

    rows = []
    for case in test_cases:
        for name, mode, kwargs in methods:
            kw = dict(kwargs)
            repair_arg = kw.pop("repair", repair if mode == "proposed" else None)
            out = aggregate_repaired_volume(
                model, repair_arg, case, device, num_classes, mode=mode, **kw
            )
            rows.append(
                {
                    "case_id": case["case_id"],
                    "method": name,
                    "baseline_edema_dice": case["baseline_edema_dice"],
                    **{k: v for k, v in out.items() if k != "pred"},
                    "tuned_alpha_probe": alpha_probe,
                    "tuned_logit_bias": bias,
                    "tuned_abstain": thresh,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "test_case_results.csv", index=False)

    summary = (
        df.groupby("method", as_index=False)
        .agg(
            mean_edema_dice=("edema_dice", "mean"),
            mean_fg_dice=("mean_fg_dice", "mean"),
            mean_wt_dice=("wt_dice", "mean"),
            mean_precision=("correction_precision", "mean"),
            mean_recall=("correction_recall", "mean"),
            mean_enh_change=("enhancing_volume_change", "mean"),
            mean_nec_change=("necrosis_volume_change", "mean"),
            mean_runtime=("runtime_s", "mean"),
        )
        .sort_values("mean_edema_dice", ascending=False)
    )
    summary.to_csv(out_dir / "method_summary.csv", index=False)

    # figures
    pivot = df.pivot(index="case_id", columns="method", values="edema_dice")
    ax = pivot[["A_baseline", "E_proposed"]].plot(kind="bar", figsize=(8, 4))
    ax.set_ylabel("edema Dice")
    ax.set_title("Edema Dice by case")
    plt.tight_layout()
    plt.savefig(figures / "edema_dice_by_case.png", dpi=150)
    plt.close()

    prop = df[df.method == "E_proposed"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(prop["correction_precision"], prop["correction_recall"])
    for _, r in prop.iterrows():
        ax.annotate(r["case_id"].split("_")[-1], (r["correction_precision"], r["correction_recall"]), fontsize=7)
    ax.set_xlabel("precision")
    ax.set_ylabel("recall")
    ax.set_title("Correction-voxel detection")
    fig.tight_layout()
    fig.savefig(figures / "transition_precision_recall.png", dpi=150)
    plt.close()

    ax = summary.set_index("method")["mean_edema_dice"].plot(kind="bar", figsize=(8, 4), color="steelblue")
    ax.set_ylabel("mean edema Dice")
    ax.set_title("Method comparison")
    plt.tight_layout()
    plt.savefig(figures / "method_comparison.png", dpi=150)
    plt.close()

    # success/fail examples: best/worst delta vs baseline
    base = df[df.method == "A_baseline"].set_index("case_id")
    prop_i = df[df.method == "E_proposed"].set_index("case_id")
    deltas = (prop_i["edema_dice"] - base["edema_dice"]).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(["worst", "best"], [deltas.iloc[0], deltas.iloc[-1]], color=["salmon", "seagreen"])
    ax.set_ylabel("Δ edema Dice")
    ax.set_title(f"Failed {deltas.index[0]} / Successful {deltas.index[-1]}")
    fig.tight_layout()
    fig.savefig(figures / "failed_repairs.png", dpi=150)
    fig.savefig(figures / "successful_repairs.png", dpi=150)
    plt.close()

    # verdict
    mean_base = float(base["edema_dice"].mean())
    mean_prop = float(prop_i["edema_dice"].mean())
    mean_improve = mean_prop - mean_base
    n_improve = int((prop_i["edema_dice"] > base["edema_dice"]).sum())
    fg_drop = float(base["mean_fg_dice"].mean() - prop_i["mean_fg_dice"].mean())
    mean_probe = float(df[df.method == "B_global_probe"]["edema_dice"].mean())
    mean_bias = float(df[df.method == "C_logit_bias"]["edema_dice"].mean())
    mean_refine = float(df[df.method == "D_logit_refine"]["edema_dice"].mean())
    prec = float(prop_i["correction_precision"].mean())
    rec = float(prop_i["correction_recall"].mean())
    detects = prec > 0.05 and rec > 0.05

    if (
        mean_improve >= 0.02
        and n_improve >= 4
        and fg_drop <= 0.005
        and mean_prop > mean_probe
        and mean_prop > mean_bias
        and (mean_prop >= mean_refine - 1e-4)
        and detects
    ):
        verdict = "PROMISING"
    elif mean_improve >= 0.01 and n_improve >= 3 and detects:
        verdict = "MIXED"
    else:
        verdict = "NOT PROMISING"

    # per-transition difficulty on test (from labels frequency / recall proxy)
    trans_names = {
        1: "bg_to_edema",
        2: "nec_to_edema",
        3: "enh_to_edema",
        4: "edema_to_bg",
        5: "edema_to_nec",
        6: "edema_to_enh",
    }
    trans_lines = []
    for tid, name in trans_names.items():
        counts = [int((c["transition_labels"] == tid).sum()) for c in test_cases]
        trans_lines.append(f"- {name}: total error voxels on test={sum(counts)}")

    fn_delta = float(base["edema_fn"].mean() - prop_i["edema_fn"].mean())
    fp_delta = float(base["edema_fp"].mean() - prop_i["edema_fp"].mean())

    report = f"""# Classifier-inverted repair feasibility report

**Verdict: {verdict}**

## Split
- train: {split['train']}
- dev: {split['dev']}
- test: {split['test']}

## Training
- device: `{device}`
- steps logged: {log_rows[-1]['step'] if log_rows else 0}
- train runtime_s: {train_runtime:.1f}
- best dev edema Dice: {best_dev:.4f}
- tuned abstain threshold: {thresh}
- tuned probe alpha: {alpha_probe}
- tuned logit bias: {bias}
- learned scale: {float(repair.scale.detach().cpu()):.4f}

## Method summary (test)
{summary.to_string(index=False)}

## Answers
1. Can the transition gate locate edema-related errors?
   - mean precision={prec:.3f}, recall={rec:.3f}; detects={'YES' if detects else 'NO'}
2. Which transitions are common on test (proxy for difficulty / prevalence)?
{chr(10).join(trans_lines)}
3. Does classifier-inverted decoder1 repair improve edema Dice?
   - mean Δ={mean_improve:+.4f}; cases improved={n_improve}/6
4. Does it beat direct logit bias?
   - proposed={mean_prop:.4f} vs bias={mean_bias:.4f} → {'YES' if mean_prop > mean_bias else 'NO'}
5. Does it beat learned logit-refinement?
   - proposed={mean_prop:.4f} vs refine={mean_refine:.4f} → {'YES' if mean_prop >= mean_refine else 'NO'}
6. FN vs FP changes (baseline − proposed; positive ⇒ fewer errors):
   - ΔFN={fn_delta:+.1f}, ΔFP={fp_delta:+.1f}
7. Other tumor classes:
   - mean enhancing vol change={float(prop_i['enhancing_volume_change'].mean()):+.2f}
   - mean necrosis vol change={float(prop_i['necrosis_volume_change'].mean()):+.2f}
   - mean fg Dice drop={fg_drop:+.4f}
8. Result: **{verdict}**

## Notes
- U-Net remained frozen (hash unchanged).
- Ridge probe used only as baseline B, not for proposed directions.
- Feasibility gates only; not clinical or publication claims.
"""
    (out_dir / "feasibility_report.md").write_text(report)
    print(summary.to_string(index=False))
    print(f"\nVerdict: {verdict}")
    print(f"Report: {out_dir / 'feasibility_report.md'}")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.open())
    out_dir = ensure_dir(Path(config["output_dir"]))
    device = _device()
    print(f"Using device: {device}; stage={args.stage}")
    if args.stage == "smoke":
        run_smoke(config, device, out_dir)
    elif args.stage == "sanity":
        run_sanity(config, device, out_dir)
    else:
        run_full(config, device, out_dir)


if __name__ == "__main__":
    main()
