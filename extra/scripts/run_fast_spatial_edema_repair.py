#!/usr/bin/env python3
"""Fast feasibility: spatially gated decoder1 edema repair (train partition only)."""

from __future__ import annotations

import argparse
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
import yaml
from tqdm import tqdm

from src.analysis.semantic_directions import SemanticDirection
from src.data.brats_dataset import load_case_cached
from src.models.spatial_edema_repair import (
    CLS_EDEMA,
    SpatialEdemaRepair,
    build_d_initial,
    build_support_mask,
)
from src.models.unet3d import DiceCrossEntropyLoss, build_model
from src.training.spatial_repair_trainer import (
    baseline_volume_prediction,
    compute_train_loss,
    get_train_partition_case_ids,
    metrics_for_prediction,
    repair_logits_from_decoder1,
    score_edema_dice_pool,
    select_feasibility_split,
    sliding_window_repair_inference,
    tumor_centered_crop,
)
from src.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast spatial edema-repair feasibility")
    p.add_argument("--config", type=Path, default=Path("extra/configs/fast_spatial_edema_repair.yaml"))
    p.add_argument("--stage", type=str, default="full", choices=["smoke", "sanity", "full"])
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--score-pool", type=int, default=None)
    return p.parse_args()


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def assert_frozen_setup(model, repair, device) -> None:
    for p in model.parameters():
        assert not p.requires_grad, "U-Net parameter requires grad"
    opt_params = set(map(id, repair.trainable_parameters()))
    assert len(opt_params) == sum(1 for _ in repair.parameters())
    # disabled repair == baseline head path
    x = torch.randn(1, 4, 64, 64, 64, device=device)
    model.eval()
    repair.eval()
    with torch.no_grad():
        h = model.forward_to_decoder1(x)
        base = model.forward_from_decoder1(h)
        support = torch.ones(1, 1, 64, 64, 64, device=device)
        edited, _ = repair(h, support, enabled=False)
        out = model.forward_from_decoder1(edited)
        assert torch.allclose(base, out, atol=1e-5)
    # gradients reach repair params (gate bias nonzero so delta/scale are in the graph)
    repair.train()
    with torch.no_grad():
        repair.gate.bias.fill_(0.5)
    h = model.forward_to_decoder1(x)
    support = torch.ones(1, 1, 64, 64, 64, device=device)
    edited, gate = repair(h.detach(), support, enabled=True)
    loss = edited.pow(2).mean() + gate.pow(2).mean() + repair.scale + repair.delta.pow(2).sum()
    loss.backward()
    assert repair.delta.grad is not None and repair.delta.grad.abs().sum() > 0
    assert repair.raw_scale.grad is not None
    assert repair.gate.weight.grad is not None and repair.gate.weight.grad.abs().sum() > 0
    repair.zero_grad(set_to_none=True)
    with torch.no_grad():
        repair.gate.bias.zero_()
        repair.delta.zero_()
        repair.raw_scale.zero_()


def run_smoke(config: dict, device: torch.device) -> None:
    model = build_model(config).to(device)
    ckpt = torch.load(config["checkpoint"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    probe = SemanticDirection.load(config["probe_direction"])
    d_probe = torch.as_tensor(probe.activation_direction, device=device)
    assert d_probe.numel() == config["model"]["base_features"]
    d0 = build_d_initial(d_probe, model.seg_head.weight)
    repair = SpatialEdemaRepair(d0, channels=d_probe.numel()).to(device)
    assert_frozen_setup(model, repair, device)
    print("SMOKE OK")


def _prepare(config: dict, device: torch.device, split_override: dict | None = None):
    out_dir = ensure_dir(Path(config["output_dir"]))
    model = build_model(config).to(device)
    ckpt = torch.load(config["checkpoint"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    probe = SemanticDirection.load(config["probe_direction"])
    d_probe = torch.as_tensor(probe.activation_direction, dtype=torch.float32, device=device)
    assert d_probe.numel() == int(model.forward_to_decoder1(torch.zeros(1, 4, 64, 64, 64, device=device)).shape[1])

    d0 = build_d_initial(d_probe, model.seg_head.weight)
    repair = SpatialEdemaRepair(d0, channels=d_probe.numel()).to(device)
    assert_frozen_setup(model, repair, device)

    if split_override is not None:
        split = split_override
    else:
        train_ids = get_train_partition_case_ids(config)
        # never use val partition
        scored = score_edema_dice_pool(
            model,
            train_ids,
            config,
            device,
            max_pool=int(config["split"]["score_pool"]),
            seed=int(config["feasibility_seed"]),
        )
        scored.to_csv(out_dir / "scored_train_pool.csv", index=False)
        split = select_feasibility_split(
            scored,
            config["split"]["n_train"],
            config["split"]["n_dev"],
            config["split"]["n_test"],
            seed=int(config["feasibility_seed"]),
        )
    (out_dir / "case_split.json").write_text(json.dumps(split, indent=2))
    return model, repair, d_probe, split, out_dir


def train_repair(
    config: dict,
    model,
    repair,
    split: dict,
    device: torch.device,
    out_dir: Path,
    max_steps: int | None = None,
) -> dict[str, Any]:
    patch_size = tuple(int(s) for s in config["data"]["patch_size"])
    overlap = float(config["inference"]["overlap"])
    num_classes = config["model"]["num_classes"]
    lambdas = {k: float(v) for k, v in config["training"]["lambdas"].items()}
    criterion = DiceCrossEntropyLoss(dice_weight=0.5, num_classes=num_classes)
    opt = torch.optim.Adam(repair.trainable_parameters(), lr=float(config["training"]["lr"]))
    rng = np.random.default_rng(int(config["feasibility_seed"]))

    # preload cases
    cache: dict[str, dict[str, Any]] = {}
    for split_name, ids in split.items():
        for case_id in ids:
            image, seg = load_case(config, case_id)
            base_logits, base_pred = baseline_volume_prediction(
                model,
                torch.from_numpy(image),
                patch_size,
                num_classes,
                overlap,
                device,
            )
            cache[case_id] = {
                "image": torch.from_numpy(image),
                "seg": torch.from_numpy(seg.astype(np.int64)),
                "baseline_logits": base_logits,
                "baseline_pred": base_pred,
                "split": split_name,
            }

    steps = int(max_steps if max_steps is not None else config["training"]["max_steps"])
    patience = int(config["training"]["patience"])
    eval_every = int(config["training"]["eval_every"])
    best_dev = -1.0
    best_state = None
    stall = 0
    log_rows = []
    train_ids = split["train"]
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        repair.train()
        case_id = train_ids[(step - 1) % len(train_ids)]
        item = cache[case_id]
        # tumor-centered using baseline WT prediction
        center = (item["baseline_pred"] > 0).astype(np.uint8)
        img_p, lab_p, origin = tumor_centered_crop(
            item["image"], item["seg"], patch_size, center, rng
        )
        z0, y0, x0 = origin
        base_p = item["baseline_logits"][
            :, z0 : z0 + patch_size[0], y0 : y0 + patch_size[1], x0 : x0 + patch_size[2]
        ]
        # pad baseline crop if needed
        if base_p.shape[1:] != patch_size:
            pad = [
                0,
                max(0, patch_size[2] - base_p.shape[3]),
                0,
                max(0, patch_size[1] - base_p.shape[2]),
                0,
                max(0, patch_size[0] - base_p.shape[1]),
            ]
            base_p = torch.nn.functional.pad(base_p.unsqueeze(0), pad).squeeze(0)
        support = build_support_mask(
            base_p, dilate_radius=int(config["training"]["dilate_radius"])
        ).view(1, 1, *patch_size)

        img_b = img_p.unsqueeze(0).to(device)
        lab_b = lab_p.unsqueeze(0).to(device)
        base_b = base_p.unsqueeze(0).to(device)
        support = support.to(device)

        opt.zero_grad(set_to_none=True)
        loss, stats = compute_train_loss(
            model, repair, img_b, lab_b, base_b, support, criterion, lambdas
        )
        loss.backward()
        opt.step()
        stats["step"] = step
        log_rows.append(stats)

        if step % eval_every == 0 or step == steps:
            repair.eval()
            dev_scores = []
            for did in split["dev"]:
                ditem = cache[did]
                out = sliding_window_repair_inference(
                    model,
                    repair,
                    ditem["image"],
                    patch_size,
                    num_classes,
                    overlap,
                    device,
                    mode="proposed",
                )
                gt = ditem["seg"].numpy()
                dev_scores.append(metrics_for_prediction(out["pred"], gt, out["soft"], out["soft_baseline"])["edema_dice"])
            mean_dev = float(np.mean(dev_scores))
            log_rows[-1]["dev_edema_dice"] = mean_dev
            print(f"step {step}: loss={stats['loss']:.4f} dev_edema_dice={mean_dev:.4f} scale={stats['scale']:.4f}")
            if mean_dev > best_dev + 1e-5:
                best_dev = mean_dev
                best_state = {
                    "repair": {k: v.detach().cpu() for k, v in repair.state_dict().items()},
                    "step": step,
                    "dev_edema_dice": mean_dev,
                }
                stall = 0
            else:
                stall += eval_every
            if stall >= patience:
                print(f"Early stop at step {step}")
                break

    if best_state is not None:
        repair.load_state_dict(best_state["repair"])
        torch.save(best_state, out_dir / "best_repair_checkpoint.pt")
    pd.DataFrame(log_rows).to_csv(out_dir / "training_log.csv", index=False)
    return {
        "best_dev_edema_dice": best_dev,
        "steps": int(log_rows[-1]["step"]) if log_rows else 0,
        "runtime_s": time.perf_counter() - t0,
        "cache": cache,
        "n_params": sum(p.numel() for p in repair.trainable_parameters()),
    }


def tune_scalar_on_dev(
    values: list[float],
    eval_fn,
) -> float:
    best_v, best_score = values[0], -1.0
    for v in values:
        score = eval_fn(v)
        if score > best_score:
            best_score, best_v = score, v
    return best_v


def evaluate_all(
    config: dict,
    model,
    repair,
    d_probe: torch.Tensor,
    split: dict,
    cache: dict,
    device: torch.device,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    patch_size = tuple(int(s) for s in config["data"]["patch_size"])
    overlap = float(config["inference"]["overlap"])
    num_classes = config["model"]["num_classes"]
    figures = ensure_dir(out_dir / "figures")
    examples = ensure_dir(out_dir / "examples")

    w = model.seg_head.weight.detach().float().reshape(model.seg_head.weight.shape[0], -1)
    d_out = w[2] - w[0]
    d_out = d_out / d_out.norm().clamp_min(1e-8)

    # tune baselines on DEV only
    def mean_dev_edema(mode: str, **kwargs) -> float:
        scores = []
        for did in split["dev"]:
            item = cache[did]
            out = sliding_window_repair_inference(
                model, repair, item["image"], patch_size, num_classes, overlap, device, mode=mode, **kwargs
            )
            scores.append(
                metrics_for_prediction(out["pred"], item["seg"].numpy(), out["soft"], out["soft_baseline"])[
                    "edema_dice"
                ]
            )
        return float(np.mean(scores))

    alpha_probe = tune_scalar_on_dev(
        [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
        lambda a: mean_dev_edema("global_probe", direction=d_probe, alpha=a),
    )
    alpha_out = tune_scalar_on_dev(
        [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
        lambda a: mean_dev_edema("global_output", direction=d_out, alpha=a),
    )
    bias = tune_scalar_on_dev(
        [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
        lambda b: mean_dev_edema("logit_bias", logit_bias=b),
    )

    methods = [
        ("A_baseline", "baseline", {}),
        ("B_global_probe", "global_probe", {"direction": d_probe, "alpha": alpha_probe}),
        ("C_global_output", "global_output", {"direction": d_out, "alpha": alpha_out}),
        ("D_logit_bias", "logit_bias", {"logit_bias": bias}),
        ("F_proposed", "proposed", {}),
    ]

    rng = np.random.default_rng(int(config["feasibility_seed"]) + 99)
    random_dirs = []
    for i in range(5):
        v = rng.standard_normal(d_probe.numel())
        v = v / (np.linalg.norm(v) + 1e-8)
        random_dirs.append(torch.as_tensor(v, dtype=torch.float32, device=device))
        methods.append((f"E_random_{i:02d}", "random_gated", {"direction": random_dirs[-1]}))

    rows = []
    for case_id in split["test"]:
        item = cache[case_id]
        gt = item["seg"].numpy()
        for method_name, mode, kwargs in methods:
            out = sliding_window_repair_inference(
                model, repair, item["image"], patch_size, num_classes, overlap, device, mode=mode, **kwargs
            )
            m = metrics_for_prediction(out["pred"], gt, out["soft"], out["soft_baseline"])
            row = {
                "case_id": case_id,
                "method": method_name,
                "mode": mode,
                **m,
                "n_edited_voxels": out["n_edited_voxels"],
                "mean_abs_gate": out["mean_abs_gate"],
                "perturbation_rms": out["perturbation_rms"],
                "cosine_d_learned_d_initial": float(repair.cosine_to_initial().detach().cpu()),
                "scale": float(repair.scale.detach().cpu()),
                "runtime_s": out["runtime_s"],
                "tuned_alpha_probe": alpha_probe,
                "tuned_alpha_output": alpha_out,
                "tuned_logit_bias": bias,
            }
            rows.append(row)
            if method_name in ("A_baseline", "F_proposed") and case_id == split["test"][0]:
                np.save(examples / f"{case_id}_{method_name}_pred.npy", out["pred"])
                np.save(examples / f"{case_id}_gt.npy", gt)
                if method_name == "F_proposed":
                    np.save(examples / f"{case_id}_gate.npy", out["gate"])
                    np.save(examples / f"{case_id}_support.npy", out["support"])
                    np.save(examples / f"{case_id}_edema_prob_diff.npy", out["edema_prob_diff"])
                    np.save(examples / f"{case_id}_baseline_pred.npy", out["baseline_pred"])

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "case_results.csv", index=False)

    summary = (
        df.groupby("method", as_index=False)
        .agg(
            mean_edema_dice=("edema_dice", "mean"),
            mean_fg_dice=("mean_fg_dice", "mean"),
            mean_wt_dice=("wt_dice", "mean"),
            mean_abs_edema_vol_err=("abs_edema_volume_error", "mean"),
            mean_enh_vol_change=("enhancing_volume_change", "mean"),
            mean_nec_vol_change=("necrosis_volume_change", "mean"),
            mean_runtime=("runtime_s", "mean"),
        )
        .sort_values("mean_edema_dice", ascending=False)
    )
    summary.to_csv(out_dir / "method_summary.csv", index=False)

    # plots
    pivot = df.pivot(index="case_id", columns="method", values="edema_dice")
    ax = pivot[["A_baseline", "F_proposed"]].plot(kind="bar", figsize=(8, 4))
    ax.set_ylabel("edema Dice")
    ax.set_title("Edema Dice by case")
    plt.tight_layout()
    plt.savefig(figures / "edema_dice_by_case.png", dpi=150)
    plt.close()

    vol_pivot = df.pivot(index="case_id", columns="method", values="abs_edema_volume_error")
    ax = vol_pivot[["A_baseline", "F_proposed"]].plot(kind="bar", figsize=(8, 4))
    ax.set_ylabel("|edema volume error|")
    plt.tight_layout()
    plt.savefig(figures / "edema_volume_error_by_case.png", dpi=150)
    plt.close()

    ax = summary.set_index("method")["mean_edema_dice"].plot(kind="bar", figsize=(8, 4), color="steelblue")
    ax.set_ylabel("mean edema Dice")
    ax.set_title("Mean Dice comparison")
    plt.tight_layout()
    plt.savefig(figures / "mean_dice_comparison.png", dpi=150)
    plt.close()

    # gate / prob diff examples if present
    gate_path = examples / f"{split['test'][0]}_gate.npy"
    if gate_path.exists():
        g = np.load(gate_path)
        mid = g.shape[0] // 2
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        axes[0].imshow(g[mid], cmap="coolwarm")
        axes[0].set_title("gate (mid slice)")
        diff = np.load(examples / f"{split['test'][0]}_edema_prob_diff.npy")
        axes[1].imshow(diff[mid], cmap="coolwarm")
        axes[1].set_title("edema Δp")
        plt.tight_layout()
        plt.savefig(figures / "gate_examples.png", dpi=150)
        plt.savefig(figures / "probability_difference_examples.png", dpi=150)
        plt.close()

    verdict = classify_feasibility(df)
    return df, summary, verdict


def classify_feasibility(df: pd.DataFrame) -> str:
    base = df[df["method"] == "A_baseline"].set_index("case_id")
    prop = df[df["method"] == "F_proposed"].set_index("case_id")
    common = base.index.intersection(prop.index)
    base = base.loc[common]
    prop = prop.loc[common]

    mean_improve = float(prop["edema_dice"].mean() - base["edema_dice"].mean())
    n_improve = int((prop["edema_dice"] > base["edema_dice"]).sum())
    fg_drop = float(base["mean_fg_dice"].mean() - prop["mean_fg_dice"].mean())

    competitors = {
        "B_global_probe": df[df["method"] == "B_global_probe"]["edema_dice"].mean(),
        "C_global_output": df[df["method"] == "C_global_output"]["edema_dice"].mean(),
        "D_logit_bias": df[df["method"] == "D_logit_bias"]["edema_dice"].mean(),
    }
    random_means = [
        df[df["method"] == f"E_random_{i:02d}"]["edema_dice"].mean() for i in range(5)
    ]
    median_random = float(np.median(random_means))
    prop_mean = float(prop["edema_dice"].mean())
    beats_all = all(prop_mean > competitors[k] for k in competitors) and prop_mean > median_random

    off = float(
        prop["enhancing_volume_change"].abs().mean() + prop["necrosis_volume_change"].abs().mean()
    )
    mainly_off = off > 500 and mean_improve < 0.05
    concentrated = float(prop["n_edited_voxels"].mean()) < 0.5 * 1e6

    if (
        mean_improve >= 0.02
        and n_improve >= 4
        and fg_drop <= 0.005
        and beats_all
        and not mainly_off
        and concentrated
    ):
        return "PROMISING"
    if mean_improve > 0 and (fg_drop > 0.005 or mainly_off or not beats_all):
        return "MIXED"
    return "NOT PROMISING"


def write_report(
    out_dir: Path,
    split: dict,
    summary: pd.DataFrame,
    df: pd.DataFrame,
    train_info: dict,
    device: torch.device,
    verdict: str,
    repair: SpatialEdemaRepair,
) -> None:
    base = df[df["method"] == "A_baseline"]
    prop = df[df["method"] == "F_proposed"]
    lines = [
        "# Fast spatial edema-repair feasibility report",
        "",
        f"**Verdict: {verdict}**",
        "",
        "## Split (train partition only; seed=feasibility_seed)",
        f"- train: {split['train']}",
        f"- dev: {split['dev']}",
        f"- test: {split['test']}",
        "",
        "## Runtime / device",
        f"- device: `{device}`",
        f"- training runtime_s: {train_info['runtime_s']:.1f}",
        f"- optimization steps: {train_info['steps']}",
        f"- trainable parameters: {train_info['n_params']}",
        f"- learned softplus scale: {float(repair.scale.detach().cpu()):.6f}",
        f"- cosine(d_learned, d_initial): {float(repair.cosine_to_initial().detach().cpu()):.6f}",
        "",
        "## Method summary (held-out test)",
        summary.to_string(index=False),
        "",
        "## Per-case baseline vs proposed (edema Dice)",
    ]
    for case_id in split["test"]:
        b = float(base.loc[base.case_id == case_id, "edema_dice"].iloc[0])
        p = float(prop.loc[prop.case_id == case_id, "edema_dice"].iloc[0])
        lines.append(f"- {case_id}: baseline={b:.4f}, proposed={p:.4f}, Δ={p-b:+.4f}")
    lines.extend(
        [
            "",
            "## Off-target soft volume changes (proposed, mean)",
            f"- enhancing: {prop['enhancing_volume_change'].mean():+.2f}",
            f"- necrosis: {prop['necrosis_volume_change'].mean():+.2f}",
            "",
            "## Notes",
            "- Feasibility gates only; not a statistical or clinical claim.",
            "- Validation (375) cases were never used.",
        ]
    )
    (out_dir / "feasibility_report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.open())
    if args.score_pool is not None:
        config["split"]["score_pool"] = args.score_pool
    device = _device()
    print(f"Using device: {device}")

    if args.stage == "smoke":
        run_smoke(config, device)
        return

    if args.stage == "sanity":
        # tiny split: 2 train, 1 dev, 1 test from scored pool
        config["split"]["n_train"] = 2
        config["split"]["n_dev"] = 1
        config["split"]["n_test"] = 1
        config["split"]["score_pool"] = min(24, int(config["split"]["score_pool"]))
        config["training"]["max_steps"] = 20
        config["training"]["patience"] = 20
        config["training"]["eval_every"] = 10
        config["output_dir"] = "outputs_fast_spatial_repair_sanity"

    max_steps = args.max_steps
    model, repair, d_probe, split, out_dir = _prepare(config, device)
    print("Split:", json.dumps(split, indent=2))

    # snapshot U-Net params
    unet_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_info = train_repair(config, model, repair, split, device, out_dir, max_steps=max_steps)
    for k, v in model.state_dict().items():
        assert torch.allclose(v.cpu(), unet_before[k]), f"U-Net changed: {k}"

    if args.stage == "sanity":
        print("SANITY: U-Net unchanged; training finished", train_info)
        # quick check repaired differs inside support on one train case
        case_id = split["train"][0]
        item = train_info["cache"][case_id]
        out_b = sliding_window_repair_inference(
            model, repair, item["image"], tuple(config["data"]["patch_size"]),
            config["model"]["num_classes"], config["inference"]["overlap"], device, mode="baseline",
        )
        out_p = sliding_window_repair_inference(
            model, repair, item["image"], tuple(config["data"]["patch_size"]),
            config["model"]["num_classes"], config["inference"]["overlap"], device, mode="proposed",
        )
        diff = (out_p["pred"] != out_b["pred"])
        inside = diff & (out_p["support"] > 0.5)
        outside = diff & (out_p["support"] <= 0.5)
        print(f"changed voxels inside support={inside.sum()} outside={outside.sum()}")
        print("SANITY OK")
        return

    df, summary, verdict = evaluate_all(
        config, model, repair, d_probe, split, train_info["cache"], device, out_dir
    )
    write_report(out_dir, split, summary, df, train_info, device, verdict, repair)
    print(summary.to_string(index=False))
    print(f"\nVerdict: {verdict}")
    print(f"Report: {out_dir / 'feasibility_report.md'}")


if __name__ == "__main__":
    main()
