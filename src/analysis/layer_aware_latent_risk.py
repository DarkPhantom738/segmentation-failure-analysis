"""Layer-Aware Latent Risk Triage — leakage-safe failure ranking from U-Net latents.

Stages
------
inventory   : write/refresh artifact inventory; validate folds & embeddings
confidence  : frozen sliding-window inference → GT-free case-level confidence
dry_check   : load artifacts, verify shapes/alignment, no model fitting
full        : nested-CV evaluation (blocked until explicitly requested)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import ndimage
from sklearn.model_selection import KFold

from src.analysis.layer_io import load_layer_index, load_layer_matrix
from src.utils.io import ensure_dir

LAYER_DIMS = {
    "encoder1": 32,
    "encoder2": 64,
    "encoder3": 128,
    "encoder4": 256,
    "bottleneck": 128,
    "decoder4": 256,
    "decoder3": 128,
    "decoder2": 64,
    "decoder1": 32,
}

GT_LEAKED_COLS = (
    "mean_entropy_error",
    "mean_entropy_nonerror",
    "entropy_error_auroc",
    "entropy_error_auprc",
    "overlap_top_5_uncertainty",
    "overlap_top_10_uncertainty",
    "overlap_top_20_uncertainty",
    "confident_false_negative_fraction",
    "false_positive_voxels",
    "false_negative_voxels",
    "missed_small_lesion_count",
)


def load_config(path: Path | str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_outer_folds(fold_csv: Path) -> dict[int, dict[str, np.ndarray]]:
    """Load reused outer folds: fold → {train: idx, test: idx} over sorted case_ids."""
    folds = pd.read_csv(fold_csv)
    required = {"case_id", "outer_fold", "split"}
    if not required.issubset(folds.columns):
        raise ValueError(f"fold_assignments missing columns: {required - set(folds.columns)}")
    case_ids = sorted(folds["case_id"].unique())
    id_to_i = {c: i for i, c in enumerate(case_ids)}
    out: dict[int, dict[str, np.ndarray]] = {}
    for fold_i, sub in folds.groupby("outer_fold"):
        tr = np.array(
            [id_to_i[c] for c in sub.loc[sub["split"] == "train", "case_id"]],
            dtype=int,
        )
        te = np.array(
            [id_to_i[c] for c in sub.loc[sub["split"] == "test", "case_id"]],
            dtype=int,
        )
        out[int(fold_i)] = {"train": np.sort(tr), "test": np.sort(te)}
    # Exactly one test membership per case
    test_count = np.zeros(len(case_ids), dtype=int)
    for parts in out.values():
        test_count[parts["test"]] += 1
    if not np.array_equal(test_count, np.ones(len(case_ids), dtype=int)):
        raise ValueError("Outer folds are not a partition (each case must be test once).")
    return out


def verify_folds_match_seed(fold_csv: Path, n: int = 375, seed: int = 42) -> bool:
    """Confirm fold_assignments match KFold(5, shuffle=True, random_state=seed)."""
    stored = load_outer_folds(fold_csv)
    expected = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(np.arange(n)))
    # stored indices are over sorted case_ids; consistency experiment used
    # case_df sorted by case_id — same ordering.
    for fold_i, (tr, te) in enumerate(expected):
        if not np.array_equal(np.sort(tr), stored[fold_i]["train"]):
            return False
        if not np.array_equal(np.sort(te), stored[fold_i]["test"]):
            return False
    return True


def assert_no_gt_leak_features(columns: list[str], extra_forbidden: list[str] | None = None) -> None:
    banned = set(GT_LEAKED_COLS)
    if extra_forbidden:
        banned.update(extra_forbidden)
    bad = [c for c in columns if c in banned]
    if bad:
        raise ValueError(f"GT-leaked features present: {bad}")


def compact_confidence_from_probabilities(
    probs: np.ndarray,
    pred: np.ndarray,
    boundary_iterations: int = 2,
) -> dict[str, float]:
    """
    Compact GT-free confidence row from softmax probs + hard pred.

    Does not retain or write probability/entropy volumes.
    """
    if probs.ndim != 4:
        raise ValueError(f"Expected probs (C,D,H,W), got {probs.shape}")
    probs = probs.astype(np.float64)
    pred = pred.astype(np.int16)

    p_clip = np.clip(probs, 1e-8, 1.0)
    entropy = -np.sum(p_clip * np.log(p_clip), axis=0)
    max_prob = probs.max(axis=0)
    part = np.partition(probs, -2, axis=0)
    margin = part[-1] - part[-2]

    fg = pred > 0
    if fg.any():
        eroded = ndimage.binary_erosion(fg, iterations=1)
        boundary = fg & ~eroded
        if boundary_iterations > 1:
            dil = ndimage.binary_dilation(fg, iterations=1)
            # 1-voxel exterior + surface band around predicted tumor
            boundary = boundary | (dil & ~fg)
    else:
        boundary = np.zeros_like(fg, dtype=bool)

    def _mean(x: np.ndarray, mask: np.ndarray | None = None) -> float:
        if mask is None:
            return float(np.mean(x))
        if not np.any(mask):
            return float("nan")
        return float(np.mean(x[mask]))

    return {
        "conf_mean_maxprob_all": _mean(max_prob),
        "conf_mean_maxprob_fg": _mean(max_prob, fg),
        "conf_mean_entropy_fg": _mean(entropy, fg),
        "conf_mean_entropy_boundary": _mean(entropy, boundary),
        "conf_entropy_p95_boundary": (
            float(np.percentile(entropy[boundary], 95)) if boundary.any() else float("nan")
        ),
        "conf_mean_margin_all": _mean(margin),
        "conf_mean_margin_fg": _mean(margin, fg),
        "conf_frac_maxprob_lt_0.60": float(np.mean(max_prob < 0.60)),
        "conf_frac_maxprob_lt_0.70": float(np.mean(max_prob < 0.70)),
        "conf_frac_maxprob_lt_0.80": float(np.mean(max_prob < 0.80)),
        "conf_frac_maxprob_lt_0.90": float(np.mean(max_prob < 0.90)),
        "conf_frac_margin_lt_0.10": float(np.mean(margin < 0.10)),
        "conf_frac_margin_lt_0.20": float(np.mean(margin < 0.20)),
        "pred_tumor_voxels": float(fg.sum()),
        "pred_boundary_voxels": float(boundary.sum()),
    }


# Keep old name as alias for tests.
def confidence_from_probabilities(
    probs: np.ndarray,
    pred: np.ndarray | None = None,
    boundary_iterations: int = 2,
) -> dict[str, float]:
    if pred is None:
        pred = np.argmax(probs, axis=0).astype(np.int16)
    out = compact_confidence_from_probabilities(probs, pred, boundary_iterations)
    # Extra fields used by earlier unit tests
    p_clip = np.clip(probs.astype(np.float64), 1e-8, 1.0)
    entropy = -np.sum(p_clip * np.log(p_clip), axis=0)
    out["conf_mean_entropy_all"] = float(np.mean(entropy))
    out["conf_fg_fraction"] = float((pred > 0).mean())
    return out


def generate_confidence_features(
    config: dict[str, Any],
    *,
    max_cases: int | None = None,
    device: Any = None,
) -> pd.DataFrame:
    """
    Frozen TTA sliding-window inference → compact GT-free confidence CSV.

    Critical: argmax masks are compared to retained hard predictions
    (`path_prediction`, typically *_pred_tta.npy). Full prob/entropy volumes
    are never written.
    """
    import torch

    from src.data.brats_dataset import load_case_cached
    from src.models.unet3d import build_model
    from src.training.tta import (
        tta_predict_probabilities,
        tta_predict_segmentation,
        tta_sliding_window_inference,
    )

    out_dir = ensure_dir(Path(config["paths"]["output_dir"]))
    out_csv = out_dir / "case_level_confidence_features.csv"
    mismatch_csv = out_dir / "confidence_mask_mismatch_report.csv"
    failure_df = pd.read_csv(config["paths"]["failure_table"])

    if max_cases is None:
        max_cases = config.get("inference", {}).get("max_cases")
    if max_cases is not None and int(max_cases) > 0:
        failure_df = failure_df.head(int(max_cases))

    # Prefer preprocessing settings embedded in the checkpoint for fidelity.
    ckpt_path = Path(config["paths"]["checkpoint"])
    ckpt_cpu = torch.load(ckpt_path, map_location="cpu")
    train_cfg = dict(ckpt_cpu.get("config") or {})
    if not train_cfg:
        with open(config["paths"]["train_config"]) as f:
            train_cfg = yaml.safe_load(f)

    data_root = Path(config["paths"].get("data_root", train_cfg["data"]["root"]))
    # Prefer local data root if configured and present; cache is keyed by preprocess params.
    if not data_root.exists():
        alt = Path("data/BraTS2021_Training")
        if alt.exists():
            data_root = alt
    cache_dir = Path(
        config["paths"].get("cache_dir")
        or train_cfg.get("data", {}).get("cache_dir")
        or "outputs_10hour/cache"
    )
    data_cfg = train_cfg["data"]

    inf_cfg = config.get("inference", {})
    use_tta = bool(inf_cfg.get("use_tta", True))
    n_tta = int(inf_cfg.get("tta_augmentations", 8))
    overlap = float(inf_cfg.get("overlap", 0.25))
    patch_size = tuple(inf_cfg.get("patch_size", data_cfg["patch_size"]))
    require_exact = bool(inf_cfg.get("require_exact_mask_match", True))
    resume = bool(inf_cfg.get("resume", True))

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model = build_model(train_cfg)
    model.load_state_dict(ckpt_cpu["model_state_dict"])
    model.to(device)
    model.eval()
    num_classes = int(train_cfg["model"]["num_classes"])

    existing_df = None
    done_ids: set[str] = set()
    rows: list[dict[str, float | str | int | bool]] = []
    if resume and out_csv.exists():
        existing_df = pd.read_csv(out_csv)
        if "mask_exact_match" in existing_df.columns:
            done = existing_df[
                existing_df["mask_exact_match"].astype(str).isin(["True", "true", "1"])
            ]
            done_ids = set(done["case_id"].astype(str))
            rows = done.to_dict(orient="records")
        else:
            rows = []

    mismatches: list[dict[str, float | str | int]] = []
    from tqdm import tqdm
    import os

    # Keep outer progress readable; silence nested TTA/sliding-window bars.
    os.environ.setdefault("TQDM_DISABLE", "0")

    print(
        f"Confidence export: device={device}, tta={use_tta}×{n_tta}, "
        f"overlap={overlap}, patch={patch_size}, data_root={data_root}, "
        f"cache={cache_dir}, n_cases={len(failure_df)}, already_done={len(done_ids)}"
    )

    pending = [
        rec
        for rec in failure_df.to_dict(orient="records")
        if str(rec["case_id"]) not in done_ids
    ]

    for rec in tqdm(pending, desc="Confidence features"):
        case_id = str(rec["case_id"])

        saved_pred_path = Path(rec["path_prediction"])
        if not saved_pred_path.exists():
            raise FileNotFoundError(f"Missing retained hard mask: {saved_pred_path}")
        saved_pred = np.load(saved_pred_path).astype(np.uint8)

        image, _gt_unused = load_case_cached(
            case_id=case_id,
            data_root=data_root,
            modalities=data_cfg["modalities"],
            target_spacing=data_cfg["target_spacing"],
            percentile_clip=data_cfg["percentile_clip"],
            cache_dir=cache_dir,
        )
        del _gt_unused  # never use GT for confidence

        image_t = torch.from_numpy(np.asarray(image)).float()
        if use_tta:
            mean_probs_t = tta_sliding_window_inference(
                model=model,
                image=image_t,
                patch_size=patch_size,
                num_classes=num_classes,
                device=device,
                overlap=overlap,
                max_augmentations=n_tta,
            )
            new_pred = tta_predict_segmentation(mean_probs_t)
            probs = tta_predict_probabilities(mean_probs_t)
        else:
            from src.training.inference import (
                predict_probabilities,
                predict_segmentation,
                sliding_window_inference,
            )

            logits, _, _ = sliding_window_inference(
                model,
                image_t,
                patch_size=patch_size,
                num_classes=num_classes,
                overlap=overlap,
                device=device,
            )
            new_pred = predict_segmentation(logits)
            probs = predict_probabilities(logits)

        if new_pred.shape != saved_pred.shape:
            raise ValueError(
                f"{case_id}: shape mismatch new {new_pred.shape} vs saved {saved_pred.shape}"
            )
        n_disagree = int(np.sum(new_pred != saved_pred))
        agree_frac = float(np.mean(new_pred == saved_pred))
        exact = n_disagree == 0

        row: dict[str, float | str | int | bool] = {
            "case_id": case_id,
            "inference_mode": "tta" if use_tta else "sliding_window",
            "tta_augmentations": n_tta if use_tta else 0,
            "overlap": overlap,
            "checkpoint_epoch": int(ckpt_cpu.get("epoch", -1)),
            "path_saved_prediction": str(saved_pred_path),
            "mask_exact_match": exact,
            "mask_agree_fraction": agree_frac,
            "mask_n_disagree_voxels": n_disagree,
            "mask_usable_with_old_dice": exact if require_exact else agree_frac > 0.999,
        }

        if exact or not require_exact:
            pred_for_conf = saved_pred if exact else new_pred
            row.update(compact_confidence_from_probabilities(probs, pred_for_conf))
        else:
            mismatches.append(
                {
                    "case_id": case_id,
                    "n_disagree": n_disagree,
                    "agree_fraction": agree_frac,
                    "path_saved_prediction": str(saved_pred_path),
                }
            )

        rows.append(row)
        pd.DataFrame(rows).sort_values("case_id").to_csv(out_csv, index=False)

        del probs, image, image_t
        if use_tta:
            del mean_probs_t

    conf_df = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    conf_df.to_csv(out_csv, index=False)
    if mismatches:
        pd.DataFrame(mismatches).to_csv(mismatch_csv, index=False)
    else:
        # Clear stale mismatch report if everything matches
        if mismatch_csv.exists() and conf_df["mask_exact_match"].all():
            mismatch_csv.unlink()

    n_ok = int(conf_df["mask_exact_match"].sum()) if "mask_exact_match" in conf_df else 0
    print(
        f"Wrote {out_csv} ({len(conf_df)} rows). "
        f"Exact mask matches: {n_ok}/{len(conf_df)}."
    )
    if n_ok < len(conf_df):
        print(
            f"WARNING: {len(conf_df) - n_ok} case(s) differ from retained hard masks. "
            f"See {mismatch_csv}. Do not pair mismatched confidence with old Dice."
        )
    return conf_df


def load_reusable_tables(config: dict[str, Any]) -> dict[str, Any]:
    """Load folds, consistency features, layer matrices; validate alignment."""
    out_dir = ensure_dir(Path(config["paths"]["output_dir"]))
    fold_csv = Path(config["paths"]["fold_assignments"])
    folds = load_outer_folds(fold_csv)
    cons = pd.read_csv(config["paths"]["consistency_features"]).sort_values("case_id").reset_index(drop=True)
    case_ids = cons["case_id"].tolist()

    morph_cols = list(config["morphology_cols"])
    probe_cols = list(config["quality_probe_cols"])
    gap_cols = [
        c
        for c in cons.columns
        if any(c.startswith(p) for p in config["inconsistency_prefixes"])
    ]
    assert_no_gt_leak_features(morph_cols + probe_cols + gap_cols)
    assert_no_gt_leak_features(list(cons.columns), list(config.get("forbidden_gt_leak_cols", [])))

    index_df = load_layer_index(config["paths"]["layer_index"]).sort_values("case_id").reset_index(drop=True)
    if list(index_df["case_id"]) != case_ids:
        # Align by merge
        index_df = index_df.set_index("case_id").loc[case_ids].reset_index()

    layers = list(config["layers"])
    layer_mats = {L: load_layer_matrix(index_df, L) for L in layers}
    for L, expected in LAYER_DIMS.items():
        if L in layer_mats and layer_mats[L].shape[1] != expected:
            raise ValueError(f"{L} dim {layer_mats[L].shape[1]} != expected {expected}")

    conf_path = out_dir / "case_level_confidence_features.csv"
    conf_df = None
    if conf_path.exists():
        conf_df = pd.read_csv(conf_path).sort_values("case_id").reset_index(drop=True)
        if list(conf_df["case_id"]) != case_ids:
            conf_df = conf_df.set_index("case_id").loc[case_ids].reset_index()
        conf_feature_cols = [c for c in conf_df.columns if c.startswith("conf_")]
        assert_no_gt_leak_features(conf_feature_cols)

    return {
        "case_ids": case_ids,
        "folds": folds,
        "consistency": cons,
        "morph_cols": morph_cols,
        "probe_cols": probe_cols,
        "gap_cols": gap_cols,
        "layer_mats": layer_mats,
        "layers": layers,
        "confidence": conf_df,
        "folds_match_seed42": verify_folds_match_seed(fold_csv, n=len(case_ids), seed=int(config["seed"])),
        "output_dir": out_dir,
    }


def write_inventory(config: dict[str, Any], tables: dict[str, Any] | None = None) -> Path:
    """Refresh artifact_inventory.md (also written at Stage 1 manually)."""
    out_dir = ensure_dir(Path(config["paths"]["output_dir"]))
    path = out_dir / "artifact_inventory.md"
    # Keep the checked-in Stage-1 inventory content; append runtime verification.
    if tables is None:
        tables = load_reusable_tables(config)
    conf = tables["confidence"]
    lines = [
        "",
        "## Runtime verification",
        "",
        f"- Cases aligned: {len(tables['case_ids'])}",
        f"- Outer folds match seed {config['seed']}: {tables['folds_match_seed42']}",
        f"- Layers loaded: {tables['layers']}",
        f"- Morphology cols: {len(tables['morph_cols'])}",
        f"- Inconsistency gap cols: {len(tables['gap_cols'])}",
        f"- Quality probe cols: {tables['probe_cols']}",
        f"- Confidence CSV present: {conf is not None}",
        f"- Confidence rows: {0 if conf is None else len(conf)}",
        "",
        "Full nested evaluation (`stage=full`) is gated until confidence features exist.",
        "",
    ]
    existing = path.read_text() if path.exists() else ""
    # Replace prior runtime section if present
    marker = "## Runtime verification"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip() + "\n"
    path.write_text(existing + "\n".join(lines))
    # Always copy fold assignments into this output dir (no overwrite of consistency).
    src_folds = Path(config["paths"]["fold_assignments"])
    dst_folds = out_dir / "fold_assignments.csv"
    if src_folds.exists():
        pd.read_csv(src_folds).to_csv(dst_folds, index=False)
    return path


def dry_check(config: dict[str, Any]) -> dict[str, Any]:
    """Validate artifacts without fitting models or running inference."""
    tables = load_reusable_tables(config)
    write_inventory(config, tables)
    report = {
        "n_cases": len(tables["case_ids"]),
        "folds_ok": tables["folds_match_seed42"],
        "n_layers": len(tables["layers"]),
        "layer_dims": {L: tables["layer_mats"][L].shape[1] for L in tables["layers"]},
        "n_morph": len(tables["morph_cols"]),
        "n_gaps": len(tables["gap_cols"]),
        "confidence_ready": tables["confidence"] is not None,
        "one_test_per_case": True,
    }
    # Sanity: each fold train/test partition
    n = len(tables["case_ids"])
    for fold_i, parts in tables["folds"].items():
        assert len(parts["train"]) + len(parts["test"]) == n
        assert len(np.intersect1d(parts["train"], parts["test"])) == 0
    return report


def run_full_nested_cv(config: dict[str, Any]) -> dict[str, Any]:
    """Full layer-aware experiment — not enabled yet; use compare_baselines."""
    tables = load_reusable_tables(config)
    if tables["confidence"] is None:
        raise RuntimeError(
            "Confidence features missing. Run stage=confidence first, then stage=full."
        )
    raise NotImplementedError(
        "Full layer-aware fusion nested-CV is not enabled yet. "
        "Use --stage compare_baselines for confidence / combined / pooled comparisons."
    )


def _impute_scale_train(
    x: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, Any]:
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(x, dtype=np.float64).copy()
    for j in range(x.shape[1]):
        col = x[:, j]
        m = float(np.nanmean(col[train_idx])) if np.isfinite(col[train_idx]).any() else 0.0
        col[~np.isfinite(col)] = m
        x[:, j] = col
    scaler = StandardScaler()
    scaler.fit(x[train_idx])
    return scaler.transform(x), scaler


def _tune_logistic(x_tr: np.ndarray, y_tr: np.ndarray, Cs: list[float], seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import StratifiedKFold

    y_tr = y_tr.astype(int)
    if y_tr.sum() < 2 or (len(y_tr) - y_tr.sum()) < 2:
        m = LogisticRegression(C=1.0, max_iter=4000, class_weight="balanced")
        m.fit(x_tr, y_tr)
        return m
    n_splits = min(4, int(y_tr.sum()), int(len(y_tr) - y_tr.sum()))
    if n_splits < 2:
        m = LogisticRegression(C=1.0, max_iter=4000, class_weight="balanced")
        m.fit(x_tr, y_tr)
        return m
    best_c, best = Cs[0], -1.0
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for c in Cs:
        scores = []
        for tr, te in splitter.split(x_tr, y_tr):
            if y_tr[tr].sum() == 0 or y_tr[tr].sum() == len(tr):
                continue
            m = LogisticRegression(C=c, max_iter=4000, class_weight="balanced")
            m.fit(x_tr[tr], y_tr[tr])
            s = m.predict_proba(x_tr[te])[:, 1]
            try:
                scores.append(average_precision_score(y_tr[te], s))
            except Exception:
                continue
        mean_s = float(np.mean(scores)) if scores else -1.0
        if mean_s > best:
            best, best_c = mean_s, c
    m = LogisticRegression(C=best_c, max_iter=4000, class_weight="balanced")
    m.fit(x_tr, y_tr)
    return m


def run_baseline_comparisons(config: dict[str, Any]) -> dict[str, Any]:
    """
    Leakage-safe nested-CV comparison of:
      confidence, morphology, inconsistency, combined(=inc+morph),
      combined_conf, pooled, combined_pooled.

    Primary questions:
      1) combined vs confidence
      2) combined+confidence vs combined+pooled
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    from src.analysis.representation_output_consistency import (
        failure_capture_at_budget,
        risk_coverage_curve,
    )

    tables = load_reusable_tables(config)
    if tables["confidence"] is None:
        raise RuntimeError("Run stage=confidence first.")
    if not bool(tables["confidence"]["mask_exact_match"].astype(str).isin(["True", "true", "1"]).all()):
        raise RuntimeError("Confidence masks do not all exactly match retained predictions.")

    out_dir = ensure_dir(Path(config["paths"]["output_dir"]) / "baseline_comparisons")
    cons = tables["consistency"]
    conf = tables["confidence"]
    case_ids = tables["case_ids"]
    n = len(case_ids)
    folds = tables["folds"]
    seed = int(config["seed"])
    Cs = list(config.get("logistic_C", [0.01, 0.1, 1.0, 10.0, 100.0]))
    n_boot = int(config.get("n_bootstrap", 2000))

    mean_fg = cons["label_mean_fg_dice"].to_numpy(dtype=float)
    conf_cols = [c for c in conf.columns if c.startswith("conf_")]
    morph_cols = tables["morph_cols"]
    gap_cols = tables["gap_cols"]

    conf_mat = conf[conf_cols].to_numpy(dtype=float)
    morph_mat = cons[morph_cols].to_numpy(dtype=float)
    gap_mat = cons[gap_cols].to_numpy(dtype=float)
    pooled = np.concatenate([tables["layer_mats"][L] for L in tables["layers"]], axis=1)

    feature_sets = {
        "confidence": conf_mat,
        "morphology": morph_mat,
        "inconsistency": gap_mat,
        "combined": np.column_stack([gap_mat, morph_mat]),  # prior "combined"
        "combined_conf": np.column_stack([gap_mat, morph_mat, conf_mat]),
        "pooled": pooled,
        "combined_pooled": np.column_stack([gap_mat, morph_mat, pooled]),
        "combined_conf_pooled": np.column_stack([gap_mat, morph_mat, conf_mat, pooled]),
    }

    fail_defs = ["lowest20_mean_fg", "mean_fg_lt_0.80", "mean_fg_lt_0.70"]
    scores = {(m, f): np.full(n, np.nan) for m in feature_sets for f in fail_defs}
    fail_labels_store = {f: np.full(n, np.nan) for f in fail_defs}

    for fold_i, parts in sorted(folds.items()):
        tr, te = parts["train"], parts["test"]
        cut20 = float(np.quantile(mean_fg[tr], 0.20))
        y_map = {
            "lowest20_mean_fg": (mean_fg < cut20).astype(int),
            "mean_fg_lt_0.80": (mean_fg < 0.80).astype(int),
            "mean_fg_lt_0.70": (mean_fg < 0.70).astype(int),
        }
        fail_labels_store["lowest20_mean_fg"][te] = y_map["lowest20_mean_fg"][te]
        fail_labels_store["mean_fg_lt_0.80"][te] = y_map["mean_fg_lt_0.80"][te]
        fail_labels_store["mean_fg_lt_0.70"][te] = y_map["mean_fg_lt_0.70"][te]

        for method, raw in feature_sets.items():
            x_full, _ = _impute_scale_train(raw, tr)
            for fdef, y_all in y_map.items():
                model = _tune_logistic(
                    x_full[tr], y_all[tr], Cs=Cs, seed=seed + fold_i
                )
                scores[(method, fdef)][te] = model.predict_proba(x_full[te])[:, 1]

    # Metrics + bootstrap
    metric_rows = []
    rc_rows = []
    boot_rows = []
    rng = np.random.default_rng(seed)

    def _boot_ci(y, s, fn, n_boot=n_boot):
        vals = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(y), size=len(y))
            try:
                vals.append(float(fn(y[idx], s[idx])))
            except Exception:
                continue
        if not vals:
            return float("nan"), float("nan"), float("nan")
        arr = np.asarray(vals)
        return float(np.mean(arr)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))

    for fdef in fail_defs:
        y = fail_labels_store[fdef].astype(int)
        for method in feature_sets:
            s = scores[(method, fdef)]
            auprc = float(average_precision_score(y, s))
            auroc = float(roc_auc_score(y, s))
            auprc_m, auprc_lo, auprc_hi = _boot_ci(y, s, average_precision_score)
            cap = failure_capture_at_budget(s, y.astype(bool))
            rc = risk_coverage_curve(s, mean_fg)
            _, cap_lo, cap_hi = _boot_ci(
                y,
                s,
                lambda yy, ss: failure_capture_at_budget(ss, yy.astype(bool))["capture_at_20"],
            )
            metric_rows.append(
                {
                    "method": method,
                    "failure_def": fdef,
                    "auprc": auprc,
                    "auroc": auroc,
                    "auprc_boot_mean": auprc_m,
                    "auprc_ci_low": auprc_lo,
                    "auprc_ci_high": auprc_hi,
                    "capture_at_10": cap["capture_at_10"],
                    "capture_at_20": cap["capture_at_20"],
                    "capture_at_30": cap["capture_at_30"],
                    "capture20_ci_low": cap_lo,
                    "capture20_ci_high": cap_hi,
                    "aurc": rc["aurc"],
                    "mean_dice_coverage_80": rc["mean_dice_coverage_80"],
                    "n_pos": int(y.sum()),
                }
            )
            rc_rows.append({"method": method, "failure_def": fdef, **rc, **cap})

        # Paired comparisons of interest
        pairs = [
            ("combined", "confidence"),
            ("combined_conf", "combined"),
            ("combined_pooled", "combined"),
            ("combined_conf", "combined_pooled"),
            ("combined_conf", "confidence"),
            ("pooled", "confidence"),
            ("pooled", "combined"),
            ("combined_conf_pooled", "combined_conf"),
            ("combined_conf_pooled", "combined_pooled"),
        ]
        for a, b in pairs:
            sa, sb = scores[(a, fdef)], scores[(b, fdef)]
            diffs = []
            for _ in range(n_boot):
                idx = rng.integers(0, n, size=n)
                try:
                    diffs.append(
                        float(average_precision_score(y[idx], sa[idx]))
                        - float(average_precision_score(y[idx], sb[idx]))
                    )
                except Exception:
                    continue
            arr = np.asarray(diffs) if diffs else np.array([np.nan])
            boot_rows.append(
                {
                    "failure_def": fdef,
                    "method_a": a,
                    "method_b": b,
                    "metric": "auprc_diff",
                    "diff_mean": float(np.nanmean(arr)),
                    "ci_low": float(np.nanquantile(arr, 0.025)),
                    "ci_high": float(np.nanquantile(arr, 0.975)),
                    "frac_a_better": float(np.mean(arr > 0)),
                }
            )

    metrics_df = pd.DataFrame(metric_rows)
    rc_df = pd.DataFrame(rc_rows)
    boot_df = pd.DataFrame(boot_rows)
    pred_df = pd.DataFrame({"case_id": case_ids, "label_mean_fg_dice": mean_fg})
    for fdef in fail_defs:
        pred_df[f"fail_{fdef}"] = fail_labels_store[fdef]
        for method in feature_sets:
            pred_df[f"score_{method}__{fdef}"] = scores[(method, fdef)]

    metrics_df.to_csv(out_dir / "classification_metrics.csv", index=False)
    rc_df.to_csv(out_dir / "risk_coverage_metrics.csv", index=False)
    boot_df.to_csv(out_dir / "bootstrap_comparisons.csv", index=False)
    pred_df.to_csv(out_dir / "heldout_predictions.csv", index=False)

    # Short report
    fdef = "lowest20_mean_fg"
    sub = metrics_df[metrics_df.failure_def == fdef].sort_values("auprc", ascending=False)
    lines = [
        "# Baseline comparisons (leakage-safe nested CV)",
        "",
        "Same outer folds as consistency experiment (seed 42). "
        "Confidence is the new GT-free TTA summaries. "
        "`combined` = inconsistency + morphology (prior definition).",
        "",
        f"## Lowest-20% mean-fg Dice (n_pos≈{int(sub.iloc[0]['n_pos'])})",
        "",
        "| Method | AUPRC | Capture@20% | AURC |",
        "|---|---:|---:|---:|",
    ]
    for _, r in sub.iterrows():
        lines.append(
            f"| {r['method']} | {r['auprc']:.3f} | {r['capture_at_20']:.3f} | {r['aurc']:.4f} |"
        )
    lines += ["", "## Key paired AUPRC differences (lowest20)", ""]
    boot_sub = boot_df[boot_df.failure_def == fdef]
    for key in [
        ("combined", "confidence"),
        ("combined_conf", "combined_pooled"),
        ("combined_conf", "combined"),
        ("combined_pooled", "combined"),
        ("pooled", "combined"),
    ]:
        row = boot_sub[(boot_sub.method_a == key[0]) & (boot_sub.method_b == key[1])].iloc[0]
        sig = "CI excludes 0" if row["ci_low"] > 0 or row["ci_high"] < 0 else "CI includes 0"
        lines.append(
            f"- **{key[0]} − {key[1]}**: Δ={row['diff_mean']:+.3f} "
            f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] "
            f"(P(a>b)={row['frac_a_better']:.2f}; {sig})"
        )
    report_path = out_dir / "comparison_report.md"
    report_path.write_text("\n".join(lines) + "\n")

    return {
        "stage": "compare_baselines",
        "n_cases": n,
        "output_dir": str(out_dir),
        "report": str(report_path),
        "lowest20_table": sub[["method", "auprc", "capture_at_20"]].to_dict(orient="records"),
    }


def run_stage(config_path: Path, stage: str, **kwargs: Any) -> dict[str, Any]:
    config = load_config(config_path)
    stage = stage or config.get("default_stage", "inventory")
    ensure_dir(Path(config["paths"]["output_dir"]))

    if stage == "inventory":
        tables = load_reusable_tables(config)
        path = write_inventory(config, tables)
        return {
            "stage": stage,
            "inventory": str(path),
            "n_cases": len(tables["case_ids"]),
            "folds_match_seed42": tables["folds_match_seed42"],
            "confidence_ready": tables["confidence"] is not None,
        }

    if stage == "confidence":
        max_cases = kwargs.get("max_cases")
        conf_df = generate_confidence_features(config, max_cases=max_cases)
        return {
            "stage": stage,
            "n_cases": len(conf_df),
            "path": str(Path(config["paths"]["output_dir"]) / "case_level_confidence_features.csv"),
        }

    if stage == "dry_check":
        return {"stage": stage, **dry_check(config)}

    if stage == "compare_baselines":
        return run_baseline_comparisons(config)

    if stage == "full":
        return run_full_nested_cv(config)

    raise ValueError(
        f"Unknown stage: {stage}. Use inventory|confidence|dry_check|compare_baselines|full."
    )
