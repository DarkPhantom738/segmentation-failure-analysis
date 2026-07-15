"""Fixed patient splits for converged multi-seed U-Net training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

from src.data.brats_dataset import (
    PREPROCESS_VERSION,
    discover_brats_cases,
    split_cases,
)


def seed_output_dirname(seed: int) -> str:
    """Directory name for a model seed (seed_042, seed_123, seed_2026, ...)."""
    if seed < 1000:
        return f"seed_{seed:03d}"
    return f"seed_{seed}"


def case_list_hash(case_ids: Sequence[str]) -> str:
    """Stable SHA256 over sorted case IDs."""
    payload = "\n".join(sorted(case_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def discover_cases_for_converged_training(
    data_root: Path,
    cache_dir: Path | None,
    modalities: Sequence[str],
    target_spacing: Sequence[float],
    percentile_clip: Sequence[float],
) -> list[str]:
    """
    Discover BraTS case IDs for the converged-training cohort.

    Prefer preprocessed cache entries that match the current preprocess key. The
    historical 1,251-case run is recoverable from cache even when only a small
    NIfTI subset is present locally. Do **not** union NIfTI-only demo cases onto
    a full cache cohort (that would change the 876/375 split).
    """
    data_root = Path(data_root)
    cache_cases: set[str] = set()

    if cache_dir is not None and Path(cache_dir).exists():
        spacing = "_".join(str(float(s)) for s in target_spacing)
        clip = "_".join(str(float(p)) for p in percentile_clip)
        mods = "-".join(modalities)
        image_tail = f"_{PREPROCESS_VERSION}_mod{mods}_sp{spacing}_pc{clip}_image.npy"
        seg_tail = f"_{PREPROCESS_VERSION}_mod{mods}_sp{spacing}_pc{clip}_seg.npy"
        cache_path = Path(cache_dir)
        for path in cache_path.glob(f"*{image_tail}"):
            case_id = path.name[: -len(image_tail)]
            if case_id and (cache_path / f"{case_id}{seg_tail}").exists():
                cache_cases.add(case_id)

    if cache_cases:
        return sorted(cache_cases)

    if data_root.exists():
        return discover_brats_cases(data_root)

    raise FileNotFoundError(
        f"No BraTS cases found under data_root={data_root} or cache_dir={cache_dir}"
    )


def build_converged_splits(
    all_cases: Sequence[str],
    *,
    data_split_seed: int = 42,
    development_split_seed: int = 31415,
    final_validation_fraction: float = 0.30,
    development_fraction_of_training_pool: float = 0.10,
) -> dict[str, list[str]]:
    """
    Build train / development / final-evaluation lists.

    Step 1 (data_split_seed): recreates the historical 876 / 375 split.
    Step 2 (development_split_seed): splits the 876-case pool into ~788 train / ~88 devel.
    """
    train_pool, final_evaluation = split_cases(
        list(all_cases),
        val_fraction=final_validation_fraction,
        seed=data_split_seed,
    )
    train_cases, development_cases = split_cases(
        train_pool,
        val_fraction=development_fraction_of_training_pool,
        seed=development_split_seed,
    )
    return {
        "train_cases": train_cases,
        "development_cases": development_cases,
        "final_evaluation_cases": final_evaluation,
        "training_pool_cases": train_pool,
    }


def assert_disjoint_splits(splits: dict[str, list[str]]) -> None:
    train = set(splits["train_cases"])
    devel = set(splits["development_cases"])
    final = set(splits["final_evaluation_cases"])
    if train & devel:
        raise ValueError(f"Train/development overlap: {sorted(train & devel)[:5]}")
    if train & final:
        raise ValueError(f"Train/final overlap: {sorted(train & final)[:5]}")
    if devel & final:
        raise ValueError(f"Development/final overlap: {sorted(devel & final)[:5]}")


def save_split_lists(splits: dict[str, list[str]], output_dir: Path) -> dict[str, str]:
    """Write shared case lists and return path map."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train_cases": output_dir / "train_cases.json",
        "development_cases": output_dir / "development_cases.json",
        "final_evaluation_cases": output_dir / "final_evaluation_cases.json",
    }
    for key, path in paths.items():
        with path.open("w") as handle:
            json.dump(splits[key], handle, indent=2)
            handle.write("\n")

    meta = {
        "n_train": len(splits["train_cases"]),
        "n_development": len(splits["development_cases"]),
        "n_final_evaluation": len(splits["final_evaluation_cases"]),
        "n_training_pool": len(splits["training_pool_cases"]),
        "hash_train": case_list_hash(splits["train_cases"]),
        "hash_development": case_list_hash(splits["development_cases"]),
        "hash_final_evaluation": case_list_hash(splits["final_evaluation_cases"]),
        "hash_training_pool": case_list_hash(splits["training_pool_cases"]),
    }
    meta_path = output_dir / "split_metadata.json"
    with meta_path.open("w") as handle:
        json.dump(meta, handle, indent=2)
        handle.write("\n")
    return {k: str(v) for k, v in paths.items()}


def load_split_lists(shared_split_dir: Path) -> dict[str, list[str]]:
    shared_split_dir = Path(shared_split_dir)
    out: dict[str, list[str]] = {}
    for key in ("train_cases", "development_cases", "final_evaluation_cases"):
        with (shared_split_dir / f"{key}.json").open() as handle:
            out[key] = list(json.load(handle))
    out["training_pool_cases"] = sorted(
        set(out["train_cases"]) | set(out["development_cases"])
    )
    return out


def prepare_or_load_shared_splits(
    config: dict,
    *,
    historical_final_ids: Sequence[str] | None = None,
    rewrite: bool = False,
) -> dict[str, list[str]]:
    """
    Build shared splits once and reuse them for every model seed.

    If historical_final_ids is provided, require an exact match to final_evaluation.
    """
    data_cfg = config["data"]
    shared_dir = Path(config["output"]["shared_split_dir"])
    required = [
        shared_dir / "train_cases.json",
        shared_dir / "development_cases.json",
        shared_dir / "final_evaluation_cases.json",
    ]
    if all(p.exists() for p in required) and not rewrite:
        splits = load_split_lists(shared_dir)
    else:
        all_cases = discover_cases_for_converged_training(
            data_root=Path(data_cfg["root"]),
            cache_dir=Path(data_cfg["cache_dir"]) if data_cfg.get("cache_dir") else None,
            modalities=data_cfg["modalities"],
            target_spacing=data_cfg["target_spacing"],
            percentile_clip=data_cfg["percentile_clip"],
        )
        max_cases = data_cfg.get("max_cases")
        if max_cases is not None and int(max_cases) > 0:
            all_cases = all_cases[: int(max_cases)]
        splits = build_converged_splits(
            all_cases,
            data_split_seed=int(data_cfg["data_split_seed"]),
            development_split_seed=int(data_cfg["development_split_seed"]),
            final_validation_fraction=float(data_cfg["final_validation_fraction"]),
            development_fraction_of_training_pool=float(
                data_cfg["development_fraction_of_training_pool"]
            ),
        )
        assert_disjoint_splits(splits)
        save_split_lists(splits, shared_dir)

    assert_disjoint_splits(splits)

    if historical_final_ids is not None:
        hist = set(historical_final_ids)
        final = set(splits["final_evaluation_cases"])
        if hist != final:
            missing = sorted(hist - final)[:5]
            extra = sorted(final - hist)[:5]
            raise ValueError(
                "Final-evaluation IDs do not match historical cohort. "
                f"missing_from_split={missing} extra_in_split={extra}"
            )
    return splits
