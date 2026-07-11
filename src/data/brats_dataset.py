"""BraTS dataset and dataloader."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.preprocessing import (
    crop_to_nonzero,
    normalize_intensities,
    random_crop,
    remap_brats_labels,
    resample_volume,
)


MODALITY_SUFFIXES = {
    "t1": "_t1.nii.gz",
    "t1ce": "_t1ce.nii.gz",
    "t2": "_t2.nii.gz",
    "flair": "_flair.nii.gz",
}


def discover_brats_cases(data_root: Path) -> list[str]:
    """Find all BraTS case directories under data_root."""
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(
            f"BraTS data root not found: {data_root}. "
            "Download BraTS and set data.root in the config."
        )

    cases = sorted(
        p.name
        for p in data_root.iterdir()
        if p.is_dir() and (p / f"{p.name}_flair.nii.gz").exists()
    )
    if not cases:
        raise FileNotFoundError(f"No BraTS cases found under {data_root}")
    return cases


def limit_cases(cases: Sequence[str], max_cases: int | None) -> list[str]:
    """Keep a deterministic subset of cases for fast experiments."""
    cases = list(cases)
    if max_cases is None or max_cases <= 0 or max_cases >= len(cases):
        return cases
    return cases[:max_cases]


def split_cases(
    cases: Sequence[str],
    val_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Deterministically split case IDs into train and validation sets."""
    cases = list(cases)
    rng = np.random.default_rng(seed)
    rng.shuffle(cases)
    n_val = max(1, int(round(len(cases) * val_fraction)))
    val_cases = sorted(cases[:n_val])
    train_cases = sorted(cases[n_val:])
    return train_cases, val_cases


class BraTSCase:
    """Load and preprocess a single BraTS case."""

    def __init__(
        self,
        case_id: str,
        data_root: Path,
        modalities: Sequence[str],
        target_spacing: Sequence[float],
        percentile_clip: Sequence[float],
    ) -> None:
        self.case_id = case_id
        self.data_root = Path(data_root)
        self.modalities = list(modalities)
        self.target_spacing = list(target_spacing)
        self.percentile_clip = list(percentile_clip)

    def _load_nifti(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        nii = nib.load(str(path))
        data = np.asarray(nii.get_fdata(dtype=np.float32))
        spacing = np.asarray(nii.header.get_zooms()[:3], dtype=np.float64)
        return data, spacing

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load multimodal image and segmentation for this case.

        Returns:
            image: (C, D, H, W) float32
            segmentation: (D, H, W) int16 with labels in {0,1,2,3}
        """
        case_dir = self.data_root / self.case_id
        raw_channels: list[np.ndarray] = []

        for modality in self.modalities:
            suffix = MODALITY_SUFFIXES[modality]
            path = case_dir / f"{self.case_id}{suffix}"
            if not path.exists():
                raise FileNotFoundError(f"Missing modality file: {path}")
            volume, vol_spacing = self._load_nifti(path)
            # Resample first; keep raw intensities for foreground masking.
            volume = resample_volume(volume, vol_spacing, self.target_spacing, order=1)
            raw_channels.append(volume)

        seg_path = case_dir / f"{self.case_id}_seg.nii.gz"
        segmentation, seg_spacing = self._load_nifti(seg_path)
        segmentation = resample_volume(
            segmentation, seg_spacing, self.target_spacing, order=0
        )
        segmentation = remap_brats_labels(segmentation.astype(np.int16))

        # Crop using the original nonzero FLAIR mask BEFORE normalization.
        # After z-score, background zeros become negative and brain voxels
        # below the mean are also negative, so image > 0 is not a brain mask.
        flair_raw = raw_channels[-1]
        _, segmentation, crop_slices = crop_to_nonzero(
            flair_raw, segmentation, margin=8
        )
        z_slice, y_slice, x_slice = crop_slices
        cropped_raw = [
            channel[z_slice, y_slice, x_slice] for channel in raw_channels
        ]

        # Normalize using nonzero brain voxels only (common BraTS practice).
        normalized_channels = []
        for channel in cropped_raw:
            foreground = channel > 0
            normalized_channels.append(
                normalize_intensities(
                    channel,
                    self.percentile_clip,
                    mask=foreground,
                )
            )
        image = np.stack(normalized_channels, axis=0)

        return image.astype(np.float32), segmentation.astype(np.int16)


# Bump this whenever preprocessing logic changes so stale caches are not reused.
PREPROCESS_VERSION = "v2_crop_raw_norm_nonzero"


def _cache_key(
    case_id: str,
    modalities: Sequence[str],
    target_spacing: Sequence[float],
    percentile_clip: Sequence[float],
) -> str:
    spacing = "_".join(str(float(s)) for s in target_spacing)
    clip = "_".join(str(float(p)) for p in percentile_clip)
    mods = "-".join(modalities)
    return f"{case_id}_{PREPROCESS_VERSION}_mod{mods}_sp{spacing}_pc{clip}"


def load_case_cached(
    case_id: str,
    data_root: Path,
    modalities: Sequence[str],
    target_spacing: Sequence[float],
    percentile_clip: Sequence[float],
    cache_dir: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a BraTS case, optionally reading/writing a preprocessed disk cache."""
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = _cache_key(case_id, modalities, target_spacing, percentile_clip)
        image_path = cache_dir / f"{key}_image.npy"
        seg_path = cache_dir / f"{key}_seg.npy"
        if image_path.exists() and seg_path.exists():
            return np.load(image_path), np.load(seg_path)

    loader = BraTSCase(
        case_id=case_id,
        data_root=data_root,
        modalities=modalities,
        target_spacing=target_spacing,
        percentile_clip=percentile_clip,
    )
    image, segmentation = loader.load()

    if cache_dir is not None:
        np.save(image_path, image)
        np.save(seg_path, segmentation)

    return image, segmentation


class BraTSTrainingDataset(Dataset):
    """Patch-based training dataset over BraTS cases."""

    def __init__(
        self,
        case_ids: Sequence[str],
        data_root: Path,
        modalities: Sequence[str],
        target_spacing: Sequence[float],
        percentile_clip: Sequence[float],
        patch_size: Sequence[int],
        patches_per_volume: int,
        seed: int,
        preload: bool = False,
        cache_dir: Path | None = None,
    ) -> None:
        self.case_ids = list(case_ids)
        self.data_root = Path(data_root)
        self.modalities = list(modalities)
        self.target_spacing = list(target_spacing)
        self.percentile_clip = list(percentile_clip)
        self.patch_size = tuple(int(s) for s in patch_size)
        self.patches_per_volume = patches_per_volume
        self.rng = np.random.default_rng(seed)
        self.preload = preload
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._cases: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        if self.preload:
            for case_id in self.case_ids:
                self._cases[case_id] = self._load_case(case_id)

    def _load_case(self, case_id: str) -> tuple[np.ndarray, np.ndarray]:
        return load_case_cached(
            case_id=case_id,
            data_root=self.data_root,
            modalities=self.modalities,
            target_spacing=self.target_spacing,
            percentile_clip=self.percentile_clip,
            cache_dir=self.cache_dir,
        )

    def __len__(self) -> int:
        return len(self.case_ids) * self.patches_per_volume

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        case_index = index % len(self.case_ids)
        case_id = self.case_ids[case_index]
        if self.preload:
            image, segmentation = self._cases[case_id]
        else:
            image, segmentation = self._load_case(case_id)
        patch_image, patch_seg = random_crop(
            image, segmentation, self.patch_size, self.rng
        )
        return {
            "image": torch.from_numpy(patch_image),
            "segmentation": torch.from_numpy(patch_seg).long(),
            "case_id": case_id,
        }


class BraTSVolumeDataset(Dataset):
    """Full-volume dataset used during validation artifact export."""

    def __init__(
        self,
        case_ids: Sequence[str],
        data_root: Path,
        modalities: Sequence[str],
        target_spacing: Sequence[float],
        percentile_clip: Sequence[float],
        preload: bool = False,
        cache_dir: Path | None = None,
    ) -> None:
        self.case_ids = list(case_ids)
        self.data_root = Path(data_root)
        self.modalities = list(modalities)
        self.target_spacing = list(target_spacing)
        self.percentile_clip = list(percentile_clip)
        self.preload = preload
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._cases: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        if self.preload:
            for case_id in self.case_ids:
                self._cases[case_id] = self._load_case(case_id)

    def _load_case(self, case_id: str) -> tuple[np.ndarray, np.ndarray]:
        return load_case_cached(
            case_id=case_id,
            data_root=self.data_root,
            modalities=self.modalities,
            target_spacing=self.target_spacing,
            percentile_clip=self.percentile_clip,
            cache_dir=self.cache_dir,
        )

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        case_id = self.case_ids[index]
        if self.preload:
            image, segmentation = self._cases[case_id]
        else:
            image, segmentation = self._load_case(case_id)
        return {
            "image": torch.from_numpy(image),
            "segmentation": torch.from_numpy(segmentation).long(),
            "case_id": case_id,
        }


def build_dataloaders(
    config: dict,
    train_cases: list[str],
    val_cases: list[str],
) -> tuple[DataLoader, DataLoader]:
    """Create train (patch) and validation (full-volume) dataloaders."""
    data_cfg = config["data"]
    train_cfg = config["training"]
    preload = bool(data_cfg.get("preload", False))
    cache_dir = data_cfg.get("cache_dir")

    train_dataset = BraTSTrainingDataset(
        case_ids=train_cases,
        data_root=Path(data_cfg["root"]),
        modalities=data_cfg["modalities"],
        target_spacing=data_cfg["target_spacing"],
        percentile_clip=data_cfg["percentile_clip"],
        patch_size=data_cfg["patch_size"],
        patches_per_volume=data_cfg["patches_per_volume"],
        seed=config["seed"],
        preload=preload,
        cache_dir=Path(cache_dir) if cache_dir else None,
    )
    val_dataset = BraTSVolumeDataset(
        case_ids=val_cases,
        data_root=Path(data_cfg["root"]),
        modalities=data_cfg["modalities"],
        target_spacing=data_cfg["target_spacing"],
        percentile_clip=data_cfg["percentile_clip"],
        preload=preload,
        cache_dir=Path(cache_dir) if cache_dir else None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader
