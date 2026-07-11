"""Standard BraTS MRI preprocessing utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.ndimage import zoom


# Original BraTS label 4 (enhancing tumor) is remapped to class index 3.
BRATS_LABEL_REMAP = {0: 0, 1: 1, 2: 2, 4: 3}


def remap_brats_labels(segmentation: np.ndarray) -> np.ndarray:
    """Remap BraTS segmentation labels {0,1,2,4} to contiguous {0,1,2,3}."""
    remapped = np.zeros_like(segmentation, dtype=np.int16)
    for original, mapped in BRATS_LABEL_REMAP.items():
        remapped[segmentation == original] = mapped
    return remapped


def resample_volume(
    volume: np.ndarray,
    original_spacing: Sequence[float],
    target_spacing: Sequence[float],
    order: int = 1,
) -> np.ndarray:
    """Resample a 3D volume to target voxel spacing using scipy zoom."""
    original_spacing = np.asarray(original_spacing, dtype=np.float64)
    target_spacing = np.asarray(target_spacing, dtype=np.float64)
    zoom_factors = original_spacing / target_spacing
    return zoom(volume, zoom_factors, order=order, mode="nearest")


def normalize_intensities(
    volume: np.ndarray,
    percentile_clip: Sequence[float] = (1.0, 99.0),
    eps: float = 1e-8,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Per-volume intensity normalization used commonly in BraTS pipelines:
    clip to [p_low, p_high] percentiles over foreground voxels, then z-score.

    If `mask` is provided, statistics are computed only where mask is True
    (typically nonzero brain voxels). Background voxels are set to 0.
    """
    volume = volume.astype(np.float32)
    if mask is None:
        mask = volume > 0
    else:
        mask = mask.astype(bool)

    if not mask.any():
        return np.zeros_like(volume, dtype=np.float32)

    foreground = volume[mask]
    p_low, p_high = percentile_clip
    lower = np.percentile(foreground, p_low)
    upper = np.percentile(foreground, p_high)
    clipped = np.clip(volume, lower, upper)

    mean = clipped[mask].mean()
    std = clipped[mask].std()
    normalized = (clipped - mean) / (std + eps)
    normalized[~mask] = 0.0
    return normalized.astype(np.float32)


def crop_to_nonzero(
    image: np.ndarray,
    segmentation: np.ndarray | None = None,
    margin: int = 0,
) -> tuple[np.ndarray, np.ndarray | None, tuple[slice, slice, slice]]:
    """
    Crop image (and optional segmentation) to the bounding box of non-zero voxels.

    Returns cropped image, cropped segmentation (or None), and the crop slices
    applied along each spatial axis.
    """
    nonzero = np.argwhere(image > 0)
    if nonzero.size == 0:
        identity = (slice(None), slice(None), slice(None))
        return image, segmentation, identity

    mins = nonzero.min(axis=0)
    maxs = nonzero.max(axis=0) + 1

    mins = np.maximum(mins - margin, 0)
    maxs = np.minimum(maxs + margin, np.array(image.shape))

    z_slice = slice(int(mins[0]), int(maxs[0]))
    y_slice = slice(int(mins[1]), int(maxs[1]))
    x_slice = slice(int(mins[2]), int(maxs[2]))
    crop_slices = (z_slice, y_slice, x_slice)

    cropped_image = image[z_slice, y_slice, x_slice]
    cropped_seg = None
    if segmentation is not None:
        cropped_seg = segmentation[z_slice, y_slice, x_slice]
    return cropped_image, cropped_seg, crop_slices


def pad_or_crop_to_shape(
    volume: np.ndarray,
    target_shape: Sequence[int],
    constant_value: float | int = 0,
) -> np.ndarray:
    """Center-pad or center-crop a 3D volume to the target shape."""
    target_shape = tuple(int(s) for s in target_shape)
    result = np.full(target_shape, constant_value, dtype=volume.dtype)

    src_slices = []
    dst_slices = []
    for src_size, dst_size in zip(volume.shape, target_shape):
        if src_size >= dst_size:
            start = (src_size - dst_size) // 2
            src_slices.append(slice(start, start + dst_size))
            dst_slices.append(slice(0, dst_size))
        else:
            start = (dst_size - src_size) // 2
            src_slices.append(slice(0, src_size))
            dst_slices.append(slice(start, start + src_size))

    result[dst_slices[0], dst_slices[1], dst_slices[2]] = volume[
        src_slices[0], src_slices[1], src_slices[2]
    ]
    return result


def random_crop(
    image: np.ndarray,
    segmentation: np.ndarray,
    patch_size: Sequence[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract a random 3D patch from (C, D, H, W) image and (D, H, W) segmentation.

    Samples a random origin in the original volume. Pads only axes that are
    smaller than `patch_size`; never center-crops large volumes first.
    """
    patch_size = tuple(int(s) for s in patch_size)

    if image.ndim != 4:
        raise ValueError(f"Expected image with shape (C, D, H, W), got {image.shape}")

    channels = image.shape[0]
    depth, height, width = image.shape[1:]

    # Random origin in the original volume (0 if that axis is smaller than patch).
    max_z = max(depth - patch_size[0], 0)
    max_y = max(height - patch_size[1], 0)
    max_x = max(width - patch_size[2], 0)
    z0 = int(rng.integers(0, max_z + 1)) if max_z > 0 else 0
    y0 = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
    x0 = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0

    z1 = min(z0 + patch_size[0], depth)
    y1 = min(y0 + patch_size[1], height)
    x1 = min(x0 + patch_size[2], width)

    patch_image = np.zeros(
        (channels, patch_size[0], patch_size[1], patch_size[2]),
        dtype=image.dtype,
    )
    patch_seg = np.zeros(patch_size, dtype=segmentation.dtype)

    patch_image[:, : z1 - z0, : y1 - y0, : x1 - x0] = image[:, z0:z1, y0:y1, x0:x1]
    patch_seg[: z1 - z0, : y1 - y0, : x1 - x0] = segmentation[z0:z1, y0:y1, x0:x1]
    return patch_image, patch_seg
