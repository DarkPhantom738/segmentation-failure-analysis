"""Dominant failure label assignment for geometry analysis."""

from __future__ import annotations

from typing import Any


def assign_dominant_failure_label(row: dict[str, Any]) -> str:
    """
    Assign one dominant failure label per case using simple priority rules.

    Rules are evaluated in order; the first matching rule wins:
      1. confident_false_negative
      2. small_lesion_miss
      3. boundary_error
      4. false_negative
      5. false_positive
      6. mixed_or_low_error
    """
    if float(row["confident_false_negative_fraction"]) > 0.25:
        return "confident_false_negative"

    if int(row["missed_small_lesion_count"]) > 0:
        return "small_lesion_miss"

    if float(row["boundary_error_fraction"]) > 0.5:
        return "boundary_error"

    false_negative_voxels = int(row["false_negative_voxels"])
    false_positive_voxels = int(row["false_positive_voxels"])

    if false_negative_voxels > false_positive_voxels:
        return "false_negative"

    if false_positive_voxels > false_negative_voxels:
        return "false_positive"

    return "mixed_or_low_error"
