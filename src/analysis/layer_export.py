"""Export globally pooled embeddings from multiple U-Net layers.

For each validation case, run sliding-window inference with layer hooks and
write one .npy vector per stage under ``{output_dir}/{layer}/{case_id}.npy``,
plus a CSV index used by consistency / triage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from src.data.brats_dataset import BraTSCase
from src.models.unet3d import LAYER_NAMES, build_model
from src.training.layer_inference import sliding_window_layer_inference
from src.utils.io import ensure_dir, save_array


def load_config(config_path: Path) -> dict:
    with config_path.open("r") as handle:
        return yaml.safe_load(handle)


def export_layer_embeddings(
    config: dict,
    checkpoint_path: Path,
    failure_table_path: Path,
    output_dir: Path,
    device: torch.device,
    overlap: float | None = None,
    max_cases: int | None = None,
) -> pd.DataFrame:
    """
    Run inference on validation cases and save per-layer globally pooled vectors.

    Writes:
      {output_dir}/{layer_name}/{case_id}.npy
      {output_dir}/layer_embedding_index.csv
    """
    output_dir = ensure_dir(output_dir)
    for layer_name in LAYER_NAMES:
        ensure_dir(output_dir / layer_name)

    failure_df = pd.read_csv(failure_table_path)
    if max_cases is not None and max_cases > 0:
        failure_df = failure_df.head(max_cases)

    model = build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    patch_size = config["data"]["patch_size"]
    num_classes = config["model"]["num_classes"]
    data_cfg = config["data"]
    if overlap is None:
        overlap = float(config.get("inference", config.get("uncertainty", {})).get("overlap", 0.25))

    index_rows: list[dict[str, str | float]] = []

    for record in tqdm(failure_df.to_dict(orient="records"), desc="Export layer embeddings"):
        case_id = str(record["case_id"])
        layer_paths: dict[str, str] = {}
        all_exist = True
        for layer_name in LAYER_NAMES:
            out_path = output_dir / layer_name / f"{case_id}.npy"
            layer_paths[layer_name] = str(out_path)
            if not out_path.exists():
                all_exist = False

        if not all_exist:
            loader = BraTSCase(
                case_id=case_id,
                data_root=Path(data_cfg["root"]),
                modalities=data_cfg["modalities"],
                target_spacing=data_cfg["target_spacing"],
                percentile_clip=data_cfg["percentile_clip"],
            )
            image, _ = loader.load()
            image_tensor = torch.from_numpy(image)
            _, layer_embeddings = sliding_window_layer_inference(
                model=model,
                image=image_tensor,
                patch_size=patch_size,
                num_classes=num_classes,
                overlap=overlap,
                device=device,
            )
            for layer_name in LAYER_NAMES:
                emb_np = layer_embeddings[layer_name].cpu().numpy().astype(np.float32)
                save_array(emb_np, layer_paths[layer_name])

        row = {
            "case_id": case_id,
            "dice": float(record.get("dice", float("nan"))),
            "path_prediction": str(record.get("path_prediction", "")),
            "path_ground_truth": str(record.get("path_ground_truth", "")),
            "path_entropy": str(record.get("path_entropy", "")),
        }
        for layer_name in LAYER_NAMES:
            row[f"path_{layer_name}"] = layer_paths[layer_name]
        index_rows.append(row)

    index_df = pd.DataFrame(index_rows)
    index_path = output_dir / "layer_embedding_index.csv"
    index_df.to_csv(index_path, index=False)
    return index_df
