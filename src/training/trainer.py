"""Training and validation artifact export."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.unet3d import DiceCrossEntropyLoss
from src.training.inference import (
    predict_probabilities,
    predict_segmentation,
    sliding_window_inference,
)
from src.training.metrics import whole_tumor_dice
from src.training.tta import (
    tta_predict_entropy,
    tta_predict_probabilities,
    tta_predict_segmentation,
    tta_sliding_window_inference,
)
from src.utils.io import ensure_dir, save_array


class Trainer:
    """Train a 3D U-Net on BraTS and export validation artifacts."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        train_cfg = config["training"]
        model_cfg = config["model"]
        self.criterion = DiceCrossEntropyLoss(
            dice_weight=train_cfg["dice_weight"],
            num_classes=model_cfg["num_classes"],
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )

        output_cfg = config["output"]
        self.output_dir = ensure_dir(output_cfg["dir"])
        self.checkpoint_dir = ensure_dir(output_cfg["checkpoint_dir"])

        self.predictions_dir = ensure_dir(self.output_dir / "predictions")
        self.ground_truth_dir = ensure_dir(self.output_dir / "ground_truth")
        self.embeddings_dir = ensure_dir(self.output_dir / "embeddings")
        self.probabilities_dir = ensure_dir(self.output_dir / "probabilities")
        self.probabilities_tta_dir = ensure_dir(self.output_dir / "probabilities_tta")
        self.uncertainty_dir = ensure_dir(self.output_dir / "uncertainty")

    def train(self) -> None:
        """Run the full training schedule."""
        epochs = self.config["training"]["epochs"]
        val_every = self.config["training"]["val_artifact_every"]
        ckpt_every = self.config["training"]["checkpoint_every"]

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(epoch)
            print(f"Epoch {epoch}/{epochs} — train loss: {train_loss:.4f}")

            if epoch % val_every == 0 or epoch == epochs:
                mean_dice = self.export_validation_artifacts(epoch)
                print(f"Epoch {epoch}/{epochs} — val whole-tumor Dice: {mean_dice:.4f}")

            if epoch % ckpt_every == 0 or epoch == epochs:
                self._save_checkpoint(epoch)

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        progress = tqdm(self.train_loader, desc=f"Train epoch {epoch}", leave=False)
        for batch in progress:
            images = batch["image"].to(self.device)
            targets = batch["segmentation"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            logits, _, _ = self.model(images)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            num_batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        return running_loss / max(num_batches, 1)

    @torch.no_grad()
    def export_validation_artifacts(self, epoch: int) -> float:
        """
        Run validation inference and save per-case artifacts.

        Saved per validation case:
          - ground-truth mask
          - predicted mask
          - softmax probability map
          - bottleneck embedding (GAP feature vector)
          - whole-tumor Dice score (recorded in metrics.csv)
        """
        self.model.eval()
        metrics_rows: list[dict[str, str | float]] = []
        dice_scores: list[float] = []

        epoch_tag = f"epoch_{epoch:03d}"
        epoch_predictions = ensure_dir(self.predictions_dir / epoch_tag)
        epoch_ground_truth = ensure_dir(self.ground_truth_dir / epoch_tag)
        epoch_embeddings = ensure_dir(self.embeddings_dir / epoch_tag)
        epoch_probabilities = ensure_dir(self.probabilities_dir / epoch_tag)

        patch_size = self.config["data"]["patch_size"]
        num_classes = self.config["model"]["num_classes"]
        overlap = self._inference_overlap()

        for batch in tqdm(self.val_loader, desc=f"Validate epoch {epoch}", leave=False):
            case_id = batch["case_id"][0]
            image = batch["image"][0]
            ground_truth = batch["segmentation"][0].cpu().numpy().astype(np.uint8)

            logits, embedding, _ = sliding_window_inference(
                model=self.model,
                image=image,
                patch_size=patch_size,
                num_classes=num_classes,
                overlap=overlap,
                device=self.device,
            )

            prediction = predict_segmentation(logits)
            probabilities = predict_probabilities(logits)
            dice = whole_tumor_dice(prediction, ground_truth)

            gt_path = save_array(
                ground_truth,
                epoch_ground_truth / f"{case_id}_gt.npy",
            )
            pred_path = save_array(
                prediction,
                epoch_predictions / f"{case_id}_pred.npy",
            )
            prob_path = save_array(
                probabilities,
                epoch_probabilities / f"{case_id}_probs.npy",
            )
            emb_path = save_array(
                embedding.cpu().numpy(),
                epoch_embeddings / f"{case_id}_embedding.npy",
            )

            metrics_rows.append(
                {
                    "epoch": epoch,
                    "case_id": case_id,
                    "dice": dice,
                    "path_ground_truth": gt_path,
                    "path_prediction": pred_path,
                    "path_probability": prob_path,
                    "path_embedding": emb_path,
                }
            )
            dice_scores.append(dice)

        self._write_metrics(metrics_rows, epoch)
        return float(sum(dice_scores) / max(len(dice_scores), 1))

    @torch.no_grad()
    def export_tta_artifacts(self, epoch: int) -> float:
        """
        Run TTA inference and save predictions + entropy maps.

        Used only when rebuilding ``failure_metrics.csv`` (via ``analyze_failures.py``).
        """
        self.model.eval()
        metrics_rows: list[dict[str, str | float]] = []
        dice_scores: list[float] = []

        epoch_tag = f"epoch_{epoch:03d}"
        epoch_predictions = ensure_dir(self.predictions_dir / epoch_tag)
        epoch_ground_truth = ensure_dir(self.ground_truth_dir / epoch_tag)
        epoch_probabilities_tta = ensure_dir(self.probabilities_tta_dir / epoch_tag)
        epoch_uncertainty = ensure_dir(self.uncertainty_dir / epoch_tag)

        patch_size = self.config["data"]["patch_size"]
        num_classes = self.config["model"]["num_classes"]
        overlap = self._inference_overlap()
        inference_cfg = self.config.get("inference", self.config.get("uncertainty", {}))

        for batch in tqdm(
            self.val_loader, desc=f"TTA export epoch {epoch}", leave=False
        ):
            case_id = batch["case_id"][0]
            image = batch["image"][0]
            ground_truth = batch["segmentation"][0].cpu().numpy().astype(np.uint8)

            mean_probabilities = tta_sliding_window_inference(
                model=self.model,
                image=image,
                patch_size=patch_size,
                num_classes=num_classes,
                device=self.device,
                overlap=overlap,
                max_augmentations=inference_cfg.get(
                    "tta_augmentations",
                    inference_cfg.get("max_augmentations"),
                ),
            )

            # Also run standard sliding-window once so embeddings exist.
            _, embedding, _ = sliding_window_inference(
                model=self.model,
                image=image,
                patch_size=patch_size,
                num_classes=num_classes,
                overlap=overlap,
                device=self.device,
            )

            prediction = tta_predict_segmentation(mean_probabilities)
            probabilities_tta = tta_predict_probabilities(mean_probabilities)
            entropy = tta_predict_entropy(mean_probabilities)
            dice = whole_tumor_dice(prediction, ground_truth)

            gt_path = self._existing_or_save_ground_truth(
                ground_truth, epoch_ground_truth, case_id
            )
            pred_path = save_array(
                prediction,
                epoch_predictions / f"{case_id}_pred_tta.npy",
            )
            prob_tta_path = save_array(
                probabilities_tta,
                epoch_probabilities_tta / f"{case_id}_probs_tta.npy",
            )
            entropy_path = save_array(
                entropy,
                epoch_uncertainty / f"{case_id}_entropy.npy",
            )

            epoch_embeddings = ensure_dir(self.embeddings_dir / epoch_tag)
            emb_path = save_array(
                embedding.cpu().numpy(),
                epoch_embeddings / f"{case_id}_embedding.npy",
            )
            prob_path, _ = self._resolve_non_tta_paths(epoch_tag, case_id)

            metrics_rows.append(
                {
                    "epoch": epoch,
                    "case_id": case_id,
                    "dice": dice,
                    "path_ground_truth": gt_path,
                    "path_prediction": pred_path,
                    "path_probability": prob_path,
                    "path_embedding": emb_path,
                    "path_tta_probability": prob_tta_path,
                    "path_entropy": entropy_path,
                }
            )
            dice_scores.append(dice)

        self._write_tta_metrics(metrics_rows, epoch)
        return float(sum(dice_scores) / max(len(dice_scores), 1))

    def _inference_overlap(self) -> float:
        cfg = self.config.get("inference", self.config.get("uncertainty", {}))
        return float(cfg.get("overlap", 0.25))

    def _existing_or_save_ground_truth(
        self,
        ground_truth: np.ndarray,
        epoch_ground_truth: Path,
        case_id: str,
    ) -> str:
        gt_path = epoch_ground_truth / f"{case_id}_gt.npy"
        if gt_path.exists():
            return str(gt_path)
        return save_array(ground_truth, gt_path)

    def _resolve_non_tta_paths(self, epoch_tag: str, case_id: str) -> tuple[str, str]:
        """Return paths to non-TTA probability / embedding files if they exist."""
        prob_path = self.probabilities_dir / epoch_tag / f"{case_id}_probs.npy"
        emb_path = self.embeddings_dir / epoch_tag / f"{case_id}_embedding.npy"
        return str(prob_path) if prob_path.exists() else "", str(emb_path) if emb_path.exists() else ""

    def _write_metrics(self, rows: list[dict], epoch: int) -> None:
        """Append validation metrics to outputs/metrics.csv."""
        metrics_path = self.output_dir / "metrics.csv"
        file_exists = metrics_path.exists()

        fieldnames = [
            "epoch",
            "case_id",
            "dice",
            "path_ground_truth",
            "path_prediction",
            "path_probability",
            "path_embedding",
        ]

        with metrics_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)

        epoch_metrics_path = self.output_dir / f"metrics_{epoch:03d}.csv"
        pd.DataFrame(rows).to_csv(epoch_metrics_path, index=False)

    def _write_tta_metrics(self, rows: list[dict], epoch: int) -> None:
        """Append TTA metrics (input for analyze_failures.py)."""
        metrics_path = self.output_dir / "metrics_uncertainty.csv"
        file_exists = metrics_path.exists()

        fieldnames = [
            "epoch",
            "case_id",
            "dice",
            "path_ground_truth",
            "path_prediction",
            "path_probability",
            "path_embedding",
            "path_tta_probability",
            "path_entropy",
        ]

        with metrics_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)

        epoch_metrics_path = self.output_dir / f"metrics_uncertainty_{epoch:03d}.csv"
        pd.DataFrame(rows).to_csv(epoch_metrics_path, index=False)

    def _save_checkpoint(self, epoch: int) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, path)
        torch.save(checkpoint, self.checkpoint_dir / "checkpoint_latest.pt")
        print(f"Saved checkpoint: {path}")
