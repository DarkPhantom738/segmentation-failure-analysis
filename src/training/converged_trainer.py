"""Train a 3D U-Net to convergence with development-set early stopping."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.brats_dataset import BraTSTrainingDataset, BraTSVolumeDataset
from src.data.converged_splits import case_list_hash, seed_output_dirname
from src.models.unet3d import DiceCrossEntropyLoss, build_model
from src.training.converged_metrics import development_dice_bundle
from src.training.inference import predict_segmentation, sliding_window_inference
from src.utils.io import ensure_dir


HISTORY_COLUMNS = [
    "epoch",
    "train_loss",
    "development_mean_foreground_dice",
    "development_necrotic_dice",
    "development_edema_dice",
    "development_enhancing_dice",
    "development_whole_tumor_dice",
    "learning_rate",
    "improved",
    "patience_counter",
]


@dataclass
class ConvergenceState:
    epoch: int
    best_epoch: int
    best_dev_mean_fg_dice: float
    patience_counter: int
    stopping_reason: str
    final_learning_rate: float
    runtime_seconds: float


class ConvergedTrainer:
    """Train with lightweight development validation and early stopping."""

    def __init__(
        self,
        *,
        config: dict,
        model_seed: int,
        train_cases: Sequence[str],
        development_cases: Sequence[str],
        final_evaluation_cases: Sequence[str],
        device: torch.device,
        smoke_test: bool = False,
    ) -> None:
        self.config = config
        self.model_seed = int(model_seed)
        self.train_cases = list(train_cases)
        self.development_cases = list(development_cases)
        self.final_evaluation_cases = list(final_evaluation_cases)
        self.final_evaluation_set = set(self.final_evaluation_cases)
        self.device = device
        self.smoke_test = smoke_test

        forbidden = self.final_evaluation_set & (
            set(self.train_cases) | set(self.development_cases)
        )
        if forbidden:
            raise ValueError(
                "Final-evaluation cases leaked into train/development: "
                f"{sorted(forbidden)[:5]}"
            )

        self.output_dir = ensure_dir(
            Path(config["output"]["root"]) / seed_output_dirname(self.model_seed)
        )
        self.checkpoint_dir = ensure_dir(self.output_dir / "checkpoints")
        self.history_path = self.output_dir / "training_history.csv"
        self.summary_path = self.output_dir / "convergence_summary.json"
        self.curve_path = self.output_dir / "training_curve.png"

        self.model = build_model(config).to(device)
        train_cfg = config["training"]
        self.criterion = DiceCrossEntropyLoss(
            dice_weight=train_cfg["dice_weight"],
            num_classes=config["model"]["num_classes"],
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )
        sched = train_cfg["scheduler"]
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode=sched["mode"],
            factor=sched["factor"],
            patience=sched["patience"],
            min_lr=sched["min_lr"],
        )

        self.validate_every = int(train_cfg["validate_every"])
        self.max_epochs = int(train_cfg["max_epochs"])
        self.early_stopping_patience = int(train_cfg["early_stopping_patience"])
        self.early_stopping_min_delta = float(train_cfg["early_stopping_min_delta"])

        self.best_dev_mean_fg_dice = float("-inf")
        self.best_epoch = 0
        self.patience_counter = 0
        self.start_epoch = 1
        self.history_rows: list[dict[str, Any]] = []

        self.train_loader = self._build_train_loader()
        self.development_loader = self._build_development_loader()
        self._assert_no_final_cases_in_loaders()

    def _data_paths(self) -> tuple[Path, Path | None]:
        data_cfg = self.config["data"]
        root = Path(data_cfg["root"])
        cache = Path(data_cfg["cache_dir"]) if data_cfg.get("cache_dir") else None
        return root, cache

    def _build_train_loader(self) -> DataLoader:
        data_cfg = self.config["data"]
        train_cfg = self.config["training"]
        root, cache = self._data_paths()
        dataset = BraTSTrainingDataset(
            case_ids=self.train_cases,
            data_root=root,
            modalities=data_cfg["modalities"],
            target_spacing=data_cfg["target_spacing"],
            percentile_clip=data_cfg["percentile_clip"],
            patch_size=data_cfg["patch_size"],
            patches_per_volume=data_cfg["patches_per_volume"],
            seed=self.model_seed,
            preload=bool(data_cfg.get("preload", False)),
            cache_dir=cache,
        )
        generator = torch.Generator()
        generator.manual_seed(self.model_seed)
        return DataLoader(
            dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            num_workers=train_cfg.get("num_workers", 0),
            pin_memory=torch.cuda.is_available(),
            generator=generator,
        )

    def _build_development_loader(self) -> DataLoader:
        data_cfg = self.config["data"]
        root, cache = self._data_paths()
        dataset = BraTSVolumeDataset(
            case_ids=self.development_cases,
            data_root=root,
            modalities=data_cfg["modalities"],
            target_spacing=data_cfg["target_spacing"],
            percentile_clip=data_cfg["percentile_clip"],
            preload=bool(data_cfg.get("preload", False)),
            cache_dir=cache,
        )
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

    def _assert_no_final_cases_in_loaders(self) -> None:
        train_ids = set(self.train_loader.dataset.case_ids)
        devel_ids = set(self.development_loader.dataset.case_ids)
        leaked = (train_ids | devel_ids) & self.final_evaluation_set
        if leaked:
            raise RuntimeError(
                f"Loader construction included final-evaluation cases: {sorted(leaked)[:5]}"
            )

    def _checkpoint_payload(self, epoch: int) -> dict[str, Any]:
        data_cfg = self.config["data"]
        return {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_dev_mean_fg_dice": self.best_dev_mean_fg_dice,
            "best_epoch": self.best_epoch,
            "patience_counter": self.patience_counter,
            "model_seed": self.model_seed,
            "data_split_seed": int(data_cfg["data_split_seed"]),
            "development_split_seed": int(data_cfg["development_split_seed"]),
            "config": self.config,
            "case_list_hashes": {
                "train": case_list_hash(self.train_cases),
                "development": case_list_hash(self.development_cases),
                "final_evaluation": case_list_hash(self.final_evaluation_cases),
            },
            "n_train": len(self.train_cases),
            "n_development": len(self.development_cases),
            "n_final_evaluation": len(self.final_evaluation_cases),
        }

    def _save_checkpoint(self, epoch: int, path: Path) -> None:
        torch.save(self._checkpoint_payload(epoch), path)

    def load_checkpoint(self, path: Path) -> int:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_dev_mean_fg_dice = float(
            checkpoint.get("best_dev_mean_fg_dice", float("-inf"))
        )
        self.best_epoch = int(checkpoint.get("best_epoch", 0))
        self.patience_counter = int(checkpoint.get("patience_counter", 0))
        return int(checkpoint.get("epoch", 0))

    def resume_from_latest(self) -> None:
        latest = self.checkpoint_dir / "checkpoint_latest.pt"
        if not latest.exists():
            raise FileNotFoundError(f"No latest checkpoint at {latest}")
        epoch = self.load_checkpoint(latest)
        self.start_epoch = epoch + 1
        if self.history_path.exists():
            self.history_rows = pd.read_csv(self.history_path).to_dict(orient="records")
        print(
            f"Resumed seed={self.model_seed} from epoch {epoch} "
            f"(best_dev={self.best_dev_mean_fg_dice:.4f}, patience={self.patience_counter})"
        )

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        running = 0.0
        n = 0
        progress = tqdm(
            self.train_loader,
            desc=f"seed={self.model_seed} train epoch {epoch}",
            leave=False,
        )
        for batch in progress:
            case_id = batch["case_id"][0] if isinstance(batch["case_id"], (list, tuple)) else batch["case_id"]
            if isinstance(case_id, (list, tuple)):
                case_id = case_id[0]
            if str(case_id) in self.final_evaluation_set:
                raise RuntimeError(f"Final-evaluation case loaded during training: {case_id}")

            images = batch["image"].to(self.device)
            targets = batch["segmentation"].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            logits, _, _ = self.model(images)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()
            running += float(loss.item())
            n += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")
        return running / max(n, 1)

    @torch.no_grad()
    def evaluate_development(self) -> dict[str, float]:
        """Sliding-window Dice on development cases only (no array dumps)."""
        self.model.eval()
        metrics: list[dict[str, float]] = []
        patch_size = self.config["data"]["patch_size"]
        num_classes = self.config["model"]["num_classes"]
        overlap = float(self.config.get("inference", {}).get("overlap", 0.25))

        for batch in tqdm(
            self.development_loader,
            desc=f"seed={self.model_seed} development",
            leave=False,
        ):
            case_id = batch["case_id"][0]
            if case_id in self.final_evaluation_set:
                raise RuntimeError(
                    f"Final-evaluation case loaded during development validation: {case_id}"
                )
            image = batch["image"][0]
            ground_truth = batch["segmentation"][0].cpu().numpy().astype(np.uint8)
            logits, _, _ = sliding_window_inference(
                model=self.model,
                image=image,
                patch_size=patch_size,
                num_classes=num_classes,
                overlap=overlap,
                device=self.device,
            )
            prediction = predict_segmentation(logits)
            metrics.append(development_dice_bundle(prediction, ground_truth))

        if not metrics:
            raise RuntimeError("Development loader produced zero cases.")

        keys = metrics[0].keys()
        return {k: float(np.mean([row[k] for row in metrics])) for k in keys}

    def _append_history(self, row: dict[str, Any]) -> None:
        self.history_rows.append(row)
        with self.history_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
            writer.writeheader()
            for item in self.history_rows:
                writer.writerow({k: item.get(k, "") for k in HISTORY_COLUMNS})

    def _write_curve(self) -> None:
        if not self.history_rows:
            return
        df = pd.DataFrame(self.history_rows)
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        ax1.plot(df["epoch"], df["train_loss"], color="#1f4e79", label="train loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Train loss", color="#1f4e79")
        ax2 = ax1.twinx()
        valid = df["development_mean_foreground_dice"].replace("", np.nan).astype(float)
        ax2.plot(
            df["epoch"],
            valid,
            color="#c45c26",
            marker="o",
            label="dev mean FG Dice",
        )
        ax2.set_ylabel("Development mean FG Dice", color="#c45c26")
        ax1.set_title(f"Converged training (seed={self.model_seed})")
        fig.tight_layout()
        fig.savefig(self.curve_path, dpi=140)
        plt.close(fig)

    def _write_summary(self, state: ConvergenceState) -> None:
        payload = {
            "model_seed": self.model_seed,
            "best_epoch": state.best_epoch,
            "stopping_epoch": state.epoch,
            "best_development_mean_foreground_dice": state.best_dev_mean_fg_dice,
            "final_learning_rate": state.final_learning_rate,
            "runtime_seconds": state.runtime_seconds,
            "stopping_reason": state.stopping_reason,
            "patience_counter": state.patience_counter,
            "n_train": len(self.train_cases),
            "n_development": len(self.development_cases),
            "n_final_evaluation": len(self.final_evaluation_cases),
            "case_list_hashes": {
                "train": case_list_hash(self.train_cases),
                "development": case_list_hash(self.development_cases),
                "final_evaluation": case_list_hash(self.final_evaluation_cases),
            },
            "absent_class_dice_rule": (
                "Per-class Dice uses eps-smoothed binary Dice over class masks. "
                "If a class is absent in both GT and prediction, Dice=1.0; "
                "if absent in GT but predicted, Dice~0; missing classes are never dropped "
                "from the three-class mean."
            ),
            "checkpoint_best": str(self.checkpoint_dir / "checkpoint_best.pt"),
            "checkpoint_latest": str(self.checkpoint_dir / "checkpoint_latest.pt"),
        }
        with self.summary_path.open("w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")

    def train(self) -> ConvergenceState:
        t0 = time.time()
        stopping_reason = "reached_max_epochs"
        last_epoch = self.start_epoch - 1

        for epoch in range(self.start_epoch, self.max_epochs + 1):
            last_epoch = epoch
            train_loss = self._train_epoch(epoch)
            print(
                f"[seed={self.model_seed}] Epoch {epoch}/{self.max_epochs} "
                f"train_loss={train_loss:.4f} lr={self._current_lr():.6g}"
            )

            should_validate = (
                epoch % self.validate_every == 0
                or epoch == self.max_epochs
                or (self.smoke_test and epoch >= 1)
            )

            if should_validate:
                dev = self.evaluate_development()
                mean_fg = float(dev["mean_foreground_dice"])
                # Scheduler tracks development mean FG Dice (maximize).
                self.scheduler.step(mean_fg)

                improved = mean_fg > (
                    self.best_dev_mean_fg_dice + self.early_stopping_min_delta
                )
                if improved:
                    self.best_dev_mean_fg_dice = mean_fg
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, self.checkpoint_dir / "checkpoint_best.pt")
                    print(
                        f"[seed={self.model_seed}] New best development mean FG Dice="
                        f"{mean_fg:.4f} at epoch {epoch}"
                    )
                else:
                    self.patience_counter += 1
                    print(
                        f"[seed={self.model_seed}] No improvement "
                        f"(dev={mean_fg:.4f}, best={self.best_dev_mean_fg_dice:.4f}, "
                        f"patience={self.patience_counter}/{self.early_stopping_patience})"
                    )

                self._append_history(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "development_mean_foreground_dice": mean_fg,
                        "development_necrotic_dice": float(dev["necrotic_dice"]),
                        "development_edema_dice": float(dev["edema_dice"]),
                        "development_enhancing_dice": float(dev["enhancing_dice"]),
                        "development_whole_tumor_dice": float(dev["whole_tumor_dice"]),
                        "learning_rate": self._current_lr(),
                        "improved": bool(improved),
                        "patience_counter": self.patience_counter,
                    }
                )
            else:
                self._append_history(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "development_mean_foreground_dice": "",
                        "development_necrotic_dice": "",
                        "development_edema_dice": "",
                        "development_enhancing_dice": "",
                        "development_whole_tumor_dice": "",
                        "learning_rate": self._current_lr(),
                        "improved": False,
                        "patience_counter": self.patience_counter,
                    }
                )

            self._save_checkpoint(epoch, self.checkpoint_dir / "checkpoint_latest.pt")

            if should_validate and self.patience_counter >= self.early_stopping_patience:
                stopping_reason = "early_stopping"
                print(
                    f"[seed={self.model_seed}] Early stopping at epoch {epoch} "
                    f"(best epoch={self.best_epoch})"
                )
                break

        # Snapshot run-end state before reloading weights. checkpoint_best.pt was
        # written at the best epoch (patience usually 0), so loading it would wipe
        # the actual early-stopping patience and the LR at stop time.
        stopping_patience = int(self.patience_counter)
        final_learning_rate = self._current_lr()
        best_epoch = int(self.best_epoch)
        best_dev = float(self.best_dev_mean_fg_dice)

        best_path = self.checkpoint_dir / "checkpoint_best.pt"
        if best_path.exists():
            self.load_checkpoint(best_path)
            print(
                f"[seed={self.model_seed}] Reloaded checkpoint_best.pt "
                f"(epoch={self.best_epoch}, dice={self.best_dev_mean_fg_dice:.4f})"
            )
        else:
            # Ensure best exists after smoke/tiny runs that always improved once.
            self._save_checkpoint(last_epoch, best_path)

        runtime = time.time() - t0
        state = ConvergenceState(
            epoch=last_epoch,
            best_epoch=best_epoch,
            best_dev_mean_fg_dice=best_dev,
            patience_counter=stopping_patience,
            stopping_reason=stopping_reason,
            final_learning_rate=final_learning_rate,
            runtime_seconds=runtime,
        )
        self._write_curve()
        self._write_summary(state)
        return state
