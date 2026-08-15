"""Training loop and validation-Dice early stopping."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from .metrics import evaluate_segmenter


@dataclass
class EarlyStopping:
    patience: int = 10
    min_delta: float = 1e-4
    best_score: float | None = None
    best_epoch: int | None = None
    bad_epochs: int = 0

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError("patience must be at least 1")
        if self.min_delta < 0:
            raise ValueError("min_delta cannot be negative")

    def step(self, score: float, epoch: int, save_best: Callable[[], None]) -> bool:
        """Save improvements and return whether training should stop."""
        if not math.isfinite(score):
            raise ValueError(f"Validation score must be finite, got {score}")
        improved = self.best_score is None or score > self.best_score + self.min_delta
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.bad_epochs = 0
            save_best()
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def train_one_epoch(model, optimizer, data_loader, device) -> float:
    import torch

    model.train()
    losses: list[float] = []
    for images, targets in data_loader:
        images = [image.to(device) for image in images]
        model_targets = [
            {
                key: value.to(device)
                for key, value in target.items()
                if isinstance(value, torch.Tensor)
            }
            for target in targets
        ]
        loss_parts = model(images, model_targets)
        loss = sum(loss_parts.values())
        value = float(loss.detach().item())
        if not math.isfinite(value):
            raise RuntimeError(f"Non-finite training loss: {value}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(value)
    if not losses:
        raise ValueError("Training loader is empty")
    return sum(losses) / len(losses)


def train_model(
    model,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    device,
    *,
    epochs: int,
    checkpoint_path: str | Path,
    patience: int = 10,
    min_delta: float = 1e-4,
    score_threshold: float = 0.5,
    mask_threshold: float = 0.5,
) -> tuple[pd.DataFrame, EarlyStopping]:
    import torch

    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    stopper = EarlyStopping(patience=patience, min_delta=min_delta)
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        summary, _ = evaluate_segmenter(
            model,
            val_loader,
            device,
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
        )
        values = summary.set_index("metric")["macro_mean"].to_dict()
        scheduler.step()

        def save_best() -> None:
            torch.save(
                {
                    "epoch": epoch,
                    "val_dice": values["dice"],
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                },
                checkpoint_path,
            )

        should_stop = stopper.step(values["dice"], epoch, save_best)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "learning_rate": learning_rate,
                **{f"val_{name}": values[name] for name in values},
            }
        )
        print(
            f"epoch={epoch}/{epochs} train_loss={train_loss:.5f} "
            f"val_dice={values['dice']:.5f} best={stopper.best_score:.5f}"
        )
        if should_stop:
            print(f"Early stopping after {stopper.bad_epochs} epochs without improvement")
            break

    if not checkpoint_path.exists():
        raise RuntimeError("Training finished without writing a best checkpoint")
    best = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    print(f"Restored best checkpoint from epoch {best['epoch']} (Dice={best['val_dice']:.5f})")
    return pd.DataFrame(history), stopper
