"""
train.py

Training pipeline for the NeuMF model.

WHY THIS MODULE EXISTS:
Model definition (model.py) and data loading (dataset.py) are separate
concerns from the actual optimization loop. This module orchestrates
training: it runs forward/backward passes, tracks train/validation loss,
persists the best-performing checkpoint, applies early stopping to avoid
wasted computation once the model stops improving, and produces a loss
curve plot for the final report.
"""

import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ensure_directories, model_cfg, paths, train_cfg
from src.dataset import get_dataloaders
from src.model import NeuMF


class EarlyStopping:
    """
    Tracks validation loss across epochs and signals when training should
    stop due to lack of improvement.

    WHY: without early stopping, training runs the full fixed number of
    epochs even after the model has started overfitting (validation loss
    rising while training loss keeps falling), wasting compute and
    yielding a worse final checkpoint than an earlier one.
    """

    def __init__(self, patience: int, min_delta: float) -> None:
        """
        Args:
            patience: number of consecutive non-improving epochs to
                tolerate before signaling a stop.
            min_delta: minimum decrease in validation loss to count as
                an improvement (guards against stopping on noise).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """
        Update internal state with the latest validation loss.

        Args:
            val_loss: validation loss for the current epoch.

        Returns:
            bool: True if this epoch's loss is a new best (i.e. the
            caller should save a checkpoint), False otherwise.
        """
        is_improvement = val_loss < (self.best_loss - self.min_delta)

        if is_improvement:
            self.best_loss = val_loss
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def train_one_epoch(
    model: NeuMF,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    epoch_num: int,
    total_epochs: int,
) -> float:
    """
    Run a single training epoch over the full training set.

    Args:
        model: the NeuMF model in training mode.
        dataloader: training DataLoader.
        optimizer: optimizer instance (Adam).
        criterion: loss function (BCELoss).
        device: 'cuda' or 'cpu'.
        epoch_num: current epoch index (1-based, for progress bar display).
        total_epochs: total number of epochs (for progress bar display).

    Returns:
        float: average training loss across all batches this epoch.
    """
    model.train()
    running_loss = 0.0
    num_samples = 0

    # WHY tqdm: gives visible, real-time feedback during potentially
    # long-running epochs over ~800K training interactions.
    progress_bar = tqdm(
        dataloader, desc=f"Epoch {epoch_num}/{total_epochs} [Train]", leave=False
    )

    for user_ids, movie_ids, labels in progress_bar:
        user_ids = user_ids.to(device)
        movie_ids = movie_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model(user_ids, movie_ids)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        num_samples += batch_size

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / num_samples


def validate_one_epoch(
    model: NeuMF,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    epoch_num: int,
    total_epochs: int,
) -> Tuple[float, float]:
    """
    Run a single validation epoch (no gradient updates).

    Args:
        model: the NeuMF model.
        dataloader: validation DataLoader.
        criterion: loss function (BCELoss).
        device: 'cuda' or 'cpu'.
        epoch_num: current epoch index (1-based, for progress bar display).
        total_epochs: total number of epochs (for progress bar display).

    Returns:
        Tuple[float, float]: (average validation loss, validation accuracy).
    """
    model.eval()
    running_loss = 0.0
    num_samples = 0
    correct_predictions = 0

    progress_bar = tqdm(
        dataloader, desc=f"Epoch {epoch_num}/{total_epochs} [Val]", leave=False
    )

    # WHY torch.no_grad(): validation does not need gradient tracking,
    # and disabling it reduces memory usage and speeds up the forward pass.
    with torch.no_grad():
        for user_ids, movie_ids, labels in progress_bar:
            user_ids = user_ids.to(device)
            movie_ids = movie_ids.to(device)
            labels = labels.to(device)

            predictions = model(user_ids, movie_ids)
            loss = criterion(predictions, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            num_samples += batch_size

            predicted_labels = (predictions >= 0.5).float()
            correct_predictions += (predicted_labels == labels).sum().item()

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / num_samples
    accuracy = correct_predictions / num_samples
    return avg_loss, accuracy


def save_checkpoint(model: NeuMF, path: str, epoch: int, val_loss: float) -> None:
    """
    Save a model checkpoint containing weights and architecture metadata.

    WHY save architecture config alongside weights: `evaluate.py` and
    `recommend.py` need to reconstruct an identical NeuMF instance before
    calling `load_state_dict`. Storing `model.get_config()` avoids
    depending on `config.py` still matching the exact hyperparameters
    used at training time.

    Args:
        model: the NeuMF model to save.
        path: destination file path.
        epoch: epoch number at which this checkpoint was saved.
        val_loss: validation loss at this checkpoint.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model.get_config(),
        "epoch": epoch,
        "val_loss": val_loss,
    }
    torch.save(checkpoint, path)


def plot_loss_curves(history: Dict[str, List[float]], save_path: str) -> None:
    """
    Plot and save training/validation loss curves.

    Args:
        history: dict with keys 'train_loss', 'val_loss', 'val_accuracy',
            each mapping to a list of per-epoch values.
        save_path: file path where the PNG plot will be saved.
    """
    epochs_range = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs_range, history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(epochs_range, history["val_loss"], label="Validation Loss", marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy Loss")
    axes[0].set_title("Training vs Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs_range, history["val_accuracy"], label="Validation Accuracy",
                 marker="o", color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy over Epochs")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot_loss_curves] Saved loss/accuracy curves to {save_path}")


def run_training_pipeline() -> Dict[str, List[float]]:
    """
    Execute the full training pipeline: data loading, model construction,
    the training/validation loop, checkpointing, early stopping, and
    result persistence.

    Returns:
        Dict[str, List[float]]: training history with keys 'train_loss',
        'val_loss', 'val_accuracy'.
    """
    ensure_directories()
    device = train_cfg.device
    print(f"[run_training_pipeline] Using device: {device}")

    print("[run_training_pipeline] Loading DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders()

    # WHY derive num_users/num_movies from the training data itself rather
    # than a hardcoded config value: the exact counts depend on how many
    # unique users/movies survived preprocessing, and using a mismatched
    # count would cause an index-out-of-range error in nn.Embedding.
    train_dataset = train_loader.dataset
    num_users = int(train_dataset.user_ids.max().item()) + 1
    num_movies = int(train_dataset.movie_ids.max().item()) + 1
    print(f"[run_training_pipeline] num_users={num_users}, num_movies={num_movies}")

    model = NeuMF(
        num_users=num_users,
        num_movies=num_movies,
        embedding_dim=model_cfg.embedding_dim,
        mlp_layers=model_cfg.mlp_layers,
        dropout=model_cfg.dropout,
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=train_cfg.learning_rate)

    # WHY ReduceLROnPlateau: automatically shrinks the learning rate once
    # validation loss stops improving, allowing finer optimization steps
    # in later epochs without manually scheduling decay points.
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=train_cfg.lr_scheduler_factor,
        patience=train_cfg.lr_scheduler_patience,
    )

    early_stopping = EarlyStopping(
        patience=train_cfg.early_stopping_patience,
        min_delta=train_cfg.early_stopping_min_delta,
    )

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(1, train_cfg.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, train_cfg.epochs
        )
        val_loss, val_accuracy = validate_one_epoch(
            model, val_loader, criterion, device, epoch, train_cfg.epochs
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{train_cfg.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        # Always keep a "last" checkpoint so training can be resumed or
        # inspected even if it was not the best epoch.
        save_checkpoint(model, paths.last_model_path, epoch, val_loss)

        is_best = early_stopping.step(val_loss)
        if is_best:
            save_checkpoint(model, paths.best_model_path, epoch, val_loss)
            print(f"  -> New best model saved (val_loss={val_loss:.4f})")

        if early_stopping.should_stop:
            print(
                f"[run_training_pipeline] Early stopping triggered at epoch {epoch} "
                f"(no improvement for {train_cfg.early_stopping_patience} epochs)."
            )
            break

    # Persist training history as CSV for reporting/reproducibility.
    history_path = paths.training_history_file
    with open(history_path, "w") as f:
        f.write("epoch,train_loss,val_loss,val_accuracy\n")
        for i in range(len(history["train_loss"])):
            f.write(
                f"{i + 1},{history['train_loss'][i]},{history['val_loss'][i]},"
                f"{history['val_accuracy'][i]}\n"
            )
    print(f"[run_training_pipeline] Training history saved to {history_path}")

    plot_path = os.path.join(paths.figures_dir, "training_curves.png")
    plot_loss_curves(history, plot_path)

    # Save a JSON summary for quick inspection.
    summary = {
        "best_val_loss": early_stopping.best_loss,
        "final_epoch": len(history["train_loss"]),
        "num_users": num_users,
        "num_movies": num_movies,
    }
    with open(os.path.join(paths.reports_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[run_training_pipeline] Training pipeline completed successfully.")
    return history


if __name__ == "__main__":
    run_training_pipeline()
