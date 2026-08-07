"""
dataset.py

PyTorch Dataset and DataLoader utilities for the NeuMF model.

WHY THIS MODULE EXISTS:
`preprocessing.py` produces clean, encoded, split CSV files, but PyTorch's
training loop needs data delivered as batched tensors via a `DataLoader`.
This module bridges that gap: it wraps the processed CSVs in a
`torch.utils.data.Dataset` subclass and exposes a factory function that
builds train/val/test `DataLoader`s with consistent settings.
"""

from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from config import paths, train_cfg


class MovieLensDataset(Dataset):
    """
    PyTorch Dataset wrapping a processed MovieLens interactions split.

    Each sample corresponds to a single (user, movie) interaction and its
    implicit-feedback label, as produced by `preprocessing.py`.

    Attributes:
        user_ids (torch.Tensor): LongTensor of encoded user indices.
        movie_ids (torch.Tensor): LongTensor of encoded movie indices.
        labels (torch.Tensor): FloatTensor of binary labels (0.0 or 1.0).
    """

    def __init__(self, csv_path: str) -> None:
        """
        Load a processed interactions CSV into memory as tensors.

        WHY load fully into memory rather than lazy row-by-row reads:
        MovieLens 1M has ~1 million interactions, which comfortably fits
        in memory as three integer/float arrays. Loading upfront avoids
        the I/O overhead of re-parsing CSV rows on every __getitem__ call,
        which would otherwise make training I/O-bound.

        Args:
            csv_path: path to a CSV file with 'user_idx', 'movie_idx',
                and 'label' columns (as produced by preprocessing.py).

        Raises:
            FileNotFoundError: if csv_path does not exist.
            ValueError: if required columns are missing.
        """
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Processed dataset not found at '{csv_path}'. "
                "Run preprocessing.py (or main.py --stage preprocess) first."
            ) from exc

        required_columns = {"user_idx", "movie_idx", "label"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV at '{csv_path}' is missing required columns: {missing}"
            )

        # WHY dtype=long for IDs: nn.Embedding requires LongTensor indices.
        self.user_ids = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.movie_ids = torch.tensor(df["movie_idx"].values, dtype=torch.long)

        # WHY dtype=float for labels: BCELoss expects float targets that
        # match the float output of the model's sigmoid layer.
        self.labels = torch.tensor(df["label"].values, dtype=torch.float)

    def __len__(self) -> int:
        """
        Return the total number of interaction samples in this split.

        Returns:
            int: number of (user, movie, label) samples.
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve a single (user_id, movie_id, label) sample by index.

        Args:
            idx: sample index in range [0, len(self)).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                (user_id, movie_id, label), each a 0-dim tensor. PyTorch's
                default collate function stacks these into batched tensors
                of shape (batch_size,).
        """
        return self.user_ids[idx], self.movie_ids[idx], self.labels[idx]


def get_dataloaders(
    batch_size: int = None,
    num_workers: int = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Construct train/validation/test DataLoaders from processed CSV splits.

    WHY a single factory function: `train.py` and `evaluate.py` both need
    DataLoaders built with identical, consistent settings (batch size,
    shuffling behavior). Centralizing construction here avoids
    configuration drift between modules.

    Args:
        batch_size: batch size for all loaders. Defaults to
            `config.train_cfg.batch_size` if not provided.
        num_workers: number of subprocess workers for data loading.
            Defaults to `config.train_cfg.num_workers` if not provided.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]:
            (train_loader, val_loader, test_loader)
    """
    batch_size = batch_size if batch_size is not None else train_cfg.batch_size
    num_workers = num_workers if num_workers is not None else train_cfg.num_workers

    train_dataset = MovieLensDataset(paths.train_file)
    val_dataset = MovieLensDataset(paths.val_file)
    test_dataset = MovieLensDataset(paths.test_file)

    # WHY shuffle=True only for training: shuffling the training set each
    # epoch prevents the model from learning spurious order-dependent
    # patterns. Validation/test order is irrelevant to correctness and
    # keeping it fixed makes evaluation runs reproducible and comparable.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick sanity check when running `python -m src.dataset` directly.
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | "
          f"Test batches: {len(test_loader)}")

    sample_users, sample_movies, sample_labels = next(iter(train_loader))
    print(f"Sample batch shapes -> users: {sample_users.shape}, "
          f"movies: {sample_movies.shape}, labels: {sample_labels.shape}")
