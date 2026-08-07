"""
config.py

Centralized configuration module for the NeuMF Movie Recommendation project.

WHY THIS FILE EXISTS:
Hardcoding paths and hyperparameters across multiple modules (preprocessing,
dataset, model, train, evaluate, recommend) leads to duplication and makes
the project error-prone to modify. By centralizing all configuration here,
every other module imports a single source of truth, ensuring consistency
(e.g. the same embedding dimension used to build the model is the same one
used to load a checkpoint at inference time).
"""

import os
from dataclasses import dataclass, field
from typing import List

import torch

# ---------------------------------------------------------------------------
# BASE DIRECTORY
# ---------------------------------------------------------------------------
# WHY: All other paths are derived relative to the project root so the
# project remains portable across machines (no absolute paths baked in).
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class PathConfig:
    """
    Holds all filesystem paths used throughout the project.

    WHY frozen=True: paths should not be mutated at runtime; treating them
    as immutable prevents accidental reassignment bugs across modules.
    """

    base_dir: str = BASE_DIR

    # Raw MovieLens 1M source files (user must place .dat files here)
    raw_data_dir: str = os.path.join(BASE_DIR, "data", "raw")
    ratings_file: str = os.path.join(raw_data_dir, "ratings.dat")
    movies_file: str = os.path.join(raw_data_dir, "movies.dat")
    users_file: str = os.path.join(raw_data_dir, "users.dat")

    # Processed / split datasets produced by preprocessing.py
    processed_data_dir: str = os.path.join(BASE_DIR, "data", "processed")
    train_file: str = os.path.join(processed_data_dir, "train.csv")
    val_file: str = os.path.join(processed_data_dir, "val.csv")
    test_file: str = os.path.join(processed_data_dir, "test.csv")
    movies_processed_file: str = os.path.join(processed_data_dir, "movies_processed.csv")
    user_encoder_file: str = os.path.join(processed_data_dir, "user_encoder.pkl")
    movie_encoder_file: str = os.path.join(processed_data_dir, "movie_encoder.pkl")
    user_positive_items_file: str = os.path.join(processed_data_dir, "user_positive_items.pkl")

    # Model checkpoints
    saved_models_dir: str = os.path.join(BASE_DIR, "saved_models")
    best_model_path: str = os.path.join(saved_models_dir, "neumf_best.pt")
    last_model_path: str = os.path.join(saved_models_dir, "neumf_last.pt")

    # Reports (metrics, JSON/CSV summaries)
    reports_dir: str = os.path.join(BASE_DIR, "reports")
    metrics_report_file: str = os.path.join(reports_dir, "evaluation_metrics.json")
    training_history_file: str = os.path.join(reports_dir, "training_history.csv")

    # Figures (plots)
    figures_dir: str = os.path.join(BASE_DIR, "figures")


@dataclass(frozen=True)
class DataConfig:
    """
    Configuration governing dataset construction and the implicit
    feedback labeling rule.
    """

    # WHY 4: The project spec defines positive interaction as rating >= 4,
    # which reflects genuine user satisfaction rather than mere exposure.
    positive_rating_threshold: int = 4

    # WHY these ratios: an 80/10/10 split gives enough training data for a
    # deep model while reserving statistically meaningful validation/test
    # sets for early stopping and unbiased final evaluation.
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # WHY fixed seed: guarantees reproducible splits across runs, which is
    # essential for fair comparison of model checkpoints and for grading.
    random_seed: int = 42

    # MovieLens .dat files use "::" as a field separator and are encoded
    # in latin-1 (movie titles contain accented characters not valid UTF-8).
    dat_separator: str = "::"
    dat_encoding: str = "latin-1"


@dataclass(frozen=True)
class ModelConfig:
    """
    Hyperparameters defining the NeuMF architecture.
    """

    # WHY 64: a moderate embedding size balances representational capacity
    # against overfitting risk on ~1M interactions with ~6000 users/~4000 movies.
    embedding_dim: int = 64

    # WHY this funnel shape [128 -> 64 -> 32]: progressively compressing the
    # concatenated user/item MLP embeddings forces the network to learn
    # increasingly abstract interaction features, a common and effective
    # design in NeuMF literature (He et al., 2017).
    mlp_layers: List[int] = field(default_factory=lambda: [128, 64, 32])

    dropout: float = 0.2

    # Populated at runtime by preprocessing.py after encoding; placeholders
    # here since the exact counts depend on the dataset.
    num_users: int = 0
    num_movies: int = 0


@dataclass(frozen=True)
class TrainConfig:
    """
    Hyperparameters and settings governing the training loop.
    """

    optimizer: str = "adam"
    learning_rate: float = 0.001
    loss_fn: str = "bce"  # Binary Cross Entropy, matching the sigmoid output
    epochs: int = 20
    batch_size: int = 256

    # WHY early stopping: prevents overfitting once validation loss stops
    # improving, saving compute and producing a more generalizable model.
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4

    # WHY a scheduler: reducing the learning rate when validation loss
    # plateaus helps the optimizer escape shallow local minima with finer
    # gradient steps in later epochs.
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 2

    num_workers: int = 2

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class EvalConfig:
    """
    Configuration for ranking-based evaluation metrics.
    """

    # WHY 10: Top-10 recommendation lists are the de facto standard in
    # recommender systems literature for Precision@K / Recall@K / Hit Rate@K.
    top_k: int = 10

    # WHY 99: for each positive test interaction we sample 99 movies the
    # user has NOT interacted with, and rank the true item among them
    # (the standard "leave-one-out" negative sampling protocol used to
    # keep ranking evaluation computationally tractable on large catalogs).
    num_negative_samples: int = 99


# ---------------------------------------------------------------------------
# INSTANTIATED CONFIG OBJECTS
# ---------------------------------------------------------------------------
# WHY module-level singletons: other modules do
#   from config import paths, data_cfg, model_cfg, train_cfg, eval_cfg
# giving a clean, explicit import surface without re-instantiating dataclasses.
paths = PathConfig()
data_cfg = DataConfig()
model_cfg = ModelConfig()
train_cfg = TrainConfig()
eval_cfg = EvalConfig()


def ensure_directories() -> None:
    """
    Create all required project directories if they do not already exist.

    WHY: preprocessing, training, and evaluation all write files to disk
    (processed data, checkpoints, reports, figures). Calling this once at
    the start of main.py avoids scattering `os.makedirs` calls throughout
    the codebase and prevents runtime FileNotFoundError crashes.
    """
    directories = [
        paths.raw_data_dir,
        paths.processed_data_dir,
        paths.saved_models_dir,
        paths.reports_dir,
        paths.figures_dir,
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


if __name__ == "__main__":
    # Quick sanity check when running `python config.py` directly.
    ensure_directories()
    print(f"Base directory: {paths.base_dir}")
    print(f"Device selected for training: {train_cfg.device}")
    print("All project directories verified/created successfully.")
