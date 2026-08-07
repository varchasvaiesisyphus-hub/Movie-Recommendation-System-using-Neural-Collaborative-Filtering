"""
evaluate.py

Evaluation pipeline for the trained NeuMF model.

WHY THIS MODULE EXISTS:
Training only tracks loss/accuracy, which is insufficient to judge a
recommender system. This module computes two complementary families of
metrics:

  1. Classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC,
     Confusion Matrix) -- measure how well the model distinguishes
     positive from negative interactions on the held-out test set.

  2. Ranking metrics (Precision@K, Recall@K, Hit Rate@K) -- measure how
     well the model performs at its actual downstream task: ranking a
     small set of candidate movies so that ones the user will like appear
     near the top. This uses the standard leave-one-out negative sampling
     protocol (for each test-positive interaction, rank the true item
     among a fixed number of unseen, randomly sampled negatives).
"""

import json
import os
import pickle
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from config import data_cfg, eval_cfg, paths, train_cfg
from src.dataset import get_dataloaders
from src.model import NeuMF


def load_model_from_checkpoint(checkpoint_path: str, device: str) -> NeuMF:
    """
    Reconstruct a NeuMF model from a saved checkpoint.

    WHY reconstruct from `model_config` rather than `config.py` defaults:
    the checkpoint stores the exact architecture hyperparameters used at
    training time (see train.py's `save_checkpoint`), so evaluation always
    matches the trained weights even if config.py changes afterward.

    Args:
        checkpoint_path: path to a .pt checkpoint file.
        device: 'cuda' or 'cpu'.

    Returns:
        NeuMF: model loaded with trained weights, in eval mode.

    Raises:
        FileNotFoundError: if the checkpoint does not exist.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. Run train.py first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint["model_config"]

    model = NeuMF(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"[load_model_from_checkpoint] Loaded model from epoch "
        f"{checkpoint.get('epoch', '?')} (val_loss={checkpoint.get('val_loss', float('nan')):.4f})"
    )
    return model


def compute_classification_metrics(
    model: NeuMF, dataloader: DataLoader, device: str
) -> Dict[str, object]:
    """
    Compute standard binary classification metrics on the test set.

    Args:
        model: trained NeuMF model in eval mode.
        dataloader: test DataLoader.
        device: 'cuda' or 'cpu'.

    Returns:
        Dict[str, object]: dict containing accuracy, precision, recall,
        f1, roc_auc, confusion_matrix (as nested list), and raw arrays
        (y_true, y_pred_proba) for later plotting.
    """
    model.eval()
    all_labels: List[float] = []
    all_probas: List[float] = []

    with torch.no_grad():
        for user_ids, movie_ids, labels in dataloader:
            user_ids = user_ids.to(device)
            movie_ids = movie_ids.to(device)

            probas = model(user_ids, movie_ids).cpu().numpy()
            all_probas.extend(probas.tolist())
            all_labels.extend(labels.numpy().tolist())

    y_true = np.array(all_labels)
    y_proba = np.array(all_probas)
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    # Keep raw arrays for plotting (not included in the JSON report).
    metrics["_y_true"] = y_true
    metrics["_y_proba"] = y_proba
    metrics["_y_pred"] = y_pred

    return metrics


def evaluate_ranking_metrics(
    model: NeuMF,
    test_df: pd.DataFrame,
    user_positive_items: Dict[int, set],
    num_movies: int,
    device: str,
    top_k: int = None,
    num_negative_samples: int = None,
) -> Dict[str, float]:
    """
    Compute ranking metrics (Precision@K, Recall@K, Hit Rate@K) using the
    leave-one-out negative sampling protocol.

    For each positive interaction in the test set:
      1. Sample `num_negative_samples` movies the user has NOT positively
         interacted with anywhere in the full dataset.
      2. Rank the true (held-out) movie among these negatives by predicted
         score.
      3. Check whether the true movie appears in the top-K of that ranked
         list.

    WHY this protocol: ranking every user against the full movie catalog
    (thousands of items) for every test interaction is computationally
    expensive and, more importantly, does not reflect how a recommender
    is actually judged in practice -- what matters is whether the model
    ranks a genuinely relevant item above a sample of irrelevant ones.
    This is the standard evaluation protocol introduced in the NeuMF paper.

    Args:
        model: trained NeuMF model in eval mode.
        test_df: test split DataFrame with 'user_idx', 'movie_idx', 'label'.
        user_positive_items: mapping of user_idx -> set of all positively
            rated movie_idx (across train+val+test), used to ensure
            sampled negatives are genuinely unseen-positive items.
        num_movies: total number of unique encoded movies (for negative
            sampling range).
        device: 'cuda' or 'cpu'.
        top_k: cutoff K for Precision/Recall/Hit Rate. Defaults to
            `config.eval_cfg.top_k` (10).
        num_negative_samples: number of negatives sampled per positive
            test interaction. Defaults to `config.eval_cfg.num_negative_samples`.

    Returns:
        Dict[str, float]: {'precision_at_k': ..., 'recall_at_k': ...,
        'hit_rate_at_k': ...}
    """
    top_k = top_k if top_k is not None else eval_cfg.top_k
    num_negative_samples = (
        num_negative_samples if num_negative_samples is not None else eval_cfg.num_negative_samples
    )

    model.eval()
    positive_test_interactions = test_df[test_df["label"] == 1]

    hits = 0
    precisions = []
    recalls = []

    all_movie_ids = set(range(num_movies))
    rng = random.Random(data_cfg.random_seed)

    with torch.no_grad():
        for _, row in positive_test_interactions.iterrows():
            user_idx = int(row["user_idx"])
            true_movie_idx = int(row["movie_idx"])

            # WHY sample from movies NOT in the user's full positive set:
            # a negative that the user actually liked (just not in the
            # test row) would be a false negative, corrupting the metric.
            watched = user_positive_items.get(user_idx, set())
            candidate_pool = list(all_movie_ids - watched - {true_movie_idx})

            if len(candidate_pool) < num_negative_samples:
                sampled_negatives = candidate_pool
            else:
                sampled_negatives = rng.sample(candidate_pool, num_negative_samples)

            # Candidate list = true item + sampled negatives.
            candidate_movies = [true_movie_idx] + sampled_negatives
            user_tensor = torch.tensor([user_idx] * len(candidate_movies), dtype=torch.long).to(device)
            movie_tensor = torch.tensor(candidate_movies, dtype=torch.long).to(device)

            scores = model(user_tensor, movie_tensor).cpu().numpy()

            # Rank candidates by predicted score, descending.
            ranked_indices = np.argsort(-scores)
            ranked_movies = [candidate_movies[i] for i in ranked_indices]

            top_k_movies = ranked_movies[:top_k]
            is_hit = true_movie_idx in top_k_movies

            hits += int(is_hit)
            # WHY precision@k = 1/K when the single held-out relevant item
            # is retrieved: with exactly one true relevant item per query
            # (the leave-one-out protocol), precision@K counts how many
            # of the K slots are relevant (1 or 0 relevant out of K).
            precisions.append((1 if is_hit else 0) / top_k)
            # WHY recall@k = hit here: with exactly one relevant item per
            # query, recall is binary (found it or not).
            recalls.append(1 if is_hit else 0)

    num_queries = len(positive_test_interactions)
    return {
        "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "hit_rate_at_k": hits / num_queries if num_queries > 0 else 0.0,
        "top_k": top_k,
        "num_queries": num_queries,
    }


def plot_confusion_matrix(cm: List[List[int]], save_path: str) -> None:
    """
    Plot and save a confusion matrix heatmap.

    Args:
        cm: confusion matrix as a nested list, [[TN, FP], [FN, TP]].
        save_path: destination PNG path.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot_confusion_matrix] Saved to {save_path}")


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, roc_auc: float, save_path: str) -> None:
    """
    Plot and save the ROC curve.

    Args:
        y_true: ground-truth binary labels.
        y_proba: predicted probabilities.
        roc_auc: precomputed ROC-AUC score (displayed in the legend).
        save_path: destination PNG path.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})", color="darkorange")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot_roc_curve] Saved to {save_path}")


def plot_embedding_pca(model: NeuMF, save_path: str, sample_size: int = 1000) -> None:
    """
    Visualize movie embeddings (GMF branch) in 2D using PCA.

    WHY PCA on embeddings: reducing the 64-dimensional learned movie
    embedding space to 2 principal components lets us visually inspect
    whether the model has learned meaningful structure (e.g. clusters of
    similar movies) purely from collaborative interaction patterns, with
    no genre information used during training.

    Args:
        model: trained NeuMF model.
        save_path: destination PNG path.
        sample_size: number of movie embeddings to sample for plotting
            (avoids an unreadably dense scatter plot for large catalogs).
    """
    movie_embeddings = model.gmf.movie_embedding.weight.detach().cpu().numpy()

    num_movies_total = movie_embeddings.shape[0]
    if num_movies_total > sample_size:
        rng = np.random.default_rng(data_cfg.random_seed)
        sampled_indices = rng.choice(num_movies_total, size=sample_size, replace=False)
        movie_embeddings = movie_embeddings[sampled_indices]

    pca = PCA(n_components=2, random_state=data_cfg.random_seed)
    reduced = pca.fit_transform(movie_embeddings)

    plt.figure(figsize=(7, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.5, s=10, color="steelblue")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Visualization of Learned Movie Embeddings (GMF branch)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot_embedding_pca] Saved to {save_path}")


def run_evaluation_pipeline() -> Dict[str, object]:
    """
    Execute the full evaluation pipeline: load the best checkpoint,
    compute classification metrics, compute ranking metrics, generate
    all required plots, and persist a JSON report.

    Returns:
        Dict[str, object]: combined classification + ranking metrics
        (JSON-serializable, raw arrays excluded).
    """
    device = train_cfg.device
    print(f"[run_evaluation_pipeline] Using device: {device}")

    model = load_model_from_checkpoint(paths.best_model_path, device)

    print("[run_evaluation_pipeline] Loading DataLoaders...")
    _, _, test_loader = get_dataloaders()

    print("[run_evaluation_pipeline] Computing classification metrics...")
    classification_metrics = compute_classification_metrics(model, test_loader, device)

    print("[run_evaluation_pipeline] Computing ranking metrics (Precision@K, Recall@K, Hit Rate@K)...")
    test_df = pd.read_csv(paths.test_file)

    with open(paths.user_positive_items_file, "rb") as f:
        user_positive_items = pickle.load(f)

    num_movies = model.num_movies
    ranking_metrics = evaluate_ranking_metrics(
        model, test_df, user_positive_items, num_movies, device
    )

    print("[run_evaluation_pipeline] Generating plots...")
    plot_confusion_matrix(
        classification_metrics["confusion_matrix"],
        os.path.join(paths.figures_dir, "confusion_matrix.png"),
    )
    plot_roc_curve(
        classification_metrics["_y_true"],
        classification_metrics["_y_proba"],
        classification_metrics["roc_auc"],
        os.path.join(paths.figures_dir, "roc_curve.png"),
    )
    plot_embedding_pca(model, os.path.join(paths.figures_dir, "embedding_pca.png"))

    # Assemble final JSON-serializable report (drop raw arrays).
    report = {
        "accuracy": classification_metrics["accuracy"],
        "precision": classification_metrics["precision"],
        "recall": classification_metrics["recall"],
        "f1_score": classification_metrics["f1_score"],
        "roc_auc": classification_metrics["roc_auc"],
        "confusion_matrix": classification_metrics["confusion_matrix"],
        "precision_at_k": ranking_metrics["precision_at_k"],
        "recall_at_k": ranking_metrics["recall_at_k"],
        "hit_rate_at_k": ranking_metrics["hit_rate_at_k"],
        "top_k": ranking_metrics["top_k"],
        "num_ranking_queries": ranking_metrics["num_queries"],
    }

    with open(paths.metrics_report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[run_evaluation_pipeline] Metrics report saved to {paths.metrics_report_file}")
    print(json.dumps(report, indent=2))

    return report


if __name__ == "__main__":
    run_evaluation_pipeline()
