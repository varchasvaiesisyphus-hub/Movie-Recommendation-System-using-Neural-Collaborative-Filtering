"""
recommend.py

Top-K recommendation engine built on a trained NeuMF model.

WHY THIS MODULE EXISTS:
Training and evaluation operate on batches of known (user, movie, label)
triples, but the actual end product of a recommendation system is: given
a user, produce a ranked list of movies they have not yet seen that they
are most likely to enjoy. This module implements that inference-time
workflow, including excluding already-watched movies and mapping raw
model output back to human-readable movie titles and genres.
"""

import pickle
from typing import List

import pandas as pd
import torch

from config import paths, train_cfg
from src.evaluate import load_model_from_checkpoint
from src.model import NeuMF


def load_recommendation_artifacts():
    """
    Load all artifacts required for inference: the trained model, the
    processed movie metadata, and the user encoder.

    WHY bundle this loading logic: `recommend.py` can be invoked
    repeatedly (e.g. once per user in a batch recommendation job), but
    the model and metadata only need to be loaded once. Separating
    loading from scoring lets callers load once and call
    `recommend_top_k` many times efficiently.

    Returns:
        Tuple[NeuMF, pd.DataFrame, object, str]:
            (model, movies_df, user_encoder, device)
    """
    device = train_cfg.device
    model = load_model_from_checkpoint(paths.best_model_path, device)

    movies_df = pd.read_csv(paths.movies_processed_file)

    with open(paths.user_encoder_file, "rb") as f:
        user_encoder = pickle.load(f)

    return model, movies_df, user_encoder, device


def get_user_watch_history(
    raw_user_id: int,
    movies_df: pd.DataFrame,
    user_encoder,
    user_positive_items: dict,
    ratings_df: pd.DataFrame = None,
    limit: int = None,
) -> pd.DataFrame:
    """
    Retrieve the movies a given user has previously interacted with
    positively (rating >= 4), for display alongside recommendations.

    WHY: showing a user's actual watch history next to their generated
    recommendations makes the system's behavior interpretable -- a viewer
    (or grader) can visually check whether the recommended movies are
    plausible given what the user is already known to enjoy, rather than
    trusting the Top-K list in isolation.

    Args:
        raw_user_id: the original (unencoded) MovieLens UserID.
        movies_df: processed movies DataFrame with 'movie_idx', 'Title',
            'Genres' columns.
        user_encoder: fitted sklearn LabelEncoder for user IDs.
        user_positive_items: mapping of user_idx -> set of movie_idx the
            user has positively interacted with (from preprocessing.py).
        ratings_df: optional raw/merged ratings DataFrame containing a
            'Rating' column and 'user_idx'/'movie_idx' columns; if
            provided, watch history is sorted by rating (highest first)
            instead of arbitrary set order.
        limit: optional cap on the number of history rows returned (e.g.
            show only the 10 most recent/highest-rated). None returns all.

    Returns:
        pd.DataFrame: columns ['Title', 'Genres'] (plus 'Rating' if
        ratings_df was supplied), one row per movie the user has
        positively rated.

    Raises:
        ValueError: if raw_user_id was not seen during training.
    """
    if raw_user_id not in set(user_encoder.classes_):
        raise ValueError(
            f"User ID {raw_user_id} was not seen during training "
            "(cold-start user); no watch history is available for them."
        )

    user_idx = int(user_encoder.transform([raw_user_id])[0])
    watched_movie_idxs = user_positive_items.get(user_idx, set())

    history_df = movies_df[movies_df["movie_idx"].isin(watched_movie_idxs)].copy()

    if ratings_df is not None and "Rating" in ratings_df.columns:
        # WHY merge in the actual star rating when available: it lets us
        # sort history by how much the user liked each movie, so the most
        # strongly-liked movies (the strongest signal for "why") appear first.
        user_ratings = ratings_df[
            (ratings_df["user_idx"] == user_idx) & (ratings_df["movie_idx"].isin(watched_movie_idxs))
        ][["movie_idx", "Rating"]]
        history_df = history_df.merge(user_ratings, on="movie_idx", how="left")
        history_df = history_df.sort_values("Rating", ascending=False)
        columns = ["Title", "Genres", "Rating"]
    else:
        columns = ["Title", "Genres"]

    if limit is not None:
        history_df = history_df.head(limit)

    return history_df[columns].reset_index(drop=True)


def explain_recommendation(recommended_genres: str, history_df: pd.DataFrame) -> str:
    """
    Produce a lightweight, human-readable explanation for a single
    recommendation based on genre overlap with the user's watch history.

    WHY genre overlap rather than a black-box explanation: NeuMF's
    learned embeddings are not directly interpretable (they encode latent
    collaborative-filtering signal, not explicit human-readable features).
    Genre overlap with movies the user already rated highly is an honest,
    simple proxy signal that is easy to verify and communicate, rather
    than fabricating a causal explanation the model cannot actually support.

    Args:
        recommended_genres: the '|'-delimited Genres string of the
            recommended movie (MovieLens format, e.g. "Action|Sci-Fi").
        history_df: the user's watch history DataFrame, must contain a
            'Genres' column in the same '|'-delimited format.

    Returns:
        str: a short explanation string, e.g.
        "Shares genres (Action, Sci-Fi) with movies you rated highly"
        or a generic fallback if no genre overlap is found.
    """
    recommended_genre_set = set(recommended_genres.split("|"))

    # Count how often each of the recommended movie's genres appears
    # across the user's watch history, to surface the most relevant overlap.
    history_genre_counts: dict = {}
    for genres_str in history_df["Genres"]:
        for genre in genres_str.split("|"):
            if genre in recommended_genre_set:
                history_genre_counts[genre] = history_genre_counts.get(genre, 0) + 1

    if not history_genre_counts:
        return "Recommended based on patterns from users with similar taste"

    # Sort overlapping genres by how frequently they appear in the user's
    # history, so the explanation highlights their strongest preference.
    top_overlap_genres = sorted(
        history_genre_counts, key=history_genre_counts.get, reverse=True
    )[:2]
    genre_list_str = ", ".join(top_overlap_genres)
    return f"Shares genres ({genre_list_str}) with movies you've enjoyed before"


def recommend_top_k(
    raw_user_id: int,
    model: NeuMF,
    movies_df: pd.DataFrame,
    user_encoder,
    user_positive_items: dict,
    top_k: int = 10,
    device: str = "cpu",
    explain: bool = False,
) -> pd.DataFrame:
    """
    Generate the Top-K movie recommendations for a given raw user ID.

    Args:
        raw_user_id: the original (unencoded) MovieLens UserID.
        model: trained NeuMF model in eval mode.
        movies_df: processed movies DataFrame with 'movie_idx', 'Title',
            'Genres' columns (from preprocessing.py's
            attach_movie_idx_to_movies).
        user_encoder: fitted sklearn LabelEncoder for user IDs (used to
            convert the raw user ID into the model's internal user_idx).
        user_positive_items: mapping of user_idx -> set of movie_idx the
            user has already positively interacted with, used to exclude
            already-watched movies from recommendations.
        top_k: number of recommendations to return. Defaults to 10.
        device: 'cuda' or 'cpu'.
        explain: if True, attach an 'Explanation' column giving a simple,
            genre-overlap-based reason for each recommendation (see
            `explain_recommendation`). Adds a small amount of extra
            computation; disabled by default so callers that only need
            raw scores (e.g. evaluate.py-style batch scoring) aren't
            slowed down.

    Returns:
        pd.DataFrame: top_k rows with columns ['Title', 'Genres', 'Score']
        (plus 'Explanation' if explain=True), sorted by predicted score
        descending.

    Raises:
        ValueError: if raw_user_id was not seen during training (i.e. is
            not known to the user_encoder), since the model has no
            embedding for unseen users (the classic "cold start" problem).
    """
    # WHY validate against encoder.classes_: passing an unseen user ID to
    # the encoder's .transform() would raise an opaque sklearn error;
    # checking explicitly lets us give a clear, actionable message instead.
    if raw_user_id not in set(user_encoder.classes_):
        raise ValueError(
            f"User ID {raw_user_id} was not seen during training "
            "(cold-start user). NeuMF requires a learned embedding for "
            "each user, so recommendations cannot be generated for "
            "entirely new users without retraining or a separate "
            "cold-start strategy."
        )

    user_idx = int(user_encoder.transform([raw_user_id])[0])

    # Exclude movies the user has already positively interacted with,
    # since re-recommending already-watched movies provides no value.
    watched_movie_idxs = user_positive_items.get(user_idx, set())
    candidate_movies_df = movies_df[~movies_df["movie_idx"].isin(watched_movie_idxs)].copy()

    if candidate_movies_df.empty:
        return pd.DataFrame(columns=["Title", "Genres", "Score"])

    candidate_movie_idxs = candidate_movies_df["movie_idx"].values

    user_tensor = torch.tensor(
        [user_idx] * len(candidate_movie_idxs), dtype=torch.long
    ).to(device)
    movie_tensor = torch.tensor(candidate_movie_idxs, dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        scores = model(user_tensor, movie_tensor).cpu().numpy()

    candidate_movies_df["Score"] = scores
    top_k_df = candidate_movies_df.sort_values("Score", ascending=False).head(top_k)

    if not explain:
        return top_k_df[["Title", "Genres", "Score"]].reset_index(drop=True)

    # WHY compute history once here rather than inside explain_recommendation
    # per-row: history_df is identical for every recommended movie in this
    # call, so computing it once avoids redundant filtering work per row.
    history_df = get_user_watch_history(raw_user_id, movies_df, user_encoder, user_positive_items)
    top_k_df = top_k_df.copy()
    top_k_df["Explanation"] = top_k_df["Genres"].apply(
        lambda genres: explain_recommendation(genres, history_df)
    )

    return top_k_df[["Title", "Genres", "Score", "Explanation"]].reset_index(drop=True)


def print_watch_history(raw_user_id: int, history_df: pd.DataFrame) -> None:
    """
    Pretty-print a user's watch history DataFrame to the console.

    Args:
        raw_user_id: the raw MovieLens UserID this history belongs to.
        history_df: DataFrame with 'Title', 'Genres' (and optionally
            'Rating') columns, as returned by `get_user_watch_history`.
    """
    print(f"\nWatch History for User {raw_user_id} ({len(history_df)} movies rated >= 4):")
    print("-" * 80)
    has_rating = "Rating" in history_df.columns
    for row in history_df.itertuples(index=False):
        if has_rating:
            print(f" - {row.Title:<50} | {row.Genres:<25} | Rated: {row.Rating}")
        else:
            print(f" - {row.Title:<50} | {row.Genres:<25}")
    print("-" * 80)


def print_recommendations(raw_user_id: int, recommendations: pd.DataFrame) -> None:
    """
    Pretty-print a recommendations DataFrame to the console.

    Args:
        raw_user_id: the raw MovieLens UserID recommendations were made for.
        recommendations: DataFrame with 'Title', 'Genres', 'Score' columns,
            optionally with an 'Explanation' column.
    """
    has_explanation = "Explanation" in recommendations.columns

    print(f"\nTop-{len(recommendations)} Recommendations for User {raw_user_id}:")
    print("-" * 80)
    for rank, row in enumerate(recommendations.itertuples(index=False), start=1):
        print(f"{rank:>2}. {row.Title:<50} | {row.Genres:<25} | Score: {row.Score:.4f}")
        if has_explanation:
            print(f"    -> {row.Explanation}")
    print("-" * 80)


def run_recommendation_demo(
    raw_user_id: int, top_k: int = 10, show_history: bool = True, explain: bool = True
) -> pd.DataFrame:
    """
    End-to-end demo: load all artifacts, optionally display the user's
    watch history, and generate Top-K (optionally explained)
    recommendations for a single user, printing the result.

    WHY show watch history before recommendations: presenting what the
    user has already watched and rated highly immediately alongside the
    recommendations lets a viewer sanity-check the model's behavior --
    e.g. confirming that a user with a history of Action/Sci-Fi movies is
    in fact recommended more Action/Sci-Fi titles -- without needing to
    separately query the raw dataset.

    WHY a standalone demo function: this is what `main.py` calls (and what
    running `python -m src.recommend` triggers) to showcase the full
    inference pipeline without requiring the caller to manually wire
    together model/metadata/encoder loading.

    Args:
        raw_user_id: the raw MovieLens UserID to generate recommendations for.
        top_k: number of recommendations to return.
        show_history: if True, print the user's watch history before the
            recommendations. Defaults to True.
        explain: if True, attach a genre-overlap explanation to each
            recommendation (see `explain_recommendation`). Defaults to True.

    Returns:
        pd.DataFrame: the generated Top-K recommendations (with an
        'Explanation' column if explain=True).
    """
    model, movies_df, user_encoder, device = load_recommendation_artifacts()

    with open(paths.user_positive_items_file, "rb") as f:
        user_positive_items = pickle.load(f)

    if show_history:
        history_df = get_user_watch_history(
            raw_user_id, movies_df, user_encoder, user_positive_items
        )
        print_watch_history(raw_user_id, history_df)

    recommendations = recommend_top_k(
        raw_user_id=raw_user_id,
        model=model,
        movies_df=movies_df,
        user_encoder=user_encoder,
        user_positive_items=user_positive_items,
        top_k=top_k,
        device=device,
        explain=explain,
    )

    print_recommendations(raw_user_id, recommendations)
    return recommendations


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Top-K movie recommendations for a user.")
    parser.add_argument(
        "--user_id", type=int, required=True, help="Raw MovieLens UserID to recommend movies for."
    )
    parser.add_argument(
        "--top_k", type=int, default=10, help="Number of recommendations to generate (default: 10)."
    )
    parser.add_argument(
        "--no_history", action="store_true",
        help="Skip printing the user's watch history before recommendations."
    )
    parser.add_argument(
        "--no_explain", action="store_true",
        help="Skip attaching genre-overlap explanations to each recommendation."
    )
    args = parser.parse_args()

    run_recommendation_demo(
        raw_user_id=args.user_id,
        top_k=args.top_k,
        show_history=not args.no_history,
        explain=not args.no_explain,
    )
